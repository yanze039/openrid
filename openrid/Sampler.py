import os
import copy
from typing import List
import logging
import numpy as np
from pathlib import Path

import mdtraj as md
import parmed as pmd
import mpiplus

from openmmtools.multistate import ReplicaExchangeSampler
from openmmtools import constants
import openmmtools as mmtools
from openmmtools import utils
import openmm.app as app
from openmm import unit
from openmm.unit.quantity import Quantity
import openmm as mm


logger = logging.getLogger(__name__)


class ParallelTemperingSampler(ReplicaExchangeSampler):

    def create(self, thermodynamic_states, sampler_states: list, storage,
               temperatures=None, **kwargs):
        # allow different initial structures
        if not isinstance(thermodynamic_states, List):
            if isinstance(thermodynamic_states, mmtools.states.ThermodynamicState):
                thermodynamic_states = [thermodynamic_states]
            else:
                raise ValueError("ParallelTempering only accepts a list of ThermodynamicState objects!\n"
                                 "If you have already set temperatures in your list of states, please use the "
                                 "standard ReplicaExchange class with your list of states.")

        if temperatures is not None:
            logger.debug("Using provided temperatures")
            if len(thermodynamic_states) == 1:
                thermodynamic_states = [copy.deepcopy(thermodynamic_states[0]) for _ in range(len(temperatures))]
            else:
                assert len(thermodynamic_states) == len(temperatures), "Number of temperatures must match number of states" \
                f"\nfound {len(thermodynamic_states)} states and {len(temperatures)} temperatures"
            for state, temperature in zip(thermodynamic_states, temperatures):
                state.temperature = temperature
        else:
            for state in thermodynamic_states:
                assert state.temperature is not None, "Temperatures must be set in the ThermodynamicState objects"

        # Initialize replica-exchange simulation.
        super(ParallelTemperingSampler, self).create(thermodynamic_states, sampler_states, storage=storage, **kwargs)


    def _compute_replica_energies(self, replica_id):
        """Compute the energy for the replica at every temperature.

        Because only the temperatures differ among replicas, we replace the generic O(N^2)
        replica-exchange implementation with an O(N) implementation.

        """
        # Initialize replica energies for each thermodynamic state.
        energy_thermodynamic_states = np.zeros(self.n_states)
        energy_unsampled_states = np.zeros(len(self._unsampled_states))

        # Determine neighborhood
        state_index = self._replica_thermodynamic_states[replica_id]
        neighborhood = self._neighborhood(state_index)
        # Only compute energies over neighborhoods
        energy_neighborhood_states = energy_thermodynamic_states[neighborhood]  # Array, can be indexed like this
        neighborhood_thermodynamic_states = [self._thermodynamic_states[n] for n in neighborhood]  # List

        # Retrieve sampler states associated to this replica.
        sampler_state = self._sampler_states[replica_id]

        # Thermodynamic state differ only by temperatures.
        reference_thermodynamic_state = self._thermodynamic_states[0]

        # Get the context, any Integrator works.
        context, integrator = self.energy_context_cache.get_context(reference_thermodynamic_state)

        # Update positions and box vectors.
        sampler_state.apply_to_context(context)

        # Compute energy.
        reference_reduced_potential = reference_thermodynamic_state.reduced_potential(context)

        # Strip reference potential of reference state's beta.
        reference_beta = 1.0 / (constants.kB * reference_thermodynamic_state.temperature)
        reference_reduced_potential /= reference_beta

        # Update potential energy by temperature.
        for thermodynamic_state_id, thermodynamic_state in enumerate(neighborhood_thermodynamic_states):
            beta = 1.0 / (constants.kB * thermodynamic_state.temperature)
            energy_neighborhood_states[thermodynamic_state_id] = beta * reference_reduced_potential

        # Return the new energies.
        return energy_neighborhood_states, energy_unsampled_states
    
    @mmtools.utils.with_timer('Propagating all replicas')
    def _propagate_replicas(self):
        """Propagate all replicas."""
        # TODO  Report on efficiency of dynamics (fraction of time wasted to overhead).
        logger.debug("Propagating all replicas...")

        # Distribute propagation across nodes. Only node 0 will get all positions
        # and box vectors. The other nodes, only need the positions that they use
        # for propagation and computation of the energy matrix entries.
        propagated_states, replica_ids = mpiplus.distribute(self._propagate_replica, range(self.n_replicas),
                                                        send_results_to=0, sync_nodes=True)

        # Update all sampler states. For non-0 nodes, this will update only the
        # sampler states associated to the replicas propagated by this node.
        for replica_id, propagated_state in zip(replica_ids, propagated_states):
            self._sampler_states[replica_id].__setstate__(propagated_state, ignore_velocities=False)

        # Gather all MCMCMoves statistics. All nodes must have these up-to-date
        # since they are tied to the ThermodynamicState, not the replica.
        all_statistics = mpiplus.distribute(self._get_replica_move_statistics, range(self.n_replicas),
                                        send_results_to='all', sync_nodes=True)
        for replica_id in range(self.n_replicas):
            if len(all_statistics[replica_id]) > 0:
                thermodynamic_state_id = self._replica_thermodynamic_states[replica_id]
                self._mcmc_moves[thermodynamic_state_id].statistics = all_statistics[replica_id]
    
    @mpiplus.on_single_node(0, broadcast_result=True, sync_nodes=True)
    def _mix_replicas(self):
        """Attempt to swap replicas according to user-specified scheme."""
        logger.debug("Mixing replicas...")

        # Reset storage to keep track of swap attempts this iteration.
        self._n_accepted_matrix[:, :] = 0
        self._n_proposed_matrix[:, :] = 0

        # Perform swap attempts according to requested scheme.
        with utils.time_it('Mixing of replicas'):
            if self.replica_mixing_scheme == 'swap-neighbors':
                self._mix_neighboring_replicas()
            elif self.replica_mixing_scheme == 'swap-all':
                nswap_attempts = self.n_replicas**3
                # Try to use numba-accelerated mixing code if possible,
                # otherwise fall back to Python-accelerated code.
                try:
                    self._mix_all_replicas_numba(
                        nswap_attempts, self.n_replicas,
                        self._replica_thermodynamic_states, self._energy_thermodynamic_states,
                        self._n_accepted_matrix, self._n_proposed_matrix
                        )
                except (ValueError, ImportError) as e:
                    logger.warning(str(e))
                    self._mix_all_replicas(nswap_attempts)
            else:
                assert self.replica_mixing_scheme is None

        # Determine fraction of swaps accepted this iteration.
        n_swaps_proposed = self._n_proposed_matrix.sum()
        n_swaps_accepted = self._n_accepted_matrix.sum()
        swap_fraction_accepted = 0.0
        if n_swaps_proposed > 0:
            swap_fraction_accepted = n_swaps_accepted / n_swaps_proposed
        logger.debug("Accepted {}/{} attempted swaps ({:.1f}%)".format(n_swaps_accepted, n_swaps_proposed,
                                                                       swap_fraction_accepted * 100.0))
        return self._replica_thermodynamic_states


class RestrainedMDSampler(object):
    def __init__(
            self, 
            topology_file, 
            temperature, 
            cv_def,
            cv_func,
            restrain_force=mm.CustomTorsionForce,
            n_steps=50000,
            traj_interval=100,
            info_interval=1000,
            output_dir=Path("data"),
            energy_function_expression="0.5*k*min(dtheta, 2*pi-dtheta)^2; dtheta = abs(theta-theta0)",
            energy_function_global_parameters={"k": 500.0*unit.kilojoules_per_mole, "pi": np.pi},
            energy_function_per_torsion_parameters=["theta0"],
            usePBC=True,
            reset_box=True,
    ) -> None:
        
        self.cv_def = np.array(cv_def, dtype=np.int32)
        
        self.cv_func = cv_func
        self.topology_file = topology_file

        self.platform = mmtools.utils.get_fastest_platform()
        logger.info(f"using {self.platform.getName()}")
        if self.platform.getName() != "CUDA":
            logger.warning("Not using CUDA can decrease the performance of the sampler.")
            raise RuntimeError("SLURM Error, please use CUDA")
        if isinstance(temperature, Quantity):
            self.temperature = temperature
        else:
            self.temperature = temperature*unit.kelvin
        
        self.n_steps = n_steps
        self.traj_interval = traj_interval
        self.info_interval = info_interval
        self.energy_function_expression = energy_function_expression
        self.energy_function_global_parameters = energy_function_global_parameters
        self.energy_function_per_torsion_parameters = energy_function_per_torsion_parameters
        self.restrain_force = restrain_force
        self.usePBC = usePBC
        self.reset_box = reset_box
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        
    def create_bias_force(self, dihedral_values, dihedral_indices, dihedral_unit="radian"):
        bias_torsion = self.restrain_force(self.energy_function_expression)
        for parm, value in self.energy_function_global_parameters.items():
            bias_torsion.addGlobalParameter(parm, value)
        for parm in self.energy_function_per_torsion_parameters:
            bias_torsion.addPerTorsionParameter(parm)
        if self.usePBC:
            bias_torsion.setUsesPeriodicBoundaryConditions(True)
        
        for _, (dih, values) in enumerate(zip(dihedral_indices, dihedral_values)):
            if dihedral_unit == "degree":
                value_rad = values*np.pi/180
            elif dihedral_unit == "radian":
                value_rad = values
            else:
                RuntimeError("dihedral_unit must be either degree or radian.")
                exit(1)
            if value_rad < -np.pi:
                value_rad += 2*np.pi
            elif value_rad > np.pi:
                value_rad -= 2*np.pi
            bias_torsion.addTorsion(*[int(d) for d in dih], [value_rad*unit.radian])

        return bias_torsion

    def create_system(self, box_vectors):
        topology = app.GromacsTopFile(self.topology_file, periodicBoxVectors=box_vectors)
        system = topology.createSystem(nonbondedMethod=app.PME,
                                 nonbondedCutoff=0.9*unit.nanometer, constraints=app.HBonds)
        system.addForce(mm.MonteCarloBarostat(1*unit.atmosphere, self.temperature))
        return system, topology

    def run(self, conformers: List[str]):
        logger.info(f" >>> Running restrained MD on {len(conformers)} conformers.")
        mpiplus.distribute(self._run, conformers, send_results_to=None)

    def _run(self, conformer):
        logger.info(f"Running restrained MD on {conformer}.")
        simulation_tag = Path(conformer).stem
        CV_out = Path(self.output_dir)/self.name_universe["CV_out"].format(simulation_tag=simulation_tag)
        traj_out = str(Path(self.output_dir)/self.name_universe["traj_out"].format(simulation_tag=simulation_tag))
        if os.path.exists(CV_out):
            logger.info("Collective variable file exists, skip.")
            return 

        if str(conformer).endswith(".gro"):
            coordinate_odj = app.GromacsGroFile(str(conformer))
        else:
            NotImplementedError("Only .gro file is supported for now.")
            exit(1)
        system, topology = self.create_system(coordinate_odj.getPeriodicBoxVectors())
        positions = np.array(coordinate_odj.positions.value_in_unit(unit.nanometer))
        dihedral_positions = positions[self.cv_def.flatten().astype(int)].reshape(-1, 4, 3)
        dihedrals = self.cv_func(dihedral_positions)
        bias_torsion = self.create_bias_force(dihedrals, self.cv_def)
        system.addForce(bias_torsion)
        integrator = mm.LangevinMiddleIntegrator(self.temperature, 1/unit.picosecond, 2*unit.femtoseconds)
        simulation = app.Simulation(topology.topology, system, integrator, self.platform)
        simulation.context.setPositions(coordinate_odj.positions)
        simulation.context.setVelocitiesToTemperature(self.temperature)
        simulation.reporters.append(pmd.openmm.NetCDFReporter(traj_out, self.traj_interval, crds=True))
        logger.info(f"Propagate system for {self.n_steps} steps ...")
        simulation.minimizeEnergy()
        simulation.step(self.n_steps)
        
        # we need to finalize the reporters to close the files, otherwise MDTraj can't read them proporly.
        for reporter in simulation.reporters:
            try:
                reporter.finalize()
            except AttributeError:
                pass
        
        initial_CV = self.clac_dihedral_angles(conformer, self.cv_def)
        CV_values = self.clac_dihedral_angles(traj_out, self.cv_def, top=conformer)
        all_CV_values = np.concatenate((initial_CV.reshape(1, -1), CV_values), axis=0)
        np.savetxt(CV_out, all_CV_values)
        logger.info(f"Collective variable values saved to {CV_out}.")
        os.remove(traj_out)
    
    @property
    def name_universe(self):
        return {
            "traj_out": "{simulation_tag}.res.nc",
            "info_out": "{simulation_tag}.info.txt",
            "CV_out": "{simulation_tag}.CV.txt",
        }

    @staticmethod
    def clac_dihedral_angles(traj, indices, top=None):
        if top is not None:
            traj_obj = md.load(traj, top=top)
        else:
            traj_obj = md.load(traj)
        values = md.compute_dihedrals(traj_obj, indices)
        return values

