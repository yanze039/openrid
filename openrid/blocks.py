import os
import time
import torch
import copy
from typing import List
import logging
from pathlib import Path
import numpy as np
import parmed as pmd
import sklearn.cluster as skcluster
from openrid.constants import Force_Group_Index
from openrid.Propagator import MTSLangevinDynamicsMove
from openrid.Sampler import ParallelTemperingSampler, RestrainedMDSampler
from openrid.colvar import calc_diherals

# openmm related
import openmm as mm
import openmmtools as mmtools
import openmmtorch as ot
from openmmtools.mcmc import LangevinDynamicsMove
import openmm.app as app
from openmm import unit
from openmm.unit.quantity import Quantity
import mpiplus

logger = logging.getLogger(__name__)


class Exploration(object):

    def __init__(
            self, 
            topology, 
            confs, 
            model_path,
            temperatures: List[Quantity], 
            pressure: Quantity, 
            timestep, 
            n_remd_iters,
            n_steps_per_iter, 
            reporter, 
            reporter_name, 
            checkpoint_interval,
            trust_lvl_1=None, 
            trust_lvl_2=None,
            platform="CUDA",
            groups=[(0,2), (1,1)], 
            collision_rate=1.0/unit.picoseconds,
            output_dir=Path("output"),
            reassign_velocities=True,
            analysis_particle_indices=None,
            resume=True
        ) -> None:

        self.output_dir = Path(output_dir)

        # create sampler states
        if isinstance(confs, str) or isinstance(confs, Path):
            confs_list = [confs]
        elif isinstance(confs, list):
            confs_list = confs
        else:
            raise TypeError("confs should be str or list of str")
        sampler_states = []
        coordinate_odj = None
        for conf in confs_list:
            
            if str(conf).endswith(".gro"):
                coordinate_odj = app.GromacsGroFile(str(conf))
            else:
                raise RuntimeError("confs must be a .gro file")
            sampler_state = mmtools.states.SamplerState(
                positions=coordinate_odj.getPositions(),box_vectors=coordinate_odj.getPeriodicBoxVectors()
            )
            sampler_states.append(sampler_state)
        
        # create system
        if isinstance(topology, str) or isinstance(topology, Path):
            if str(topology).endswith(".top"):
                assert coordinate_odj is not None, "coordinate_odj must be specified!"
                topology = app.GromacsTopFile(topology, periodicBoxVectors=coordinate_odj.getPeriodicBoxVectors())
            else:
                raise RuntimeError("topology must be a .top file")
        else:
            raise TypeError("topology should be str or Path")

        self._system = topology.createSystem(nonbondedMethod=app.PME,
                            nonbondedCutoff=0.9*unit.nanometer, constraints=app.HBonds)
        self._system.addForce(mm.MonteCarloBarostat(pressure, temperatures[0]))
        
        thermodynamic_state_list = []
        if model_path is None:
            move = LangevinDynamicsMove(timestep=timestep,
                             collision_rate=collision_rate, n_steps=n_steps_per_iter)
            thermodynamic_state_list = [
                mmtools.states.ThermodynamicState(self._system, temperature=temperatures[0])
            ]
        else:
            assert trust_lvl_1 is not None, "trust_lvl_1 must be specified!"
            assert trust_lvl_2 is not None, "trust_lvl_2 must be specified!"
            if isinstance(trust_lvl_1, float):
                trust_lvl_1 = [trust_lvl_1] * len(temperatures)
            if isinstance(trust_lvl_2, float):
                trust_lvl_2 = [trust_lvl_2] * len(temperatures)
            
            for ii in range(len(temperatures)):
                self.jitted_model_path = self.output_dir / (Path(model_path).stem + f"_state_{ii}_jitted.pt")
                self.jit_model(model_path, trust_lvl_1[ii], trust_lvl_2[ii], self.jitted_model_path)
                mpicomm = mpiplus.get_mpicomm()
                try:
                    mpicomm.barrier()
                except AttributeError:
                    pass
                torch_force = ot.TorchForce(str(self.jitted_model_path))
                torch_force.setOutputsForces(True)
                torch_force.setUsesPeriodicBoundaryConditions(False)
                system = copy.deepcopy(self._system)
                system.addForce(torch_force)
                system = self.set_force_group(system)
                assert groups is not None, "force groups must be specified!"
                thermodynamic_state_list.append(mmtools.states.ThermodynamicState(system, temperature=temperatures[ii]))
            move = MTSLangevinDynamicsMove(timestep=timestep, collision_rate=collision_rate,
                    n_steps=n_steps_per_iter, groups=groups, reassign_velocities=reassign_velocities)
        
        reporter_path = self.output_dir / reporter_name
        self._sampler = ParallelTemperingSampler(mcmc_moves=move, number_of_iterations=n_remd_iters, online_analysis_interval=None)
        self.analysis_particle_indices = np.array(analysis_particle_indices).flatten()
        self.reporter = reporter(reporter_path, checkpoint_interval=checkpoint_interval, analysis_particle_indices=self.analysis_particle_indices)
        self.max_attempt_number = 5
        if resume and os.path.exists(reporter_path):
            is_sucessful = False
            attempts = 0
            while attempts < self.max_attempt_number:
                try:
                    self._sampler = self._sampler.from_storage(self.reporter)
                    is_sucessful = True
                    break
                except FileNotFoundError:
                    attempts += 1
                    logger.info("Attemp to open storage files failed, try again.")
                    time.sleep(3)
                    continue
            if not is_sucessful:
                raise RuntimeError(f"Attempts to open file {str(reporter_path)} exceeds {self.max_attempt_number} times, raise an error.")
        else:
            self._sampler.create(thermodynamic_state_list, sampler_states, storage=self.reporter, temperatures=temperatures)
        
        
    def run(self):
        self._sampler.run()
    
    @mpiplus.on_single_node(0, sync_nodes=True)
    def jit_model(self, model_path, e0, e1, output_path):
        _model = torch.load(model_path)
        _model.eval()
        _model.set_e0(e0)
        _model.set_e1(e1)
        torch.jit.script(_model).save(output_path)
    
    def set_force_group(self, system):
        for force in system.getForces():
            if type(force) in Force_Group_Index:
                force.setForceGroup(Force_Group_Index[type(force)])
            else:
                assert type(force) == mm.Force
                force.setForceGroup(Force_Group_Index["SlowForce"])
        return system


class Selector(object):

    def __init__(
            self,
            topology,
            cv_def,
            name="",
            model_path = None,
            reporter = mmtools.multistate.MultiStateReporter, 
            n_cv = 18,
            model_devi_threshold = 0.1,
            n_cluaster_threshold = 8,
            n_cluster_lower_bound = 2,
            n_cluster_upper_bound = 4,
            distance_threshold = [0.01],
            threshold_mode="search",
            pick_mode="all",
            output_dir = Path("select"),
            overwrite = True,
        ) -> None:
        self.model_path = model_path
        if self.model_path is not None:
            logger.info(f" >>> Loading model from {model_path}")
            self.model = torch.load(self.model_path)
            self.model.eval()
        self.name = name
        # self.topology = pmd.load_file(topology)
        self.topology_file = topology
        if topology.endswith(".top"):
            self.topology = app.GromacsTopFile(topology)
        else:
            raise RuntimeError("topology must be a .top file")
        self.dihedral_calculator = calc_diherals
        self.reporter = reporter
        self.n_cv = n_cv
        self.cv_def = cv_def
        self.n_cluster_lower_bound = n_cluster_lower_bound
        self.n_cluster_upper_bound = n_cluster_upper_bound
        self.max_selection = self.n_cluster_upper_bound

        self.distance_threshold = distance_threshold
        self.distance_threshold_file = output_dir / "distance_threshold_{state}.txt"
        self.model_devi_threshold = model_devi_threshold
        self.threshold_mode = threshold_mode
        self.output_dir = Path(output_dir)
        self.n_cluaster_threshold = n_cluaster_threshold
        if not self.output_dir.exists():
            os.makedirs(self.output_dir, exist_ok=True)

        self.overwrite = overwrite
        self.mpicomm = mpiplus.get_mpicomm()
        self.pick_mode = pick_mode

        assert self.pick_mode in ["all", "model"], "pick_mode must be all or model, found: {}".format(self.pick_mode)
        assert self.threshold_mode in ["search", "fixed"], "threshold_mode must be search or fixed, found: {}".format(self.threshold_mode)

    def pick_by_model(self, storage):
        reporter = self.reporter(storage, open_mode='r')
        for state_index in range(reporter.n_states):
            self.pick_per_state(reporter, state_index)
    
    def select(self, storage):
        reporter = self.reporter(storage, open_mode='r')
        states_on_node = [s for s in range(reporter.n_states) if s % self.mpicomm.size == self.mpicomm.rank]
        for state_index in states_on_node:
            self.pick_per_state(reporter, state_index)
    
    def select_by_model(self, torsion):
        assert self.model is not None, "model must be specified"
        self.model.eval()
        # mean forces: [n_model, 1, n_frames, n_cv]
        mean_forces = self.model.get_mean_force_from_torsion(torsion.to(self.model.device)).detach().cpu()
        model_devi = (torch.mean( torch.var(mean_forces, dim=0), dim=-1 ) ** 0.5).flatten()
        selected_idx = torch.where(model_devi > self.model_devi_threshold )[0]
        return selected_idx
    
    def pick_per_state(self, reporter, state_index):

        check_point_index = reporter.read_checkpoint_iterations()
        
        tmp_cv_slice = reporter.read_sampler_states(0, analysis_particles_only=True)[state_index].positions.value_in_unit(unit.nanometer)
        tmp_cv_slice = tmp_cv_slice.reshape(self.n_cv, -1, 3)
        cv_def_dim = tmp_cv_slice.shape[1]

        all_positions = torch.empty((len(check_point_index), self.n_cv, cv_def_dim, 3))

        for ii in check_point_index:
            cv_sampler = reporter.read_sampler_states(ii, analysis_particles_only=True)[state_index]
            pos = torch.from_numpy(cv_sampler.positions.value_in_unit(unit.nanometer))
            all_positions[ii] = pos.reshape(self.n_cv, cv_def_dim, 3)
            
        dih = self.dihedral_calculator(all_positions.reshape(-1, 4, 3)).reshape((-1, self.n_cv))
 
        if self.pick_mode == "all":
            logger.info(" >>> select all conformers")
            selected_idx = torch.arange(len(dih))
        elif self.pick_mode == "model":
            logger.info(" >>> select conformers by model")
            selected_idx = self.select_by_model(dih)
        else:
            raise ValueError("pick_mode must be all or model")
        
        selection_rate = 100 * len(selected_idx) / len(dih)
        logger.info(f"Selection rate {selection_rate}%")
        if selection_rate < 3:
            logger.warn("Selection rate is too low, the simulation may almost converge or consider using a larger model_devi_threshold")
        selected_dih = dih[selected_idx]
        
        diff = selected_dih.reshape(1, -1, self.n_cv) - selected_dih.reshape(-1, 1, self.n_cv)
        diff[diff < -np.pi] += 2 * torch.pi
        diff[diff >  np.pi] -= 2 * torch.pi
        dist_map = torch.norm(diff, dim=-1)
        
        state_threshold = self.distance_threshold[state_index]
        if self.threshold_mode == "fixed":
            if len(dist_map) == 1:
                self.write_n_cluster(1, self.output_dir / f"n_cluster_{state_index}.txt")
                logger.info(" >>> only one conformer, skip clustering,")
                self.write_to_file(reporter, selected_idx[0], state_index)
                return
            elif len(dist_map) == 0:
                self.write_n_cluster(0, self.output_dir / f"n_cluster_{state_index}.txt")
                logger.info(" >>> no cluster found, skip clustering,")
                return
            else:
                logger.info(f" >>> distance_threshold for state {state_index}: {state_threshold}")
            cluster = skcluster.AgglomerativeClustering(n_clusters=None,
                                                linkage='average',
                                                metric='precomputed',
                                                distance_threshold=state_threshold
                                                )
            cluster.fit(dist_map)
            if len(np.unique(cluster.labels_)) > self.max_selection:
                cluster, state_threshold = self.search_distance_threshold(dist_map)
        
        elif self.threshold_mode == "search":
            cluster, state_threshold = self.search_distance_threshold(dist_map)
        else:
            raise ValueError("threshold_mode must be search or fixed")
            
        
        self.write_threshold(str(self.distance_threshold_file).format(state=state_index), state_threshold)

        self.output_dir.mkdir(exist_ok=True)
        self.write_n_cluster(len(np.unique(cluster.labels_)), self.output_dir / f"n_cluster_{state_index}.txt")
        for i_cluster in range(len(np.unique(cluster.labels_))):
            logger.info(f" >>> processing cluster: {i_cluster}")
            member_index = torch.tensor(np.where(cluster.labels_ == i_cluster)[0], dtype=torch.int)
            if len(member_index) < 3:
                cluster_center = member_index[0]
            else:
                member_dist_map = dist_map[member_index][:, member_index]
                rmsd_in_cluster = torch.sum(member_dist_map**2, dim=-1)
                cluster_center = member_index[torch.argmin(rmsd_in_cluster)].item()
            cluster_center = selected_idx[cluster_center].item()
            sampler_index = check_point_index[cluster_center]
            self.write_to_file(reporter, sampler_index, state_index)
            logger.info(f" >>> cluster {i_cluster} center: {sampler_index} at state {state_index} done")
    
    @staticmethod
    def write_n_cluster(n_cluster, output_path):
        with open(output_path, "w") as f:
            f.write(str(n_cluster))
    
    def write_to_file(self, reporter, sampler_index, state_index):
        positions = reporter.read_sampler_states(
                        sampler_index, analysis_particles_only=False
                    )[state_index].positions
        box_vectors = reporter.read_sampler_states(
                        sampler_index, analysis_particles_only=False
                    )[state_index].box_vectors
        pmd_topology = pmd.load_file(self.topology_file)
        pmd_topology.box_vectors = box_vectors
        pmd_topology.positions = positions
        out_conf = str(self.output_dir / f"conf_{str(self.name)}_{state_index}_{sampler_index}.gro")
        pmd_topology.save(out_conf, overwrite=self.overwrite)
        logger.info(f" >>> Saved to file {out_conf}")
    
    def write_threshold(self, output_path, value):
        with open(output_path, "w") as f:
            f.write(str(value))

    def search_distance_threshold(self, dist_map, init_threshold=0.01):
        if dist_map.shape[0] < self.n_cluster_lower_bound:
            logger.warn("n_cluster_lower_bound is larger than the number of conformers, set to the number of conformers"
                        f" n_cluster_lower_bound: {self.n_cluster_lower_bound} -> {(dist_map.shape[0])} "
                        "please consider using a smaller n_cluster_lower_bound")
            self.n_cluster_lower_bound = dist_map.shape[0]
        
        state_threshold = init_threshold
        n_cluster = 0
        cluster = None
        assert self.n_cluster_lower_bound is not None, "n_cluster_lower_bound must be specified"
        assert self.n_cluster_upper_bound is not None, "n_cluster_upper_bound must be specified"

        while n_cluster < self.n_cluster_lower_bound or n_cluster > self.n_cluster_upper_bound:
            cluster = skcluster.AgglomerativeClustering(n_clusters=None,
                                                linkage='average',
                                                metric='precomputed',
                                                distance_threshold=state_threshold
                                                )
            cluster.fit(dist_map)
            n_cluster = len(np.unique(cluster.labels_))
            logger.info(f" >>> n_cluster: {n_cluster}")
            if n_cluster < self.n_cluster_lower_bound:
                state_threshold *= 0.9
            elif n_cluster > self.n_cluster_upper_bound:
                state_threshold *= 1.1
            else:
                logger.info(f"distance_threshold: {state_threshold}")
                break

        assert cluster is not None, "cluster is None, something wrong"
        logger.info(f" >>> Final searching result: n_cluster: {n_cluster}")

        return cluster, state_threshold


class RestrainedMDLabeler():
    def __init__(
            self,
            topology_file, 
            conformers,
            cv_def,
            restraint_constant: Quantity,
            cv_func,
            temperature, 
            n_steps=50000,
            traj_interval=100,
            info_interval=1000,
            output_dir=Path("data"),
            label_data_file_name="data.raw.npy",
            reset_box=True
    ) -> None:
        self.sampler = RestrainedMDSampler(
            topology_file, 
            temperature, 
            cv_def=cv_def,
            cv_func=cv_func,
            restrain_force=mm.CustomTorsionForce,
            n_steps=n_steps,
            traj_interval=traj_interval,
            info_interval=info_interval,
            output_dir=output_dir,
            energy_function_expression="0.5*k*min(dtheta, 2*pi-dtheta)^2; dtheta = abs(theta-theta0)",
            energy_function_global_parameters={"k": restraint_constant, "pi": np.pi},
            energy_function_per_torsion_parameters=["theta0"],
            usePBC=True,
            reset_box=reset_box,
        ) 
        self.output_dir = output_dir
        self.cv_def = cv_def
        self.conformers = conformers
        self.restraint_constant = restraint_constant
        self.label_data_file_name = label_data_file_name

    def run(self):
        self.sampler.run(self.conformers)
        cv_outputs = self.collect_outputs()
        n_cv = len(self.cv_def)
        n_data = len(cv_outputs)
        labels = np.zeros((n_data, n_cv*2))
        for i, cv_output in enumerate(cv_outputs):
            cv_label_data = self.calc_mean_force(cv_output)
            labels[i] = cv_label_data
        self.label_data_file = self.output_dir/self.label_data_file_name
        np.save(self.label_data_file, labels)

    def collect_outputs(self):
        tags = [Path(s).stem for s in self.conformers]
        cv_outputs = [ Path(self.output_dir)/self.sampler.name_universe["CV_out"].format(simulation_tag=tag) for tag in tags]
        return cv_outputs

    def calc_mean_force(self, cv_output, tail=0.9):
        data = np.loadtxt(cv_output)
        centers = data[0,:]
        nframes = data.shape[0]
        for ii in range(1, nframes):
            current_angle = data[ii,:]
            angular_diff = current_angle - centers
            current_angle[angular_diff < -np.pi] += 2 * np.pi
            current_angle[angular_diff >= np.pi] -= 2 * np.pi
            data[ii,:] = current_angle

        start_f = int(nframes * (1-tail))
        avgins = np.average(data[start_f:, :], axis=0)

        diff = avgins - centers
        diff[diff < -np.pi] += 2 * np.pi
        diff[diff >=  np.pi] -= 2 * np.pi
        ff = np.multiply(np.float32(self.restraint_constant.value_in_unit(unit.kilojoules_per_mole / (unit.radian**2))), diff)
        cv_label_data = np.concatenate((centers, ff))

        return cv_label_data


if __name__ == "__main__":
    import netCDF4 as nc
    # data = nc.Dataset("/home/gridsan/ywang3/Project/rid_openmm/output/test.reporter.nc")
    reporter = mmtools.multistate.MultiStateReporter("/home/gridsan/ywang3/Project/rid_openmm/output/test.reporter.nc", open_mode='r')
    for ii in range(11):
        try:
            tmp_cv_slice = reporter.read_sampler_states(ii, analysis_particles_only=False)[0].positions.value_in_unit(unit.nanometer)
            print(f"found {ii}")
        except:
            print(ii, "can't find")

    # sele = Selector(
    #     model_list=["../data/model_0.pt", "../data/model_1.pt", "../data/model_2.pt", "../data/model_3.pt"],
    # )