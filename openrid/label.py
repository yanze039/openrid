import os
import torch
from parmed import load_file
from pathlib import Path
import MDAnalysis as mda

import openmm.app as app
import openmm as mm
import openmm.unit as unit
import openmmtorch as ot
import openmmtools as mmtools
from openmmtools.multistate import MultiStateSampler, MultiStateReporter

from openrid.common import prep_dihedral
from openrid.Propagator import LangevinMiddleDynamicsMove


### prepare initial conformal
data = Path("../data/")
top = load_file(str(data/'topol.top'))
gro = load_file(str(data/'npt.gro'))
top.box = gro.box[:]

### Create Systems
system = top.createSystem(nonbondedMethod=app.PME,
        nonbondedCutoff=1*unit.nanometer, constraints=app.HBonds)
platform = mm.Platform.getPlatformByName('CUDA')


### Load model
u = mda.Universe(data/"npt.gro")
dih_index = prep_dihedral("../data/npt.gro")
my_model = torch.load("../model/model_99.pt")
print("jiting ...")
torch.jit.script(my_model).save('../data/model_scripted.pt')
torch_force = ot.TorchForce('../data/model_scripted.pt')
torch_force.setOutputsForces(True)
torch_force.setUsesPeriodicBoundaryConditions(True)


system.addForce(torch_force)
barostat = system.addForce(mm.MonteCarloBarostat(1*unit.atmosphere, 300*unit.kelvin))

### Create Simulation
thermodynamic_state = mmtools.states.ThermodynamicState(system, temperature=300*unit.kelvin)

sampler_states = mmtools.states.SamplerState(positions=gro.positions,box_vectors=system.getDefaultPeriodicBoxVectors())

move = LangevinMiddleDynamicsMove(timestep=2.0*unit.femtosecond, frictionCoeff=1.0/unit.picoseconds,
                 n_steps=500, reassign_velocities=True)
simulation = MultiStateSampler(mcmc_moves=move, number_of_iterations=10)

storage_path = "label.nc"
if os.path.exists(storage_path):
    os.remove(storage_path)

reporter = MultiStateReporter(storage_path, checkpoint_interval=2, analysis_particle_indices=dih_index.flatten())
temperatures = [300.0*unit.kelvin, 310.0*unit.kelvin]
simulation.create(thermodynamic_state, sampler_states, storage=reporter, temperatures=temperatures, n_temperatures=len(temperatures))

simulation.run()

