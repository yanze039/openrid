
import openmmtorch as ot
import torch
from model import DihedralBias
import openmm.app as app
import openmm as mm
import openmm.unit as unit
from sys import stdout
from parmed import load_file
from parmed.openmm.reporters import NetCDFReporter
from parmed import unit as u
import parmed
from pathlib import Path
import MDAnalysis as mda
import numpy as np
from colvar import DihedralAngle
from common import prep_dihedral


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
# print(torch_force)
# help(torch_force)
# exit(0)
# torch_force.setProperty("useCUDAGraphs", "true")
# torch_force.setProperty("CUDAGraphWarmupSteps", "12")
system.addForce(torch_force)


### Create Simulation
barostat = system.addForce(mm.MonteCarloBarostat(1*unit.atmosphere, 300*unit.kelvin))
integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 2*unit.femtoseconds)
new_simulation = app.Simulation(top.topology, system, integrator, platform)
new_simulation.context.setPositions(gro.positions)
new_simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
new_simulation.minimizeEnergy(maxIterations=5000)
new_simulation.reporters.append(parmed.openmm.NetCDFReporter('../data/chignolin.nc', 5000, crds=True))
new_simulation.reporters.append(app.StateDataReporter(stdout, 5000, step=True,
        potentialEnergy=True, temperature=True, density=True))


### Benchmark Simulation
import time
start_time = time.time()

new_simulation.step(20000)
end_time = time.time()
print("simultsion time for 20000 steps: ", end_time - start_time, "s")
    