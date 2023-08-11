
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


data = Path("../data/")
top = load_file(str(data/'topol.top'))
gro = load_file(str(data/'npt.gro'))
top.box = gro.box[:]

system = top.createSystem(nonbondedMethod=app.PME,
        nonbondedCutoff=1*unit.nanometer, constraints=app.HBonds)
barostat = system.addForce(mm.MonteCarloBarostat(1*unit.atmosphere, 300*unit.kelvin))
integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 2*unit.femtoseconds)
platform = mm.Platform.getPlatformByName('CUDA')
simulation = app.Simulation(top.topology, system, integrator, platform)

simulation.context.setPositions(gro.positions)
simulation.context.setVelocitiesToTemperature(300*unit.kelvin)


# Add the NNP to the system

simulation.minimizeEnergy(maxIterations=5000)
simulation.reporters.append(parmed.openmm.NetCDFReporter('../data/chignolin.nc', 1000, crds=True))
simulation.reporters.append(app.StateDataReporter(stdout, 1000, step=True,
        potentialEnergy=True, temperature=True, density=True))
simulation.step(10000)
    