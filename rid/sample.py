import openmm.app as app
import openmm as mm
import openmm.unit as unit
from sys import stdout

# class BiasSampler():


inpcrd = app.AmberInpcrdFile('input.inpcrd')
prmtop = app.AmberPrmtopFile('input.prmtop', periodicBoxVectors=inpcrd.boxVectors)
forcefield = mm.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
system = forcefield.createSystem(prmtop.topology, nonbondedMethod=app.PME,
        nonbondedCutoff=1*unit.nanometer, constraints=app.HBonds)
integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.004*unit.picoseconds)
simulation = mm.Simulation(prmtop.topology, system, integrator)
simulation.context.setPositions(inpcrd.positions)
simulation.minimizeEnergy()
simulation.reporters.append(app.PDBReporter('output.pdb', 1000))
simulation.reporters.append(app.StateDataReporter(stdout, 1000, step=True,
        potentialEnergy=True, temperature=True))
simulation.step(10000)
    