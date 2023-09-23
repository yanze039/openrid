from openmmtools.constants import *
import openmm as mm


Force_Group_Index = {
        mm.HarmonicBondForce:     0,
        mm.HarmonicAngleForce:    0,
        mm.PeriodicTorsionForce:  0,
        mm.CustomTorsionForce:    0,
        mm.CMAPTorsionForce:      0,
        mm.CustomNonbondedForce:  0,
        mm.NonbondedForce:        0,
        mm.CMMotionRemover:       0,
        mm.MonteCarloBarostat:    0,
        "SlowForce":              1
}