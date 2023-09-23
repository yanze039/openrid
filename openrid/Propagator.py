import openmm
from openmmtools import mcmc
from openmm import unit
from openmm.unit import MOLAR_GAS_CONSTANT_R


class _MTSLangevinIntegrator(openmm.MTSLangevinIntegrator):
    """Subclass of OpenMM LangevinIntegrator that exposes get/set methods for temperatures.
       This is required by ThermostatedIntegrator::restore_interface().
    """

    def getTemperature(self):
        return self.temperature
    
    def setTemperature(self, temperature):
        self.temperature = temperature
        self.setGlobalVariableByName('kT', temperature * MOLAR_GAS_CONSTANT_R)


class MTSLangevinDynamicsMove(mcmc.BaseIntegratorMove):
    """Multi-time step Langevin dynamics in a specified force group."""

    def __init__(self, timestep=1.0*unit.femtosecond, collision_rate=10.0/unit.picoseconds,
                 n_steps=1000, groups=None, reassign_velocities=False, constraint_tolerance=1e-8, **kwargs):
        super(MTSLangevinDynamicsMove, self).__init__(n_steps=n_steps,
                                                   reassign_velocities=reassign_velocities,
                                                   **kwargs)
        self.timestep = timestep
        self.collision_rate = collision_rate
        self.constraint_tolerance = constraint_tolerance
        self.groups = groups

    def apply(self, thermodynamic_state, sampler_state, context_cache=None):
        """Apply the Multi-time step Langevin dynamics MCMC move.

        This modifies the given sampler_state. The temperature of the
        thermodynamic state is used in Langevin dynamics.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The thermodynamic state to use to propagate dynamics.
        sampler_state : openmmtools.states.SamplerState
           The sampler state to apply the move to. This is modified.
        context_cache : openmmtools.cache.ContextCache
            Context cache to be used during propagation with the integrator.

        """
        super(MTSLangevinDynamicsMove, self).apply(thermodynamic_state, sampler_state,
                                                context_cache=context_cache)
    
    def __getstate__(self):
        # add attribute "groups"
        serialization = super(MTSLangevinDynamicsMove, self).__getstate__()
        serialization['timestep'] = self.timestep
        serialization['collision_rate'] = self.collision_rate
        serialization['constraint_tolerance'] = self.constraint_tolerance
        serialization['groups'] = self.groups
        return serialization

    def __setstate__(self, serialization):
        # add attribute "groups"
        super(MTSLangevinDynamicsMove, self).__setstate__(serialization)
        self.timestep = serialization['timestep']
        self.collision_rate = serialization['collision_rate']
        self.constraint_tolerance = serialization['constraint_tolerance']
        self.groups = serialization['groups']

    def _get_integrator(self, thermodynamic_state):
        """Implement BaseIntegratorMove._get_integrator()."""
        integrator = _MTSLangevinIntegrator(thermodynamic_state.temperature,
                                                     self.collision_rate, self.timestep, self.groups)
        integrator.setConstraintTolerance(self.constraint_tolerance)
        return integrator


class LangevinMiddleDynamicsMove(mcmc.BaseIntegratorMove):
    """Multi-time step Langevin dynamics in a specified force group."""

    def __init__(self, timestep=1.0*unit.femtosecond, frictionCoeff=10.0/unit.picoseconds,
                 n_steps=1000, reassign_velocities=False, constraint_tolerance=1e-8, **kwargs):
        super(LangevinMiddleDynamicsMove, self).__init__(n_steps=n_steps,
                                                   reassign_velocities=reassign_velocities,
                                                   **kwargs)
        self.timestep = timestep
        self.frictionCoeff = frictionCoeff
        self.constraint_tolerance = constraint_tolerance

    def apply(self, thermodynamic_state, sampler_state, context_cache=None):
        """Apply the Multi-time step Langevin dynamics MCMC move.

        This modifies the given sampler_state. The temperature of the
        thermodynamic state is used in Langevin dynamics.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The thermodynamic state to use to propagate dynamics.
        sampler_state : openmmtools.states.SamplerState
           The sampler state to apply the move to. This is modified.
        context_cache : openmmtools.cache.ContextCache
            Context cache to be used during propagation with the integrator.

        """
        super(LangevinMiddleDynamicsMove, self).apply(thermodynamic_state, sampler_state,
                                                context_cache=context_cache)
    
    def __getstate__(self):
        # add attribute "groups"
        serialization = super(LangevinMiddleDynamicsMove, self).__getstate__()
        serialization['timestep'] = self.timestep
        serialization['frictionCoeff'] = self.frictionCoeff
        serialization['constraint_tolerance'] = self.constraint_tolerance
        return serialization

    def __setstate__(self, serialization):
        # add attribute "groups"
        super(LangevinMiddleDynamicsMove, self).__setstate__(serialization)
        self.timestep = serialization['timestep']
        self.frictionCoeff = serialization['frictionCoeff']
        self.constraint_tolerance = serialization['constraint_tolerance']

    def _get_integrator(self, thermodynamic_state):
        """Implement BaseIntegratorMove._get_integrator()."""
        integrator = openmm.LangevinMiddleIntegrator(thermodynamic_state.temperature,
                                                     self.frictionCoeff, self.timestep)
        integrator.setConstraintTolerance(self.constraint_tolerance)
        return integrator
