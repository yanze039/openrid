import torch
import torch.nn as nn
import math
from colvar import DihedralAngle
# from common import prep_dihedral
# import MDAnalysis as mda
from pathlib import Path
from typing import Dict, List, Tuple, Optional
# from torch import vmap


class MLP(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    """ we can concatenate multiple linear layers to form a multi-linear layer """
    def __init__(self, size_in, size_out, features=[64,64,64,64]):
        super().__init__()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.Sequence = nn.Sequential(
            nn.Linear(size_in, features[0]),
            self.activation,
            self.dropout,
            nn.Linear(features[0], features[1]),
            self.activation,
            self.dropout,
            nn.Linear(features[1], features[2]),
            self.activation,
            self.dropout,
            nn.Linear(features[2], features[3]),
            self.activation,
            self.dropout,
            nn.Linear(features[3], size_out)
        )

    def forward(self, torsion):
        # x: [N_sample, N_CVs] --> [N_sample, 1, N_CVs]
        output = self.Sequence(torsion)
        return output




class DihedralBias(nn.Module):
    """Free energy biasing potential."""

    def __init__(self, 
                 colvar_fn, colvar_idx, n_models, n_cvs, 
                 features=[64, 64, 64, 64],
                 e0=2,
                 e1=3
                 ):
        """Initialize the biasing potential.

            Parameters
            ----------
            colvar : torch.nn.Module
                  The collective variable to bias.
            kT : float
                  The temperature in units of energy.
            target : float
                  The target value of the collective variable.
            width : float
                  The width of the biasing potential.
        """
        super().__init__()
        self.colvar_fn = colvar_fn()
        self.colvar_idx = torch.from_numpy(colvar_idx)
        self.n_models = n_models
        self.n_cvs = n_cvs

        self.layer_norm = nn.LayerNorm(2*n_cvs)

        self.model1 = MLP(2 * n_cvs, 1, features=features)
        self.model2 = MLP(2 * n_cvs, 1, features=features)
        self.model3 = MLP(2 * n_cvs, 1, features=features)
        self.model4 = MLP(2 * n_cvs, 1, features=features)
        self.model_list = [self.model1, self.model2, self.model3, self.model4]
        self.e0 = e0
        self.e1 = e1
        # just calculate once
        self.e1_m_e0 = e1-e0
        self.loss_fn = nn.MSELoss()
            

    def forward(self, positions, boxvectors):
        """The forward method returns the energy computed from positions.

            Parameters
            ----------
            positions : torch.Tensor with shape (nparticles, 3)
            positions[i,k] is the position (in nanometers) of spatial dimension k of particle i

            Returns
            -------
            potential : torch.Scalar
            The potential energy (in kJ/mol)
        """
        positions.requires_grad_(True)
        boxsize = boxvectors.diag()
        positions = positions - torch.floor(positions/boxsize)*boxsize  # remove PBC
        selected_positions = positions[self.colvar_idx.flatten()].reshape([-1, 4, 3])  # [N_CVs, 4, 3]
        # calculate CVs
        cvs = self.colvar_fn(selected_positions)
        energy, mean_sforces = self.get_energy_mean_force_from_torsion(cvs)
        energy_ave = torch.mean(energy)
        forces = torch.autograd.grad([energy_ave,], [positions,], allow_unused=True, create_graph=True, retain_graph=True)[0]
        assert forces is not None
        model_div = torch.mean(torch.var(mean_sforces, dim=-1)) ** 0.5
        sigma = self.uncertainty_weight(model_div)

        return (energy_ave * sigma, forces * sigma)
    

    def uncertainty_weight(self, model_div):
        iswitch = (self.e1-model_div)/self.e1_m_e0
        # use heaviside function to make the gradient zero when iswitch is zero
        uncertainty_weight = torch.heaviside(torch.div(iswitch, 1, rounding_mode='floor'), 0.5*(1+torch.cos(torch.pi*(1-iswitch))))
        return uncertainty_weight

    
    def get_energy_from_torsion(self, torsion):
        torsion = torch.cat([torch.cos(torsion), torch.sin(torsion)], dim=-1)
        # torsion = self.layer_norm(torsion).reshape([-1, 2*self.n_cvs])
        torsion = torsion.reshape([-1, 2*self.n_cvs])
        # energy_list = torch.zeros(torsion.shape[0], self.n_models)
        # energy_list = []
        # for i, model in enumerate(self.model_list):
        #     energy = model(torsion).reshape(-1)
        #     # energy_list[:,i] = energy
        #     energy_list.append( energy )
        return torch.stack([
                self.model1(torsion).reshape(-1),
                self.model2(torsion).reshape(-1),
                self.model3(torsion).reshape(-1),
                self.model4(torsion).reshape(-1)
            ], dim=-1)
        # return torch.stack(energy_list, dim=-1)
    

    def get_energy_mean_force_from_torsion(self, torsion):
        model_energy = self.get_energy_from_torsion(torsion)
        # print(model_energy)
        force_list = []
        grad_outputs : List[Optional[torch.Tensor]] = [ torch.ones_like(model_energy[:,0]) ]
        for imodel in range(self.n_models):
            mean_forces = torch.autograd.grad([model_energy[:,imodel],], [torsion,], grad_outputs=grad_outputs, retain_graph=True )[0]
            assert mean_forces is not None
            force_list.append(mean_forces)
        force_list = torch.stack(force_list, dim=-1)
        return model_energy, force_list
    

    def mseloss(self, torsion):
        energy_list = self.get_energy_from_torsion(self, torsion)
        force_list = torch.zeros(energy_list.shape[0], self.n_models, self.n_cvs)
        for imodel in range(self.n_models):
            mean_forces = torch.autograd.grad(energy_list[:,imodel], torsion, retain_graph=True )[0]
            force_list[imodel] = mean_forces
        return energy_list, force_list
