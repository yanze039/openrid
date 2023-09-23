import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from colvar import DihedralAngle
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
# from torch import vmap
import logging
from functorch import combine_state_for_ensemble
import math


class MLP(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    """ we can concatenate multiple linear layers to form a multi-linear layer """
    def __init__(self, size_in, size_out, dropout_rate=0.1, features=[64,64,64,64]):
        super().__init__()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
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

    def forward(self, x):
        # x: [N_sample, N_CVs] --> [N_sample, 1, N_CVs]
        output = self.Sequence(x)
        return output


class MultiLinearLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    """ we can concatenate multiple linear layers to form a multi-linear layer """
    def __init__(self, n_models, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out, self.n_models = size_in, size_out, n_models
        weights = torch.Tensor(n_models, size_in, size_out)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(n_models, 1, size_out)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        # print(x.shape)
        # x: [N_sample, N_CVs] --> [N_sample, 1, N_CVs, 1]
        # self.weights: [N_models, N_CVs, N_features]
        w_times_x= torch.matmul( x, self.weights).reshape(self.n_models, -1, self.size_out)
        # w_times_x: [N_models, N_sample, N_features]
        # self.bias: [N_models, 1, N_features]
        return torch.add(w_times_x, self.bias)  # w times x + b


class DihedralBiasVMap2(nn.Module):
    """Free energy biasing potential."""

    def __init__(
            self, 
            colvar_fn, 
            colvar_idx, 
            n_models, 
            n_cvs, 
            dropout_rate=0.1,
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

    
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.Sequence = nn.Sequential(
            MultiLinearLayer( n_models, 2 * n_cvs, features[0]),
            self.activation,
            self.dropout,
            MultiLinearLayer(n_models, features[0], features[1]),
            self.activation,
            self.dropout,
            MultiLinearLayer(n_models, features[1], features[2]),
            self.activation,
            self.dropout,
            MultiLinearLayer(n_models, features[2], features[3]),
            self.activation,
            self.dropout,
            MultiLinearLayer(n_models, features[3], 1)
        )

        self.e0 = e0
        self.e1 = e1
        # just calculate once
        self.e1_m_e0 = e1-e0
        self.loss_fn = nn.MSELoss()
        self.clac_mf = torch.func.jacrev(self.get_energy_from_torsion, argnums=0)
        self.batched_clac_mf = torch.vmap(torch.func.jacrev(self.get_energy_from_torsion, argnums=0), in_dims=0, randomness='same')
            
    def get_model_div(self, torsion):
        _, force_list = self.get_energy_mean_force_from_torsion(torsion)
        model_div = torch.mean(torch.var(force_list, dim=-1)) ** 0.5
        return model_div

    def uncertainty_weight(self, model_div):
        iswitch = (self.e1-model_div)/self.e1_m_e0
        # use heaviside function to make the gradient zero when iswitch is zero
        uncertainty_weight = torch.heaviside(torch.div(iswitch, 1, rounding_mode='floor'), 0.5*(1+torch.cos(torch.pi*(1-iswitch))))
        return uncertainty_weight

    def get_energy_from_torsion(self, torsion):
        torsion = torch.cat([torch.cos(torsion), torch.sin(torsion)], dim=-1)
        return self.Sequence(torsion).squeeze(-1)

    def get_energy_mean_force_from_torsion(self, torsion):
        model_energy = self.get_energy_from_torsion(torsion)
        force_list = []
        grad_outputs : List[Optional[torch.Tensor]] = [ torch.ones_like(model_energy[:,0]) ]
        for imodel in range(self.n_models):
            mean_forces = torch.autograd.grad([model_energy[:,imodel],], [torsion,], grad_outputs=grad_outputs, retain_graph=True )[0]
            assert mean_forces is not None
            force_list.append(mean_forces)
        force_list = torch.stack(force_list, dim=-1)
        return model_energy, force_list
    
    def get_batched_mean_force_from_torsion(self, torsion):
        return self.batched_clac_mf(torsion)
    
    def get_mean_force_from_torsion(self, torsion):
        return self.clac_mf(torsion)

    def mseloss(self, torsion):
        energy_list = self.get_energy_from_torsion(torsion)
        force_list = torch.zeros(energy_list.shape[0], self.n_models, self.n_cvs)
        for imodel in range(self.n_models):
            mean_forces = torch.autograd.grad(energy_list[:,imodel], torsion, retain_graph=True )[0]
            force_list[imodel] = mean_forces
        return energy_list, force_list

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


if __name__ == "__main__":
    # n_models, size_in, size_out = 5, 2, 3
    # model = MultiLinearLayer(size_in, n_models, size_out)
    # weights = torch.Tensor(n_models, size_in, size_out)
    # bias = torch.Tensor(n_models, 1, size_out)
    # X = torch.rand(5, 1, size_in)
    # y = model(X)
    # print(y.shape)
    # y1 = (torch.matmul( X, weights)) 
    # y2 = torch.matmul( X[0], weights[0] )
    # print(y1.shape, y2.shape, )
    # print(y1[0], y2)
    # print(torch.allclose(y1[0], y2))
    
    dih_index = np.array([0,1,2,3])
    model2 = DihedralBiasVMap2(
        colvar_fn=DihedralAngle, 
        colvar_idx=dih_index, 
        n_models=4, 
        n_cvs=9, 
        dropout_rate=0.1,
        features=[64, 64, 64, 64],
        e0=2,
        e1=3
    )
    X_new = torch.rand( 9)
    y_new = model2.get_energy_from_torsion(X_new)
    
    print(y_new.shape)
    
    