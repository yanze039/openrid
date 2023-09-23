import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from colvar import DihedralAngle
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
# from torch import vmap
import logging
import math
from functorch import combine_state_for_ensemble
from colvar import calculate_dihedral_and_derivatives, calc_diherals
from torch.func import jacrev


logger = logging.getLogger(__name__)


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
        bias = torch.zeros(n_models, 1, size_out)
        self.bias = nn.Parameter(bias)
        # initialize weights and biases
        torch.nn.init.xavier_normal_(self.weights, gain=0.1)
        # nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        # bound = 1 / math.sqrt(fan_in)
        # nn.init.uniform_(self.bias, -bound, bound)  # bias init
        # self.bias.data.fill_(0.00)

    def forward(self, x):
        # print("weights", self.weights.shape)
        # x: [N_sample, N_CVs] --> [N_sample, 1, N_CVs, 1]
        # self.weights: [N_models, N_CVs, N_features]
        # CHANGE
        w_times_x= torch.matmul( x, self.weights).reshape(self.n_models, -1, self.size_out)
        # w_times_x= torch.matmul( x.reshape(self.n_models, 1, -1), self.weights).reshape(self.n_models, -1)
        # print("w_times_x", w_times_x.shape)
        # w_times_x: [N_models, N_sample, N_features]
        # self.bias: [N_models, 1, N_features]
        return torch.add(w_times_x, self.bias)  # w times x + b


class DihedralBiasVmap(nn.Module):
    """Free energy biasing potential."""

    def __init__(
            self, 
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
        if isinstance(colvar_idx, list):
            colvar_idx = np.array(colvar_idx)
        self.colvar_idx = torch.from_numpy(colvar_idx)
        self.n_models = n_models
        self.n_cvs = n_cvs

        self.layer_norm = nn.LayerNorm(2*n_cvs)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.Sequence = nn.Sequential(
            self.layer_norm,
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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def to_device(self, device):
        self.to(device)
        self.colvar_idx = self.colvar_idx.to(device)

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
        out = self.Sequence(torsion).squeeze(-1)
        return out, out
    
    def get_energy_from_torsion_pure(self, torsion):
        torsion = torch.cat([torch.cos(torsion), torch.sin(torsion)], dim=-1)
        out = self.Sequence(torsion).squeeze(-1)
        return out
    
    def get_mf(self):
        return jacrev(self.get_energy_from_torsion, argnums=(0), has_aux=True)

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
        cvs, dcvs_dx = calculate_dihedral_and_derivatives(selected_positions)  # [*, 4, 3]
        
        cvs = cvs.reshape([1, 1, -1])
        # energy = self.get_energy_from_torsion_pure(cvs)  # [4, *]
        mean_forces, energy = self.get_mf()(cvs)
        # exit(0)
        # force_list = []
        # for imodel in range(self.n_models):
        #     mean_forces = torch.autograd.grad([energy[imodel],], [cvs,], retain_graph=True)[0]
        #     assert mean_forces is not None
        #     force_list.append(mean_forces)
        # mean_forces = torch.stack(force_list, dim=0)
        model_div = torch.mean(torch.var(mean_forces, dim=0).flatten()) ** 0.5
        sigma = self.uncertainty_weight(model_div)
        mean_forces = mean_forces.mean(0)
        forces = torch.zeros_like(positions, device=mean_forces.device).index_add_(
            0, self.colvar_idx.flatten(), (mean_forces.reshape(-1, 1, 1) * dcvs_dx.reshape(-1, 4, 3)).reshape(-1, 3)
        )
        
        return (energy.mean() * sigma, forces * sigma)
    
    # def get_mean_force_from_torsion(self, torsion):
    #     energy = self.get_energy_from_torsion(torsion).flatten()
    #     force_list = []
    #     for imodel in range(self.n_models):
    #         mean_forces = torch.autograd.grad([energy[imodel],], [torsion,], retain_graph=True )[0]
    #         assert mean_forces is not None
    #         force_list.append(mean_forces)
    #         # print(mean_forces.shape)
    #     mean_forces = torch.stack(force_list, dim=0)
    #     return mean_forces
    
    def train_mean_force_from_torsion(self, torsion):
        # torsion [4, 64, 18]
        energy = self.get_energy_from_torsion(torsion)
        force_list = []
        for imodel in range(self.n_models):
            mean_forces = torch.autograd.grad([(energy[imodel]),], [torsion,], grad_outputs=[torch.ones_like(energy[imodel]),], create_graph=True, retain_graph=True)[0]
            assert mean_forces is not None
            force_list.append(mean_forces[imodel])
        mean_forces = torch.stack(force_list, dim=0)
        return mean_forces
    
    def validation_mean_force_from_torsion(self, torsion):
        # torsion [4, 64, 18]
        energy = self.get_energy_from_torsion(torsion)
        force_list = []
        for imodel in range(self.n_models):
            # mean_forces = torch.autograd.grad([(energy[imodel]).mean(),], [torsion,], create_graph=True, retain_graph=True)[0]
            mean_forces = torch.autograd.grad([(energy[imodel]),], [torsion,], grad_outputs=[torch.ones_like(energy[imodel]),], create_graph=True, retain_graph=True)[0]
            assert mean_forces is not None
            force_list.append(mean_forces.squeeze(0))
        mean_forces = torch.stack(force_list, dim=0)
        return mean_forces
    
    def get_auto_grad_forces_from_torsion(self, torsion):
        energy = self.get_energy_from_torsion(torsion)
        energy_ave = torch.mean(energy)
        forces = torch.autograd.grad([energy_ave,], [torsion,], allow_unused=True, create_graph=True, retain_graph=True)[0]
        return forces


class DihedralBias(nn.Module):
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

        self.model1 = MLP(2 * n_cvs, 1, dropout_rate=dropout_rate, features=features)
        self.model2 = MLP(2 * n_cvs, 1, dropout_rate=dropout_rate, features=features)
        self.model3 = MLP(2 * n_cvs, 1, dropout_rate=dropout_rate, features=features)
        self.model4 = MLP(2 * n_cvs, 1, dropout_rate=dropout_rate, features=features)
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
        force_list = []
        grad_outputs : List[Optional[torch.Tensor]] = [ torch.ones_like(model_energy[:,0]) ]
        for imodel in range(self.n_models):
            mean_forces = torch.autograd.grad([model_energy[:,imodel],], [torsion,], grad_outputs=grad_outputs, retain_graph=True )[0]
            assert mean_forces is not None
            force_list.append(mean_forces)
        force_list = torch.stack(force_list, dim=-1)
        return model_energy, force_list
    

    def mseloss(self, torsion):
        energy_list = self.get_energy_from_torsion(torsion)
        force_list = torch.zeros(energy_list.shape[0], self.n_models, self.n_cvs)
        for imodel in range(self.n_models):
            mean_forces = torch.autograd.grad(energy_list[:,imodel], torsion, retain_graph=True )[0]
            force_list[imodel] = mean_forces
        return energy_list, force_list

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mymodel = DihedralBiasVmap(
        colvar_idx=[[1,2,3,4], [5,6,7,8], [7,8,9,10], [3,4,5,6]], 
        n_models=4, 
        n_cvs=4, 
        dropout_rate=0.1,
        features=[64, 64, 64, 64],
        e0=2,
        e1=3
    ).to(device)
    mymodel.to_device(device)
    mymodel.eval()
    jitted_model = torch.jit.script(mymodel)

    # old_model = DihedralBias(
    #     colvar_idx=np.array([[1,2,3,4], [5,6,7,8], [7,8,9,10], [3,4,5,6]]),
    #     colvar_fn=DihedralAngle,
    #     n_models=4,
    #     n_cvs=4,
    #     dropout_rate=0.1,
    #     features=[64, 64, 64, 64],
    #     e0=2,
    #     e1=3
    # ).to(device)
    # old_model.eval()

    positions = torch.randn(20, 3).to(device)
    boxvectors = torch.randn(3, 3).to(device)*100

    y = mymodel(positions, boxvectors)

    exit(0)
    
    import time
    import tqdm

    time_1 = time.time()
    
    cycles = 1
    for i in tqdm.tqdm(range(cycles)):
        y = mymodel(positions, boxvectors)
    time_2 = time.time()
    # exit(0)
    # for i in tqdm.tqdm(range(cycles)):
    #     y2 = old_model(positions, boxvectors)
    # time_3 = time.time()

    print("new model time: ", time_2-time_1)
    print("old model time: ", time_3-time_2)

    

