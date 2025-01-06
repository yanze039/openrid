import os
import logging
import math
import shutil
from typing import Tuple, Optional, List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from openrid.colvar import DihedralAngle
from openrid.colvar import calculate_dihedral_and_derivatives
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)

EV_TO_KJ_PER_MOL = 96.485
KJ_PER_MOL_TO_EV = 1/EV_TO_KJ_PER_MOL


class BaseDataset(Dataset):
    def __init__(self, data, device):
        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data).float().to(device)
        else:
            self.data = (data).float().to(device)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


class MultiModelDataset(Dataset):
    def __init__(self, data, n_models=4):
        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data).float()
        else:
            self.data = (data).float()
        self.n_models = n_models
        self.index_list = [
            torch.randperm(len(self.data)) for _ in range(n_models)
        ]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = torch.zeros((self.n_models, self.data.shape[1]))
        for i, index in enumerate(self.index_list):
            sample[i] = self.data[index[idx]]
        return sample


def l2_loss(f_0, f_hat):
    f_diff = f_0 - f_hat
    loss = torch.mean(torch.sum(f_diff**2, dim=-1), dim=-1)
    return loss


class NeuralNetworkManager:

    def __init__(
            self,
            model = None,
            model_path = None,
            data_path = "data.txt",
            output_dir = None,
            batch_size = 128,
            optimizer = torch.optim.Adam,
            learning_rate = 0.001,
            decayStep = 120,
            decayRate = 0.96,
            epochs = 100,
            loss_fn = nn.MSELoss(),
            cv_num = 18,
            training_data_portion = 0.9,
            print_interval = 100
    ) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if model is None:
            assert model_path is not None
            self.model = torch.load(model_path).to(self.device)
        else:
            self.model = model.to(self.device)
            # self.model.apply(self.init_weights)
        self.data_path = data_path
        logger.info(f"Using data {str(self.data_path)}")
        if output_dir is None:
            self.output_dir = Path("output")
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decayRate = decayRate
        self.decayStep = decayStep
        self.epochs = epochs
        self.cv_num = cv_num
        self.training_data_portion = training_data_portion
        assert self.training_data_portion <= 1.0
        self.loss_fn = loss_fn
        self.print_interval = print_interval
        self.n_models = self.model.n_models
        self.optimizer = optimizer
    
    def get_dataset(self):
        if hasattr(self, 'dataset') and hasattr(self, 'validation_dataset'):
            return self.dataset
        dataset_path = Path(self.data_path)
        if dataset_path.suffix == ".npy":
            all_data = np.load(dataset_path)
        elif dataset_path.suffix == ".txt":
            all_data = np.loadtxt(dataset_path)
        else:
            raise NotImplementedError
        n_data = len(all_data)
        n_train = int(n_data * self.training_data_portion)
        n_val = int(n_data * (1 - self.training_data_portion))
        if n_val == 0 and self.training_data_portion < 1.0:
            n_val = 1
            logger.error("Validation data is empty, however training data portion < 1. set val to 1"
                         "this may be due to too few data selected from labeling step, which can indicates"
                         "the simulation has converged or you have a unreasonable threshold for labeling selection")
            n_train = n_data - 1
        all_data = np.random.permutation(all_data)
        train_data = all_data[:n_train]
        self.training_dataset = MultiModelDataset(train_data, n_models=self.n_models)
        if self.training_data_portion == 1.0:
            self.validation_dataset = None
        else:
            val_data = all_data[n_train:n_train+n_val]
            self.validation_dataset = BaseDataset(val_data, self.device)
        self.dataset = {
            "training": self.training_dataset,
            "validation": self.validation_dataset,
        }
        return self.dataset
    
    def get_dataloader(self):
        if hasattr(self, 'dataloader'):
            return self.dataloader
        dataset = self.get_dataset()
        self.training_dataloader = DataLoader(dataset["training"], batch_size=self.batch_size, shuffle=True)
        if dataset["validation"] is None:
            self.validation_dataloader = None
        else:
            self.validation_dataloader = DataLoader(dataset["validation"], batch_size=len(dataset["validation"]), shuffle=False)
        self.dataloader = {
            "training": self.training_dataloader,
            "validation": self.validation_dataloader,
        }
        return self.dataloader
    
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight, gain=0.1)
            m.bias.data.fill_(0.00)
    
    def save(self, output_path):
        torch.save(self.model, output_path)

    def script(self, output_path):
        torch.jit.script(self.model).save(output_path)
    
    def train(
        self,
    ): 
        optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=self.decayRate
        )
        train_error_list = []
        val_error_list = []
        self.dataloader = self.get_dataloader()
        force_loss_validation = None
        batch = 0
        miminum_validation_error = 1e10
        miminum_validation_error_epoch = 0
        for epoch in range(self.epochs):
            loss_accum = 0
            self.model.train()
            for batch, data in enumerate(self.dataloader["training"]):
                optimizer.zero_grad()

                X = data.to(self.device)[...,:self.cv_num].requires_grad_(True).transpose(0,1)
                y = data.to(self.device)[...,self.cv_num:].transpose(0,1) * KJ_PER_MOL_TO_EV
                mean_forces = self.model.train_mean_force_from_torsion(X)
                force_loss = l2_loss(mean_forces, y).mean()
                force_loss.backward()
                optimizer.step()
                loss_accum += force_loss.detach().cpu().item()

            train_error_list.append(loss_accum/(batch+1))
            if epoch % self.decayStep == 0 and epoch > 0:
                scheduler.step()

            if self.validation_dataloader is not None:
                self.model.eval()
                for batch, data in enumerate(self.dataloader["validation"]):
                    X = data.to(self.device)[...,:self.cv_num].requires_grad_(True).reshape(1, -1, self.cv_num)
                    y = data.to(self.device)[...,self.cv_num:] * KJ_PER_MOL_TO_EV
                    mean_forces = self.model.validation_mean_force_from_torsion(X).mean(0)
                    # force_loss_validation = self.loss_fn(mean_forces, y)
                    force_loss_validation = l2_loss(mean_forces, y)
            assert force_loss_validation is not None
            val_error_list.append(force_loss_validation.detach().cpu().item())
            if force_loss_validation < miminum_validation_error:
                miminum_validation_error = force_loss_validation
                miminum_validation_error_epoch = epoch
                self.save(self.output_dir / f"model_{epoch}.pt")
            if epoch % self.print_interval == 0:
                logger.info("Epoch: {}, train loss: {}, val loss: {}".format(epoch, train_error_list[-1], val_error_list[-1]))
        logger.info("miminum_validation_error: {} at epoch {}".format(miminum_validation_error, miminum_validation_error_epoch))
        shutil.copy(self.output_dir / f"model_{miminum_validation_error_epoch}.pt", self.output_dir / f"model_best.pt")
        np.savetxt(self.output_dir / "train_error.txt", train_error_list)
        np.savetxt(self.output_dir / "val_error.txt", val_error_list)


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
        # torch.nn.init.xavier_normal_(self.weights, gain=0.1)
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
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
            e0=2.,
            e1=3.
        ):
        super().__init__()
        if isinstance(colvar_idx, list):
            colvar_idx = np.array(colvar_idx)
        self.colvar_idx = torch.from_numpy(colvar_idx)
        self.n_models = n_models
        self.n_cvs = n_cvs
        self.activation = nn.Tanh()
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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.colvar_idx = self.colvar_idx.to(self.device)
        self.boost_factor = 1.0 * EV_TO_KJ_PER_MOL
    
    def get_e0(self):
        return self.e0
    
    def get_e1(self):
        return self.e1
    
    def set_e0(self, e0):
        self.e0 = e0
        self.e1_m_e0 = self.e1 - e0
    
    def set_e1(self, e1):
        self.e1 = e1
        self.e1_m_e0 = e1 - self.e0
    
    def to_device(self, device):
        self.to(device)
        self.colvar_idx = self.colvar_idx.to(device)

    def uncertainty_weight(self, model_div):
        iswitch = (self.e1-model_div)/self.e1_m_e0
        # use heaviside function to make the gradient zero when iswitch is zero
        uncertainty_weight = torch.heaviside(torch.div(iswitch, 1, rounding_mode='floor'), 0.5*(1+torch.cos(torch.pi*(1-iswitch))))
        return uncertainty_weight

    def get_energy_from_torsion(self, torsion):
        # unit # [eV]
        torsion = torch.cat([torch.cos(torsion), torch.sin(torsion)], dim=-1)
        return self.Sequence(torsion).squeeze(-1)

    # def forward(self, positions, boxvectors):
    def forward(self, positions):
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
        selected_positions = positions[self.colvar_idx.flatten()].reshape([-1, 4, 3])  # [N_CVs, 4, 3]
        cvs, dcvs_dx = calculate_dihedral_and_derivatives(selected_positions)  # [*, 4, 3]
        cvs = cvs.reshape([1, 1, -1])
        energy = self.get_energy_from_torsion(cvs) * self.boost_factor # [4, *]
        force_list = []
        for imodel in range(self.n_models):
            # add `-` to `energy`?
            # mean_forces = torch.autograd.grad([ - energy[imodel],], [cvs,], retain_graph=True)[0]
            mean_forces = torch.autograd.grad([  energy[imodel],], [cvs,], retain_graph=True)[0]
            assert mean_forces is not None
            force_list.append( mean_forces)
        mean_forces = torch.stack(force_list, dim=0)
        model_div = torch.mean(torch.var(mean_forces, dim=0)) ** 0.5
        sigma = self.uncertainty_weight(model_div)
        mean_forces = mean_forces.mean(0)
        forces = torch.zeros_like(positions, device=mean_forces.device).index_add_(
            0, self.colvar_idx.flatten(), (mean_forces.reshape(-1, 1, 1) * dcvs_dx.reshape(-1, 4, 3)).reshape(-1, 3)
        )
        return (energy.mean() * sigma, forces * sigma)
    
    def get_mean_force_from_torsion(self, torsion):
        torsion = torsion.reshape([1, -1, self.n_cvs]).requires_grad_(True)
        energy = self.get_energy_from_torsion(torsion)  # [4, *]
        force_list = []
        for imodel in range(self.n_models):
            mean_forces = torch.autograd.grad([(-energy[imodel]),], [torsion,], grad_outputs=[torch.ones_like(energy[imodel]),], create_graph=True, retain_graph=True)[0]
            assert mean_forces is not None
            force_list.append(mean_forces)
        mean_forces = torch.stack(force_list, dim=0)
        return mean_forces
    
    def train_mean_force_from_torsion(self, torsion):
        # torsion [4, 64, 18]
        torsion = torsion.reshape([self.n_models, -1, self.n_cvs]).requires_grad_(True)
        energy = self.get_energy_from_torsion(torsion)
        force_list = []
        for imodel in range(self.n_models):
            mean_forces = torch.autograd.grad([(-energy[imodel]),], [torsion,], grad_outputs=[torch.ones_like(energy[imodel]),], create_graph=True, retain_graph=True)[0]
            assert mean_forces is not None
            force_list.append(mean_forces[imodel])
        mean_forces = torch.stack(force_list, dim=0)
        return mean_forces
    
    def validation_mean_force_from_torsion(self, torsion):
        # torsion [4, 64, 18]
        torsion = torsion.reshape([1, -1, self.n_cvs])
        energy = self.get_energy_from_torsion(torsion)
        force_list = []
        for imodel in range(self.n_models):
            # mean_forces = -torch.autograd.grad([(energy[imodel]).mean(),], [torsion,], create_graph=True, retain_graph=True)[0]
            mean_forces = torch.autograd.grad([(-energy[imodel]),], [torsion,], grad_outputs=[torch.ones_like(energy[imodel]),], create_graph=True, retain_graph=True)[0]
            assert mean_forces is not None
            force_list.append(mean_forces.squeeze(0))
        mean_forces = torch.stack(force_list, dim=0)
        return mean_forces


class BiasForceNet(nn.Module):
    """Free energy biasing potential."""

    def __init__(self, model, CVFn, CVIndex, e0=2., e1=3., bias_factor=1.0):
        super().__init__()
        self.e0 = e0
        self.e1 = e1
        # just calculate once
        self.e1_m_e0 = e1-e0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if isinstance(CVIndex, list):
            CVIndex = np.array(CVIndex)
        self.colvar_idx = torch.from_numpy(CVIndex)
        self.colvar_idx = self.colvar_idx.to(self.device)
        self.model = model
        self.CVFn = CVFn
        self.n_models = model.n_models
        self.bias_factor = bias_factor * EV_TO_KJ_PER_MOL
    
    def get_uncertainty_threshold(self):
        return [self.e0, self.e1]

    def set_uncertainty_threshold(self, e0, e1):
        self.e0 = e0
        self.e1 = e1
        self.e1_m_e0 = e1 - e0

    def forward(self, positions):
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
        selected_positions = positions[self.colvar_idx.flatten()].reshape([-1, 4, 3])  # [N_CVs, 4, 3]
        cvs, dcvs_dx = self.CVFn(selected_positions)
        cvs = cvs.reshape([1, 1, -1])
        energy = self.model.get_energy_from_CV(cvs) * self.bias_factor
        force_list = []
        for imodel in range(self.n_models):
            bias_forces = torch.autograd.grad([  energy[imodel],], [cvs,], retain_graph=True)[0]
            assert bias_forces is not None
            force_list.append( bias_forces)
        bias_forces = torch.stack(force_list, dim=0)
        model_div = torch.mean(torch.var(bias_forces, dim=0)) ** 0.5
        sigma = self.uncertainty_weight(model_div)
        bias_forces = bias_forces.mean(0)
        forces = torch.zeros_like(positions, device=bias_forces.device).index_add_(
            0, self.colvar_idx.flatten(), (bias_forces.reshape(-1, 1, 1) * dcvs_dx.reshape(-1, 4, 3)).reshape(-1, 3)
        )
        return (energy.mean() * sigma, forces * sigma)
    
    def uncertainty_weight(self, model_div):
        iswitch = (self.e1-model_div)/self.e1_m_e0
        # use heaviside function to make the gradient zero when iswitch is zero
        uncertainty_weight = torch.heaviside(torch.div(iswitch, 1, rounding_mode='floor'), 0.5*(1+torch.cos(torch.pi*(1-iswitch))))
        return uncertainty_weight


class AbstractCVNet(nn.Module, ABC):
    def __init__(
            self, 
            n_models, 
            n_cvs, 
            dropout_rate=0.1,
            features=[64, 64, 64, 64]
        ):
        super().__init__()
        self.n_models = n_models
        self.n_cvs = n_cvs
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.hiddens = nn.ModuleList()
        self.input_layer = MultiLinearLayer( n_models, 2 * n_cvs, features[0])
        for ii in range(len(features)-1):
            self.hiddens.append(MultiLinearLayer(n_models, features[ii], features[ii+1]))
        self.final_layer = MultiLinearLayer(n_models, features[-1], 1)
        
    def forward(self, cvs) -> Tuple[torch.Tensor, torch.Tensor]:
        energy = self.get_energy_from_CV(cvs) # [4, *]
        force_list = []
        for imodel in range(self.n_models):
            # add `-` to `energy`?
            # mean_forces = torch.autograd.grad([ - energy[imodel],], [cvs,], retain_graph=True)[0]
            mean_forces = torch.autograd.grad([  energy[imodel],], [cvs,], retain_graph=True)[0]
            assert mean_forces is not None
            force_list.append( mean_forces)
        mean_forces = torch.stack(force_list, dim=0)
        return energy, mean_forces
    
    def train_mean_force(self, cvs):
        cvs = cvs.reshape([self.n_models, -1, self.n_cvs]).requires_grad_(True)
        energy = self.get_energy_from_CV(cvs)
        force_list = []
        for imodel in range(self.n_models):
            mean_forces = torch.autograd.grad([(-energy[imodel]),], [cvs,], grad_outputs=[torch.ones_like(energy[imodel]),], create_graph=True, retain_graph=True)[0]
            assert mean_forces is not None
            force_list.append(mean_forces[imodel])
        mean_forces = torch.stack(force_list, dim=0)
        return mean_forces
    
    def validation_mean_force(self, cvs):
        cvs = cvs.reshape([1, -1, self.n_cvs])
        energy = self.get_energy_from_CV(cvs)
        force_list = []
        for imodel in range(self.n_models):
            mean_forces = torch.autograd.grad([(-energy[imodel]),], [cvs,], grad_outputs=[torch.ones_like(energy[imodel]),], create_graph=True, retain_graph=True)[0]
            assert mean_forces is not None
            force_list.append(mean_forces.squeeze(0))
        mean_forces = torch.stack(force_list, dim=0)
        return mean_forces

    @abstractmethod
    def get_energy_from_CV(self, cvs) -> torch.Tensor:
        pass        


class DistanceCVNet(AbstractCVNet):
    def __init__(
            self, 
            n_models, 
            n_cvs, 
            dropout_rate=0.1,
            features=[64, 64, 64, 64]
        ):
        super().__init__(n_models, n_cvs, dropout_rate, features)

    def get_energy_from_CV(self, cvs):
        # unit # [eV]
        cvs = self.activation(self.input_layer(cvs))
        cvs = self.dropout(cvs)
        for layer in self.hiddens:
            cvs = self.activation(layer(cvs)) + cvs
            cvs = self.dropout(cvs)
        output = self.final_layer(cvs).squeeze(-1)
        return output


class PeriodicCVNet(AbstractCVNet):
    def __init__(
            self, 
            n_models, 
            n_cvs, 
            dropout_rate=0.1,
            features=[64, 64, 64, 64]
        ):
        super().__init__(n_models, n_cvs, dropout_rate, features)

    def get_energy_from_CV(self, cvs):
        # unit # [eV]
        cvs = torch.cat([torch.cos(cvs), torch.sin(cvs)], dim=-1)
        cvs = self.activation(self.input_layer(cvs))
        cvs = self.dropout(cvs)
        for layer in self.hiddens:
            cvs = self.activation(layer(cvs)) + cvs
            cvs = self.dropout(cvs)
        output = self.final_layer(cvs).squeeze(-1)
        return output
    



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    w = torch.randn(4, 5, 3).to(device)
    x = torch.randn(1, 1, 5).to(device)
    w_x = torch.matmul(x, w)
    w_x_2= torch.matmul(x.flatten(), w)
    exit(0)

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

    old_model = DihedralBias(
        colvar_idx=np.array([[1,2,3,4], [5,6,7,8], [7,8,9,10], [3,4,5,6]]),
        colvar_fn=DihedralAngle,
        n_models=4,
        n_cvs=4,
        dropout_rate=0.1,
        features=[64, 64, 64, 64],
        e0=2,
        e1=3
    ).to(device)
    old_model.eval()

    positions = torch.randn(20000, 3).to(device)
    boxvectors = torch.randn(3, 3).to(device)*100
    
    import time
    import tqdm

    time_1 = time.time()
    
    cycles = 1000
    for i in tqdm.tqdm(range(cycles)):
        y = mymodel(positions, boxvectors)
    time_2 = time.time()
    # exit(0)
    for i in tqdm.tqdm(range(cycles)):
        y2 = old_model(positions, boxvectors)
    time_3 = time.time()

    print("new model time: ", time_2-time_1)
    print("old model time: ", time_3-time_2)

    

