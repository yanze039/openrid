import torch
import torch.nn as nn
import math
from colvar import DihedralAngle

class MultiLinearLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    """ we can concatenate multiple linear layers to form a multi-linear layer """
    def __init__(self, size_in, n_models, size_out):
        super().__init__()
        self.size_in, self.size_out, self.n_models = size_in, size_out, n_models
        weights = torch.Tensor(n_models, size_in, size_out)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(n_models, size_out)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        # x: [N_sample, N_CVs] --> [N_sample, 1, N_CVs, 1]
        # self.weights: [N_models, N_CVs, N_features]
        w_times_x= torch.matmul( x[..., None, None, :], self.weights).squeeze()  
        # w_times_x: [N_sample, N_models, N_features]
        # self.bias: [N_models, N_features] --> [1, N_models, N_features]
        return torch.add(w_times_x, self.bias[None, ...])  # w times x + b


class FreeEnergyBias(nn.Module):
    """Free energy biasing potential."""

    def __init__(self, colvar_fn, colvar_idx, kT, n_models, n_cvs, features=[64, 64, 64, 64]):
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
        self.colvar_fn = colvar_fn
        self.kT = kT
        self.colvar_idx = colvar_idx
        self.n_models = n_models


        self.dense_list = [MultiLinearLayer(n_cvs, n_models, features[0])]
        for ii in range(len(features)):
            self.dense_list.append(MultiLinearLayer(features[ii], n_models, features[ii+1]))
        self.dense_list.append(MultiLinearLayer(features[-1], n_models, n_cvs))
        self.activation = nn.ReLU()
            

    def forward(self, positions):
        """The forward method returns the energy computed from positions.

            Parameters
            ----------
            positions : torch.Tensor with shape (nparticles,3)
            positions[i,k] is the position (in nanometers) of spatial dimension k of particle i

            Returns
            -------
            potential : torch.Scalar
            The potential energy (in kJ/mol)
        """
            
        colvar = self.colvar_fn(positions[self.colvar_idx])

        for layer in self.dense_list:
            colvar = self.activation(layer(colvar))

        return colvar
    



# Render the compute graph to a TorchScript module
# module = torch.jit.script(ForceModule())

# Serialize the compute graph to a file
# module.save('model.pt')

if __name__ == "__main__":
    mymodel = MultiLinearLayer(3, 2, 4)
    mymodel2 = FreeEnergyBias(DihedralAngle(), [0,1,2,3], 1.0, 2, 4)
    x = torch.randn((6,3))
    mymodel(x)
    # print(x)
    # print(mymodel(x))