

# class MultiLinearLayer(nn.Module):
#     """ Custom Linear layer but mimics a standard linear layer """
#     """ we can concatenate multiple linear layers to form a multi-linear layer """
#     def __init__(self, size_in, n_models, size_out):
#         super().__init__()
#         self.size_in, self.size_out, self.n_models = size_in, size_out, n_models
#         weights = torch.Tensor(n_models, size_in, size_out)
#         self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
#         bias = torch.Tensor(1, n_models, size_out)
#         self.bias = nn.Parameter(bias)

#         # initialize weights and biases
#         nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
#         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
#         bound = 1 / math.sqrt(fan_in)
#         nn.init.uniform_(self.bias, -bound, bound)  # bias init

#     def forward(self, x):
#         # x: [N_sample, N_CVs] --> [N_sample, 1, N_CVs, 1]
#         # self.weights: [N_models, N_CVs, N_features]
#         w_times_x= torch.matmul( x, self.weights).reshape(-1, self.n_models, self.size_out)
#         # w_times_x: [N_sample, N_models, N_features]
#         # self.bias: [N_models, N_features] --> [1, N_models, N_features]
#         return torch.add(w_times_x, self.bias)  # w times x + b

# class DihedralBias_test(nn.Module):
#     """Free energy biasing potential."""

#     def __init__(self, 
#                  colvar_fn, colvar_idx, n_models, n_cvs, 
#                  features=[64, 64, 64, 64],
#                  e0=2,
#                  e1=3
#                  ):
#         """Initialize the biasing potential.

#             Parameters
#             ----------
#             colvar : torch.nn.Module
#                   The collective variable to bias.
#             kT : float
#                   The temperature in units of energy.
#             target : float
#                   The target value of the collective variable.
#             width : float
#                   The width of the biasing potential.
#         """
#         super().__init__()
#         self.colvar_fn = colvar_fn()
#         self.colvar_idx = colvar_idx
#         self.n_models = n_models
#         self.n_cvs = n_cvs

#         self.activation = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.1)
#         self.layer_norm = nn.LayerNorm(2*n_cvs)

#         self.dense1 = MultiLinearLayer(2 * n_cvs, n_models, features[0])
#         self.dense2 = MultiLinearLayer(features[0], n_models, features[1])
#         self.dense3 = MultiLinearLayer(features[1], n_models, features[2])
#         self.dense4 = MultiLinearLayer(features[2], n_models, features[3])
#         self.final_layer = MultiLinearLayer(features[3], n_models, 1)
#         self.dense_list = [self.dense1, self.dense2, self.dense3, self.dense4, self.final_layer]
#         self.e0 = e0
#         self.e1 = e1
#         # just calculate once
#         self.e1_m_e0 = e1-e0
            

#     def forward(self, positions):
#         """The forward method returns the energy computed from positions.

#             Parameters
#             ----------
#             positions : torch.Tensor with shape (nparticles, 3)
#             positions[i,k] is the position (in nanometers) of spatial dimension k of particle i

#             Returns
#             -------
#             potential : torch.Scalar
#             The potential energy (in kJ/mol)
#         """
#         # boxsize = boxvectors.diag()
        
#         positions = positions[self.colvar_idx.flatten()].reshape([-1, 4, 3])  # [N_CVs, 4, 3]
#         # positions = positions - torch.floor(positions/boxsize)*boxsize
#         # calculate CVs
#         positions = self.colvar_fn(positions).reshape(-1, self.colvar_idx.shape[0])
#         energy = self.get_energy_from_torsion(positions)
#         print(positions.shape, energy.shape)
#         forces = self.get_mean_force_from_torsion(positions)
#         print(forces)
#         exit(0)
#         model_div = torch.std(forces, dim=-1)
#         uncertainty_weight = self.uncertsinty_weight(model_div)
#         return (torch.Scalar(energy.mean() * uncertainty_weight), forces)
    

#     def uncertsinty_weight(self, model_div):
#         iswitch = (self.e1-model_div)/self.e1_m_e0
#         # use heaviside function to make the gradient zero when iswitch is zero
#         uncertainty_weight = torch.heaviside(iswitch, 0.5*(1+torch.cos(torch.pi*(1-iswitch))))
#         return uncertainty_weight

    
#     def get_energy_from_torsion(self, torsion):
#         torsion = torch.cat([torch.cos(torsion), torch.sin(torsion)], dim=-1)
#         torsion = self.layer_norm(torsion).reshape([-1, 1, 1, 2*self.colvar_idx.shape[0]])
#         for layer in self.dense_list[:-1]:
#             torsion = self.dropout(self.activation(layer(torsion)))
#             torsion = torsion.reshape([-1, self.n_models, 1, torsion.shape[-1]])
#         output = self.dense_list[-1](torsion).squeeze(-1)  # [N_sample, N_models]
#         return output
    

#     def get_mean_force_from_torsion(self, torsion):
#         model_energy = self.get_energy_from_torsion(torsion)
#         torsion = torsion.repeat(5,1)
#         print(torsion.shape)
#         print(model_energy.shape)
#         mean_forces = torch.autograd.grad(model_energy, torsion, grad_outputs=torch.ones_like(model_energy))
#         return mean_forces
    

# """