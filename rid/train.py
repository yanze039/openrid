import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import DihedralBias
from colvar import DihedralAngle
from common import prep_dihedral
import MDAnalysis as mda

# import matplotlib.pyplot as plt


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight, gain=0.1)
        m.bias.data.fill_(0.00)



# Define your custom dataset class
class MyDataset(Dataset):
    def __init__(self, data, device):
        self.data = torch.from_numpy(data).float().to(device)
        # Add any necessary preprocessing steps here

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        # Add any necessary data transformations here
        return sample

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set the path to your dataset
dataset_path = Path("/home/gridsan/ywang3/Project/rid_openmm/data/data.raw.npy")
all_data = np.load(dataset_path)
# all_data[:,18:] -= np.mean(all_data[:,18:], axis=0)
# all_data[:,18:] /= np.std(all_data[:,18:], axis=0)
n_data = len(all_data)
n_train = int(n_data * 0.9)
n_val = int(n_data * 0.1)
random_data = np.random.permutation(all_data)
del all_data
train_data = random_data[:n_train]
val_data = random_data[n_train:n_train+n_val]

# Create an instance of your dataset
train_dataset = MyDataset(train_data, device)
val_dataset = MyDataset(val_data, device)

# Set the batch size for training
batch_size = 128

# Training loop
learning_rate = 0.0008
decayRate = 0.96
epochs = 100
cv_num = 9


# Create an instance of your model
data = Path("../data/")
u = mda.Universe(data/"npt.gro")
dih_index = prep_dihedral("../data/npt.gro")
model = DihedralBias(DihedralAngle, dih_index, 4, len(dih_index), features=[80,80,80,80]).to(device)
# model = DihedralBias(DihedralAngle, [1,2], 4, 18, features=[80,80,80,80]).to(device)
model.apply(init_weights)
# Create a data loader
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=10000)
del random_data
# Set the loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
loss_fn = nn.MSELoss()


train_error_list = []
val_error_list = []

for epoch in range(epochs):
    print("Epoch: {}".format(epoch))
    loss_accum = 0
    model.train()
    for batch, data in enumerate(train_data_loader):
        
        optimizer.zero_grad()
        X = data.to(device)[:,:18].requires_grad_(True)
        y = data.to(device)[:,18:]
        
        model_energy = model.get_energy_from_torsion(X)
        force_loss = 0
        for imodel in range(4):
            mean_forces = torch.autograd.grad([model_energy[:,imodel],], [X,], grad_outputs=[torch.ones_like(model_energy[:,imodel]),], create_graph=True,  retain_graph=True )[0]
            force_loss += loss_fn(mean_forces, y)
        
        force_loss.backward()
        optimizer.step()
        
        loss_accum += force_loss.detach().cpu().item()

    train_error_list.append(loss_accum/(batch+1))
    my_lr_scheduler.step()

    
    model.eval()
    for batch, data in enumerate(val_data_loader):
        X = data.to(device)[:,:18].requires_grad_(True)
        y = data.to(device)[:,18:]
        model_energy = model.get_energy_from_torsion(X)
        force_loss = 0
        for imodel in range(4):
            mean_forces = torch.autograd.grad(model_energy[:,imodel], X, grad_outputs=torch.ones_like(model_energy[:,imodel]), create_graph=True,  retain_graph=True )[0]
            force_loss += loss_fn(mean_forces, y)
        val_error_list.append(force_loss.detach().cpu().item())

    
    torch.save(model, f"../model/model_{epoch}.pt")
    print("Epoch: {}, train loss: {}, val loss: {}".format(epoch, train_error_list[-1], val_error_list[-1]))

np.savetxt("../data/train_error.txt", train_error_list)
np.savetxt("../data/val_error.txt", val_error_list)
