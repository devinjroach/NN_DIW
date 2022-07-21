"""
Simple example of NN regression - training a NN to reproduce a parabola in 1D.
"""

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch import tensor

def create_torch_network(layer_sizes):
    """
    Creates a pytorch network architecture from layer sizes.
    This also performs standarization in the first linear layer.
    This only supports softplus as the nonlinear activation function.

        Parameters:
            layer_sizes (list of ints): Size of each network layers

        Return:
            Network Architecture of type neural network sequential

    """
    layers = []
    try:
        layers.append(torch.nn.Linear(layer_sizes[0], layer_sizes[0]))
        for i, layer in enumerate(layer_sizes):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(torch.nn.Softplus())
    except IndexError:
        layers.pop()
    return torch.nn.Sequential(*layers)

class FitTorch(torch.nn.Module):
    """
    FitSNAP PyTorch Neural Network Architecture Model
    Currently only fits on energies
    """

    def __init__(self, network_architecture):
        """
        Saves lammps ready pytorch model.

            Parameters:
                network_architecture : A nn.Sequential network architecture

        """
        super().__init__()
        self.network_architecture = network_architecture
        
    def forward(self, x):
        """
        Saves lammps ready pytorch model.

            Parameters:
                x (tensor of floats): Array of inputs with size (ndata, nfeatures)

        """

        return self.network_architecture(x)

    def import_wb(self, weights, bias):
        """
        Imports weights and bias into FitTorch model

            Parameters:
                weights (list of numpy array of floats): Network weights at each layer
                bias (list of numpy array of floats): Network bias at each layer

        """

        assert len(weights) == len(bias)
        imported_parameter_count = sum(w.size + b.size for w, b in zip(weights, bias))
        combined = [None] * (len(weights) + len(bias))
        combined[::2] = weights
        combined[1::2] = bias

        assert len([p for p in self.network_architecture.parameters()]) == len(combined)
        assert sum(p.nelement() for p in self.network_architecture.parameters()) == imported_parameter_count

        state_dict = self.state_dict()
        for i, key in enumerate(state_dict.keys()):
            state_dict[key] = torch.tensor(combined[i])
        self.load_state_dict(state_dict)

# make fitting data
# X is a 30x1 tensor, rows are data points and columns are features (i.e. number of network inputs)
# Y is a 30x1 tensor, rows are data point targets and columns are number of targets for each data point (i.e. number of network outputs)

X = torch.linspace(-2., 2., 30).unsqueeze(1)
print(X.shape)
Y = torch.square(X)
targets = Y
print(Y.shape)

# set neural network parameters

layer_sizes = [1,10,1]

# make neural network architecture

network_architecture = create_torch_network(layer_sizes)

#for name, param in network_architecture.named_parameters():
#    print("------")
#    print(name)
#    print(param)

# make the model

model = FitTorch(network_architecture)

# get ready for fitting

nconfigs=30
num_atoms = torch.ones(nconfigs,dtype=torch.int32) # number of atoms per config
indices = np.linspace(0,nconfigs-1,30).astype(int)
indices = torch.from_numpy(indices)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# training loop

for epoch in range(1000):
    #outputs = model(X, indices, num_atoms)
    outputs = model(X)
    # only use first output in loss function (first column), likewise for targets 
    loss = loss_function(outputs[:,0], targets[:,0])
    if (epoch % 10 == 0):
        print(f'epoch: {epoch}, loss = {loss.item():.4f}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# plot

predicted = model(X).detach().numpy()
plt.plot(X.numpy(), Y.numpy(), 'ro')
plt.plot(X.numpy(), predicted, 'b')
plt.savefig("fit.png")


