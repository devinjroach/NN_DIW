"""
Simple example of NN regression with DIW training set. 
"""

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch import tensor
import scipy.io as sio

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
# get inputs and outputs from .mat files

input_mat = sio.loadmat("../../inputs.mat")
inputs = input_mat["inputs"].astype(np.double).T
output_mat = sio.loadmat("../../outputs.mat")
outputs = output_mat["outputs"].astype(np.double).T



# convert numpy arrays to torch tensors

X = torch.from_numpy(inputs).float()
Y = torch.from_numpy(outputs).float()
print(X.shape)
print(Y.shape)
targets = Y

# set neural network parameters

#layer_sizes = [4,10,100,100,100,1]
#layer_sizes = [4,4,4,4,4,1]
layer_sizes = [4,4,4,4,4,4]

# make neural network architecture

network_architecture = create_torch_network(layer_sizes)

#for name, param in network_architecture.named_parameters():
#    print("------")
#    print(name)
#    print(param)

# make the model

model = FitTorch(network_architecture)

# get ready for fitting

nepochs = 100000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# training loop

for epoch in range(nepochs):
    #outputs = model(X, indices, num_atoms)
    outputs_model = model(X)
    # only use first output in loss function (first column), likewise for targets 
    loss = loss_function(outputs_model[:,0], targets[:,0])
    # if we wanna force another output to be near some value (e.g. 2) like this:
    # loss = loss + weight*loss_function(outputs_model[:,1], 2)
    if (epoch % 10 == 0):
        print(f'epoch: {epoch}, loss = {loss.item():.4f}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# plot

xaxis = np.arange(0,196,1,dtype=int)
yaxis = outputs[:,0]
#print(np.shape(xaxis))
#print(np.shape(yaxis))
#print(yaxis)

plt.plot(xaxis, yaxis, 'r-')
plt.plot(xaxis, outputs_model.detach().numpy()[:,0], 'b-')
#plt.plot(X.numpy(), predicted, 'b')
#plt.savefig("fit.png")
plt.show()


