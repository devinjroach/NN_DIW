# import modules

from matplotlib import pyplot as plt
import torch
from torch import nn
from torch import tensor
import scipy.io as sio
import time
import numpy as np

"""
Class describing our pytorch model or neural network architecture.
It inherits objects from the nn.Module class. 
Here we define our neural network structure, which has 1 hidden layer and a variable number of nodes in the layer.
"""
class Nonlinear(nn.Module):
    
    # Initializer.
    #def __init__(self, hidden_size=2):
    def __init__(self, layer_sizes):
        super().__init__()

        self.layer_sizes = layer_sizes
        self.nlayers = len(layer_sizes)

        #self.layer_0_weights = nn.Parameter(torch.randn(layer_sizes[0]))
        #self.layer_0_bias = nn.Parameter(torch.randn(layer_sizes[0]))

        for indx, layer_size in enumerate(layer_sizes[:-1]):
            print(indx)
            setattr(self, f"layer_{indx}_weights", 
                          nn.Parameter(torch.randn(layer_size,layer_sizes[indx+1])) )
            setattr(self, f"layer_{indx}_bias",
                          nn.Parameter(torch.randn(layer_size)) )

        
        
        """
        self.layer_1_weights = nn.Parameter(torch.randn(1, hidden_size))
        self.layer_1_bias = nn.Parameter(torch.randn(hidden_size)) 
        
        self.layer_2_weights = nn.Parameter(torch.randn(hidden_size, 1) ) 
        self.layer_2_bias = nn.Parameter(torch.randn(1))
        """
        
    # Feed fordward function - here we use matrix/tensor operations to calculate the output of the neural network
    #                          given some inputs "x".
    def forward(self, x):
       
        #print(x.size())
        #print(self.layer_0_weights.size())
        #""" 
        for indx, layer_size in enumerate(self.layer_sizes[:-1]):
            weights = getattr(self, f"layer_{indx}_weights")
            bias = getattr(self, f"layer_{indx}_bias")
            x = x.matmul(weights).add(bias)
            if (indx<(self.nlayers-2)): # don't apply activation on last layer
                #x = x.sigmoid() # need to use leaky relu
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.03)
        #"""
        """
        x = x.matmul(self.layer_1_weights).add(self.layer_1_bias)
        x = x.sigmoid()
        x = x.matmul(self.layer_2_weights).add(self.layer_2_bias)
        """
        return x.squeeze()
    
    # Optional function for setting the NN weights manually instead of random initialization. 
    def nonrandom_init(self):
        self.layer_1_weights.data = tensor([[1.1, 0.8]])
        self.layer_1_bias.data = tensor([0.5 , -0.7]) 
        self.layer_2_weights.data = tensor([[0.3], [-0.7]])
        self.layer_2_bias.data = tensor([0.2])

# get data

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

model = torch.load("network.pt")

print(model)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())


preds = model(X)


# plot the model vs. target

xaxis = np.arange(0,196,1,dtype=int)
yaxis = outputs[:,0]

plt.plot(xaxis, yaxis, 'r-')
plt.plot(xaxis, preds.detach().numpy()[:,0], 'bo')
#plt.plot(X.numpy(), predicted, 'b')
plt.savefig("evaluate.png", dpi=500)
#plt.show()