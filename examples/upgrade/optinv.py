"""
Optimize the inverse network problem
"""

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

model = torch.load("model.pt")

print(model)

# Print model's state_dict
#print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor)
    #print(model.state_dict()[param_tensor])
    #print(param_tensor, "\t", model.state_dict()[param_tensor].size())

weights = []
weights_inv = []
biases = []
for i, param_tensor in enumerate(model.state_dict()):
    if ("weights" in param_tensor):
        w = model.state_dict()[param_tensor].numpy()
        weights.append(model.state_dict()[param_tensor].numpy())
        weights_inv.append(np.linalg.inv(w))
    elif ("bias" in param_tensor):
        arr = model.state_dict()[param_tensor].numpy()
        #arr = np.array([arr]).T
        biases.append(arr)

# reverse weights list so we can go backwards later

weights_inv = weights_inv[::-1]
biases_inv = biases[::-1]

num_hidden_layers = len(weights)
assert(len(weights)==len(biases))

def leaky_relu(x):
    y1 = ((x > 0) * x)                                                 
    y2 = ((x <= 0) * x * leaky_relu_slope)                                         
    value = y1 + y2  
    return value

def inv_leaky_relu(x):
    y1 = ((x > 0) * x)
    y2 = ((x <=0) * x * (1./leaky_relu_slope))
    value = y1 + y2
    return value
    

size = 4
leaky_relu_slope = 0.03

test_value = -100
# this asserts that the inverse activation is coded correctly
assert (test_value == inv_leaky_relu(leaky_relu(test_value)))

x = np.random.rand(1,size)
x_torch = torch.from_numpy(x).float()
print("----- inputs:")
print(x)
print(x_torch.size())

model.eval()
outputs_model = model(x_torch)
print("----- outputs_model:")
print(outputs_model)

# Apply this input through all the layers
for l in range(0,num_hidden_layers):
    if (l==0):
        input = x 
    else:
        input = output
    if (l<num_hidden_layers-1):
        #output = leaky_relu(np.matmul(weights[l], input) + biases[l])
        output = leaky_relu(np.matmul(input, weights[l]) + biases[l])
    elif (l==num_hidden_layers-1):
        output = np.matmul(input, weights[l]) + biases[l]
    print(np.shape(output))

print(output)
#assert(False)

# now go backwards to invert 
print("Inverting...")
start = time.time()

# Apply this input through all the layers
for l in range(0,num_hidden_layers):
    if (l==0):
        input = outputs_model.detach().numpy()
    else:
        input = output
    if (l<num_hidden_layers-1):
        #output = inv_leaky_relu(np.matmul(weights_inv[l],input - biases_inv[l]))
        output = inv_leaky_relu(np.matmul(input - biases_inv[l],weights_inv[l]))
    elif (l==num_hidden_layers-1):
        #output = np.matmul(weights_inv[l],input - biases_inv[l])
        output = np.matmul(input - biases_inv[l],weights_inv[l])
    print(np.shape(output))

print(time.time()-start)
print("----- Recovered input:")
print(output)

# Let's demonstrate an optimization example.
# Say we're starting from some known linewidth delta=419, associated with:
# D,V,A,H = 0.84, 10, 0.6, 2
# We wanna go to a new linewidth delta=294, associated with:
# D,V,A,H = 0.84, 5, 0.6, 1.5
# while keeping D,V,A,H minimum
# So create a loss = (D1-D2)^2 + (V1-V2)^2 + (A1-A2)^2 + (H1-H2)^2

outputs_model = model(X)
"""
print("Mean/max 2nd output:")
print(torch.mean(outputs_model[:,1]))
print(torch.max(torch.abs(outputs_model[:,1])))
print(f"min max: {torch.min(outputs_model[:,1])} {torch.max(outputs_model[:,1])}")
print("Mean/max 3rd output:")
print(torch.mean(outputs_model[:,2]))
print(torch.max(torch.abs(outputs_model[:,2])))
print(f"min max: {torch.min(outputs_model[:,2])} {torch.max(outputs_model[:,2])}")
print("Mean/max 4th output:")
print(torch.mean(outputs_model[:,3]))
print(torch.max(torch.abs(outputs_model[:,3])))
print(f"min max: {torch.min(outputs_model[:,3])} {torch.max(outputs_model[:,3])}")
"""
R1_min = torch.min(outputs_model[:,1]).detach().numpy()
R1_max = torch.max(outputs_model[:,1]).detach().numpy()
R2_min = torch.min(outputs_model[:,2]).detach().numpy()
R2_max = torch.max(outputs_model[:,2]).detach().numpy()
R3_min = torch.min(outputs_model[:,3]).detach().numpy()
R3_max = torch.max(outputs_model[:,3]).detach().numpy()

# example output, which is input to the reverse network

print("example input:")
print(X[0].unsqueeze(dim=0)) # this has the current D,V,A,H
print("model prediction:")
print(model(X)[0].unsqueeze(dim=0)) # this has the associated linewidth and 3 RVs

# current params

X1 = X[0].unsqueeze(dim=0).detach().numpy() # current D,V,A,H
Y1 = model(X)[0].unsqueeze(dim=0).detach().numpy() # associated linewidth and 3 RVs
delta1 = Y1[0,0]
print(Y1)
print(delta1)

# now we wanna sample on a grid of R1,R2,R3

print(f"R1: {R1_min} {R1_max}")
print(f"R2: {R2_min} {R2_max}")
print(f"R3: {R3_min} {R3_max}")

# make grids

R1_grid = np.linspace(R1_min, R1_max, 30)
R2_grid = np.linspace(R2_min, R2_max, 30)
R3_grid = np.linspace(R3_min, R3_max, 30)

rv1 = R1_grid[15]
rv2 = R2_grid[15]
rv3 = R3_grid[15]

y_tmp = np.zeros((1,size))

y_tmp[0,0] = delta1
y_tmp[0,1] = Y1[0,1] #rv1
y_tmp[0,2] = Y1[0,2] #rv2
y_tmp[0,3] = Y1[0,3] #rv3
print(y_tmp)
#y_tmp = torch.tensor([[delta1]])

# put this temporary output into the reverse network

for l in range(0,num_hidden_layers):
    if (l==0):
        input = y_tmp
    else:
        input = l_tmp
    if (l<num_hidden_layers-1):
        #output = inv_leaky_relu(np.matmul(weights_inv[l],input - biases_inv[l]))
        l_tmp = inv_leaky_relu(np.matmul(input - biases_inv[l],weights_inv[l]))
    elif (l==num_hidden_layers-1):
        #output = np.matmul(weights_inv[l],input - biases_inv[l])
        x_tmp = np.matmul(input - biases_inv[l],weights_inv[l])
    
print(np.shape(x_tmp))
print(x_tmp)

#for rv1 in R1_grid




#y = np.random.rand(1,size)
#y = np.array([419,])
#y_torch = torch.from_numpy(x).float()