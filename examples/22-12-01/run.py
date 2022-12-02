"""
Optimize the inverse network problem.
Run this script with 
    python run.py
Tweak the indices and change_lim variable below to try different training points and grid sampling.

This produces 5 plots:
1. surface_D.png : surface of D vs. change in chi1 and chi2
2. surface_V.png : surface of V vs. change in chi1 and chi2
3. surface_A.png : surface of A vs. change in chi1 and chi2
4. surface_H.png : surface of H vs. change in chi1 and chi2
5. loss_vs_V_A.png : surface of loss function vs. V and A
"""

# import modules

from matplotlib import pyplot as plt
import torch
from torch import nn
from torch import tensor
import scipy.io as sio
import time
import numpy as np
from matplotlib import cm

# sample index to start from
starting_indx = 193
# sample index representing the target
target_indx = 100
# how far we sample from the target known unused outputs
change_lim = 10.0 # 10.0 works well with 4x4x4 network

class Nonlinear(nn.Module):
    """
    Class describing our pytorch model or neural network architecture.
    It inherits objects from the nn.Module class. 
    Here we define our neural network structure, which has 1 hidden layer and a variable number of 
    nodes in the layer.
    """
    
    def __init__(self, layer_sizes):
        super().__init__()

        self.layer_sizes = layer_sizes
        self.nlayers = len(layer_sizes)

        for indx, layer_size in enumerate(layer_sizes[:-1]):
            #print(indx)
            setattr(self, f"layer_{indx}_weights", 
                          nn.Parameter(torch.randn(layer_size,layer_sizes[indx+1])) )
            setattr(self, f"layer_{indx}_bias",
                          nn.Parameter(torch.randn(layer_size)) )

    def forward(self, x):
       
        for indx, layer_size in enumerate(self.layer_sizes[:-1]):
            weights = getattr(self, f"layer_{indx}_weights")
            bias = getattr(self, f"layer_{indx}_bias")
            x = x.matmul(weights).add(bias)
            if (indx<(self.nlayers-2)): # don't apply activation on last layer
                #x = x.sigmoid() # need to use leaky relu
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.03)

        return x.squeeze()

    def nonrandom_init(self):
        """ Optional function for setting the NN weights manually instead of random initialization. """
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
#print(X.shape)
#print(Y.shape)
targets = Y

model = torch.load("model.pt")

# Print model's state dictionary.
#print("Model's state_dict:")
#for param_tensor in model.state_dict():
#    print(param_tensor)
#    print(model.state_dict()[param_tensor])
#    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

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

outputs_model = model(X)

print("-------------------------------------")

# current params

#starting_indx = 193
X1 = X[starting_indx].unsqueeze(dim=0).detach().numpy() # current D,V,A,H
Y1 = model(X)[starting_indx].unsqueeze(dim=0).detach().numpy() # associated linewidth and 3 RVs
delta1 = Y1[0,0]
print(f"Starting linewidth: {Y1[0,0]}")
print(f"Starting params: {X[starting_indx,0]} {X[starting_indx,1]} {X[starting_indx,2]} {X[starting_indx,3]}")

# new params

#target_indx = 100
Y2 = model(X)[target_indx].unsqueeze(dim=0).detach().numpy()
delta2 = Y2[0,0]
rvar1 = Y2[0,1]
rvar2 = Y2[0,2]
rvar3 = Y2[0,3]

print("-------------------------------------")
print(f"Target linewidth: {Y2[0,0]}")
print(f"Target params: {X[target_indx,0]} {X[target_indx,1]} {X[target_indx,2]} {X[target_indx,3]}")
print(f"Target unused output: {Y2[0,1]} {Y2[0,2]} {Y2[0,3]}")

y_tmp = np.zeros((1,size))
y_tmp[0,0] = Y2[0,0]
y_tmp[0,1] = Y2[0,1]
y_tmp[0,2] = Y2[0,2]
y_tmp[0,3] = Y2[0,3]

# put this temporary output into the reverse network

for l in range(0,num_hidden_layers):
    if (l==0):
        input = y_tmp
    else:
        input = l_tmp
    if (l<num_hidden_layers-1):
        l_tmp = inv_leaky_relu(np.matmul(input - biases_inv[l],weights_inv[l]))
    elif (l==num_hidden_layers-1):
        x_tmp = np.matmul(input - biases_inv[l],weights_inv[l])

print("-------------------------------------")
print(f"Actual data point: {X[100].detach().numpy()}")
print(f"Recovered data point: {x_tmp}")

print("-------------------------------------")

# now we wanna sample on a grid of R1,R2,R3

y_tmp = np.zeros((1,size))

#change_lim = 10.0 # 10.0 worked well with 4x4x4 network
change1_grid = np.linspace(-1.0*change_lim,change_lim,50)
change2_grid = np.linspace(-1.0*change_lim,change_lim,50)
change3_grid = np.linspace(-1.0*change_lim,change_lim,50)
y_tmp[0,0] = Y2[0,0]

# make a loss function to minimize
# squared loss between starting parameters and any target parameters
# we will loop through all sampled values of d,v,a,h and find the one that minimizes this loss

w_d = 1.0e3
w_v = 1.0
w_a = 1.0
w_h = 1.0
loss_function = lambda d,v,a,h : w_d*(X[starting_indx,0]-d)**2 + \
                                     (X[starting_indx,1]-v)**2 + \
                                     (X[starting_indx,2]-a)**2 + \
                                     (X[starting_indx,3]-h)**2

data = []
count_1 = 0
for change1 in change1_grid:
    if count_1 % 10 == 0:
        print(count_1)
    count_1 += 1
    for change2 in change2_grid:
        for change3 in change3_grid:

            #data_tmp = []

            y_tmp[0,1] = Y2[0,1] + change1
            y_tmp[0,2] = Y2[0,2] + change2 
            y_tmp[0,3] = Y2[0,3] + change3 

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
            loss_tmp = loss_function(x_tmp[0,0], x_tmp[0,1], x_tmp[0,2], x_tmp[0,3])
            data.append([change1, change2, change3, x_tmp[0,0], x_tmp[0,1], x_tmp[0,2], x_tmp[0,3], loss_tmp])

data = np.array(data)

# find the argument that gave the min loss

losses = data[:,-1]
min_loss = np.min(losses)
min_indx = np.argmin(losses)
print(f"Minimum of {min_loss} at indx {min_indx}")
print(losses[min_indx])
print("Optimized parameters:")
print(data[min_indx,3:7])

# Plot the data
#param_indx =  # 3: D
               # 4: V
               # 5: A
               # 6: H
hashmap = {3: 'D', 4: 'V', 5: 'A', 6: 'H'}
# declare chi indices
# can be 0, 1, or 2 for the three unused outputs chi1, chi2, and chi3
chi_indx1 = 0
chi_indx2 = 1

for param_indx in range(3,6+1):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    Xs = data[:,chi_indx1]
    Ys = data[:,chi_indx2]
    Zs = data[:,param_indx]
    surf = ax.plot_trisurf(Xs-Xs.mean(), Ys-Ys.mean(), Zs, cmap=cm.jet) #, linewidth=0)
    fig.colorbar(surf, location='left')

    zlim_low = np.min(data[:,param_indx])
    zlim_high = np.max(data[:,param_indx])
    ax.set_zlim(zlim_low, zlim_high)
    ax.set_xlabel(fr'$\Delta \chi_{chi_indx1+1}$')
    ax.set_ylabel(fr'$\Delta \chi_{chi_indx2+1}$')
    ax.set_zlabel(f"{hashmap[param_indx]}")

    plt.savefig(f"surface_{hashmap[param_indx]}.png", dpi=500)
    plt.clf()

# Plot the loss function surface.
# We have 3 main variables to care about: V, A, H since D should not change much.
# So let's plot loss vs. (V,A), (A,H), and (V,H)
# For now we're just plotting in terms of (V,A)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

Xs = data[:,4]
Ys = data[:,5]
Zs = data[:,-1]
surf = ax.plot_trisurf(Xs, Ys, Zs, cmap=cm.jet) #, linewidth=0)
fig.colorbar(surf, location='left')

zlim_low = np.min(data[:,-1])
zlim_high = np.max(data[:,-1])
xlim_low = np.min(data[:,4])
xlim_high = np.max(data[:,4])
ylim_low = np.min(data[:,5])
ylim_high = np.max(data[:,5])

ax.set_xlim(xlim_low, xlim_high)
ax.set_ylim(ylim_low, ylim_high)
ax.set_zlim(zlim_low, zlim_high)
ax.set_xlabel(fr'$V$')
ax.set_ylabel(fr'$A$')
ax.set_zlabel(f"Loss (MSE)")

plt.savefig(f"loss_vs_V_A.png", dpi=500)
plt.clf()