# Import modules.

from matplotlib import pyplot as plt
import torch
from torch import nn
from torch import tensor
import scipy.io as sio
import time
import numpy as np
from configuration import Configuration

from dataloader import InRAMDatasetPyTorch, torch_collate, DataLoader

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
            #print(indx)
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

def train(X, Y, model, loss_function, optim, num_epochs):
    loss_history = []

    for epoch in range(num_epochs):
        
        epoch_loss = 0.0
        
        Y_pred = model(X)
        #print(Y_pred.shape)
        #loss = loss_function(Y_pred[:,0], Y[:,0]) #+ loss_function(Y_pred[:,1], Y[:,1])
        #loss = loss_function(Y_pred[:,0], Y[:,0]) + 100.*loss_function(Y_pred[:,1], Y[:,1])
        loss = loss_function(Y_pred[:,0], Y[:,0]) #+ loss_function(Y_pred[:,1], Y[:,0]) #+ 0.5*loss_function(Y_pred[:,0], Y_pred[:,1])
        #loss = loss_function(Y_pred[:,0], Y[:,0]) + loss_function(Y_pred[:,1], Y[:,0]) + loss_function(Y_pred[:,2], Y[:,0]) + loss_function(Y_pred[:,3], Y[:,0])
        if (epoch % 10 == 0): 
            print(f'epoch: {epoch}, loss = {loss.item():.4f}')
        
        loss.backward()
        optim.step()
        optim.zero_grad()

    return Y_pred

# make fitting data
# get inputs and outputs from .mat files

input_mat = sio.loadmat("../../inputs.mat")
inputs = input_mat["inputs"].astype(np.double).T
output_mat = sio.loadmat("../../outputs.mat")
outputs = output_mat["outputs"].astype(np.double).T

ndat = np.shape(outputs)[0]

# convert numpy arrays to torch tensors

X = torch.from_numpy(inputs).float()
Y = torch.from_numpy(outputs).float()
print(X.shape)
print(Y.shape)
targets = Y

#configs = ndat*[Configuration()]
#print(configs)
# loop thru all configs and set inputs/targets
configs = []
for i in range(0,ndat):
    configs.append(Configuration(X[i],Y[i]))
    #configs[i].inputs=X[i]
    #configs[i].targets=Y[i]
    #print(configs[i].inputs)
    #print(configs[i].targets)

# set neural network parameters

#layer_sizes = [4,4,4] # 3 layer total (input, hidden, and output)
#layer_sizes = [4,4,4,4,4,4]
#layer_sizes = [4,4,4,4,4,4,4,4,4]
#layer_sizes = [4,4,4,4,4,4,4,4,4]
layer_sizes = [4,4,4,4,4]

# define model

model = Nonlinear(layer_sizes)

# get random training indices

p = 0.8
training_bool_indices = np.random.choice(a=[True, False], size=ndat, p=[p, 1-p])
total_data = InRAMDatasetPyTorch(configs)
training_indices = [i for i, x in enumerate(training_bool_indices) if x]
testing_indices = [i for i, x in enumerate(training_bool_indices) if not x]
training_data = torch.utils.data.Subset(total_data, training_indices)
validation_data = torch.utils.data.Subset(total_data, testing_indices)

"""
# randomly shuffle and split into training and testing fractions

training_fraction = 0.8
train_size = int(training_fraction * len(total_data))
test_size = len(total_data) - train_size
training_data, validation_data = \
    torch.utils.data.random_split(total_data, 
                                  [train_size, test_size])


#training_data = torch.utils.data.Subset(X, training_indices)
"""

# NOTE: If shuffling, the final training data indices will be all fucked up when plotting, so it's 
#       best to just save the model and then re-evaluate it on everything maybe. 
training_loader = DataLoader(training_data,
                                  batch_size=int(len(training_data)/10),
                                  shuffle=True, #True,
                                  collate_fn=torch_collate,
                                  num_workers=0)
validation_loader = DataLoader(validation_data,
                                  batch_size=len(validation_data),
                                  shuffle=False, #True,
                                  collate_fn=torch_collate,
                                  num_workers=0)


optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_function = nn.MSELoss()


train_losses_epochs = []
val_losses_epochs = []
train_pred_epochs = []
val_pred_epochs = []
train_target_epochs = []
val_target_epochs = []
nepochs = 5000 # 15000 if batch size = len(training_data)/3
for epoch in range(nepochs):
    print(f"----- epoch: {epoch}")
    #start = time()

    # loop over training data

    train_losses_step = []
    train_pred_step = []
    train_target_step = []
    loss = None
    model.train()
    for i, batch in enumerate(training_loader):
        inputs = batch['inputs'].requires_grad_(True)
        targets = batch['targets'].requires_grad_(True)
        predictions = model(inputs)
        if (len(predictions.size())==1):
            predictions = torch.unsqueeze(predictions,dim=0)

        #print(inputs)
        #print(targets)
        #print(predictions)
        #assert(False)

        loss = loss_function(predictions[:,0], targets[:,0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses_step.append(loss.item())

        train_pred_step.append(predictions.detach().numpy())
        train_target_step.append(targets.detach().numpy())

    # loop over validation data

    val_losses_step = []
    val_pred_step = []
    val_target_step = []
    model.eval()
    for i, batch in enumerate(validation_loader):
        inputs = batch['inputs']
        targets = batch['targets']
        predictions = model(inputs)

        #print(predictions)
        #print(targets)

        loss = loss_function(predictions[:,0], targets[:,0])
        val_losses_step.append(loss.item())

        val_pred_step.append(predictions.detach().numpy())
        val_target_step.append(targets.detach().numpy())

    print(f"Batch averaged train/val loss: {np.mean(np.asarray(train_losses_step))} {np.mean(np.asarray(val_losses_step))}")
    train_losses_epochs.append(np.mean(np.asarray(train_losses_step)))
    val_losses_epochs.append(np.mean(np.asarray(val_losses_step)))
    #print(f"Batch averaged train/val loss: {np.mean(np.asarray(train_losses_step))} {np.mean(np.asarray(val_losses_step))}")

    #train_pred_step = np.concatenate(train_pred_step, axis=0)
    #print(np.shape(train_pred_step))
    #train_pred_epochs.append(train_pred_step)

    if (epoch==nepochs-1):
        train_pred_step = np.concatenate(train_pred_step, axis=0)
        val_pred_step = np.concatenate(val_pred_step, axis=0)
        train_target_step = np.concatenate(train_target_step, axis=0)
        val_target_step = np.concatenate(val_target_step, axis=0)

torch.save(model, "model.pt")

# plot loss vs. epochs

epochs_nums = np.arange(nepochs)
plt.plot(epochs_nums, train_losses_epochs, 'b-', linewidth=3)
plt.plot(epochs_nums, val_losses_epochs, 'r-', linewidth=3)
#plt.plot(lims, lims, 'k-')
plt.xlabel("Epochs")
plt.ylabel(r'Loss function')
plt.yscale('log')
plt.legend(["Train", "Validation"])
#plt.xlim(lims[0], lims[1])
#plt.ylim(lims[0], lims[1])
plt.savefig("error_vs_epochs.png", dpi=500)

# plot predictions vs. actual

train_preds = np.array(train_pred_step)
val_preds = np.array(val_pred_step)
train_target = np.array(train_target_step)
val_target = np.array(val_target_step)
print(train_preds.shape)
xaxis = np.arange(0,196,1,dtype=int)
plt.clf()
plt.plot(xaxis, outputs[:,0], 'k-')
plt.plot(xaxis[training_indices], train_preds[:,0], 'ko')
plt.plot(xaxis[testing_indices], val_preds[:,0], 'ro')
plt.legend(["Truth", "Training", "Validation"])
#plt.plot(xaxis, outputs_model.detach().numpy()[:,1], 'ko', markersize=0.5)
#plt.plot(xaxis, outputs_model.detach().numpy()[:,2], 'go', markersize=0.5)
#plt.plot(xaxis, outputs_model.detach().numpy()[:,3], 'ro', markersize=0.5)
plt.savefig("all_output.png", dpi=500)

# plot the model vs. target
plt.clf()
xaxis = np.arange(0,196,1,dtype=int)
yaxis = outputs[:,0]
preds = model(X)
plt.plot(xaxis, yaxis, 'k-')
plt.plot(xaxis, preds.detach().numpy()[:,0], 'ko')
plt.plot(xaxis[testing_indices], val_preds[:,0], 'ro')
#plt.plot(X.numpy(), predicted, 'b')
plt.savefig("evaluate.png", dpi=500)
#plt.show()


# plot the model vs. target

plt.clf()
plt.plot(train_target[:,0], train_preds[:,0], 'bo')
plt.plot(val_target[:,0], val_preds[:,0], 'ro')
#plt.plot(X.numpy(), predicted, 'b')
plt.savefig("comparison.png", dpi=500)
#plt.show()