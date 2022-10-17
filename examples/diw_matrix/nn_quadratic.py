# Import modules.

from matplotlib import pyplot as plt
import torch
from torch import nn
from torch import tensor

# Define the parabolic dataset.
X = torch.linspace(-2., 2., 30).unsqueeze(1)
print("X shape:")
print(X.shape)
# Define the target data to fit to, e.g. a parabola Y = X**2
Y = torch.square(X).flatten()
print("Y shape:")
print(Y.shape)
#plt.plot(X.squeeze().numpy(), Y.numpy(), 'r.')
#plt.xlabel("X")
#plt.ylabel("Y")

"""
Function for training the NN.
Inputs: X - training data inputs tensor with shape [ndata, nfeatures].
        Y - training data target tensor with shape [ndata]
        model - a pytorch model
        loss_function - how to calculate the loss or error
        optim - optimizer 
        num_epochs - number of training iterations
"""
def train(X, Y, model, loss_function, optim, num_epochs):
    loss_history = []
    
    def extra_plot(*args):
        plt.plot(X.squeeze(1).numpy(), Y.numpy(), 'r.', label="Ground truth")
        plt.plot(X.squeeze(1).numpy(), model(X).detach().numpy(), '-', label="Model")
        plt.title("Prediction")
        plt.legend(loc='lower right')
    
    #liveloss = PlotLosses(extra_plots=[extra_plot], plot_extrema=False)

    for epoch in range(num_epochs):
        
        epoch_loss = 0.0
        
        Y_pred = model(X)
        #print(Y_pred.shape)
        loss = loss_function(Y_pred, Y)
        if (epoch % 10 == 0): 
            print(f'epoch: {epoch}, loss = {loss.item():.4f}')
        
        loss.backward()
        optim.step()
        optim.zero_grad()

"""
Class describing our pytorch model or neural network architecture.
It inherits objects from the nn.Module class. 
Here we define our neural network structure, which has 1 hidden layer and a variable number of nodes in the layer.
"""
class Nonlinear(nn.Module):
    
    # Initializer.
    def __init__(self, hidden_size=2):
        super().__init__()
        
        self.layer_1_weights = nn.Parameter(torch.randn(1, hidden_size))
        self.layer_1_bias = nn.Parameter(torch.randn(hidden_size)) 
        
        self.layer_2_weights = nn.Parameter(torch.randn(hidden_size, 1) ) 
        self.layer_2_bias = nn.Parameter(torch.randn(1))
        
    # Feed fordward function - here we use matrix/tensor operations to calculate the output of the neural network
    #                          given some inputs "x".
    def forward(self, x):
        print(x.size())
        print(self.layer_1_weights.size())
        x = x.matmul(self.layer_1_weights).add(self.layer_1_bias)
        x = x.sigmoid()
        x = x.matmul(self.layer_2_weights).add(self.layer_2_bias)
        return x.squeeze()
    
    # Optional function for setting the NN weights manually instead of random initialization. 
    def nonrandom_init(self):
        self.layer_1_weights.data = tensor([[1.1, 0.8]])
        self.layer_1_bias.data = tensor([0.5 , -0.7]) 
        self.layer_2_weights.data = tensor([[0.3], [-0.7]])
        self.layer_2_bias.data = tensor([0.2])

# Define model.
nonlinear_model = Nonlinear(hidden_size=5)
nonlinear_model.nonrandom_init()

# Define optimizer.
#optim = torch.optim.SGD(nonlinear_model.parameters(), lr=0.2)
optim = torch.optim.Adam(nonlinear_model.parameters(), lr=0.1)
loss_function = nn.MSELoss()

# Train the model 
train(X, Y, nonlinear_model, loss_function, optim, num_epochs=200)

# Now we can see how well our neural network models the data.
predicted = nonlinear_model(X).detach().numpy()

plt.plot(X.numpy(), Y.numpy(), 'ro')
plt.plot(X.numpy(), predicted, 'b')
plt.show()


