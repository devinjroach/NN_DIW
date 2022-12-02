This directly illustrates how to solve the optimization problem.

The `model.pt` came from running `python nn_2.py` on a 4x4x4 network. Here we solve the inverse 
optimization problem on this model.

Simple do 

    python run.py

which will generate a grid of unused outputs and find the minimum such that the target process
params are closest to our current params. The starting and target sample indices are declared as 
variable in this script, along with the size of the grid to sample from.

This creates the five 3D plots in this directory.