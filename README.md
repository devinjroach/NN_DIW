# DIW neural net


## Training data

`inputs.mat` - a 4x196 array. Rows are features and columns are data points

`outputs.mat` - a 1x196 array. Single output target row and columns are data points

## Simple starting examples

`examples/simple/nn_1.py`

Simple 1D regression network that fits a parabola. This should be easily extended to the 4 input example here. 

`examples/diw/nn_2.py`

Applying this simple regression example to the DIW training set. Layer sizes and training weights can be tweaked to give desired results.

`examples/diw_matrix/nn_quadratic.py`

Simple 1D regression network that fits a parabola again, except this time using matrix multiplication and weight matrices. From this formulation, it's easier to do the inverse operation by inverting the matrices and multiplying backwards. This may only be possible, however, if all matrices in the layers have the shame shape, due to inverse properties of matrix multiplication.

## 22-07-29
We successfuly made 4x4 layer models with 4 outputs, the structure is invertible.
Now convert this to the diw_matrix example format.
Then train the matrix format model.
Then invert the matrices (weights, activations, etc.) and go backwards. That’s our new inverse model for production use.
