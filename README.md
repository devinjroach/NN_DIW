# DIW neural net

## Training data

`inputs.mat` - a 4x196 array. Rows are features and columns are data points

`outputs.mat` - a 1x196 array. Single output target row and columns are data points

## Simple starting examples

`examples/simple/nn_1.py`

Simple 1D regression network that fits a parabola. This should be easily extended to the 4 input example here. 

`examples/diw/nn_2.py`

Applying this simple regression example to the DIW training set. Layer sizes and training weights can be tweaked to give desired results.
