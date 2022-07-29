"""
Simple example showing how a 4x4x4x4 network with invertible actions (leaky ReLU)
is invertible. 
"""

import numpy as np

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
leaky_relu_slope = 0.05

test_value = -100
# this asserts that the inverse activation is coded correctly
assert (test_value == inv_leaky_relu(leaky_relu(test_value)))

x = np.random.rand(size,1)
# layer 1 weights and biases
w1 = np.random.rand(size,size)
b1 = np.random.rand(size,1)
# layer 2 weights and biases
w2 = np.random.rand(size,size)
b2 = np.random.rand(size,1)
# layer 3 weights and biases
w3 = np.random.rand(size,size)
b3 = np.random.rand(size,1)
y = np.random.rand(size,1)

print("Initial x:")
print(x)

# hidden layers
l1 = leaky_relu(np.matmul(w1, x) + b1)
l2 = leaky_relu(np.matmul(w2, l1) + b2)
# output
y = np.matmul(w3, l2) + b3

#print(y)

# now go backwards to invert 
print("Inverting...")

w1inv = np.linalg.inv(w1)
w2inv = np.linalg.inv(w2)
w3inv = np.linalg.inv(w3)

l2_back = inv_leaky_relu(np.matmul(w3inv,y - b3))
l1_back = inv_leaky_relu(np.matmul(w2inv,l2_back - b2))
x_back = np.matmul(w1inv,l1_back - b1)

print("Recovered x:")
print(x_back)
