import numpy as np


size = 4

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
#print(w1)
#print(y)

l1 = np.tanh(np.matmul(w1, x))
l2 = np.tanh(np.matmul(w2, l1))

y = np.matmul(w3, l2)

#print(y)

# now go backwards
print("Inverting...")

w1inv = np.linalg.inv(w1)
w2inv = np.linalg.inv(w2)
w3inv = np.linalg.inv(w3)

l2_back = np.arctanh(np.matmul(w3inv,y))
l1_back = np.arctanh(np.matmul(w2inv,l2_back))
x_back = np.matmul(w1inv,l1_back)

print("Recovered x:")
print(x_back)
