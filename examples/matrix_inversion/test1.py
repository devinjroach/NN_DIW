import numpy as np


size = 4

x = np.random.rand(size,1)
w1 = np.random.rand(size,size)
w2 = np.random.rand(size,size)
w3 = np.random.rand(size,size)
y = np.random.rand(size,1)

print(x)
#print(w1)
#print(y)

l1 = np.matmul(w1, x)
l2 = np.matmul(w2, l1)

y = np.matmul(w3, l2)

#print(y)

# now go backwards

w1inv = np.linalg.inv(w1)
w2inv = np.linalg.inv(w2)
w3inv = np.linalg.inv(w3)

l2_back = np.matmul(w3inv,y)
l1_back = np.matmul(w2inv,l2_back)
x_back = np.matmul(w1inv,l1_back)

print(x_back)
