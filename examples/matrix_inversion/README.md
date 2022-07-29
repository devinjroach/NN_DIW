### Inverting matrices

Here we show examples that illustrate that some network architectures are invertible.

This requires thinking of the network as a sequence of matrices being multiplied.

`test1.py` - simple inversion of a small matrix product

`test2.py` - inversion of a small network with tanh activations and no biases
           - here we found that y=tanh(x) activations aren't always invertible across the domain of x

`test3.py` - inversion of a 4 node network with leaky ReLU activations, which are invertible across the entire domain

### Why does this work?

This simple  example below illustrates the matrix form of networks, and how we invert them, for a 2-node network:

![simple_network](https://user-images.githubusercontent.com/11083811/181841443-f0cb6114-9d0d-458f-8ac9-36cb7cbd1a27.jpg)
