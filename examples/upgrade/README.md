# Upgraded example

To run a fit, do 

    python fit.py

This creates a `model.pt` file, which we'll use in `evaluate.py` to perform the inverse with

    python evaluate.py

# Showing inversion of pytorch model

Do

    python optinv.py

And see

    example input:
    tensor([[0.2500, 1.0000, 5.0000, 0.0150]])
    model prediction:
    tensor([[ 63.5977,  -9.4492,   7.7778, -14.7959]],
          grad_fn=<UnsqueezeBackward0>)
    [[ 63.597736   -9.449163    7.7778387 -14.795891 ]]
    63.597736
    R1: -382.2424011230469 0.13716652989387512
    R2: -15.729334831237793 322.5234680175781
    R3: -167.4440155029297 -14.618879318237305
    [[ 63.59773636  -9.44916344   7.77783871 -14.79589081]]
    (1, 4)
    [[0.25000191 0.99999668 5.00000436 0.01499993]]

(or something like that, it'll change depending on model).

To invert and observe sensitive behavior of random vars, do:

    python optinv-d2.py
