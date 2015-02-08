#! /usr/bin/python3
# -*- coding: utf-8 -*-
""" 
Example of use multi-layer perceptron
=====================================

Task: Approximation function: 1/2 * sin(x)

"""

import neurolab as nl
import numpy as np

def func1(x):
    if x < 0:
        return (10*np.exp(-.03*x*40 - 0.3)*np.sin((0.6*x*40-0.1))+40*x+.5)/20.
    if x >= 0:
        return (10*np.exp(-.03*x*40 - 0.3)*np.sin((0.3*x*40-0.1))+40*x+.5)/20.

# Create train samples
x = np.linspace(-0.5, 0.5, 50)
y = np.vectorize(func1)(x)
size = len(x)
y = y + (np.random.random(size)-0.5)*5/10.

inp = x.reshape(size,1)
tar = y.reshape(size,1)

# Create network with 2 layers and random initialized
net = nl.net.newff([[-1, 1]],[16, 1])

# Train network
error = net.train(inp, tar, epochs=500, show=100, goal=0.001)

# Simulate network
out = net.sim(inp)

# Plot result
import pylab as pl
pl.subplot(211)
pl.plot(error)
pl.xlabel('Epoch number')
pl.ylabel('error (default SSE)')


y3 = out.reshape(size)

pl.subplot(212)
pl.plot(x , y, '.', x, y3, 'p')
pl.legend(['train target', 'net output'])
pl.show()
