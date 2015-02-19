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
        return 10*np.exp(-.03*x*40 - 0.3)*np.sin((0.6*x*40-0.1))+40*x+.5
    if x >= 0:
        return 10*np.exp(-.03*x*40 - 0.3)*np.sin((0.3*x*40-0.1))+40*x+.5

# Create train samples
x = np.arange(-0.5, 0.5, 0.5/40)
o = np.vectorize(func1)(x)/30
size = len(x)
#print(size)
y = o + (np.random.random(size)-0.5)*1./30.

inp = x.reshape(size,1)
tar = y.reshape(size,1)

n = nl.net.newff([[np.min(inp), np.max(inp)]], [35, 1])
n.trainf = nl.train.train_gd
#import pdb; pdb.set_trace()
error = n.train(inp, tar, epochs=200, goal=0.0, show=0, adapt=True)
y3 = np.array(error)
out = n.sim(inp)
print(error[-1])
# Plot result
import pylab as pl
pl.subplot(211)
pl.plot(error)
pl.xlabel('Epoch number')
pl.ylabel('error (default SSE)')


y3 = out.reshape(size)

pl.subplot(212)
pl.plot(x , y, '-', x, y3, '--',x,o)
pl.legend(['train target', 'net output'])
pl.show()
