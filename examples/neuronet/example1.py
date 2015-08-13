#! /usr/bin/python3
# -*- coding: utf-8 -*-
""" 
Example of use multi-layer perceptron
=====================================

Task: Approximation function: 1/2 * sin(x)

"""

import neurolab as nl
import numpy as np
from scipy import signal
def func1(x):
    if x < -2:
        return -2.186*x-12.864
    elif -2<=x<0:
        return 4.246*x
    elif 0<=x<10:
        return 10*np.exp(-0.05*x-0.5)*np.sin((0.03*x+0.7)*x) 

# Create train samples

x = np.arange(-10, 10, 0.1)

o = np.vectorize(func1)(x)
size = len(x)
#print(size)
y = o #/+ (np.random.random(size)-0.5)*0.1
x=x-np.min(x)
x=x/np.max(x)-0.5
y=y-np.min(y)
y=y/np.max(y)-0.5
inp = x.reshape(size,1)
tar = y.reshape(size,1)

n = nl.net.newff([[np.min(inp), np.max(inp)]], [50, 1])
#n.trainf = nl.train.train_bfgs
n.trainf = nl.train.train_gd
#import pdb; pdb.set_trace()
error = n.train(inp, tar, epochs=1000, goal=0.0, show=1, adapt=False, lr=0.00005)
e = np.array(error)
out = n.sim(inp)
print(error[-1])
# Plot result
import pylab as plb
from matplotlib import rc
rc('text', usetex=True)
rc('text.latex', unicode=True)
rc('text.latex', preamble=r'\usepackage[russian]{babel}')
rc('font',**{'size':'19'})
plb.rc('font', family='serif')
plb.rc('font', size=13)
plb.figure('Апроксимация')
plb.subplot(211)
plb.plot(x, y, label='$\check{y}(t)$')
plb.xlabel('$\check{t}$')
plb.plot(x, out, linestyle='--', label='$\hat{y}(t)$')
plb.legend(loc=4)
plb.subplot(212)
plb.ylabel('Энергия ошибки, E')
plb.plot(e)
plb.xlabel('Эпохи, n')
plb.show()
