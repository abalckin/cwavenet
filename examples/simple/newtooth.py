#! /usr/bin/python3

import sys
import tool
sys.path.append('../../bin/')
import wavenet as wn
import math
import pylab as plb
import numpy as np
import cregister as cr
from scipy import signal
class Caller(object):
    def __call__(self, prg):
        pass

xmin=0.
xmax=0.5
y0 = -1.
p0=.7
p1=1.2
a0=.1
a1=.7
nc=10
w0= -.1
w1= 0.1

cb = cr.Caller()
cal = Caller()
cb.setHandler(cal)

def func1(x):
    return (0.7+x)*np.exp(-.3*x*40 - 0.3)+0.05

#inp = range(xmin, xmax, 1)
#tar = [math.sin(x/1440*3.14*2.) for x in inp]
inp = np.arange(xmin, xmax, 0.01)
tar = np.vectorize(func1)(inp)
eps = 0.1*(np.random.random(tar.shape)-0.5)
tar = tar+eps
w = wn.Net(nc, xmin, xmax, y0,
                         a0, a1,  w0, w1, p0, p1, wn.ActivateFunc.POLYWOG, 4)
track = w.train(cb, inp, inp, tar, wn.TrainStrategy.Gradient, 1000, 0.0, 1, True, True)
tool.plot(inp, inp, tar, w, track)
plb.show()
sys.exit()
