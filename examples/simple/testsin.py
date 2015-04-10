#! /usr/bin/python3
import sys
import tool
sys.path.append('../../bin/')
import wavenet as wn
import math
import pylab as plb
xmax = 150
xmin = 0
c0 = 0.
a0 = 10.
a1 = 10.
w0 = 0.005
w1 = -0.005
p0 = 0.3
p1 = 0.3
ncount = 15
inp = range(xmin, xmax, 1)
t = [x/150. for x in inp]
tar = [math.sin(x/10*3.14*2.) for x in inp]
w = wn.Net(ncount,
           c=c0,
           a0=a0,
           a1=a1,
           w0=w0,
           w1=w1,
           p0=p0,
           p1=p1)
track = w.train(t, inp, tar, wn.TrainStrategy.BFGS, 100, 0.01, 1)
tool.plot(inp, inp, tar, w, track)
plb.show()
sys.exit()
