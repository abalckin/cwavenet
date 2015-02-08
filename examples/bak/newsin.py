#! /usr/bin/python3
import sys
import tool
sys.path.append('../python/')
import wavenet as wn
import math
import pylab as plb
xmax = 100
xmin = 0
y0 = 0.
a0 = 10.
a1 = 11.
w0 = 0.1
w1 = -0.1
p0 = 1.
p1 =2.
ncount = 10
inp = range(xmin, xmax, 1)
tar = [math.sin(x/10.) for x in inp]
w = wn.Net(ncount, xmin, xmax, y0,
                         a0, a1,  w0, w1, p0, p1)
track = w.train(inp, inp, tar, wn.TrainStrategy.CG, 100, 0.2, 1)
tool.plot(inp, inp, tar, w, track)
plb.show()
sys.exit()
