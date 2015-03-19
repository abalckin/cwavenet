#! /usr/bin/python3
import sys
import tool
sys.path.append('../../bin/')
import wavenet as wn
import math
import pylab as plb
xmax = 144
xmin = 1
y0 = 0.
a0 = 99.
a1 = 101.
w0 = 0.005
w1 = -0.005
p0 = 0.29
p1 =0.31
ncount = 9
inp = range(xmin, xmax, 1)
tar = [math.sin(x/1440*3.14*2.) for x in inp]
w = wn.Net(ncount, xmin, xmax, y0,
                         a0, a1,  w0, w1, p0, p1, wn.ActivateFunc.Morlet, 2)
track = w.train(inp, inp, tar, wn.TrainStrategy.BFGS, 50, 0.2, 1)
tool.plot(inp, inp, tar, w, track)
plb.show()
sys.exit()
