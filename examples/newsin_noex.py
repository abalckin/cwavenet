#! /usr/bin/python3
import sys
import tool
sys.path.append('../python/')
import wavenet as wn
import math
xmax = 100
xmin = 0
y0 = 0.
a0 = 10.
w0 = 0.1
p0 = 1.
ncount = 10
inp = range(xmin, xmax, 1)
tar = [math.sin(x/10.) for x in inp]
w = wn.Net(ncount, xmin, xmax, y0,
                         a0, w0)
track = w.train(inp, tar, wn.TrainStrategy.CG, 50, 0.2, 1, False, False)
tool.show(inp, tar, w, track)
sys.exit()
