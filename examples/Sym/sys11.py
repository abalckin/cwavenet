#! /usr/bin/python3
import sys
import tool
sys.path.append('../../bin/')
import wavenet as wn
import numpy as np
import pylab as plb
def sys1(x, xmax):
    l = len(x)
    if l >= xmax:
        return x
    else:
        x.append(
           (0.8*x[-1]+1)
            )
        return np.array(sys1(x, xmax))

xmax = 100
xmin = 1
y0 = 0.
a0 = 3.
a1 = 3.
w0 = 0.05
w1 = -0.05
p0 = 0.3
p1 =0.3
ncount = 16
tl = 30
t = np.arange(0., tl)
inp = np.ones(tl)
tar = sys1([0], tl)
w = wn.Net(ncount,y0,
                         a0, a1,  w0, w1, p0, p1, 1, 1., 1., wn.ActivateFunc.POLYWOG, 2)
track = w.train(t, inp, tar, wn.TrainStrategy.Gradient, 50, 0., 1)
tool.plot(t, inp,  w, track, orig=tar)
plb.show()
sys.exit()
