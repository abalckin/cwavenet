#! /usr/bin/python3
import sys
import tool
sys.path.append('../../Release/')
import wavenet as wn
import numpy as np
import pylab as plb
def sys1(x, xmax):
    l = len(x)
    if l >= xmax:
        return x
    else:
        x.append(
           (0.9*x[-1]-0.3*x[-2]+1)
            )
        return np.array(sys1(x, xmax))

xmax = 100
xmin = 1
y0 = 0.
a0 = 3.
a1 = 5.
w0 = 0.005
w1 = -0.005
p0 = 0.29
p1 =0.31
ncount = 40
tl = 30
t = np.arange(0., tl)
inp = np.ones(tl)
tar = sys1([0, 0], tl)
w = wn.Net(ncount,y0,
                         a0, a1,  w0, w1, p0, p1, 1, 0.01, 0.01, wn.ActivateFunc.POLYWOG, 2)
track = w.train(t, inp, tar, wn.TrainStrategy.Gradient, 50, 0.1, 1)
tool.plot(t, inp, tar, w, track)
plb.show()
sys.exit()
