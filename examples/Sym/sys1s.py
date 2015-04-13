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

def sys1s(x, xmax):
    l = len(x)
    if l >= xmax:
        return x
    else:
        x.append(
           (0.8*x[-1]+0.1*np.sin(l/11+1)+1)
            )
        return np.array(sys1s(x, xmax))

def sys2s(x, xmax):
    l = len(x)
    if l >= xmax:
        return x
    else:
        x.append(
           (0.8*x[-1]+0.2*np.sin(l/11+1)+1)
            )
        return np.array(sys1s(x, xmax))

y0 = 0.
a0 = 10.
a1 = 10
w0 = 0.1
w1 = -0.1
p0 = 1.
p1 =1.
ncount = 10
tl = 50
t = np.arange(0., tl)
inp1 = np.ones(tl)
inp1s = 0.1*np.sin(t/11+1)+1
inp2s = 0.2*np.sin(t/11+1)+1
tar1 = sys1([0], tl)
tar1s = sys1s([0], tl)
tar2s = sys2s([0], tl)
time = np.concatenate([t, t])
inp = np.concatenate([inp1, inp1s])
tar = np.concatenate([tar2s, tar2s]) 

w = wn.Net(ncount,y0,
                         a0, a1,  w0, w1, p0, p1, 1, 1., 1., wn.ActivateFunc.Morlet, 4)
track = w.train(time, inp, tar, wn.TrainStrategy.Gradient, 100, 0., 1)
tool.plot(t, inp2s,  w, track, orig=tar2s)
plb.show()
sys.exit()
