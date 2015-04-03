#! /usr/bin/python3
import sys
import tool
sys.path.append('../../bin/')
import wavenet as wn
import pylab as plb
import numpy as np
from scipy.integrate import odeint
import cregister as cr

class Caller(object):
    def __call__(self, prg):
        pass

# Mодельная система


def ksi2(t):
    return (np.sin(7*t)+np.cos(9*t))*0.1    


def func1c(y, t):
    y1, y2  = y
    return [y2, -5*y2-(6+3*np.sin(0.5*t))*y1+uc(t)+ksi2(t)]


def func2c(y, t):
    y1, y2  = y
    return [y2, -(2+1*np.sin(0.5*t))*y2-5*y1+uc(t)+ksi2(t)]


def func1g(y, t):
    y1, y2  = y
    return [y2, -5*y2-(6+2*np.sin(0.5*t))*y1+ug(t)+ksi2(t)]


def func2g(y, t):
    y1, y2  = y
    return [y2, -(2+1*np.sin(0.5*t))*y2-5*y1+ug(t)+ksi2(t)]


def uc(t):
    return 1.


def ug(t):
    return 1+0.2*np.cos(2*t)

cb = cr.Caller()
cal = Caller()
cb.setHandler(cal)
k2 = 0.05
N = 100
np.random.seed()
c0 = 0.1
a0 = 0.5
a1 = 0.6
w0 = -0.2
w1 = 0.1
p0 = .5 
p1 = .5
nc = 10
t = np.arange(0, 10, 0.1)
eps = np.random.normal(0., np.sqrt(0.5), t.shape[-1])*k2
inpc = np.vectorize(uc)(t)
inpc = inpc+eps*np.abs(inpc)
eps = np.random.normal(0., np.sqrt(0.5), t.shape[-1])*k2
inpg = np.vectorize(ug)(t)
inpg = inpg+eps*np.abs(inpg)
sysag = odeint(func1g, [1, 1], t)[:, 0]
sysac = odeint(func1c, [1, 1], t)[:, 0]
sysbg = odeint(func2g, [1, 1], t)[:, 0]
sysbc = odeint(func2c, [1, 1], t)[:, 0]
ulist = [inpg, inpc, inpg, inpc]
tarlist = [sysag, sysac, sysbg, sysbc]
eps = np.random.normal(0., np.sqrt(0.5), sysbg.shape)*k2
d = sysbc+eps*np.abs(sysbc)
w = wn.Net(nc, 0., 10., c0,
                          a0, a1,  w0, w1, p0, p1, wn.ActivateFunc.Morlet, 1)
track = w.train(cb, t, inpc, d, wn.TrainStrategy.Gradient, N, 0., 1, True, True)
tool.plot(t, inpc, w, track, orig=sysbc, target=d)
plb.show()

tool.plot(t, inpg, w, track, orig=sysbg)
plb.show()
sys.exit()


















