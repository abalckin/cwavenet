#! /usr/bin/python3
import sys
import tool
sys.path.append('../../bin/')
import wavenet as wn
import pylab as plb
import numpy as np
from scipy.integrate import odeint
import cregister as cr
from scipy import signal

class Caller(object):
    def __call__(self, prg):
        pass

# Mодельная система


def ksi2(t):
    return (np.sin(7*t)+np.cos(9*t))*0.1    


def func1c(y, t):
    y1, y2  = y
    return [y2, -5*y2-(6+2*np.sin(0.5*t))*y1+uc(t)+ksi2(t)]


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
    return 0.


def ug(t):
    return 0.

cb = cr.Caller()
cal = Caller()
cb.setHandler(cal)
k2 = 0.05
N = 200
np.random.seed()
c0 = 1.0
a0 = 1.
a1 = 1.
w0 = -0.
w1 = 0.
p0 = 1. 
p1 = 1.
nc = 10
znum = 2
t = np.arange(0, 10, 0.1)
eps = np.random.normal(0., k2, t.shape[-1])
inpc = np.vectorize(uc)(t)
inpc = inpc+eps*np.abs(inpc)
eps = np.random.normal(0., k2, t.shape[-1])
inpg = np.vectorize(ug)(t)
inpg = inpg+eps*np.abs(inpg)
sysag = odeint(func1g, [6., 5.], t)[:, 0]
sysac = odeint(func1c, [6., 5.], t)[:, 0]
sysbg = odeint(func2g, [6., 5.], t)[:, 0]
sysbc = odeint(func2c, [6., 5.], t)[:, 0]
ulist = [inpg, inpc, inpg, inpc]
tarlist = [sysag, sysac, sysbg, sysbc]
eps = np.random.normal(0., k2, sysbg.shape)
dg2 = sysbg+eps
#dg = sysag+eps*np.abs(sysag)
w = wn.Net(nc,c0,a0, a1,  w0, w1, p0, p1,znum,0.5,1., wn.ActivateFunc.Morlet, 4)
track = w.train(t, t, dg2, wn.TrainStrategy.BFGS, N, 0., 1, True, True)
#track = w.train(t, t, dg, wn.TrainStrategy.BFGS, N, 0., 1, False, False)
tool.plot(t, t, w, track, orig=sysbg, target=dg2)
plb.show()

#tool.plot(t, inpg, w, track, orig=sysbg)
#plb.show()
sys.exit()


















