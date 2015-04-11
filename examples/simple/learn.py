#! /usr/bin/python3
import sys
import tool
sys.path.append('../../bin/')
import wavenet as wn
import pylab as plb
import numpy as np
from scipy.integrate import odeint
import multiprocessing
# Mодельная система


def ksi2(t):
    return (np.sin(7*t)+np.cos(9*t))*0.

def func1c(y, t):
    y1, y2 = y
    return [y2, -5*y2-6*y1+uc(t)+ksi2(t)]


def func2c(y, t):
    y1, y2 = y
    return [y2, -2*y2-5*y1+uc(t)+ksi2(t)]

def func2t(y, t):
    y1, y2 = y
    return [y2, -2*y2-5*y1+ut(t)+ksi2(t)]

def func2g(y, t):
    y1, y2 = y
    return [y2, -2*y2-5*y1+ug(t)+ksi2(t)]


def func1g(y, t):
    y1, y2 = y
    return [y2, -5*y2-6*y1+ug(t)+ksi2(t)]


def uc(t):
    return 1.

def ut(t):
    return 1.+0.1*np.sin(2*t)

def ug(t):
    return 1.+0.2*np.sin(2*t)

cpu_count = multiprocessing.cpu_count()
k2 = 0.0
N = 100
np.random.seed()
c0 = 0.
a0 = 0.7
a1 = 0.7
w0 = -0.01
w1 = 0.01
p0 = 1.
p1 = 1.
nc = 10
fcount = 0
bcount = 0
f1 = .0
f0 = .0
fb = .01
d0 = 0.1
d1 = 0.1
t = np.arange(0, 10., 0.1)
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
sysbc = np.sin(t)
inpt = np.vectorize(ut)(t)
inpt = inpt+eps*np.abs(inpt)
sysbt = odeint(func2t, [1, 1], t)[:, 0]
ulist = [inpg, inpc, inpg, inpc]
tarlist = [sysag, sysac, sysbg, sysbc]
eps = np.random.normal(0., np.sqrt(0.5), sysbc.shape)*k2
dc = sysbc+eps*np.abs(sysbc)
dg = sysbg+eps*np.abs(sysbg)
w = wn.Net(nc, c0,
           a0=a0, a1=a1,
           w0=w0, w1=w1,
           p0=p0, p1=p0,
           f=wn.ActivateFunc.Morlet,
           numberOfThreads=cpu_count,
           f0=f0,
           fcount=fcount,
           fbcoef=fb)
time = np.ma.concatenate([t, t])
inputs = np.ma.concatenate([inpc, inpg])
targets = np.ma.concatenate([dc, dg])
#track = w.train(t, inpg, d, wn.TrainStrategy.Gradient, N, 0., 1, True, True)
track = w.train(time, inputs, targets, wn.TrainStrategy.Gradient, N, 0., 1, True, True)

#import pdb; pdb.set_trace()
tool.plot(t, inpc, dc, w, track, orig=sysbc)
plb.show()
tool.plot(t, inpt, dc, w, track, orig=sysbt)
plb.show()
sys.exit()


















