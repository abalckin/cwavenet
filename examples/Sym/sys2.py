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
    return (np.sin(7*t)+np.cos(9*t))*0.1

def func1c(y, t):
    y1, y2 = y
    return [y2, -5*y2-(6+1*np.sin(0.5*t))*y1+uc(t)+ksi2(t)]


def func2c(y, t):
    y1, y2 = y
    return [y2, -(2+1*np.sin(0.5*t))*y2-5*y1+uc(t)+ksi2(t)]


def func1g(y, t):
    y1, y2 = y
    return [y2, -5*y2-(6+2*np.sin(0.5*t))*y1+ug(t)+ksi2(t)]


def func2g(y, t):
    y1, y2 = y
    return [y2, -(2+1*np.sin(0.5*t))*y2-5*y1+ug(t)+ksi2(t)]


def uc(t):
    return 1.+0.1*np.sin(2*t)


def ug(t):
    return 1+0.3*np.sin(2*t)

#def func5(y, tmax):
#    if len(y)==tmax:
#        return y
#    else:
#        t = len(y)
#        y.append(y[-1]*y[-2]/(1+y[-1]**2+y[-2]**2)+un(t))
#        return  np.array(func5(y, tmax))

cpu_count = multiprocessing.cpu_count()
k2 = 0.05
N = 300
np.random.seed()
c0 = 0.0
a0 = .7
a1 = .7
w0 = -0.001
w1 = 0.001
p0 = .7
p1 = .7
nc = 10
fcount = 2
f0 = 0.1
fb = 0.1
t = np.arange(0, 10, 0.1)
eps = np.random.normal(0., np.sqrt(0.5), t.shape[-1])*k2
inpc = np.vectorize(uc)(t)
inpc = inpc+eps*np.abs(inpc)
eps = np.random.normal(0., np.sqrt(0.5), t.shape[-1])*k2
inpg = np.vectorize(ug)(t)
inpg = inpg+eps*np.abs(inpg)
#inpn = np.vectorize(un)(t)
#sysnl = func5([0., 0.], 100)
sysag = odeint(func1g, [1, 1], t)[:, 0]
sysac = odeint(func1c, [1, 1], t)[:, 0]
sysbg = odeint(func2g, [1, 1], t)[:, 0]
sysbc = odeint(func2c, [1, 1], t)[:, 0]

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
#import pdb; pdb.set_trace()
#track = w.train(t, inpg, d, wn.TrainStrategy.Gradient, N, 0., 1, True, True)
time = np.ma.concatenate([t, t])
inputs = np.ma.concatenate([inpc, inpg])
targets = np.ma.concatenate([dc, dg])

track = w.train(time, inputs, targets, wn.TrainStrategy.BFGS, N, 0., 1, True, True)

#ans = w.sim(t, inpc)
#import pdb; pdb.set_trace()
tool.plot(t, inpc, w, track, orig=sysbc, target=dc)
plb.show()

tool.plot(t, inpg, w, track, orig=sysbg)
plb.show()
sys.exit()


















