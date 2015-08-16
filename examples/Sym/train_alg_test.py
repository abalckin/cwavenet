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
    return [y2, -5*y2-(6+3*np.sin(0.5*t))*y1+uc(t)+ksi2(t)]


def func2c(y, t):
    y1, y2 = y
    return [y2, -(2+1*np.sin(0.5*t))*y2-5*y1+uc(t)+ksi2(t)]


def func1g(y, t):
    y1, y2 = y
    return [y2, -5*y2-(6+20*np.sin(3*t))*y1+ug(t)+ksi2(t)]


def func2g(y, t):
    y1, y2 = y
    return [y2, -(2+1*np.sin(0.5*t))*y2-5*y1+ug(t)+ksi2(t)]


def uc(t):
    return 1.


def ug(t):
    return 1+0.2*np.sin(2*t)

def  un(t):
    return 0.8*np.sin(2*np.pi*t/50)

def func5(y, tmax):
    if len(y)==tmax:
        return y
    else:
        t = len(y)
        y.append(y[-1]*y[-2]/(1+y[-1]**2+y[-2]**2)+un(t))
        return  np.array(func5(y, tmax))

cpu_count = multiprocessing.cpu_count()
k2 = 0.05
N = 5000
np.random.seed()
c0 = 0.0
a0 = 1.
a1 = 1.
w0 = 0.001
w1 = 0.001
p0 = 1.
p1 = 1.
nc = 10
fcount = 0
f0 = 0.01
fb = 0.01
t = np.arange(0, 10, 0.1)
eps = np.random.normal(0., np.sqrt(0.5), t.shape[-1])*k2
inpc = np.vectorize(uc)(t)
inpc = inpc+eps*np.abs(inpc)
eps = np.random.normal(0., np.sqrt(0.5), t.shape[-1])*k2
inpg = np.vectorize(ug)(t)
inpg = inpg+eps*np.abs(inpg)
inpn = np.vectorize(un)(t)
sysnl = func5([0., 0.], 100)
sysag = odeint(func1g, [1, 1], t)[:, 0]
sysac = odeint(func1c, [1, 1], t)[:, 0]
sysbg = odeint(func2g, [1, 1], t)[:, 0]
sysbc = odeint(func2c, [1, 1], t)[:, 0]

ulist = [inpg, inpc, inpg, inpc]
tarlist = [sysag, sysac, sysbg, sysbc]
eps = np.random.normal(0., np.sqrt(0.5), sysbc.shape)*k2
d = sysag+eps*np.abs(sysag)
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

##track = w.train(t, inpg, sysag, wn.TrainStrategy.Gradient, N, 0.015, 1, True, True)
##track = w.train(t, inpg, sysag, wn.TrainStrategy.CG, N, 0.015, 1, True, True)
track = w.train(t, inpg, sysag, wn.TrainStrategy.BFGS, N, 0.015, 1, True, True)
print(len(track["e"][0]))
ans = w.sim(t, inpg)
#import pdb; pdb.set_trace()
tool.plot(t, inpg, w, track, orig=sysag, target=d)
plb.show()

#tool.plot(t, inpg, w, track, orig=sysbg)
#plb.show()
sys.exit()


















