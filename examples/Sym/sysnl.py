#! /usr/bin/python3
import sys
import tool
sys.path.append('../../bin/')
import wavenet as wn
import pylab as plb
import numpy as np
import multiprocessing
# Mодельная система


def func1(y, t):
    n = len(y)
    if n<t:
        y.append(y[-1]*y[-2]/(1+y[-1]**2+y[-2]**2)+u1(n)+np.random.normal(0., ks)+1.)
        return func1(y, t)
    else:
        return(y)


        
def func2(y, t):
    n = len(y)
    if n<t:
        y.append(y[-1]*y[-2]/(1+y[-1]**2+y[-2]**2)+u2(n)+np.random.normal(0., ks)+1.)
        return func2(y, t)
    else:
        return(y)

def func3(y, t):
    n = len(y)
    if n<t:
        y.append(y[-1]*y[-2]/(1+y[-1]**2+y[-2]**2)+u3(n)+np.random.normal(0., ks)+1.)
        return func3(y, t)
    else:
        return(y)

def func4(y, t):
    n = len(y)
    if n<t:
        y.append(y[-1]*y[-2]/(1+y[-1]**2+y[-2]**2)+u4(n)+np.random.normal(0., ks)+1.)
        return func4(y, t)
    else:
        return(y)

def u1(t):
    return 1.

def u2(t):
    return 1.5*np.sin(np.pi*3*t/50+1)+1

def u3(t):
    return np.sin(np.pi*2*t/50)+1

def u4(t):
    return 1. if t == 50 else 0.

cpu_count = multiprocessing.cpu_count()
N = 500
np.random.seed()
c0 = 0.0
a0 = 148.5
a1 = 151.5
w0 = -0.001
w1 = 0.001
p0 = 1.
p1 = 1.
nc = 12
fcount = 0
f0 = 0.
fb = 1.
T=150
ky=0.1
ks=0.02
ku=0.02
t = np.arange(0, T, 1.)
inp1 = np.vectorize(u1)(t)
inp2 = np.vectorize(u2)(t)
inp3 = np.vectorize(u3)(t)
inp4 = np.vectorize(u4)(t)

sys1 = func1([.0, .0], T)
sys2 = func2([.0, .0], T)
sys3 = func3([.0, .0], T)
sys4 = func4([.0, .0], T)

eps = np.random.normal(0., ku, t.shape[-1])
inp1= inp1+eps
eps = np.random.normal(0., ku, t.shape[-1])
inp2 = inp2+eps
eps = np.random.normal(0., ku, t.shape[-1])
inp3 = inp3+eps
eps = np.random.normal(0., ky, t.shape[-1])
tar1 = sys1+eps
eps = np.random.normal(0., ky, t.shape[-1])
tar2 = sys2+eps
eps = np.random.normal(0., ky, t.shape[-1])
tar3 = sys3+eps
eps = np.random.normal(0., ky, t.shape[-1])
tar4 = sys4+eps

w = wn.Net(nc, c0,
           a0=a0, a1=a1,
           w0=w0, w1=w1,
           p0=p0, p1=p0,
           f=wn.ActivateFunc.RASP1,
           numberOfThreads=cpu_count,
           f0=f0,
           fcount=fcount,
           fbcoef=fb)
#import pdb; pdb.set_trace()
time = np.ma.concatenate([t, t])
inputs = np.ma.concatenate([inp1, inp2])
targets = np.ma.concatenate([tar1, tar2])

track = w.train(time, inputs, targets, wn.TrainStrategy.Gradient, N, 0., 1, True, True)
#track = w.train(t, inp1, tar1, wn.TrainStrategy.Gradient, N, 0., 1, True, True)
#ans = w.sim(t, inpc)
#import pdb; pdb.set_trace()
tool.plot(t, inp3, w, track, orig=sys3, target=tar3)
plb.show()
#plb.plot(t, sys2, t, w.sim(t, inp2))
#plb.show()
sys.exit()


















