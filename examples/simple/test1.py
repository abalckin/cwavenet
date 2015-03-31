#! /usr/bin/python3
import sys
import tool
sys.path.append('../../bin/')
import wavenet as wn
import pylab as plb
import numpy as np
from scipy.integrate import odeint

k2=0.1

class Caller(object):
    def __call__(self, prg):
        pass
import cregister as cr
cb = cr.Caller()
cal = Caller()
cb.setHandler(cal)

# Mодельная система
def ksi1(t):
    #random.seed(t)
    #return random.random()*k2
    #return (np.sin(np.sin(t**2)+5*t))*k2
    return np.sin(5*t)*k2

def ksi2(t):
    return np.sin(7*t)*k2
    
def func1(y, t):
    y1, y2  = y
    #import pdb; pdb.set_trace()
    return [y2, -5*y2-(6-np.sin(0.5*t))*y1+u(t)+ksi2(t)]

def func2(y, t):
    y1, y2  = y
    return [y2, (-2-np.sin(0.3*t))*y2-5*y1+u(t)+ksi2(t)]


def u(t):
    return 0.8+0.2*np.sin(2*t)+ksi1(t)


c0 = 0
a0 = .1
a1 = 3.
w0 = -0.5
w1 = 0.5
p = 1.
xmin=0
xmax=10 
ncount = 10
t = np.arange(xmin, xmax, 0.1)
inp = u(t)
#import pdb; pdb.set_trace()
#tar = odeint(func3, [1], t)[:, 0]
#tar = odeint(func2, [1, 1], t)[:, 0]
tar = odeint(func1, [1, 1], t)[:, 0]
#tar = func4(t)
d = tar+(np.random.random(tar.shape)-.5)*0.1
#tar = odeint(func3, [1], t)[:, 0]
c0 = 0. #np.average(d)
#import pdb; pdb.set_trace()
w = wn.Net(ncount, xmin, xmax, c0,
                         a0, a1,  w0, w1, p, p, wn.ActivateFunc.RASP, 4)
track = w.train(cb, t, inp, d, wn.TrainStrategy.Gradient, 300, 0., 1, False, False)
#import pdb; pdb.set_trace()
tool.plot(t, inp, d, w, track, orig=tar)
plb.show()
## inp = np.sin(3*t)
## tar = odeint(func4_, [1, 1, 1], t)[:, 0]
## res = w.sim(t, inp)
## plb.plot(t,inp,'.', t, tar, '-',t, res,'--')
#plb.show()
sys.exit()








