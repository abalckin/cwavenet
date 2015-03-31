#! /usr/bin/python3
import sys
import tool
sys.path.append('../../bin/')
import wavenet as wn
import pylab as plb
import numpy as np
from scipy.integrate import odeint


class Caller(object):
    def __call__(self, prg):
        pass
import cregister as cr
cb = cr.Caller()
cal = Caller()
cb.setHandler(cal)

def heaviside(t):
    return t*0.+1.

## def func2(t):
##     return np.cos(2*t)*np.exp(-t) + np.sin(2*t)*np.exp(-t) - np.cos(2*t)*np.exp(-t)*(heaviside(t)/5 - (np.exp(t)*heaviside(t)*(2*np.cos(2*t) - np.sin(2*t)))/10) - np.sin(2*t)*np.exp(-t)*(heaviside(t)/10 - (np.exp(t)*heaviside(t)*(np.cos(2*t) + 2*np.sin(2*t)))/10)
## def func1(t):
##     return 4*np.exp(-2*t) - 3*np.exp(-3*t) + (np.exp(-2*t)*heaviside(t)*(np.exp(2*t) - 1))/2 - (np.exp(-3*t)*heaviside(t)*(np.exp(3*t) - 1))/3

## def func3(t):
##     return np.exp(-3*t) + (np.exp(-3*t)*heaviside(t)*(np.exp(3*t) - 1))/3

## def func4(t):
##     return 6*np.exp(-t) - 8*np.exp(-2*t) + 3*np.exp(-3*t) - (np.exp(-2*t)*heaviside(t)*(np.exp(2*t) - 1))/2 + (np.exp(-3*t)*heaviside(t)*(np.exp(3*t) - 1))/6 + (np.exp(-t)*heaviside(t)*(np.exp(t) - 1))/2
def func1(y, p):
    y1, y2  = y
    return [y2, -5*y2-6*y1+1]

def func2(y, p):
    y1, y2  = y
    return [y2, -2*y2-5*y1+1]

def func2_(y, t):
    y1, y2  = y
    return [y2, -2*y2-5*y1+u_t(t)]


def func3_(y, t):
    return u_t(t)-3*y

def func3(y, p):
    return 1-3*y

def func4(y, par_):
     #import pdb; pdb.set_trace()
     y1, y2, y3  = y
     return [y2 ,y3, -6*y3-11*y2-6*y1+ 1]

def u_t(t):
    return 0.4*np.sin(t+np.pi/4)+1.

def func4_(y, t):
     #import pdb; pdb.set_trace()
     y1, y2, y3  = y
     return [y2 ,y3, -6*y3-11*y2-6*y1+ u_t(t)]

xmax = 5.
xmin = 0
a0 = 1
a1 = 5
w0 = -1.5
w1 = 1.5
p0 =2.6
p1 =1.4
p=2
ncount = 10
t = np.arange(xmin, xmax, 0.1)
inp = np.sin(t)
inp = inp+(np.random.random(inp.shape)-0.5)*0.1
#import pdb; pdb.set_trace()
#tar = odeint(func3, [1], t)[:, 0]
#tar = odeint(func2, [1, 1], t)[:, 0]
tar = odeint(func1, [1, 1], t)[:, 0]
#tar = func4(t)
d = tar+(np.random.random(tar.shape)-.5)*0.1
#tar = odeint(func3, [1], t)[:, 0]
c0 = 0 #np.average(d)
#import pdb; pdb.set_trace()
##w = wn.Net(ncount, xmin, xmax, c0,
##                         a0, a1,  w0, w1, p0, p1, wn.ActivateFunc.POLYWOG, 4)
##track = w.train(cb, t, inp, d, wn.TrainStrategy.BFGS, 500, 0., 1)
w = wn.Net(ncount, xmin, xmax, c0,
                          a0, a1,  w0, w1, p, p, wn.ActivateFunc.Morlet, 4)
track = w.train(cb, t, inp, d, wn.TrainStrategy.BFGS, 500, 0., 1, True, True)
tool.plot(t, inp, d, w, track, orig=tar)
plb.show()
inp = u_t(t)
tar = odeint(func2_, [1, 5], t)[:, 0]
res = w.sim(t, inp)
plb.figure()
plb.plot(t, inp, '.', label='Вход')
plb.plot(t, tar, '-', label='Выход модели')
plb.plot(t, res, '--', label='Выход сети')
plb.legend(loc=0)
plb.show()
sys.exit()








