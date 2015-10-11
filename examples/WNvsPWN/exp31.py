#! /usr/bin/python3
from matplotlib import rc
rc('text', usetex=True)
rc('text.latex', unicode=True)
rc('text.latex', preamble=r'\usepackage[russian]{babel}')
rc('font',**{'size':'19'})

import sys
sys.path.append('../../bin/')
import wavenet as wn
import pylab as plb
import numpy as np

def Morlet(p, tau):
    return np.cos(p*tau)*np.exp(-0.5*tau**2)

t = np.arange(-5., 5., 0.1)
m1 = Morlet(1, t)
m2 = Morlet(3, t)
m3 = Morlet(6, t)
plb.plot(t, m1, label='p=1', linestyle='-')
plb.plot(t, m2, label='p=3', linestyle='--')
plb.plot(t, m3, label='p=6', linestyle=':')
plb.legend(loc=0)
plb.show()










