#! /usr/bin/python3
import sys

sys.path.append('../../bin')
import cregister as cr
import wavenet as wn
import wavelet as wt
import pylab as plb
import numpy as np
import scipy.stats
import scipy as sp
import time
from scipy.integrate import odeint
import pylab as plb
import random

k2 = 0.1

class Caller(object):
    def __call__(self, prg):
        pass
        
#доверительный интервал
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h

# Mодельная система
def ksi1(t):
    #random.seed(t)
    #return random.random()*k2
    #return (np.sin(np.sin(t**2)+5*t))*k2
    return (np.sin(5*t)+np.sin(7*t))*k2

def ksi2(t):
    return (np.sin(7*t)+np.sin(9*t))*k2
    
def func1(y, t):
    y1, y2  = y
    #import pdb; pdb.set_trace()
    return [y2, -5*y2-(6-np.sin(0.5*t))*y1+u(t)+ksi2(t)]

def func2(y, t):
    y1, y2  = y
    return [y2, (-2-np.sin(0.3*t))*y2-5*y1+u(t)+ksi2(t)]



def u(t):
    return 3.+0.2*np.sin(2*t)+ksi1(t)

def main():
    # Инициалиазация
    Exp_list=[]
    N = 200
    np.random.seed()
    test_num = 30
    c0 = 0
    a0 = .1
    a1 = 3.
    w0 = -0.4
    w1 = 0.4
    p = 2.
    nc=10
    #text_file = open("log.txt", "w")
    #text_file.write('\n\t|Morlet\t\t|POLYWOG')
    #text_file.write('# \t|\tS\t|\tn\t|\tM\t||\tn\t|\tM\t|\tdE')
    print('\n\t|Полим.\t\t\t\t\t\t|Традиц.')
    print('\nsys \t|\tn\t|\tE\t|\tT\t|\tn\t|\tE\t|\tT')
    #import pdb; pdb.set_trace()
    t = np.arange(0, 10, 0.1)
    inp = np.vectorize(u)(t)
    sysa = odeint(func1, [1, 1], t)[:, 0]
    sysb = odeint(func2, [1, 1], t)[:, 0]
    #plb.plot(sysa)
    plb.show()
    tarlist = [sysa, sysb]
    tarnum = ['a', 'b']
    S_list = []
    #import pdb; pdb.set_trace()
    for tar, tarn in zip(tarlist, tarnum):
        n_list = [[], []]
        E_list = [[], []]
        T_list = [[], []]
        #import pdb; pdb.set_trace()
        for types, wavenum in zip([True, False], [0, 1]):
            for i in range(test_num):
                T = tar.shape[-1]
                eps = (np.random.random(T)-0.5)*k2
                dparam = tar+eps
                ts = wn.TrainStrategy.Gradient
                w = wn.Net(nc, np.min(t), np.max(t), c0,
                            a0, a1, w0, w1, p, p,  wn.ActivateFunc.RASP1, 4)
                #import pdb; pdb.set_trace()
                cb = cr.Caller()
                cal = Caller()
                cb.setHandler(cal)
                track = w.train(cb, t, inp, dparam, ts, N, 0.0, 1, types, types)
                Ay = (np.sum(tar**2)/T)**0.5
                Aeps = (np.sum((dparam-tar)**2)/T)**0.5
                S = (Ay/Aeps)**2
                S_list.append(S)
                E = track['e']
                #import pdb; pdb.set_trace()
                y = np.array(E)[0]
                T = track['t'][0]
                try:
                    y010 = np.extract(y < y[0]*0.1, y)[0]
                    x010 = np.nonzero(y == y010)[0][0]
                    n_list[wavenum].append(x010)
                    T_list[wavenum].append(T[x010])


                except:
                    #import pdb; pdb.set_trace()
                    pass
                #M = (yinf-err_min)/err_min
                #ans = w.sim(t, inp)
                E = w.energy(t, inp, tar)
                #import pdb; pdb.set_trace()
                E_list[wavenum].append(E)
                #import pdb; pdb.set_trace()
        n0, dn0 = mean_confidence_interval(n_list[0])
        E0, dE0 = mean_confidence_interval(E_list[0])
        n1, dn1 = mean_confidence_interval(n_list[1])
        E1, dE1 = mean_confidence_interval(E_list[1])
        T0, dT0 = mean_confidence_interval(T_list[0])
        T1, dT1 = mean_confidence_interval(T_list[1])
        str = '\n{tarnum}\t|{n0:.1f}~{dn0:.1f}\t|{E0:.3f}~{dE0:.3f}\t|{T0:.2f}~{dT0:.2f}\t|{n1:.1f}~{dn1:.1f}\t|{E1:.3f}~{dE1:.3f}\t|{T1:.2f}~{dT1:.2f}'.format(
            tarnum=tarn,
            n0=n0,
            dn0=dn0,
            E0=E0,
            dE0=dE0,
            n1=n1,
            dn1=dn1,
            E1=E1,
            dE1=dE1,
            T0=T0,
            T1=T1,
            dT0=dT0,
            dT1=dT1)
        print(str)
        Exp_list.append([tarnum, n0, dn0, E0, dE0, n1, dn1, E1,
                        dE1])
        #text_file.write(str)
    S, dS = mean_confidence_interval(S_list)
    print ('S={S:.1f}~{dS:.1f}'.format(S=S, dS=dS))
    np.savetxt('result.txt', np.array(Exp_list), delimiter=', ')
    #text_file.close()

if __name__ == '__main__':
    main()
    sys.exit()









