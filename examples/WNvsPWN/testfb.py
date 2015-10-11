#! /usr/bin/python3
import sys
sys.path.append('../../bin')
import cregister as cr
import wavenet as wn
import numpy as np
import scipy.stats
import scipy as sp
from scipy.integrate import odeint


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
    #import pdb; pdb.set_trace()
    return [y2, -5*y2-(6+2*np.sin(0.5*t))*y1+ug(t)+ksi2(t)]


def func2g(y, t):
    y1, y2  = y
    return [y2, -(2+1*np.sin(0.5*t))*y2-5*y1+ug(t)+ksi2(t)]


def uc(t):
    return 1.


def ug(t):
    return 1+0.2*np.sin(2*t)


def main():
    # Инициалиазация
    cb = cr.Caller()
    cal = Caller()
    cb.setHandler(cal)
    k2 = 0.05
    N = 400
    np.random.seed()
    test_num = 30
    c0 = 0.
    a0 = 1.
    a1 = 1.
    w0 = -0.
    w1 = 0. 
    p1 = 2.7
    p2 = 2.7
    nc = 10
    y0 = 0.
    y1 = 0.
    t = np.arange(0, 10, 0.1)
    eps = np.random.normal(0., np.sqrt(0.5), t.shape[-1])*k2
    inpc = np.vectorize(uc)(t)
    inpc = inpc+eps*np.abs(inpc)
    eps = np.random.normal(0., np.sqrt(0.5), t.shape[-1])*k2
    inpg = np.vectorize(ug)(t)
    inpg = inpg+eps*np.abs(inpg)
    sysag = odeint(func1g, [y0, y1], t)[:, 0]
    sysac = odeint(func1c, [y0, y1], t)[:, 0]
    sysbg = odeint(func2g, [y0, y1], t)[:, 0]
    sysbc = odeint(func2c, [y0, y1], t)[:, 0]
    ulist = [inpg, inpc, inpg, inpc]
    tarlist = [sysag, sysac, sysbg, sysbc]
    tarnum = ['a_garmonic', 'a_constant', 'b_garmonic', 'b_constant']
    for uin,  tar, tarn in zip(ulist, tarlist, tarnum):
        n_list = [[], []]
        E_list = [[], []]
        T_list = [[], []]
        print ('\n', tarn)
        for wttype, wtname in zip([wn.ActivateFunc.Morlet,
                                wn.ActivateFunc.POLYWOG,
                                wn.ActivateFunc.RASP1],
                                ["Morlet", "POLYWOG", "RASP"]):
            print('\n', wtname)
            print('\n|\tE\t|\tn\t|\tT\t|')
            for types, wavenum in zip([True, False], [0, 1]):
                for i in range(test_num):
                    T = tar.shape[-1]
                    eps = np.random.normal(0., np.sqrt(0.5), T)*k2
                    dparam = tar+eps*np.abs(tar)
                    ts = wn.TrainStrategy.BFGS
                    w = wn.Net(nc, c0,
                                    a0, a1, w0, w1, p1, p2,2,.0, 1.,  wttype, 4)
                    track = w.train(t, t, dparam, ts, N, 0.0, 1, types, types)
                    #print(types)
                    E = track['e']
                    y = np.array(E)[0]
                    T = track['t'][0]
                    try:
                        y010 = np.extract(y < y[0]*0.1, y)[0]
                        x010 = np.nonzero(y == y010)[0][0]
                        n_list[wavenum].append(x010)
                        T_list[wavenum].append(T[x010])   
                    except:
                        pass
                    E = w.energy(t, t, tar)
                    E_list[wavenum].append(E)
            n0, dn0 = mean_confidence_interval(n_list[0])
            E0, dE0 = mean_confidence_interval(E_list[0])
            n1, dn1 = mean_confidence_interval(n_list[1])
            E1, dE1 = mean_confidence_interval(E_list[1])
            T0, dT0 = mean_confidence_interval(T_list[0])
            T1, dT1 = mean_confidence_interval(T_list[1])
            str = '\n{E0:.3f}~{dE0:.3f}\t|{n0:.1f}~{dn0:.1f}\t|{T0:.2f}~{dT0:.2f}\n{E1:.3f}~{dE1:.3f}\t|{n1:.1f}~{dn1:.1f}\t|{T1:.2f}~{dT1:.2f}'.format(
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

if __name__ == '__main__':
    main()
    sys.exit()









