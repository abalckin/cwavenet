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
def func2(t):
    return np.cos(2*t)*np.exp(-t) + np.sin(2*t)*np.exp(-t) - np.cos(2*t)*np.exp(-t)*(heaviside(t)/5 - (np.exp(t)*heaviside(t)*(2*np.cos(2*t) - np.sin(2*t)))/10) - np.sin(2*t)*np.exp(-t)*(heaviside(t)/10 - (np.exp(t)*heaviside(t)*(np.cos(2*t) + 2*np.sin(2*t)))/10)
def func1(t):
##import pdb; pdb.set_trace()
##    return np.exp(-2*t) - np.exp(-3*t) + (np.exp(-2*t)*heaviside(t)*(np.exp(2*t) - 1))/2 ##- (np.exp(-3*t)*heaviside(t)*(np.exp(3*t) - 1))/3
    return 4*np.exp(-2*t) - 3*np.exp(-3*t) + (np.exp(-2*t)*heaviside(t)*(np.exp(2*t) - 1))/2 - (np.exp(-3*t)*heaviside(t)*(np.exp(3*t) - 1))/3

def u_t(t):
    return 0.4*np.sin(t+np.pi/4)+1.


def func3(t):
    return np.exp(-3*t) + (np.exp(-3*t)*heaviside(t)*(np.exp(3*t) - 1))/3

def func4(y, t):
    y1, y2  = y
    return [y2, (-5-5*np.sin(t))*y2-6*y1+u_t(t)]

def u_t(t):
    return 0.4*np.sin(t+np.pi/4)+1.


def func5(y, tmax):
#    import pdb; pdb.set_trace()
    if len(y)==tmax:
        return y
    else:
        t = len(y)
        y.append(y[-1]*y[-2]/(1+y[-1]**2+y[-2]**2)+0.8*np.sin(2*np.pi*t/50))
        return  np.array(func5(y, tmax))


def main():
    # Инициалиазация
    Exp_list=[]
    N = 300
    np.random.seed()
    test_num = 10
    c0 = 0.
    a0 = 90
    a1 = 110
    w0 = -60.
    w1 = 40
    p0 =1.
    p1 =1

    nc = 20
    
    text_file = open("log.txt", "w")
    #text_file.write('\n\t|Morlet\t\t|POLYWOG')
    #text_file.write('# \t|\tS\t|\tn\t|\tM\t||\tn\t|\tM\t|\tdE')
    print('\n\t|Полим.\t\t\t\t\t\t|Традиц.')
    print('\nsys \t|\tn\t|\tE\t|\tT\t|\tn\t|\tE\t|\tT')
    #import pdb; pdb.set_trace()
    t = np.arange(0, 500, 1.)
    inp = 0.8*np.sin(2*np.pi*t/50)
    #sys4 = odeint(func4, [1, 1], t)[:, 0]
    sys5 = func5([0., 0.1], 500)

    tarlist = [sys5]
    tarnum = ['e']
    k2 = 0.15
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
                ts = wn.TrainStrategy.BFGS
                w = wn.Net(nc, np.min(t), np.max(t), c0,
                            a0, a1, w0, w1, p0, p1,  wn.ActivateFunc.RASP1, 4)
                #import pdb; pdb.set_trace()
                cb = cr.Caller()
                cal = Caller()
                cb.setHandler(cal)
                shtamp = time.clock()
                track = w.train(cb, t, inp, dparam, ts, N, 0.0, 1, types, types)
                now = time.clock()
                Ay = (np.sum(tar**2)/T)**0.5
                Aeps = (np.sum((dparam-tar)**2)/T)**0.5
                S = (Ay/Aeps)**2
                S_list.append(S)
                E = track['e']
                y = np.array(E)[0]
                try:
                    y010 = np.extract(y < y[0]*0.1, y)[0]
                    x010 = np.nonzero(y == y010)[0][0]
                    n_list[wavenum].append(x010)
                except:
                    pass
                #M = (yinf-err_min)/err_min
                #ans = w.sim(t, inp)
                E = w.energy(t, inp, tar)
                #import pdb; pdb.set_trace()
                E_list[wavenum].append(E)
                #import pdb; pdb.set_trace()
                T = now - shtamp
                T_list[wavenum].append(T/N*x010)
        n0, dn0 = mean_confidence_interval(n_list[0])
        E0, dE0 = mean_confidence_interval(E_list[0])
        n1, dn1 = mean_confidence_interval(n_list[1])
        E1, dE1 = mean_confidence_interval(E_list[1])
        T0, dT0 = mean_confidence_interval(T_list[0])
        T1, dT1 = mean_confidence_interval(T_list[1])
        str = '\n{tarnum}\t|{n0:.1f}~{dn0:.1f}\t|{E0:.2f}~{dE0:.2f}\t|{T0:.2f}~{dT0:.2f}\t|{n1:.1f}~{dn1:.1f}\t|{E1:.2f}~{dE1:.2f}\t|{T1:.2f}~{dT1:.2f}'.format(
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
        text_file.write(str)
    S, dS = mean_confidence_interval(S_list)
    print ('S={S:.1f}~{dS:.1f}'.format(S=S, dS=dS))
    np.savetxt('result.txt', np.array(Exp_list), delimiter=', ')
    text_file.close()

if __name__ == '__main__':
    main()
    sys.exit()









