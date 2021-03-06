#! /usr/bin/python3
import sys

sys.path.append('../../bin')

import wavenet as wn
import numpy as np
import scipy.stats
import scipy as sp
import neurolab as nl
import cregister as cr
import pylab as plb
import time
#from scipy import signal
import tool

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

# Mодельная функция
def func1(x):
        return (0.7+2*x)*np.exp(-.3*x*40 - 0.3)
def main():
    # Инициалиазация
    Exp_list = []
    N = 200
    np.random.seed()
    test_num = 30
    cb = cr.Caller()
    cal = Caller()
    cb.setHandler(cal)
    c0 = .5
    p0 = 1.
    p1 = 1.
    a0 = .4
    a1 = .4
    nc = 10
    w0 = 0.00001
    w1 = -0.00001

    print('\n\t\t\t|Полиморфная вейвсеть\t\t\t\t|Нейронная сеть')
    print('\nk2\t|\tS\t|\tn\t|\tM\t|\tdE\t|\tn\t|\tM\t|\tdE')
    #import pdb; pdb.set_trace()
    klist = [1.*1.5**i/50. for i in range(-1, 10)]
    T_list = [[], []]
    for k2 in klist:
        S_list = []
        n_list = [[], []]
        M_list = [[], []]
        ME_list = [[], []]
        for ff in [True, False]:
            for i in range(test_num):
                inp = np.arange(0., 0.5, 0.01)
                tar = np.vectorize(func1)(inp)
                T = tar.shape[-1]
                eps = (np.random.random(T)-0.5)*k2
                d = tar+eps
                start=0.
                end=0.
                if ff == True:
                    inp_ff = inp.reshape(T, 1)
                    tar_ff = d.reshape(T, 1)
                    n = nl.net.newff([[-1., 1.]], [50, 1])
                    n.trainf = nl.train.train_gd
                    #import pdb; pdb.set_trace()
                    start= time.clock()
                    E = n.train(inp_ff, tar_ff, epochs=N, goal=0.0, show=0, adapt=True)
                    end= time.clock()
                    total = (end-start)/N
                    y = np.array(E)
                    ans = n.sim(inp_ff).reshape(T)
                    #import pdb; pdb.set_trace()
                    #import pdb; pdb.set_trace()
                else:
                    ts = wn.TrainStrategy.Gradient
                    w = wn.Net(nc, c0,
                            a0, a1, w0, w1, p0, p1, wn.ActivateFunc.POLYWOG, 4)
                    start = time.clock()
                    track = w.train(inp, inp, d, ts, N, 0.0, 1, False, True)
                    end= time.clock()
                    ans = w.sim(inp, inp)
                    #tool.plot(inp, inp, w, track, orig=tar, target=d)
                    #plb.show()
                    E = track['e']
                    y = np.array(E)[0]
                Ay = (np.sum(tar**2)/T)**0.5
                Aeps = (np.sum((d-tar)**2)/T)**0.5
                S = (Ay/Aeps)**2
                S_list.append(S)
                try:
                    y010 = np.extract(y < y[0]*0.1, y)[0]
                    x010 = np.nonzero(y == y010)[0][0]
                    n_list[ff].append(x010)
                except:
                    pass
                #M = (yinf-err_min)/err_min
                Ae = (np.sum((ans-tar)**2)/T)**0.5 
                M = (Ay/Ae)**2
                #import pdb; pdb.set_trace()
                M_list[ff].append(M)
                #import pdb; pdb.set_trace()
                dEinf = (y[-2]-y[-1])
                ME_list[ff].append(dEinf)
                total += end-start
                T_list[ff].append(total)
        S, dS = mean_confidence_interval(S_list)
        n0, dn0 = mean_confidence_interval(n_list[0])
        M0, dM0 = mean_confidence_interval(M_list[0])
        ME0, dME0 = mean_confidence_interval(ME_list[0])
        n1, dn1 = mean_confidence_interval(n_list[1])
        M1, dM1 = mean_confidence_interval(M_list[1])
        ME1, dME1 = mean_confidence_interval(ME_list[1])
        print('\n{k2:.3f}\t|{S:.1f}~{dS:.1f}\t|{n0:.1f}~{dn0:.1f}\t|{M0:.1f}~{dM0:.1f}\t|{ME0:.3f}~\
{dME0:.3f}\t|{n1:.1f}~{dn1:.1f}\t|{M1:.1f}~{dM1:.1f}\t|{ME1:.3f}~{dME1:.3f}'.format(
            k2=k2,
            S=S,
            dS=dS,
            n0=n0,
            dn0=dn0,
            M0=M0,
            dM0=dM0,
            ME0=ME0,
            dME0=dME0,
            n1=n1,
            dn1=dn1,
            M1=M1,
            dM1=dM1,
            ME1=ME1,
            dME1=dME1
            ))
        Exp_list.append([k2, S, dS, n0, dn0, M0, dM0, ME0, dME0, n1, dn1, M1,
                        dM1, ME1, dME1])
        np.savetxt('result.txt', np.array(Exp_list), delimiter=', ')
    T0, dT0 = mean_confidence_interval(T_list[0])
    print("Время ПВС - {0}=+-{1}".format(T0, dT0))
    T1, dT1 = mean_confidence_interval(T_list[1])
    print("Время НС - {0}=+-{1}".format(T1, dT1))

if __name__ == '__main__':
    main()
    sys.exit()









