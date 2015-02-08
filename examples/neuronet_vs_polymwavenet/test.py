#! /usr/bin/python3
import sys

sys.path.append('../../bin')

import wavenet as wn
import numpy as np
import scipy.stats
import scipy as sp
import neurolab as nl
#доверительный интервал
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h

# Mодельная функция
def func1(x):
    if x < 0:
        return 10*np.exp(-.03*x*40 - 0.3)*np.sin((0.6*x*40-0.1))+40*x+.5
    if x >= 0:
        return 10*np.exp(-.03*x*40 - 0.3)*np.sin((0.3*x*40-0.1))+40*x+.5

def main():
    # Инициалиазация
    Exp_list = []
    N = 200
    np.random.seed()
    test_num = 3
    p0=2./40
    p1=2./40
    a0=3./40
    a1=5./40
    nc=16
    w0=-.5/20
    w1=2.0/20
    print('\n\t\t\t|Полиморфная вейвсеть\t\t\t\t|Нейронная сеть')
    print('\nk2\t|\tS\t|\tn\t|\tM\t|\tdE\t|\tn\t|\tM\t|\tdE')
    #import pdb; pdb.set_trace()
    klist = [1.*1.5**i for i in range(1, 11)]
    for k2 in klist:
        S_list = []
        n_list = [[], []]
        M_list = [[], []]
        ME_list = [[], []]
        for ff in [True, False]:
            for i in range(test_num):
                inp = np.arange(-0.5, 0.5, 0.5/40.)
                tar = np.vectorize(func1)(inp)/20
                T = tar.shape[-1]
                eps = (np.random.random(T)-0.5)*k2/20.
                d = tar+eps
                #err_min = 0.5*np.sum(((d-tar)**2))
                if ff == True:
                    inp_ff = inp.reshape(T, 1)
                    tar_ff = tar.reshape(T, 1)
                    n = nl.net.newff([[np.min(inp_ff), np.max(inp_ff)]], [16 ,1])
                    n.trainf = nl.train.train_gd
                    #import pdb; pdb.set_trace()
                    E = n.train(inp_ff, tar_ff, epochs=N, goal=0.0, adapt=True, show=0)
                    y = np.array(E)
                    ans = n.sim(inp_ff).reshape(T)
                    #import pdb; pdb.set_trace()
                    #import pdb; pdb.set_trace()
                else:
                    ts = wn.TrainStrategy.Gradient
                    w = wn.Net(nc, np.min(inp), np.max(inp), np.average(d),
                            a0, a1, w0, w1, p0, p1)
                    track = w.train(inp, inp*40, d, ts, N, 0.0, 1, True, True)
                    ans = w.sim(inp, inp*40)
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

if __name__ == '__main__':
    main()
    sys.exit()









