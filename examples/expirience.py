#! /usr/bin/python3
import sys
import tool
sys.path.append('../python/')

import wavenet as wn
import pylab as plb
import numpy as np
import scipy.stats
import scipy as sp
#доверительный интервал
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h

# Mодельная функция
def func1(x, k1):
    if x < 0:
        return k1*(np.exp(-.03*x - 0.3)*np.sin((0.6*x-0.1))+.1*x+.1)
    if x >= 0:
        return k1*(np.exp(-.03*x - 0.3)*np.sin((0.3*x-0.1))+.1*x+.1)

def main():
    # Инициалиазация
    k1 = 10.
    Exp_list=[]
    S_list = []
    n_list = [[], []]
    M_list = [[], []]
    ME_list = [[], []]
    N = 200
    np.random.seed()
    test_num = 10
    p0=1.9
    p1=2.1
    a0=4.9
    a1=5.1
    nc = 10
    w0 = .1
    w1 = -.1
    print('\n\t\t\t|Обычная вейвсеть\t\t\t\t|Полиморфная вейвсеть')
    print('\nk2\t|\tS\t|\tn\t|\tM\t|\tdE\t|\tn\t|\tM\t|\tdE')
    #import pdb; pdb.set_trace()
    klist = [0.0001*5**i for i in range(1, 8)]
    for k2 in klist:
        for polymorph in [False, True]:
            for i in range(test_num):
                #print('Iter {}...'.format(i))
                inp = np.arange(-20, 20, 0.5)
                tar = np.vectorize(func1)(inp, k1)
                T = tar.shape[-1]
                eps = np.random.random(T)*k2
                d = tar+eps
                err_min = 0.5*np.sum(((d-tar)**2))
                ts = wn.TrainStrategy.Gradient
                w = wn.Net(nc, np.min(inp), np.max(inp), np.average(0),
                            a0, a1, w0, w1, p0, p1)
                track = w.train(inp, inp, d, ts, N, 0.0, 1, polymorph, polymorph)
                Ay = (np.sum(tar**2)/T)**0.5
                Aeps = (np.sum((d-tar)**2)/T)**0.5
                S = 20*np.log10(Ay/Aeps)
                #import pdb; pdb.set_trace()
                S_list.append(S)
                E = track['e']
                y = np.array(E)[0]
                try:
                    y010 = np.extract(y < y[0]*0.1, y)[0]
                    x010 = np.nonzero(y == y010)[0][0]
                    n_list[polymorph].append(x010)
                except:
                    pass
                #M = (yinf-err_min)/err_min
                ans = w.sim(inp, inp)
                Ae = (np.sum((ans-tar)**2)/T)**0.5 
                M = 20*np.log10(Ay/Ae)
                #import pdb; pdb.set_trace()
                M_list[polymorph].append(M)
                #import pdb; pdb.set_trace()
                dEinf = (y[-2]-y[-1])/err_min
                ME_list[polymorph].append(dEinf)
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
    ## E = np.array([t['e'] for t in list_track])
    ## Eav = np.array(np.average(E, axis=0)[0])
    ## Ereal = np.array([np.average(Ereal_list)])
    ## Emin = np.array([np.average(Emin_list)])
    ## np.savetxt('Eav.txt', Eav, delimiter=', ')
    ## np.savetxt('Ereal.txt', Ereal, delimiter=', ')
    ## np.savetxt('Emin.txt', Emin, delimiter=', ')
    ## sys.exit()

if __name__ == '__main__':
    main()
    sys.exit()









