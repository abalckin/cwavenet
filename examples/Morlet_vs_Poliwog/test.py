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


def func3(t):
    return np.exp(-3*t) + (np.exp(-3*t)*heaviside(t)*(np.exp(3*t) - 1))/3

def heaviside(t):
    return t*0.+1.

def main():
    # Инициалиазация
    Exp_list=[]
    N = 100
    np.random.seed()
    test_num = 30
    c0 = 1.
    a0 = .1
    a1 = 2.
    w0 = -0.1
    w1 = 0.1
    p0 = 0.1
    p1 = 2.
    nc = 10
    
    text_file = open("log.txt", "w")
    #text_file.write('\n\t|Morlet\t\t|POLYWOG')
    #text_file.write('# \t|\tS\t|\tn\t|\tM\t||\tn\t|\tM\t|\tdE')
    print('\n\t|Morlet\t\t\t\t|POLYWOG')
    print('\nsys \t|\tn\t|\tE\t|\tn\t|\tE\t')
    #import pdb; pdb.set_trace()
    t = np.arange(0, 5, 0.01)
    inp = heaviside(t)
    tarlist = [func1(t), func2(t), func3(t)]
    tarnum = ['a', 'b', 'c']
    k2 = 0.5

    #import pdb; pdb.set_trace()
    for tar, tarn in zip(tarlist, tarnum):
        S_list = []
        n_list = [[], []]
        E_list = [[], []]
        IE_list = [[], []]
        #import pdb; pdb.set_trace()
        for wavelet, wavenum in zip([wn.ActivateFunc.Morlet, wn.ActivateFunc.POLYWOG], [0, 1]):
            for i in range(test_num):
                T = tar.shape[-1]
                eps = (np.random.random(T)-0.5)*k2
                dparam = tar+eps
                ts = wn.TrainStrategy.CG
                w = wn.Net(nc, np.min(inp), np.max(inp), c0,
                            a0, a1, w0, w1, p0, p1,  wavelet, 4)
                #import pdb; pdb.set_trace()
                cb = cr.Caller()
                cal = Caller()
                cb.setHandler(cal)
                track = w.train(cb, t, inp, dparam, ts, N, 0.0, 1, True, True)
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
                dEinf = y[-2]-y[-1]
                IE_list[wavenum].append(dEinf)
        n0, dn0 = mean_confidence_interval(n_list[0])
        E0, dE0 = mean_confidence_interval(E_list[0])
        n1, dn1 = mean_confidence_interval(n_list[1])
        E1, dE1 = mean_confidence_interval(E_list[1])
        str = '\n{tarnum}\t|{n0:.1f}~{dn0:.1f}\t|{E0:.2f}~{dE0:.2f}\t|{n1:.1f}~{dn1:.1f}\t|{E1:.2f}~{dE1:.2f}'.format(
            tarnum=tarn,
            n0=n0,
            dn0=dn0,
            E0=E0,
            dE0=dE0,
            n1=n1,
            dn1=dn1,
            E1=E1,
            dE1=dE1)
        print(str)
        Exp_list.append([tarnum, n0, dn0, E0, dE0, n1, dn1, E1,
                        dE1])
        text_file.write(str)
    np.savetxt('result.txt', np.array(Exp_list), delimiter=', ')
    text_file.close()

if __name__ == '__main__':
    main()
    sys.exit()









