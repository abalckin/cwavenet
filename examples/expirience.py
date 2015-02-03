#! /usr/bin/python3
import sys
import tool
sys.path.append('../python/')

#import wavenet as wn
import pylab as plb
import numpy as np
import scipy.stats

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
    S_list = []
    x010_list = [[], []]
    list_track = []
    list_minerr = []
    Ereal_list = []
    M_list = []
    dEinf_list =[]
    k1 = 10.
    np.random.seed()
    test_num = 20
    p0=2.
    a0=5.
    nc = 10
    w0 = .1
    w1 = -.1
    print('\n\t\t\t|Обычная вейвсеть\t\t\t\t|Полиморфная вейвсеть')
    print('\nk2\t|\tS\t|\tn\t|\tM\t|\tdE\t|n\t|\tM\t|\tdE')
    import pdb; pdb.set_trace()
    for k2 in [0.1, 0.5, 10]:
        for polymorph in [False, True]:
            for i in range(test_num):
                print('Iter {}...'.format(i))
                inp = np.arange(-20, 20, 0.5)
                tar = np.vectorize(func1)(inp, k1)
                T = tar.shape[-1]
                eps = np.random.random(T)*k2
                d = tar+eps
                err_min = 0.5*sum(((d-tar)**2))
                ts = wn.TrainStrategy.Gradient
                w = wn.Net(nc, np.min(inp), np.max(inp), np.average(0),
                                 a0, w0, w1, p0)
                track = w.train(inp, inp, d, ts, 200, 0.05, 1, polymorph, polymorph)
                Ay = (np.sum(tar**2)/T)**0.5
                Aeps = (np.sum(tar**2)/T)**0.5
                S = 20*np.log10(Ay/Aeps)
                S_list.append(S)
                E = track['e']
                y = np.array(E)[0]
                y010 = np.extract(y < y[0]*0.1, y)[0]
                yinf = y[-1]
                x010 = np.nonzero(y == y010)[0][0]
                x010_list[polymorph].append(x010)
                M = (yinf-err_min)/err_min
                M_list.append(M)
                dEinf = y[-2]-y[-1]
                dEinf = dEinf_list.append()
        S, dS = mean_confidence_interval(S_list)
        n0, dn0 = 
        print('\n{k2}\t|\t{S}~{dS}\t|\tn\t|\tM\t|\tdE\t|n\t|\tM\t|\tdE'.format(
            k2=k2,
            S=S,
            dS=dS
            ))
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









