#! /usr/bin/python3
import sys
import tool
sys.path.append('../python/')
import wavenet as wn
import pylab as plb
import numpy as np

def func1(x):
    if x < 0:
        return 10*np.exp(-.03*x - 0.3)*np.sin((0.6*x-0.1))+x+1
    if x >= 0:
        return 10*np.exp(-.03*x - 0.3)*np.sin((0.3*x-0.1))+x+1


def calc():
    list_track = []
    list_minerr = []
    Ereal_list=[]
    Emin_list=[]
    x010_list=[]
    test_num = 20
for k2 in [0.1, 0.5, 10]:
        for i in range(test_num):
                print('Iter {}...'.format(i))
                inp = np.arange(-20, 20, 0.5)
                tar = np.vectorize(func1)(inp)
                size = len(inp)
                d = tar+np.random.random(size)
                list_minerr.append(0.5*sum(((d-tar)**2)))
                p0=2.
                a0=5.
                nc = 10
                w0 = .1
                w1 = -.1
                #ts = wn.TrainStrategy.BFGS
                ts = wn.TrainStrategy.Gradient
                w = wn.Net(nc, np.min(inp), np.max(inp), np.average(0),
                                 a0, w0, w1, p0)
                track = w.train(inp, inp, d, ts, 200, 0.05, 1, False, False)
                #track = w.train(inp,inp, tar, ts, 200, 0.05, 1, True, True)
                list_track.append(track)
                Ereal_list.append(w.energy(inp, inp, d))
                Emin_list.append(0.5*sum(((d-tar)**2)))
                E = track['e']
                x = np.arange(0, 201)
                y = np.array(E)[0]
                y010 = np.extract(y < y[0]*0.1, y)[0]
                yinf = y[-1]
                x010 = np.nonzero(y == y010)[0][0]
                xinf = x[-1]
                x010_list.append(x010)
            print (np.std(x010_list))

>>>>>>> 16d2f556314b7892b07e390265cbdf7d2d8b7e3f
    E = np.array([t['e'] for t in list_track])
    Eav = np.array(np.average(E, axis=0)[0])
    Ereal = np.array([np.average(Ereal_list)])
    Emin = np.array([np.average(Emin_list)])
    np.savetxt('Eav.txt', Eav, delimiter=', ')
    np.savetxt('Ereal.txt', Ereal, delimiter=', ')
    np.savetxt('Emin.txt', Emin, delimiter=', ')
    sys.exit()

if __name__ == '__main__':
    calc()









