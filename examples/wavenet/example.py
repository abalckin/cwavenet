#! /usr/bin/python3
import sys
import tool
sys.path.append('../../bin/')
import wavenet as wn
import pylab as plb
import numpy as np
import csv
import datetime as dt
from scipy import signal
class Test():
    def func1(self, x):
        if x<0:
            return 10*np.exp(-.03*x - 0.3)*np.sin((0.6*x-0.1))+x+10
        if x>=0:
            return 10*np.exp(-.03*x - 0.3)*np.sin((0.3*x-0.1))+x+10
        
    def func2(self, x):
        if x < 0:
            return 10*np.exp(-.03*x*40 - 0.3)*np.sin((0.6*x*40-0.1))+x*40+.5
        if x >= 0:
            return 10*np.exp(-.03*x*40 - 0.3)*np.sin((0.3*x*40-0.1))+x*40+.5 

        
    def calc(self):
        list_track = []
        list_minerr = []
        test_num =1
        for i in range(test_num):
            inp = np.arange(-0.5, 0.5, 0.5/40)
            tar = np.vectorize(self.func2)(inp)/20
            size = len(inp)
            d = tar+(np.random.random(size)-0.5)*5.062/20
            list_minerr.append(0.5*sum(((d-tar)**2)))
            #tar -= np.min(tar)
            #tar /= np.max(tar)
            p0=2./40
            p1=2./40
            a0=3./40
            a1=5./40
            nc=16
            w0=-.5/20
            w1=2.0/20
            #ts = wn.TrainStrategy.BFGS
            ts = wn.TrainStrategy.Gradient
            w = wn.Net(nc, np.min(inp), np.max(inp), np.average(0),
                             a0, a1, w0, w1, p0, p1)
            #track = w.train(inp, inp, d, ts, 200, 0.05, 1, False, False)
            track = w.train(inp, inp*40 , tar, ts, 100, 0.05, 1, True, True)
            list_track.append(track)
        E= np.array([t['e'] for t in list_track])
        Eav=np.average(E, axis=0)[0]
        np.savetxt('Eav.txt', Eav, delimiter=', ')
        #import pdb; pdb.set_trace()    
        #track = w.train(inp, tar, ts, 600, 100000, 1, False, False)
        #import pdb; pdb.set_trace()
        tool.plot(inp, inp*40, d, w, track, orig=tar, xlabel='x', ylabel='f(x)')
        #print (w.energy(inp, inp, tar))
        #print (w.energy(inp, inp, d))
        ans = w.sim(inp, inp*40)
        #print (w.energy(inp, inp, tar))
        print (0.5*np.sum(((d-tar)**2)))
        print (0.5*np.sum(((ans-tar)**2)))
        #print (0.5*sum(((d-tar)**2)))
        plb.show()
        sys.exit()

    def __init__(self):
        pass
        #self.fileName = "../data/spidr_1420457768932_0.txt"
        #self.load()
def main():
    t = Test()
    t.calc()
if __name__ == '__main__':
    main()








