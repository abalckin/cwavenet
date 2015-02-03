#! /usr/bin/python3
import sys
import tool
sys.path.append('../python/')
import wavenet as wn
import pylab as plb
import numpy as np
import csv
import datetime as dt
from scipy import signal
class Test():
    def load(self):
        with open(self.fileName, 'rt') as csvdata:
            date = []
            value = []
            self.header = []
            for row in csv.reader(csvdata, delimiter=' '):
                if ('#' in row[0]):
                    self.header.append(row)
                else:
                    date.append(' '.join([row[0], row[1]]))
                    value.append(row[2])
        signal = np.array((date, value), dtype=np.dtype('a25'))
        self.value = signal[1, :].astype(np.float)
        self.time = signal[0, :].astype(np.datetime64).astype(dt.datetime)

    def func1(self, x):
        if x<0:
            return 10*np.exp(-.03*x - 0.3)*np.sin((0.6*x-0.1))+x+10
        if x>=0:
            return 10*np.exp(-.03*x - 0.3)*np.sin((0.3*x-0.1))+x+10
        
    def func2(self, x):
        if x < 0:
            return 10*(np.exp(-.03*x - 0.3)*np.sin((0.6*x-0.1))+.1*x+.01)
        if x >= 0:
            return 10*(np.exp(-.03*x - 0.3)*np.sin((0.3*x-0.1))+.1*x+.01) 

        
    def calc(self):
        list_track = []
        list_minerr = []
        test_num =3
        for i in range(test_num):
            inp = np.arange(-10, 10, 0.1)
            tar = np.vectorize(self.func2)(inp)
            size = len(inp)
            d = tar+(np.random.random(size)-0.5)*20
            list_minerr.append(0.5*sum(((d-tar)**2)))
            #tar -= np.min(tar)
            #tar /= np.max(tar)
            p0=1.0
            p1=2.0
            a0=3.5
            a1=7.0
            nc = 20
            w0 = -0.5
            w1 = 2.
            #ts = wn.TrainStrategy.BFGS
            ts = wn.TrainStrategy.Gradient
            w = wn.Net(nc, np.min(inp), np.max(inp), np.average(0),
                             a0, a1, w0, w1, p0, p1)
            #track = w.train(inp, inp, d, ts, 200, 0.05, 1, False, False)
            track = w.train(inp,inp, tar, ts, 200, 0.05, 1, True, True)
            list_track.append(track)
        E= np.array([t['e'] for t in list_track])
        Eav=np.average(E, axis=0)[0]
        np.savetxt('Eav.txt', Eav, delimiter=', ')
        #import pdb; pdb.set_trace()    
        #track = w.train(inp, tar, ts, 600, 100000, 1, False, False)
        #import pdb; pdb.set_trace()
        tool.plot(inp, inp, d, w, track, orig=tar, xlabel='x', ylabel='f(x)')
        #print (w.energy(inp, inp, tar))
        #print (w.energy(inp, inp, d))
        ans = w.sim(inp, inp)
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








