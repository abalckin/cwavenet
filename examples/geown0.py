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
    
        
    def calc(self):
        list_track = []
        list_minerr = []
        for i in range(1):
            inp = np.arange(-20, 20, 0.5)
            tar = np.vectorize(self.func1)(inp)
            size = len(inp)
            d = tar+np.random.random(size)
            list_minerr.append(0.5*sum(((d-tar)**2)))
            #tar -= np.min(tar)
            #tar /= np.max(tar)
            p0=2.
            a0=5.
            nc = 10
            w0 = .1
            w1 = -.1
            ts = wn.TrainStrategy.BFGS
            #ts = wn.TrainStrategy.Gradient
            w = wn.Net(nc, np.min(inp), np.max(inp), np.average(d),
                             a0, w0, w1, p0)
            #track = w.train(inp, inp, d, ts, 200, 0.05, 1, False, False)
            track = w.train(inp, inp, tar, ts, 200, 0.05, 1, True, True)
            list_track.append(track)
        #track = w.train(inp, tar, ts, 600, 100000, 1, False, False)
        #import pdb; pdb.set_trace()
        tool.plot(inp, inp, d, w, track, xlabel='x', ylabel='f(x)')
        #print (w.energy(inp, inp, tar))
        #print (w.energy(inp, inp, d))
        print (w.energy(inp, inp, tar))
        print (w.energy(inp, inp, d))
        print (0.5*sum(((d-tar)**2)))
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








