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
        return np.sin(x)+np.random.random()-0.5
        
        
    def calc(self):
        t = np.arange(0, 5, 0.1)
        inp = np.vectorize(self.func1)(t)
        inp1 = np.vectorize(self.func1)(t)
        
        tar = np.sin(t)
        size = len(inp)
        p0=2.
        a0=2.
        nc = 20
        w0 = 0.1
        w1 = 0.1
        #ts = wn.TrainStrategy.BFGS
        ts = wn.TrainStrategy.Gradient
        w = wn.Net(nc, np.min(inp), np.max(inp), np.average(tar),
                         a0, w0, w1, p0)
        track = w.train(t, t, inp, ts, 100, 0.3, 1, True, True)
        track = w.train(t, t, inp1, ts, 100, 0.3, 1, True, True)
        #track = w.train(inp, inp, tar, ts, 100, 0.3, 1, False, False)
        
        #import pdb; pdb.set_trace()
        tool.plot(t, t, tar, w, track, xlabel='x', ylabel='f(x)')
        print (w.energy(t, t, tar))
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








