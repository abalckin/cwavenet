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
        if x < -2:
            return -2.186*x-12.846
        elif -2 <= x < 0:
            return 4.246*x
        #elif -1 <= x < 0:
        #    return np.sin(10*x)
        elif 0 <= x:
            return 10*np.exp(-.05*x - 0.5)*np.sin((0.03*x + 0.7)*x)
        
    def calc(self):
        inp = np.arange(-10, 10, 0.3)
        tar = np.vectorize(self.func1)(inp)
        tar -= np.min(tar)
        tar /= np.max(tar)
        size = len(inp)
        p0=3.3
        a0=6.
        nc = 10
        w0 = .1
        w1 = 0.2
        #ts = wn.TrainStrategy.BFGS
        ts = wn.TrainStrategy.Gradient
        w = wn.Net(nc, np.min(inp), np.max(inp), np.average(tar),
                         a0, w0, w1, p0)
        track = w.train(inp, tar, ts, 10, 0.3, 1, False, False)
        #track = w.train(inp, tar, ts, 600, 100000, 1, True, True)
        #import pdb; pdb.set_trace()
        tool.plot(inp, tar, w, track, xlabel='x', ylabel='f(x)')
        print (w.energy(inp, tar))
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








