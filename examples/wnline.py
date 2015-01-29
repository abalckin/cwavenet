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
        return np.sin(-3*x)-100
        
    def calc(self):
        t = np.arange(-10, 100, 5.)
        tar = np.vectorize(self.func1)(t)
        #tar -= np.min(tar)
        #tar /= np.max(tar)
        size = len(t)
        p0=3.3
        a0=6.
        inp=t-100/33
        nc = 20
        #ts = wn.TrainStrategy.BFGS
        ts = wn.TrainStrategy.Gradient
        w = wn.Net(nc, np.min(inp), np.max(inp), np.average(tar),
                         a0, .01, 0.01, p0)
        track = w.train(t, inp, tar, ts, 100, 0.3, 1, False, False)
        #track = w.train(inp, tar, ts, 600, 100000, 1, True, True)
        #import pdb; pdb.set_trace()
        tool.plot(t, inp, tar, w, track)
        print (w.energy(t, inp, tar))
        plb.show()
        sys.exit()
    def __init__(self):
        self.fileName = "../data/spidr_1420457768932_0.txt"
        self.load()
def main():
    t = Test()
    t.calc()
if __name__ == '__main__':
    main()








