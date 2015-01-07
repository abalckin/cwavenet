#! /usr/bin/python3
import sys
import tool
sys.path.append('../python/')
import neurolab as nl
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

    def calc(self):
        x = plb.date2num(self.time)
        y = self.value
        x -= np.min(x)
        delta = np.max(x)-np.min(x)
        k = 4*24/delta
        x *= k
        #y-=np.average(y)
        y/=10000
        #x-=np.average(x)
        #x = np.linspace(-7, 7, 20)
        #y = np.sin(x) * 0.5
        size = len(x)
        inp = x.reshape(size, 1)
        tar = y.reshape(size, 1)
        net = nl.net.newff([[np.min(x), np.max(x)]], [50, 1])
        net.trainf = nl.train.train_cg
        for l in net.layers:
            l.np['b'][:] = np.average(y)
        ##    print (l.np)
        error = net.train(inp, tar, epochs=50, goal=1e-7)
        # Simulate network
        out = net.sim(inp)
        #print(out)
        # Plot result
        plb.subplot(211)
        plb.plot(error)
        plb.xlabel('Epoch number')
        plb.ylabel('error (default SSE)')


        plb.subplot(212)
        plb.plot(inp, tar, inp, out)
        plb.legend(['train target', 'net output'])
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





