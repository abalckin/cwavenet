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
        inp = inp.reshape(size, 1)
        tar = tar.reshape(size, 1)
        net = nl.net.newff([[np.min(inp), np.max(inp)]], [10, 1])
        net.trainf = nl.train.train_gd
        #print(dir(net))
        #for l in net.layers:
        #    l.initf = nl.init.InitRand([-10., 10.], 'wb')
        ##    print (l.np)
        error = net.train(inp, tar, epochs=100, goal=0.3, adapt=True)
        # Simulate network
        out = net.sim(inp)
        #print(out)
        # Plot result
        plb.subplot(211)
        plb.plot(error)
        plb.xlabel('Эпоха')
        plb.ylabel('Суммарная квардратичная ошибка E')


        plb.subplot(212)
        plb.plot(inp, tar, linestyle='-')
        plb.plot(inp, out, linestyle='--')
        plb.xlabel('x')
        plb.ylabel('f(x)')
        plb.legend(['Модельная функция', 'Аппроксимация'], loc=0)
        plb.show()
        sys.exit()

    def __init__(self):
        plb.rc('font', family='serif')
        plb.rc('font', size=14)
        self.fileName = "../data/spidr_1420457768932_0.txt"
        self.load()


def main():
    t = Test()
    t.calc()
if __name__ == '__main__':
    main()





