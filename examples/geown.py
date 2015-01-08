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
        
    def calc(self):
        p0=3.
        a0=15.
        #ts = wn.TrainStrategy.BFGS
        ts = wn.TrainStrategy.Gradient
        inp = plb.date2num(self.time)
        tar = self.value-np.average(self.value)
        inp -= np.min(inp)
        delta = np.max(inp)-np.min(inp)
        k = 4*24/delta
        inp *= k
        w = wn.Net(5, np.min(inp), np.max(inp), np.average(tar),
                         a0, .01, p0)
        #track = w.train(inp, tar, ts, 200, 300000, 1, False, False)
        track = w.train(inp, tar, ts, 300, 100000, 1, True, True)
        print('first')
        for c in track.keys():
            print(c)
            a= np.array(track[c])
            print (np.transpose(a[:, 0]))
        print('end')
        for c in track.keys():
            print(c)
            a= np.array(track[c])
            print (np.transpose(a[:, -1]))
        #import pdb; pdb.set_trace()
        tool.plot(inp, tar, w, track)
        we = wn.Net(5, np.min(inp), np.max(inp), np.average(tar),
                         a0, .01, p0)
        plb.show()
        tracke = we.train(inp, tar, ts, 300, 100000, 1, False, False)
        #import pdb; pdb.set_trace()
        plb.title('Суммарная квадратичная ошибка')
        plb.plot(tracke['e'][0], label='Обычная вейвсеть')
        plb.plot(track['e'][0], linestyle='--', label='Полиморфная вейвсеть')
        plb.xlabel('Эпохи')
        plb.legend()
        plb.show()
        tool.plot(inp, tar, we, tracke)
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








