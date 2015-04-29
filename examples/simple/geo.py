#! /usr/bin/python3
import sys
import tool
import time
from datetime import datetime
sys.path.append('../../bin/')
import wavenet as wn
import numpy as np
import pylab as plb
from matplotlib import rc
rc('text', usetex=True)
rc('text.latex', unicode=True)
rc('text.latex', preamble=r'\usepackage[russian]{babel}')
rc('font',**{'size':'21'})

fname = './data/h_.txt'
date = []
value = []
with open(fname) as f:
    content = f.readlines()
    for l in content:
        if l[0]=='#':
            continue
        else:
            date.append(datetime.strptime(l[0:16], '%Y-%m-%d %H:%M'))
            value.append(float(l[17:24]))
t=plb.date2num(date)[0:2160]
#time=np.reshape(t, [2160/9, 9])[:, 5]
v=np.array(value)[0:2160]
#val = np.sort(np.reshape(v, [2160/9, 9]))[:, 5]
#plb.plot_date(time, val)
#plb.show()
#import pdb; pdb.set_trace()
inp = t
y0 = v[0]
a0 = .1
a1 = .1
w0 = 0
w1 = 0
p0 = 0.1
p1 = 0.1
ncount = 22
import pdb; pdb.set_trace()
start = time.clock()
w = wn.Net(ncount,y0,
                         a0, a1,  w0, w1, p0, p1, 0, 0.01, 0.01, wn.ActivateFunc.Morlet, 4)
track = w.train(t, t, v, wn.TrainStrategy.Gradient, 20, 0.1, 1)
print (time.clock()-start)
#tool.plot(t, t, v, w, track)
plb.plot_date(t, w.sim(t, t), label='$\hat{S}_H(t)$', linestyle='--', marker='')
plb.plot_date(t, v, label='H', linestyle='-', marker='')
plb.xlabel('Время')
plb.ylabel('нТ')
plb.legend(loc=0)
plb.show()

fname = './data/d_.txt'
date = []
value = []
with open(fname) as f:
    content = f.readlines()
    for l in content:
        if l[0]=='#':
            continue
        else:
            date.append(datetime.strptime(l[0:16], '%Y-%m-%d %H:%M'))
            value.append(float(l[17:24]))
t=plb.date2num(date)[0:2160]
#time=np.reshape(t, [2160/9, 9])[:, 5]
v=np.array(value)[0:2160]
#val = np.sort(np.reshape(v, [2160/9, 9]))[:, 5]
#plb.plot_date(time, val)
#plb.show()
#import pdb; pdb.set_trace()
inp = t
y0 = v[0]
a0 = .1
a1 = .1
w0 = 0
w1 = 0
p0 = 0.1
p1 = 0.1
ncount = 22
w = wn.Net(ncount,y0,
                         a0, a1,  w0, w1, p0, p1, 0, 0.01, 0.01, wn.ActivateFunc.Morlet, 4)
track = w.train(t, t, v, wn.TrainStrategy.Gradient, 20, 0.1, 1)
#tool.plot(t, t, v, w, track)
plb.plot_date(t, w.sim(t, t), label='$\hat{S}_D(t)$', linestyle='--', marker='')
plb.plot_date(t, v, label='D', linestyle='-', marker='')
plb.xlabel('Время')
plb.ylabel('$min$/10')
plb.legend(loc=0)
plb.show()

sys.exit()
