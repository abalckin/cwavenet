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
def getKinx(e):
    if 0<=e<5:
        return 0
    elif 5<=e<10:
        return 1
    elif 10<=e<20:
        return 2
    elif 20<=e<40:
        return 3
    elif 40<=e<70:
        return 4
    elif 70<=e<120:
        return 5
    elif 120<=e<200:
        return 6
    elif 200<=e<300:
        return 7
    elif 330<=e<550:
        return 8
    elif 500<=e:
        return 8
   
    else:
        return np.NAN








fname = './data/x_.txt'
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
t=plb.date2num(date)[1440/2:-1440/2]
#time=np.reshape(t, [2160/9, 9])[:, 5]
x=np.array(value)[1440/2:-1440/2]
fname = './data/y_.txt'
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
#t=plb.date2num(date)[0:2160]
#time=np.reshape(t, [2160/9, 9])[:, 5]
y=np.array(value)[1440/2:-1440/2]

#val = np.sort(np.reshape(v, [2160/9, 9]))[:, 5]
#plb.plot_date(time, val)
#plb.show()
#import pdb; pdb.set_trace()
vH =-np.sqrt(x**2+y**2)
inp = t
y0 = vH[0]
a0 = .1
a1 = .1
w0 = 0
w1 = 0
p0 = 0.1
p1 = 0.1
ncount = 22
#import pdb; pdb.set_trace()
#start = time.clock()
w = wn.Net(ncount,y0,
                         a0, a1,  w0, w1, p0, p1, 0, 0.01, 0.01, wn.ActivateFunc.Morlet, 4)
track = w.train(t, t, vH, wn.TrainStrategy.Gradient, 20, 0.1, 1)
#print (time.clock()-start)
#tool.plot(t, t, v, w, track)
plb.plot_date(t, w.sim(t, t), label='$\hat{S}_H(t)$', linestyle='--', marker='')
plb.plot_date(t, vH, label='H', linestyle='-', marker='')
plb.xlabel('Время')
plb.ylabel('нТ')
plb.legend(loc=0)
#plb.show()
SqH=w.sim(t, t)
vD = np.arctan(y/x)*180/np.pi*600
#val = np.sort(np.reshape(v, [2160/9, 9]))[:, 5]
#plb.plot_date(time, val)
plb.show()
#import pdb; pdb.set_trace()
inp = t
y0 = vD[0]
a0 = .05
a1 = .05
w0 = 0
w1 = 0
p0 = 0.1
p1 = 0.1
ncount = 22
w = wn.Net(ncount,y0,
                         a0, a1,  w0, w1, p0, p1, 0, 0.01, 0.01, wn.ActivateFunc.Morlet, 4)
track = w.train(t, t, vD, wn.TrainStrategy.Gradient, 20, 0.1, 1)
#tool.plot(t, t, v, w, track)
plb.plot_date(t, w.sim(t, t), label='$\hat{S}_D(t)$', linestyle='--', marker='')
plb.plot_date(t, vD, label='D', linestyle='-', marker='')
plb.xlabel('Время')
plb.ylabel('$min$/10')
plb.legend(loc=0)
plb.show()
SqD=w.sim(t, t)
#sys.exit()
#D K Index
np.set_printoptions(precision=1)
epsD = np.abs(SqD-vD)
epsD = np.reshape(epsD, [16, 180])
epsD = np.max(epsD, axis=1)[4:-4]
epsH = np.abs(SqH-vH)
epsH = np.reshape(epsH, [16, 180])
epsH = np.max(epsH, axis=1)[4:-4]
KD = np.vectorize(getKinx)(epsD)
KH = np.vectorize(getKinx)(epsH)
K =np.max([KD, KH],axis=0)
print ('\neD\n', epsD,'\nKD\n', KD,'\neH\n', epsH,'\nKH\n', KH,'\nK\n', K)
