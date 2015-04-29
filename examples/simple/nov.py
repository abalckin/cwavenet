#! /usr/bin/python3
import sys
#import tool
import time
import itertools
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
def ff (i , j):
    if i==j:
        return -2
    elif i-j == 1 or i-j == -1:
        return 1
    else:
        return 0

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
x = t
y = v
delta1 = 5e-3
delta2 = 5e-3
M = 20/3
N = 2160
hi = np.ones(N)/delta1
A = np.zeros([N, N])
np.fill_diagonal(A, hi)
D = np.array([ff(i, j) for i, j in itertools.product(range(N), range(N))]).reshape([N, N])
dD = np.dot(np.transpose(D),D)
a = np.dot(A,A) + dD 
c = np.dot(A, y)
s = np.linalg.solve(a, c)
d=(y-s).reshape(60, 36)
Vj=np.max(d, axis=0)-np.min(d, axis=0)
hi = np.ones(N)
for j in range(0, 36):
    for k in range(0, 60):
        hi[j*60+k]=np.exp(1-Vj[j]/M)/delta2
#import pdb; pdb.set_trace()
A = np.zeros([N, N])
np.fill_diagonal(A, hi)
a = np.dot(A,A) + dD
c = np.dot(A, y)
s = np.linalg.solve(a, c)
#import pdb; pdb.set_trace()
#print (time.clock()-start)
plb.plot_date(x, s, label='$\hat{S}_D(t)$', linestyle='--', marker='')
plb.plot_date(x, y, label='D', linestyle='-', marker='')
plb.xlabel('Время')
plb.ylabel('$min$/10')
plb.legend(loc=0)
plb.show()
#sys.exit()
