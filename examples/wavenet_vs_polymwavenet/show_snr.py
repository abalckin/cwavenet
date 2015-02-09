#! /usr/bin/python3
import pylab as plb
import numpy as np

from matplotlib import rc
rc('text', usetex=True)
rc('text.latex', unicode=True)
rc('text.latex', preamble=r'\usepackage[russian]{babel}')
#rc('font',**{'family':'serif'})
rc('font',**{'size':'22'})
res = np.loadtxt('result.txt', delimiter=', ')[0:7]
#import pdb; pdb.set_trace()
#plb.barh(y_pos, performance, xerr=error, align='center', alpha=0.4)
#plb.yscale('linear')
plb.errorbar(res[:, 1], res[:, 5], yerr=res[:, 6], label='Стандартная вейвсеть', linestyle='--', marker='*', color='black')
plb.errorbar(res[:, 1], res[:, 11], yerr=res[:, 12], label='Полиморфная вейвсеть', marker='o', color='green')
plb.errorbar(res[:, 1], res[:, 1],  yerr=res[:, 2], label='Отношение сигнал/шум на входе системы, S', color='blue')
#import pdb; pdb.set_trace()
plb.fill_between(res[:, 1], res[:, 1], res[:, 1]-np.max(res[:, 1]), res[:, 1], alpha=0.1, color='blue')
plb.xscale('log')
plb.legend(loc=0)
plb.xlim(res[-1, 1]-0.1, res[0, 1]+20)
plb.ylim(0, 310)
plb.gca().set_xticks(res[:, 1])
#plb.gca().xaxis.set_major_locator(plb.LogLocator(numticks=50))
plb.gca().xaxis.set_major_formatter(plb.ScalarFormatter())
plb.ylabel('Отношение сигнал/шум на выходе сети, M')
plb.xlabel('Отношение сигнал/шум на входе системы, S')
plb.annotate('Область применимости вейвсетей', [10, 175])
plb.show()