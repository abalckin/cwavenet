#! /usr/bin/python3
import pylab as plb
import numpy as np

from matplotlib import rc
rc('text', usetex=True)
rc('text.latex', unicode=True)
rc('text.latex', preamble=r'\usepackage[russian]{babel}')
#rc('text.latex', preamble=r'\usepackage[lmodern]')
#rc('font',**{'family':'sans-serif','sans-serif':['Times']})
rc('font', **{'size':'22'})
res = np.loadtxt('result.txt', delimiter=', ')[0:7]
#import pdb; pdb.set_trace()
#plb.barh(y_pos, performance, xerr=error, align='center', alpha=0.4)
#plb.yscale('linear')
plb.errorbar(res[:, 1], res[:, 3], yerr=res[:, 4], label='Традиционная вейвлет-сеть', linestyle='--', marker='*', color='black')
plb.errorbar(res[:, 1], res[:, 9], yerr=res[:, 10], label='Полиморфная вейвлет-сеть', marker='o', color='green')
#plb.errorbar(res[:, 1], res[:, 1],  yerr=res[:, 2], label='Отношение сигнал/шум на входе системы', color='blue')
#import pdb; pdb.set_trace()
#plb.fill_between(res[:, 1], res[:, 1], res[:, 1]-np.max(res[:, 1]), res[:, 1], alpha=0.1, color='blue')
plb.xscale('log')
plb.legend(loc=0)
plb.xlim(res[-1, 1]-0.1, res[0, 1]+20)
#plb.ylim(0, 310)
plb.gca().set_xticks(res[:, 1])
#plb.gca().xaxis.set_major_locator(plb.LogLocator(numticks=50))
plb.gca().xaxis.set_major_formatter(plb.ScalarFormatter())
plb.ylabel(u'Продолжительность сходимости вейвлет-сети, $\hat{n}$')
plb.xlabel('Отношение сигнал/шум для временного ряда $d(t), S$')
#plb.annotate('Область применимости вейвлет-сетей', [15, 175])
plb.show()
polym_higest=res[:, 11]>res[:, 1]
polym_avg=res[polym_higest, 9][1:-2]
std_higest=res[:, 5]>res[:, 1]
std_avg=res[std_higest, 3][:-2]
inp_avg=res[std_higest, 1][:-2]
polym_min=res[polym_higest, 9][1:-2]-res[polym_higest, 10][1:-2]
polym_max=res[polym_higest, 9][1:-2]+res[polym_higest, 10][1:-2]
std_max=res[std_higest, 3][:-2]+res[std_higest, 4][:-2]
std_min=res[std_higest, 3][:-2]-res[std_higest, 4][:-2]
print('Улучшение в среднем на {}%'.format(np.average((std_avg-polym_avg)/std_avg*100)))
print('Улучшение по диапазону на {0}-{1}%'.format((np.average((std_min-polym_min)/std_min*100)),
      np.average((std_max-polym_max)/std_max*100)))
print('Улучшение по диапазону на {0}-{1}'.format((np.average((std_min-polym_min))),
      np.average((std_max-polym_max))))















