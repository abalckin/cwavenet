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
res = np.loadtxt('result.txt', delimiter=', ')[0:10]
#import pdb; pdb.set_trace()
#plb.barh(y_pos, performance, xerr=error, align='center', alpha=0.4)
#plb.yscale('linear')
plb.errorbar(res[:, 1], res[:, 3], yerr=res[:, 4], label='Полиморфная вейвлет-сеть', linestyle='--', marker='*', color='black')
plb.errorbar(res[:, 1], res[:, 9], yerr=res[:, 10], label='Многослойная нейронная сеть', marker='o', color='green')
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
plb.xlabel('Отношение сигнал/шум на входе сети, S')
plb.annotate('Область применимости вейвлет-сетей', [15, 175])
plb.show()

a = 3
b = -3
neur_avg = res[a:b, 9]
poly_avg = res[a:b, 3]
neur_min = res[a:b, 9]-res[a:b, 10]
neur_max = res[a:b, 9]+res[a:b, 10]
poly_min = res[a:b, 3]-res[a:b, 4]
poly_max = res[a:b, 3]+res[a:b, 4]
#import pdb; pdb.set_trace()
print('Улучшение в среднем на {}%'.format(np.average((neur_avg/poly_avg))))

#print('Улучшение в среднем на {}%'.format(np.average((neur_avg-poly_avg)/neur_avg*100)))
#print('Улучшение по диапазону на {0}-{1}%'.format((np.average((neur_min-poly_min)/neur_min*100)),
#      np.average((neur_max-poly_max)/neur_max*100)))
#print('Улучшение по диапазону на {0}-{1}'.format((np.average((neur_min-poly_min))),
#      np.average((neur_max-poly_max))))














