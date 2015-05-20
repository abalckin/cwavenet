#! /usr/bin/python3
import pylab as plb
import numpy as np

from matplotlib import rc
rc('text', usetex=True)
rc('text.latex', unicode=True)
rc('text.latex', preamble=r'\usepackage[russian]{babel}')
#rc('font',**{'family':'serif'})
rc('font',**{'size':'21'})
res = np.loadtxt('result.txt', delimiter=', ')[0:10]
#import pdb; pdb.set_trace()
#plb.barh(y_pos, performance, xerr=error, align='center', alpha=0.4)
#plb.yscale('linear')
plb.errorbar(res[:, 1], res[:, 5], yerr=res[:, 6], label='Полиморфная вейвлет-сеть', linestyle='--', marker='*', color='black')
plb.errorbar(res[:, 1], res[:, 11], yerr=res[:, 12], label='Многослойная нейронная сеть', marker='o', color='green')
plb.errorbar(res[:, 1], res[:, 1],  yerr=res[:, 2], label='M=S', color='blue')
#import pdb; pdb.set_trace()
plb.fill_between(res[:, 1], res[:, 1], res[:, 1]-np.max(res[:, 1]), res[:, 1], alpha=0.1, color='blue')
plb.xscale('log')
plb.legend(loc=0)
plb.xlim(res[-1, 1]-0.1, res[0, 1]+50)
plb.ylim(0, 2100)
plb.gca().set_xticks(res[:, 1])
#plb.gca().xaxis.set_major_locator(plb.LogLocator(numticks=50))
plb.gca().xaxis.set_major_formatter(plb.ScalarFormatter())
plb.ylabel('Отношение сигнал/шум на выходе сети, M')
plb.xlabel('Отношение сигнал/шум на входе сети, S')
#plb.annotate('Область применимости нейронных сетей', [13, 700])
plb.show()
a = 3
b = -3
neur_avg = res[a:b, 11]
poly_avg = res[a:b, 5]
neur_min = res[a:b, 11]-res[a:b, 12]
neur_max = res[a:b, 11]+res[a:b, 12]
poly_min = res[a:b, 5]-res[a:b, 6]
poly_max = res[a:b, 5]+res[a:b, 6]
import pdb; pdb.set_trace()
print('Улучшение в среднем на {}%'.format(np.average((poly_avg-neur_avg)/neur_avg*100)))
print('Улучшение в по диапазону на {0}-{1}%'.format(np.average((poly_min-neur_min)/neur_min*100),
      np.average((poly_max-neur_max)/neur_max*100)))


















