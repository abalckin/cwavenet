#! /usr/bin/python3
import pylab as plb
import numpy as np
import sys

from matplotlib import rc
rc('text', usetex=True)
rc('text.latex', unicode=True)
rc('text.latex', preamble=r'\usepackage[russian]{babel}')
rc('font',**{'family':'serif'})
rc('font',**{'size':'17'})
Eav = np.loadtxt('Eav.txt', delimiter=', ')
Ereal = np.loadtxt('Ereal.txt', delimiter=', ')
Emin = np.loadtxt('Emin.txt', delimiter=', ')
x = np.arange(0, 201)
y = Eav
y010 = np.extract(y < y[0]*0.1, y)[0]
yinf = y[-1]
x010 = np.nonzero(y == y010)[0][0]
xinf = x[-1]

plb.plot(x, y, linestyle='-')
plb.ylabel('Усредненная энергия ошибки, $E_{av}(n)$')
plb.xlabel('Количество итераций, N')
plb.hlines(linewidth=1, color='black', xmin=0., xmax=x010, y=y010, linestyle='--')
plb.vlines(linewidth=1, color='black', ymin=0., ymax=y010, x=x010, linestyle='--')
plb.hlines(linewidth=1, color='black', xmin=0., xmax=xinf, y=yinf, linestyle='--')
plb.axvline(linewidth=1, color='black')
#plb.ylim((0., .1))
plb.annotate('$0,1E_{{av}}(0)$', xy=(0, y010), xytext=(-30, 400),
            arrowprops=dict(facecolor='black', shrink=0.1, width=0.5),
            )
plb.annotate('$E_{{av}}(\infty)$={E:.1f}'.format(E=yinf), xy=(200, yinf), xytext=(170, 400),
            arrowprops=dict(facecolor='black', shrink=0.1, width=0.5),
            )
plb.annotate('', xy=(0, -5), xytext=(x010, -5),
             arrowprops=dict(arrowstyle="<->",
                                connectionstyle="bar,fraction=-0.3",
                                ec="k",
                                shrinkA=15, shrinkB=15,))

plb.annotate('Скорость сходимости, n={}'.format(x010),xy=(0,0), xytext=(-15., -600.))
            ## arrowprops=dict(arrowstyle="-[", facecolor='black',
            ##                 widthA=1.0,lengthA=0.2),
            ## )
plb.gca().xaxis.set_label_coords(0.8, -0.05)

plb.gca().yaxis.set_major_locator(plb.MultipleLocator(1500))
#plb.gca().yaxis.set_major_locator(plb.NullLocator())
plb.show()
sys.exit()

















