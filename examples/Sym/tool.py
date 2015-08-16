#! /usr/bin/python3
import pylab as plb
import numpy as np
from matplotlib import rc
rc('text', usetex=True)
rc('text.latex', unicode=True)
rc('text.latex', preamble=r'\usepackage[russian]{babel}')
rc('font',**{'size':'19'})



def plot(t, inp, wavenet, param, orig=None, target=None, xlabel='', ylabel=''):
    plb.rc('font', family='serif')
    plb.rc('font', size=13)
    plb.figure('Апроксимация')
    #plb.subplot(212)
    if target is not None:
        plb.plot(t, target, label='d(t)', linestyle=':')
    if orig is not None:
        plb.plot(t, orig, label='y(t)')
    plb.xlabel('n')
    plb.plot(t, wavenet.sim(t, inp), linestyle='--', label='$\hat{y}(t)$')
    plb.legend(loc=0)
    #_____________
    leng= len(param['e'][0])
    plb.figure('Уточненная ошбика')
    plb.subplot(211)
    plb.ylabel('Энергия ошибки, E')
    plb.plot(param['e'][0][0:leng//10])
    plb.xlabel('Эпохи, n')
    plb.subplot(212)
    plb.ylabel('Энергия ошибки, E')
    plb.plot(np.arange(leng//10, leng), param['e'][0][leng//10:leng])
    plb.xlabel('Эпохи, n')
    plb.show()
    #_____________
    plb.figure("Основные веса")
    plb.subplot(231)
    plb.title('Масштабы, $\overline{a}$')
    plb.plot(np.transpose(param['a']))
    plb.subplot(232)
    plb.title('Сдвиги, $\overline{b}$')
    plb.plot(np.transpose(param['b']))
    plb.subplot(233)
    plb.title('Веса, $\overline{w}$')
    plb.plot(np.transpose(param['w']))
##    plb.figure("Расширенные веса")
    plb.subplot(234)
    plb.title('Параметры, $\overline{p}$')
    plb.plot(np.transpose(param['p']))
    plb.subplot(235)
    plb.title('Смещение, c')
    plb.plot(param['c'][0])
##    plb.subplot(133)
##    plb.title('Обратные связи, $\overline{r}$')
##    plb.plot(np.transpose(param['f']))

    #plb.show()


