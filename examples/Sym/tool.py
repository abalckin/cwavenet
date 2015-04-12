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
    plb.subplot(212)
    if target is not None:
        plb.plot(t, target, label='d(t)', linestyle=':')
    if orig is not None:
        plb.plot(t, orig, label='y(t)')
    plb.xlabel('t')
    plb.plot(t, wavenet.sim(t, inp), linestyle='--', label='$\hat{y}(t)$')
    plb.legend(loc=0)
    plb.subplot(211)
    plb.ylabel('Энергия ошибки, E')
    plb.plot(param['e'][0])
    plb.xlabel('Эпохи, n')

    plb.figure("Основные веса")
    plb.subplot(131)
    plb.title('Масштабы, a')
    plb.plot(np.transpose(param['a']))
    plb.subplot(132)
    plb.title('Сдвиги, b')
    plb.plot(np.transpose(param['b']))
    plb.subplot(133)
    plb.title('Веса, w')
    plb.plot(np.transpose(param['w']))
    plb.figure("Расширенные веса")
    plb.subplot(131)
    plb.title('Параметры, p')
    plb.plot(np.transpose(param['p']))
    plb.subplot(132)
    plb.title('Смещение, c')
    plb.plot(param['c'][0])
    plb.subplot(133)
    plb.title('Обратная связь, f')
    plb.plot(np.transpose(param['f']))

    #plb.show()


