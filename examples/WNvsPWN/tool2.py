#! /usr/bin/python3
import pylab as plb
import numpy as np



def plot(t, inp, target, wavenet, param, orig=None, xlabel='', ylabel=''):
    plb.rc('font', family='serif')
    plb.rc('font', size=13)
    plb.figure('Апроксимация')
    plb.subplot(212)
    plb.plot(t, target, label='Модельный сигнал')
    if orig is not None:
        plb.plot(t, orig, label='Оригинал')
    plb.plot(t, wavenet.sim(t, inp), linestyle='--', label='Аппроксимация')
    plb.legend(loc=0)
    plb.subplot(211)
    #plb.title('Суммарная квадратичная ошибка')
    plb.plot(param['e'][0])
    plb.xlabel('Эпохи')
    plb.ylabel('Энергия ошибки, E')
    
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
    plb.title('Обратные связи, f')
    plb.plot(np.transpose(param['f']))
    #import pdb; pdb.set_trace()
  
    #plb.show()


