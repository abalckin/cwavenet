import pylab as plb
import numpy as np


def show(t, target, wavenet, param):
    plb.rc('font', family='serif')
    plb.rc('font', size=13)
    plb.figure('Апроксимация')
    plb.subplot(211)
    #plb.title('Вейвсеть из 30 вейвлетов Морле')
    plb.plot(t, target, label='Модельный сигнал')
    plb.plot(t, wavenet.sim(t), label='Аппроксимация')
    plb.legend()
    plb.subplot(212)
    plb.title('Суммарная квадратичная ошибка')
    plb.plot(param['e'][0])
    plb.xlabel('Эпохи')

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
    plb.subplot(121)
    plb.title('Параметры, p')
    plb.plot(np.transpose(param['p']))
    plb.subplot(122)
    plb.title('Смещение, c')
    plb.plot(param['c'][0])
    plb.show()


