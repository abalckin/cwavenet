#! /usr/bin/python3
import unittest
import sys
sys.path.append('../python/')
import wavenet as wn
import numpy as np


class TestSequenceFunctions(unittest.TestCase):
    def setUp(self):
        self.xmax = 10.
        self.xmin = 0.
        self.ymin = 0.
        self.a0 = 10.
        self.w0 = 0.1
        self.p0 = 1.
        self.ncount = 2
        self.wn = wn.Net(self.ncount, self.xmin, self.xmax, self.ymin,
                         self.a0, self.w0)
        self.eps = 1e-10
        self.inp = [0, 5., 10]
        self.morlet = wn.wavelet.Morlet()

    def test_sim(self):
        sim = np.array(self.wn.sim(self.inp))
        ans = []
        delta = (self.xmax - self.xmin)/self.ncount
        for t in self.inp:
            res = 0.
            for n in range(self.ncount):
                b = n*delta
                tau = (t-b)/self.a0
                res += self.morlet.h(tau, self.p0)*self.w0
            ans.append(res*t+self.ymin)
        calc = np.array(ans)
        self.assertTrue(np.all(sim == calc))

    def test_enegry(self):
        targ = np.array([0, 0.5, 1.])
        E = np.array(self.wn.energy(self.inp, targ))
        sim = np.array(self.wn.sim(self.inp))
        calc = np.sum((sim-targ)**2/2)
        self.assertTrue(calc == E)

    def test_gradient(self):
        targ = np.array([0, 0.5, 1.])
        gr = np.array(self.wn.gradientVector(self.inp, targ))
        sim = np.array(self.wn.sim(self.inp))
        e = targ - sim
        c = gr[0]
        self.assertEqual(-np.sum(e), c)
        ws = gr[1:].reshape(2, 4)
        #print(w.shape)
        #import pdb; pdb.set_trace()
        delta = (self.xmax - self.xmin)/self.ncount
        i = np.array(self.inp)
        for n in range(self.ncount):
            #print (n)
            b = n*delta
            tau = (i-b)/self.a0
            self.p0 = float(self.p0)
            htau = np.vectorize(self.morlet.h)(tau, self.p0)
            da = ws[n, 0]
            db = ws[n, 1]
            dp = ws[n, 2]
            dw = ws[n, 3]
            cw = -np.sum(i*htau*e)
            self.assertEqual(dw, cw)
            d = e*i*self.w0*np.vectorize(self.morlet.db)(tau,
                                            htau, self.a0, self.p0)
            cb = -np.sum(d)
            self.assertEqual(db, cb)
            ca = -np.sum(d*tau)
            self.assertEqual(ca, da)
            p = e*i*self.w0*np.vectorize(self.morlet.dp)(tau, self.p0)
            cp = -np.sum(p)
            self.assertEqual(cp, dp)
    def test_sim_speed(self):
        inp = range(1, 1000000)
        sim = np.array(self.wn.sim(inp))
if __name__ == '__main__':
    unittest.main()
