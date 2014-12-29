#! /usr/bin/python3
import unittest
import sys
sys.path.append('../python/')
import math
import wavenet as wn


class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.delta = 1e-6
        self.eps = 1e-10
        self.wavelets = []
        self.morlet = wn.Morlet()
        self.wavelets.append(self.morlet)

    def test_tau(self):
        self.assertTrue(self.morlet.tau(10, 2, 4) == 3.0)

    def test_morlet(self):
        self.assertLess(math.fabs(self.morlet.h(math.pi/4, 5) -
                        -0.519442723414381), self.eps)

    def test_wavelets(self):
        t = 5
        a0 = 1.0
        b0 = 10
        b1 = b0+self.delta
        p0 = 1.0
        p1 = p0+self.delta
        for w in self.wavelets:
                tau0 = w.tau(t, a0, b0)
                htau0 = w.h(tau0, p0)
                tau1 = w.tau(t, a0, b1)
                htau1 = w.h(tau1, p0)
                eb = math.fabs((htau1-htau0)/(b1-b0)-w.db(tau0, htau0, a0, p0))
                self.assertLess(eb, self.eps)
                htau1 = w.h(tau0, p1)
                ep = math.fabs((htau1-htau0)/(p1-p0)-w.dp(tau0, p0))
                self.assertLess(ep, self.eps)

if __name__ == '__main__':
    unittest.main()
