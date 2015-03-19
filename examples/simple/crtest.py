#! /usr/bin/python3
"""
#########################################################################
in Python, register for and handle event callbacks from the C language;
compile and link the C code, and launch this with 'python register.py'
#########################################################################
"""

####################################
# C calls these Python functions;
# handle an event, return a result
####################################
class Caller(object):
    def __init__(self, str):
        self.str=str
    def __call__(self, count):
        print(self.str.format(count))

#######################################
# Python calls a C extension module
# to register handlers, trigger events
#######################################
import sys
sys.path.append('../../bin/')
import cregister
cal = Caller("Helllo{0}")
cb = cregister.Caller()
print('\nTest1:')
cb.setHandler(cal)      # register callback function
#import pdb; pdb.set_trace()

for num in range(1, 10):
    res = cb.triggerEvent(num)         # simulate events caught by C layer











