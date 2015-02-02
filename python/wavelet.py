# This file was automatically generated by SWIG (http://www.swig.org).
# Version 2.0.12
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.





from sys import version_info
if version_info >= (2,6,0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_wavelet', [dirname(__file__)])
        except ImportError:
            import _wavelet
            return _wavelet
        if fp is not None:
            try:
                _mod = imp.load_module('_wavelet', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _wavelet = swig_import_helper()
    del swig_import_helper
else:
    import _wavelet
del version_info
try:
    _swig_property = property
except NameError:
    pass # Python < 2.2 doesn't have 'property'.
def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "thisown"): return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    if (name == "thisown"): return self.this.own()
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError(name)

def _swig_repr(self):
    try: strthis = "proxy of " + self.this.__repr__()
    except: strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0


class Wavelet(_object):
    """Proxy of C++ Wavelet class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Wavelet, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Wavelet, name)
    def __init__(self, *args, **kwargs): raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    def h(self, *args):
        """
        h(Wavelet self, double tau, double p) -> double

        Parameters:
            tau: double
            p: double

        """
        return _wavelet.Wavelet_h(self, *args)

    def db(self, *args):
        """
        db(Wavelet self, double tau, double htau, double a, double p) -> double

        Parameters:
            tau: double
            htau: double
            a: double
            p: double

        """
        return _wavelet.Wavelet_db(self, *args)

    def dp(self, *args):
        """
        dp(Wavelet self, double tau, double p) -> double

        Parameters:
            tau: double
            p: double

        """
        return _wavelet.Wavelet_dp(self, *args)

    def tau(*args):
        """
        tau(double t, double a, double b) -> double

        Parameters:
            t: double
            a: double
            b: double

        """
        return _wavelet.Wavelet_tau(*args)

    if _newclass:tau = staticmethod(tau)
    __swig_getmethods__["tau"] = lambda x: tau
    __swig_destroy__ = _wavelet.delete_Wavelet
    __del__ = lambda self : None;
Wavelet_swigregister = _wavelet.Wavelet_swigregister
Wavelet_swigregister(Wavelet)

def Wavelet_tau(*args):
  """
    Wavelet_tau(double t, double a, double b) -> double

    Parameters:
        t: double
        a: double
        b: double

    """
  return _wavelet.Wavelet_tau(*args)

class Morlet(Wavelet):
    """Proxy of C++ Morlet class"""
    __swig_setmethods__ = {}
    for _s in [Wavelet]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, Morlet, name, value)
    __swig_getmethods__ = {}
    for _s in [Wavelet]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, Morlet, name)
    __repr__ = _swig_repr
    def __init__(self): 
        """__init__(Morlet self) -> Morlet"""
        this = _wavelet.new_Morlet()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _wavelet.delete_Morlet
    __del__ = lambda self : None;
Morlet_swigregister = _wavelet.Morlet_swigregister
Morlet_swigregister(Morlet)

# This file is compatible with both classic and new-style classes.


