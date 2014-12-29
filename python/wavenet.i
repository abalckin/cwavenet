%module wavenet
%feature("autodoc", "3");
%import "wavelet.i"
%include "std_vector.i"
namespace std {
   %template(std_vector) vector<double>;
};
%{
#include "../wavenet/net.hpp"
%}
%include ../wavenet/net.hpp
