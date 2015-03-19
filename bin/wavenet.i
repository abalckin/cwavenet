%module wavenet
%feature("autodoc", "3");
//%feature("director") Callback;
%import "wavelet.i"
%include "std_vector.i"
%include "std_list.i"
%include "std_string.i"
%include "std_map.i"
namespace std {
   %template(std_vector) vector<double>;
   %template(param_series) list<double>;
 };
%{
#include "../wavenet/net.hpp"
%}
%include ../wavenet/net.hpp
namespace std {
   %template(train_set) vector<param_series>;
   %template(train_res) map<string, train_set>;
};


