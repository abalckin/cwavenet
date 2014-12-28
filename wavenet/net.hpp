#include "./wavelet.hpp"
#include <vector>
#include <dlib/optimization.h>
#pragma once
typedef dlib::matrix<double,0,1> column_vector;
typedef std::vector<double> std_vector;
struct wavelon
{
  double a;
  double b;
  double p;
  double w;
};
struct ActivateFunc {
    enum Func {Morlet,SLOG,POLYWOG};
};
typedef ActivateFunc::Func ActFunc;
class Net 
{
 public:
  Net(int ncount, double xmin, double xmax, double ymin, double a0=10., double w0=0.1, ActFunc f = ActivateFunc::Morlet);
  std_vector sim(const std_vector&  t);
  std_vector gradientVector(const std_vector& t, const std_vector& target);
  double energy(const std_vector& t, const std_vector& target);
 private:
  int nc;
  int wcount;
  column_vector weight;
  Wavelet* wt;
  wavelon* wn;
  column_vector gradient(const std_vector&  t, const std_vector&  target);
};

