#include "./wavelet.hpp"
#include <vector>
#include <dlib/optimization.h>
using namespace dlib;
using namespace std;
#pragma once
typedef matrix<double,0,1> column_vector;
struct wavelon
{
  double a;
  double b;
  double p;
  double w;
};
class Net 
{
 public:
  Net(Wavelet *wt, int ncount, double xmax, double xmin, double ymin, double a0);
  ~Net();
  column_vector sim(const column_vector&  t);
 private:
  int nc;
  column_vector weight;
  Wavelet* wt;
  wavelon* wn;
  column_vector gradient(const column_vector&  t);

};

