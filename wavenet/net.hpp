#include "./wavelet.hpp"
#include <vector>
#include <dlib/optimization.h>
using namespace dlib;
using namespace std;
//using namespace vector;
#pragma once
typedef matrix<double,0,1> column_vector;
typedef std::vector<double> std_vector;
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
  //~Net();
  std_vector sim(const std_vector&  t);
 private:
  int nc;
  int wcount;
  column_vector weight;
  Wavelet* wt;
  wavelon* wn;
  column_vector gradient(const std_vector&  t, const std_vector&  target);
  double energy(const std_vector& t, const std_vector& target);
};

