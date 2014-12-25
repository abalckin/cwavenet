#include "./wavelet.hpp"
#include <vector>
using namespace std;
#pragma once
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
  vector<double> sim(vector<double> t);
 private:
  int nc;
  double* weight;
  Wavelet* wt;
  wavelon* wn;
  vector <double> gradient(vector <double> t);

};

