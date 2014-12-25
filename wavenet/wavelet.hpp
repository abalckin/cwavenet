#pragma once
#include <math.h>
class Wavelet
{
public:
  virtual double h(double t, double a, double b, double p) const = 0;
protected:
  inline static double getTau(double t, double a, double b){return (t-b)/a;}
};
class Morlet :public Wavelet
{
  virtual double h(double t, double a, double b, double p) const 
  {
    double tau = getTau(t, a, b);
    return cos(p*tau)*exp(-0.5*tau*tau);
  }
};
