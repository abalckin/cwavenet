#pragma once
#include <math.h>
class Wavelet
{
public:
  virtual double h(double tau,  double p) const = 0;
  virtual double db(double tau, double htau, double a, double p) const = 0;
  virtual double dp(double tau, double p) const = 0;
  static double tau(double t, double a, double b){return (t-b)/a;}
};
class Morlet :public Wavelet
{
  /**Return wavelet value*/
  virtual double h(double tau, double p) const 
  {
    return cos(p*tau)*exp(-0.5*tau*tau);
  }
  virtual double db(double tau, double htau, double a, double p) const 
  {
    return p*sin(p*tau)*exp(-0.5*tau*tau)+tau*htau/a;
  }
  virtual double dp(double tau, double p) const
  {
    return -sin(p*tau)*tau*exp(-0.5*tau*tau);
  }
};
class POLYWOG :public Wavelet
{
  /**Return wavelet value*/
  virtual double h(double tau, double p) const 
  {
    return p*tau*exp(-0.5*tau*tau);
  }
  virtual double db(double tau, double htau, double a, double p) const 
  {
    return (p*tau*tau-1)*exp(-0.5*tau*tau)/a;
  }
  virtual double dp(double tau, double p) const 
  {
    return tau*exp(-0.5*tau*tau);
  }
};
class RASP1 :public Wavelet
{
  /**Return wavelet value*/
  virtual double h(double tau, double p) const 
  {
    return tau/(tau*tau+p)/(tau*tau+p);
  }
  virtual double db(double tau, double htau, double a, double p) const 
  {
    return (3*tau*tau-p)/a/(tau*tau+p)/(tau*tau+p)/(tau*tau+p);
  }
  virtual double dp(double tau, double p) const 
  {
    return -2*tau/(tau*tau+p)/(tau*tau+p)/(tau*tau+p);
  }
};
