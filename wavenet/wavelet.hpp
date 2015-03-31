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
    return p*(tau*tau-1)*exp(-0.5*tau*tau)/a;
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
    return p*tau/(tau*tau+1)/(tau*tau+1);
  }
  virtual double db(double tau, double htau, double a, double p) const 
  {
    return p*(3*tau*tau-1)/a/(tau*tau+1)/(tau*tau+1)/(tau*tau+1);
  }
  virtual double dp(double tau, double p) const 
  {
    return tau/(tau*tau+1)/(tau*tau+1);
  }
};
class RASP2 :public Wavelet
{
  /**Return wavelet value*/
  virtual double h(double tau, double p) const 
  {
    return sin(p*tau)/(tau*tau-1);
  }
  virtual double db(double tau, double htau, double a, double p) const 
  {
    double m1 = 2*tau/a;
    double s1 = m1*sin(p*tau);
    double m2 = (tau*tau-1)*p/a;
    double s2 = m2*cos(p*tau);
    double z = (tau*tau-1);
    return (s1-s2)/z/z;
  }
  virtual double dp(double tau, double p) const 
  {
    return p*cos(p*tau)/(tau*tau-1);
  }
};

class RASP4 :public Wavelet
{
  /**Return wavelet value*/
  virtual double h(double tau, double p) const 
  {
    return tau/(tau*tau+p)/(tau*tau+p);
  }
  virtual double db(double tau, double htau, double a, double p) const 
  {
    return (3*tau*tau-1)/a/(tau*tau+p)/(tau*tau+p)/(tau*tau+p);
  }
  virtual double dp(double tau, double p) const 
  {
    return (-2*tau/(tau*tau+p)/(tau*tau+p)/(tau*tau+p));
  }
};

class RASP3 :public Wavelet
{
  /**Return wavelet value*/
  virtual double h(double tau, double p) const 
  {
    return tau/(tau*tau+1)/(tau*tau+1/p);
  }
  virtual double db(double tau, double htau, double a, double p) const 
  {
    return (3*tau*tau-1)/a/p/(tau*tau+1)/(tau*tau+1)/(tau*tau+1);
  }
  virtual double dp(double tau, double p) const 
  {
    return -tau/(tau*tau+1)/(tau*tau+1)/p/p;
  }
};

class RASP :public Wavelet
{
  /**Return wavelet value*/
  virtual double h(double tau, double p) const 
  {
    return tau*cos(tau)/(tau*tau+1);
  }
  virtual double db(double tau, double htau, double a, double p) const 
  {
    double m1 = tau*(tau*tau+1)/a;
    double s1 = p*m1*sin(p*tau);
    double m2 = (tau*tau-1)/a;
    double s2 = m2*cos(p*tau);
    double z = (tau*tau+1);
    return (s1-s2)/z/z;
  }
  virtual double dp(double tau, double p) const 
  {
    return -tau*tau*sin(p*tau)/(tau*tau+1);
  }
};
