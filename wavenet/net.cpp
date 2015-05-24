#include "net.hpp"
#include <iostream>
#include <stdlib.h>     /* srand, rand */
#include <time.h> 
using namespace std;
Net::Net(int ncount, double c0,
 double a0, double a1, double w0, double w1, double p0, double p1,
	 int fcount, double f0, double fbcoef, ActFunc f, int numberOfThreads)
  {
  wcount = ncount*4+1+fcount;
  switch (f)
     {
     case  ActivateFunc::Morlet:
       wt = new Morlet();
       break;
     case ActivateFunc::POLYWOG:
       wt= new POLYWOG();
       break;
     case ActivateFunc::RASP:
       wt = new RASP();
       break;
     case ActivateFunc::RASP1:
       wt = new RASP1();
       break;
     case ActivateFunc::RASP2:
       wt = new RASP2();
       break;
     case ActivateFunc::RASP3:
       wt = new RASP3();
       break;

     default:
       break;
     }
  nc = ncount;
  fc = fcount;
  fb = fbcoef;
  weight.set_size(wcount);
  weight(0) = c0;
  wn =(wavelon *) (&weight(0) + 1 + fc);
  srand (time(NULL));
  omp_set_num_threads(numberOfThreads);
  float dw = w1-w0;
  float da = a1-a0;
  float dp = p1-p0;
  for (int i=0; i<nc; i++)
    {
      float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      wn[i].w = (r*dw)+w0;
      r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      wn[i].a = (r*da)+a0;
      r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      wn[i].p = (r*dp)+p0;
    }
  double *fcoef = (&weight(0) + 1);
  for (int i=0; i<fc; i++)
      fcoef[i]=f0;
  }

std_vector Net::sim(const std_vector& t, const std_vector&  inp)
{
  return _sim(t, inp, weight);
}

std_vector Net::_sim(const std_vector& t,const std_vector&  inp, const column_vector& weight)
{
  wavelon *wn =(wavelon *) (&weight(0) + 1+fc);
  const double *f = &weight(0) + 1;
  std_vector out(t.size());
  std_vector ans(t.size());
  #pragma omp parallel for
  for (uint i=0; i<t.size();i++)
    {
      double nans = 0.;
      #pragma omp parallel for reduction(+:nans)
      for( int j=0; j<nc; j++)
	{
	  double tau = wt->tau(t[i], wn[j].a, wn[j].b);
	  nans+=wt->h(tau, wn[j].p)*wn[j].w;
	}
      out[i]=nans*inp[i]+ weight(0);
    }
  for(uint i=0; i<t.size(); i++)
    {
      ans[i]=0.;
      for (int j=0; j<fc; j++)
	{
	  double fval = 0.;
	  int inx = i-j;
	  if (inx>=0)
	    fval=out[inx];
	  else
	    fval=out[0];
	  ans[i]+=f[j]*fval*fb;
	}
    }
  return ans;
}

std_vector Net::gradient(const std_vector& t, const std_vector&  inp, const std_vector& target)
{
  column_vector gt = _gradient(t, inp, target, weight, true, true);
  std_vector ans(wcount);
  for(int k=0; k<wcount; k++ ) ans[k]=gt(k);
  return ans;
  
}

column_vector Net::_gradient(const std_vector& t, const std_vector&  inp, const std_vector& target, const column_vector& weight, bool varc, bool varp)
 {
   wavelon *wn =(wavelon *) (&weight(0) + 1 + fc);   
   column_vector gd;
   gd.set_size(wcount);
   for(int k=0; k<wcount; k++ ) gd(k)=0.;
   std_vector ans = sim(t, inp);
   wavelon* gdw =(wavelon *) (&gd(0) + 1 + fc);
   double* gf = (&gd(0) + 1);
   #pragma omp parallel for
   for (uint j=0; j<t.size(); j++)
   {
     double e = target[j]-ans[j];
     if (varc)
       gd(0)+=-e;
     for (int i=0; i<nc; i++)
       {
	 double tau = wt->tau(t[j], wn[i].a, wn[i].b);
	 double htau = wt->h(tau, wn[i].p);
	 gdw[i].w+=-e*htau*inp[j];
	 double d = e*inp[j]*wn[i].w*wt->db(tau, htau, wn[i].a, wn[i].p);
	 gdw[i].b += -d;
	 gdw[i].a += -d*tau;
	 if (varp)
	   gdw[i].p += -e*inp[j]*wn[i].w*wt->dp(tau, wn[i].p);
       }
   }
   for(int j=0; j<fc; j++)
     {
       for(uint i=0; i<t.size(); i++)
	 {
	   double e = target[i]-ans[i];
	   double a = 0.;
	   int inx = i-j;
	   if (inx>=0)
	     a=ans[inx];
	   gf[j]-=e*a;

	 }
     }
   return gd; 
 }

double Net::energy(const std_vector& t, const std_vector&  inp, const std_vector& target)
{
  return _energy(t, inp, target, weight);
}
double Net::_energy(const std_vector& t, const std_vector&  inp, const std_vector& target, const column_vector& weight)
{
  double ener = 0.;
  std_vector ans = _sim(t, inp, weight);
  for(uint i=0; i<t.size(); i++)
    {
      double err = target[i] - ans[i];
      ener += err*err;
    }
  return ener/2;
}
Net::~Net()
{
  delete[] wt;
}



