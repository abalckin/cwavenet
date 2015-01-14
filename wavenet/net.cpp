#include "net.hpp"
#include <iostream>
using namespace std;
Net::Net(int ncount, double xmin, double xmax, double ymin, double a0, double w0, double p0, ActFunc f)
  {
  wcount = ncount*4+1;
  switch (f)
     {
     case  ActivateFunc::Morlet:
       wt = new Morlet();
       break;
     default:
       break;
     }
  nc = ncount;
  weight.set_size(wcount);
  weight(0) = ymin;
  double delta = (xmax - xmin)/nc;
  wn =(wavelon *) (&weight(0) + 1); 
  for (int i=0; i<nc; i++)
    {
      wn[i].a = a0;
      wn[i].b = xmin+i*delta+delta/2;
      wn[i].p = p0;
      wn[i].w =w0;
    }
  }

std_vector Net::sim(const std_vector& t)
{
  return _sim(t, weight);
}

std_vector Net::_sim(const std_vector& t, const column_vector& weight)
{
  wavelon *wn =(wavelon *) (&weight(0) + 1);   
  std_vector out(t.size());
  for (uint i=0; i<t.size();i++)
    {
      double nans = 0.;
      for( int j=0; j<nc; j++)
	{
	  double tau = wt->tau(t[i], wn[j].a, wn[j].b);
	  nans+=wt->h(tau, wn[j].p)*wn[j].w;
	}
      out[i]=nans*t[i]+ weight(0);
    }
  return out;
}

std_vector Net::gradient(const std_vector& t, const std_vector& target)
{
  column_vector gt = _gradient(t, target, weight, true, true);
  std_vector ans(wcount);
  for(int k=0; k<wcount; k++ ) ans[k]=gt(k);
  return ans;
  
}

column_vector Net::_gradient(const std_vector& t, const std_vector& target, const column_vector& weight, bool varc, bool varp)
 {
   wavelon *wn =(wavelon *) (&weight(0) + 1);   
   column_vector gd;
   gd.set_size(wcount);
   for(int k=0; k<wcount; k++ ) gd(k)=0.;
   std_vector ans = sim(t);
   wavelon* gdw =(wavelon *) (&gd(0) + 1);
   for (uint j=0; j<t.size(); j++)
   {
     double e = target[j]-ans[j];
     if (varc)
       gd(0)+=-e;
     for (int i=0; i<nc; i++)
       {
	 double tau = wt->tau(t[j], wn[i].a, wn[i].b);
	 double htau = wt->h(tau, wn[i].p);
	 gdw[i].w+=-e*htau*t[j];
	 double d = e*t[j]*wn[i].w*wt->db(tau, htau, wn[i].a, wn[i].p);
	 gdw[i].b += -d;
	 gdw[i].a += -d*tau;
	 if (varp)
	   gdw[i].p += -e*t[j]*wn[i].w*wt->dp(tau, wn[i].p);
       }
   }
   return gd; 
 }

double Net::energy(const std_vector& t, const std_vector& target)
{
  return _energy(t, target, weight);
}
double Net::_energy(const std_vector& t, const std_vector& target, const column_vector& weight)
{
  double ener = 0.;
  std_vector ans = _sim(t, weight);
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



