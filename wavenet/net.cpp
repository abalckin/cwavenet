#include "net.hpp"
#include <iostream>
using namespace std;
Net::Net(int ncount, double xmin, double xmax, double ymin, double a0, double w0, ActFunc f)
  {
  wcount = ncount*4+1;
  //wt = new Morlet();
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
      wn[i].b = i*delta;
      wn[i].p = wt->p0();
      wn[i].w =w0;
    }
  }

std_vector Net::sim(const std_vector& t)
{
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
column_vector Net::gradient(const std_vector& t, const std_vector& target)
 {
   column_vector gd;
   gd.set_size(wcount);
   for(int k=0; k<wcount; k++ ) gd(k)=0.;
   std_vector ans = sim(t);
   wavelon* gdw =(wavelon *) (&gd(0) + 1);
   for (uint j=0; j<t.size(); j++)
   {
     double e = target[j]-ans[j];
     gd(0)+=-e;
     for (int i=0; i<nc; i++)
       {
	 double tau = wt->tau(t[j], wn[i].a, wn[i].b);
	 double htau = wt->h(tau, wn[i].p);
	 gdw[i].w+=-e*htau*t[j];
	 double d = e*t[j]*wn[i].w*wt->db(tau, htau, wn[i].a, wn[i].p);
	 gdw[i].b += -d;
	 gdw[i].a += -d*tau;
	 gdw[i].p += -e*t[j]*wn[i].w*wt->dp(tau, wn[i].p);
       }
   }
   return gd; 
 }
std_vector Net::gradientVector(const std_vector& t, const std_vector& target)
{
  column_vector gt = gradient(t, target);
  std_vector ans(wcount);
  for(int k=0; k<wcount; k++ ) ans[k]=gt(k);
  return ans;
}
double Net::energy(const std_vector& t, const std_vector& target)
{
  double ener = 0.;
  std_vector ans = sim(t);
  for(uint i=0; i<t.size(); i++)
    {
      double err = target[i] - ans[i];
      ener += err*err;
    }
  return ener/2;
}



