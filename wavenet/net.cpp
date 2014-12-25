#include "net.hpp"
#include <iostream>

Net::Net(Wavelet* wavelet, int ncount, double xmin, double xmax, double ymin, double a0)
{
  wcount = ncount*4+1;
  wt = wavelet;
  nc = ncount;
  weight.set_size(wcount);
  weight(0) = ymin;
  double delta = (xmax - xmin)/wcount;
  wn =(wavelon *) (&weight(0) + 1); 
  for (int i=0; i<nc; i++)
    {
      wn[i].a = a0;
      wn[i].b = i*(delta+1);
      wn[i].p = 1.;
      wn[i].w =0.05;
    }  
}

std_vector Net::sim(const std_vector& t)
{
  std_vector out(t.size());
  for (int i=0; i<t.size();i++)
    {
      double nans = 0.;
      for( int j=0; j<nc; j++)
	{
	  double tau = wt->tau(t[j], wn[j].a, wn[j].b);
	  nans+=wt->h(tau, wn[j].p)*wn[j].w;
	}
      out[i]=nans*t[i]+ weight(0);
    }
  return out;
}
column_vector Net::gradient(const std_vector& t, const std_vector& target)
 {
   int len = t.size();
   column_vector gd(len);
   for(int k=0; k<wcount; k++ ) gd=0.;
   std_vector ans = sim(t);
   wavelon* gdw =(wavelon *) (&gd(0) + 1);
   for (int j=0; j<t.size(); j++)
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
	 gdw[i].p += e*t[j]*wn[i].w*wt->dp(tau, wn[i].p);
       }
   }
   return gd;
 }

double Net::energy(const std_vector& t, const std_vector& target)
{
  double ener = 0.;
  std_vector ans = sim(t);
  for(int i=0; i<t.size(); i++)
    {
      double err = target[i] - ans[i];
      ener += err*err;
    }
  return ener/2;
}



