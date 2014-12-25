#include "core.hpp"
#include <iostream>

Net::Net(Wavelet* wavelet, int ncount, double xmin, double xmax, double ymin, double a0)
{
  int wcount = ncount*4+1;
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
column_vector Net::sim(const column_vector& t)
{
  column_vector ans(t.size());
  for (int i=0; i<t.size();i++)
    {
      double wans = 0.;
      for( int j=0; j<nc; j++)
	{
	  wans+=wt->h(t(j), wn[j].a, wn[j].b, wn[j].p)*wn[j].w;
	}
      ans(i)=wans*t(i)+ weight(0);
    }
  return ans;
}
column_vector Net::gradient(const column_vector& t)
 {
   ;
 }
Net::~Net()
{
  ;
}

