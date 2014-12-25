#include "core.hpp"
#include <iostream>

Net::Net(Wavelet* wavelet, int ncount, double xmin, double xmax, double ymin, double a0)
{
  
  int wcount = ncount*4+1;
  wt = wavelet;
  nc = ncount;
  weight = new double[wcount];
  weight[0] = ymin;
  double delta = (xmax - xmin)/wcount;
  wn =(wavelon *) (weight + 1); 
  for (int i=0; i<nc; i++)
    {
      wn[i].a = a0;
      wn[i].b = i*(delta+1);
      wn[i].p = 1.;
      wn[i].w =0.05;
    }  
}
vector<double> Net::sim(vector<double> t)
{
  vector<double> ans(t.size());
  for (int i=0; i<t.size();i++)
    {
      double wans = 0.;
      for( int j=0; j<nc; j++)
	{
	  wans+=wt->h(t[j], wn[j].a, wn[j].b, wn[j].p)*wn[j].w;
	}
      ans[i]=wans*t[i]+weight[0];
      //std::cout<<ans[i]<<std::endl;
    }
  return ans;
}
 vector <double> gradient(vector <double> t)
 {
   
 }
Net::~Net()
{
  delete [] weight;
}

