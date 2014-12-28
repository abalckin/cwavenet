#include "../wavenet/net.hpp"
#include <iostream>
#include <vector>
int main()
{
  Wavelet* wt = new Morlet();
  int c = 10;
  Net n = Net(2, 0., 10., 0.);
  std::vector<double> t(c);
  std::vector<double> tar(c);

  for (int i =0; i<c; i++)
    {
      t[i]=i/10.;//= {1, 2, 3, 4, 5};
  //std::vector<double> t (arr, arr + sizeof(arr) / sizeof(arr[0]) );
      tar[i]=i;
    }
  column_vector grad =  n.gradient(t, tar);
   std::cout<<grad;
}
