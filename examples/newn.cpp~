#include "../wavenet/core.hpp"
#include <iostream>
#include <vector>
int main()
{
  Wavelet* wt = new Morlet();
  int c = 1000000;
  Net n = Net(wt, 10, 0., 100., 1000., 16);
  std::vector<double> t(c);
  for (int i =0; i<c; i++)
    t[i]=i/10;//= {1, 2, 3, 4, 5};
  //std::vector<double> t (arr, arr + sizeof(arr) / sizeof(arr[0]) );
  
  n.sim(t);
  //  std::cout<<n.sim(t)[0];
}
