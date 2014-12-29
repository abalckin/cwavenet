#include "../wavenet/net.hpp"
#include <iostream>
#include <vector>
int main()
{
  int c = 10;
  Net n = Net(10, 0., 10., 0.);
  std::vector<double> t(c);
  std::vector<double> tar(c);

  for (int i =0; i<c; i++)
    {
      t[i]=i/10.;//= {1, 2, 3, 4, 5};
  //std::vector<double> t (arr, arr + sizeof(arr) / sizeof(arr[0]) );
      tar[i]=t[i]*t[i];
    }
  double e =n.train(t, tar, TrainStrategy::CG, 3000, 0.0000000000000000003);
  std::cout << e <<std::endl;
}
