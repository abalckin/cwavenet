#include "../dlib/matrix.h"
class Net {
 public:
  Net(int ncount);
  ~Net();
 private:
  double* weight;

};

Net::Net(int ncount, )
{
  weight = new double[ncount*4+1];
}

Net::~Net()
{
  delete [] weight;
}

struct wavelon
{
  double a;
  double b;
  double p;
  double w;
};

