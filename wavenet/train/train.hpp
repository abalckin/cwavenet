#include <dlib/matrix.h>
#include "../net.hpp"
class Trainer
{
public:
  Trainer(const Net &net);
  matrix<double> learn(Train train, int epochs, int goal, int show);
private:
  Net* net;
};
