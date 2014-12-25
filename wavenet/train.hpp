#include "net.hpp"
#include <dlib/optimization.h>
#include <list>
#include <dlib/optimization/optimization_search_strategies.h>
#include <dlib/optimization/optimization_stop_strategies.h>
#include <dlib/optimization/optimization_line_search.h>
using namespace std;
using namespace dlib;
typedef list<std_vector> list_array;
class Trainer
{
public:
  Trainer(const Net &net);
  double f (const column_vector& m);
  const column_vector der (const column_vector& m);

  list_array learn(search_strategy_type search_strategy,
		       stop_strategy_type stop_strategy,
		       int epochs, double goal, int show);
private:
  Net* net;
  column_vector x;
};
