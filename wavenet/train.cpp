#include "net.hpp"

using namespace std;
using namespace dlib;

double Net::f (const column_vector& x)
{
  column_vector temp = weight;
  weight = x;
  double energ = energy(inp, targ);
  weight = temp;
  return energ;
}

column_vector Net::der (const column_vector& x)
{
  column_vector temp = weight;
  weight = x;
  column_vector d = gradient(inp, targ);
  weight = temp;
  return d;
}

train_res Net::train(const std_vector t, const std_vector target,
		     TrainStrat train_strategy=TrainStrategy::CG,
		     int epochs=30, double goal = 0.3, int show=1)
{
     switch (train_strategy)
     {
       case TrainStrategy::CG:
      return _train(t, target, cg_search_strategy(),
		    epochs, goal, show);
       break;
     default:
       return _train(t, target, cg_search_strategy(),
		    epochs, goal, show);
       break;
     }
}
template <typename search_strategy_type>
train_res Net::_train(const std_vector input, const std_vector target,
		      search_strategy_type search_strategy,
		     int epochs=30, double goal = 0.3, int show=1)
{
  NetF func(this);
  NetDer deriv(this);
   inp = input;
   targ = target;
   column_vector g,s;
  train_res res();
  double f_value = f(weight);
  g = der(weight);
  int iter = 1;
  if (!is_finite(f_value))
    throw error("The objective function generated non-finite outputs");
  if (!is_finite(g))
    throw error("The objective function generated non-finite outputs");

  while(iter<=epochs && f_value > goal)
    {
      s = search_strategy.get_next_direction(weight, f_value, g);

      double alpha = line_search(
				 make_line_search_function(func, weight, s, f_value),
				 f_value,
				 make_line_search_function(deriv ,weight,s,g),
				 dot(g,s), // compute initial gradient for the line search
				 search_strategy.get_wolfe_rho(), search_strategy.get_wolfe_sigma(), goal,
				 search_strategy.get_max_line_search_iterations());

      // Take the search step indicated by the above line search
      weight += alpha*s;

      if (!is_finite(f_value))
	throw error("The objective function generated non-finite outputs");
      if (!is_finite(g))
	throw error("The objective function generated non-finite outputs");
    }

  return  res;
}



