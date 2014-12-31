#include "net.hpp"

using namespace dlib;

double Net::f (const column_vector& x)
{
  return _energy(inp, targ, x);
}

column_vector Net::der (const column_vector& x)
{
  return _gradient(inp, targ, x);
}

double Net::train(const std_vector& t, const std_vector& target,
		     TrainStrat train_strategy,
		  int epochs, double goal, int show)
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
double Net::_train(const std_vector& input, const std_vector& target,
		      search_strategy_type search_strategy,
		      int epochs, double goal, int show)
{
  inp = input;
  targ = target;
  NetF func(this);
  NetDer deriv(this);
  column_vector g,s;
  double f_value = f(weight);
  g = der(weight);
  train_res tr_res();
  if (!is_finite(f_value))
    throw error("The objective function generated non-finite outputs");
  if (!is_finite(g))
    throw error("The objective function generated non-finite outputs");

  for (int iter = 1; (epochs==0 || iter<=epochs) && f_value > goal; iter++)
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
      f_value = f(weight);
      g = der(weight);
      if (!is_finite(f_value))
	throw error("The objective function generated non-finite outputs");
      if (!is_finite(g))
	throw error("The objective function generated non-finite outputs");
    }
  return  f_value;
}



