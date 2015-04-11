#include "net.hpp"
#include "train.hpp"
#include <time.h>
using namespace dlib;
using namespace std;
double Net::f (const column_vector& x)
{
  return _energy(t, inp, targ, x);
}

column_vector Net::der (const column_vector& x)
{
  return _gradient(t, inp, targ, x, varc, varp);
}

train_res Net::train(const std_vector& t, const std_vector&  inp, const std_vector& target,
		     TrainStrat train_strategy,
		     int epochs, double goal, int show, bool varc, bool varp, Caller cb)
{
  double max = std::max_element(t.begin(), t.end())[0];
  double min = std::min_element(t.begin(),t.end())[0];
  double delta = (max - min)/nc;
  max += 2*delta;
  min -= 2*delta;
  delta =(max-min)/nc;
  wn =(wavelon *) (&weight(0) + 1 + fc);
  for (int i=0; i<nc; i++)
    {
      wn[i].b = min+i*delta+delta/2.;
    }

  this->varc = varc;
  this->varp = varp;
  switch (train_strategy)
    {
    case TrainStrategy::CG:
      return _train(t, inp, target, cg_search_strategy(),
		    epochs, goal, show, cb);
      break;
    case TrainStrategy::Gradient:
      return _train(t, inp, target, gr_search_strategy(),
		    epochs, goal, show, cb);
      break;
    case TrainStrategy::BFGS:
      return _train(t, inp, target, bfgs_search_strategy(),
		    epochs, goal, show, cb);
      break;

    default:
      return _train(t, inp, target, cg_search_strategy(),
		    epochs, goal, show, cb);
      break;
    }
}
template <typename search_strategy_type>
train_res Net::_train(const std_vector& t, const std_vector&  input, const std_vector& target,
		      search_strategy_type search_strategy,
		      int epochs, double goal, int show, Caller& cb)
{
  train_res tr_res;
  tr_res[std::string("a")] = train_set(nc);
  tr_res[std::string("b")] = train_set(nc);
  tr_res[std::string("c")] = train_set(1);
  tr_res[std::string("e")] = train_set(1);
  tr_res[std::string("p")] = train_set(nc);
  tr_res[std::string("w")] = train_set(nc);
  tr_res[std::string("f")] = train_set(fc);
  tr_res[std::string("t")] = train_set(1);
  this->t = t;
  inp = input;
  targ = target;
  NetF func(this);
  NetDer deriv(this);
  column_vector g,s;
  double f_value = f(weight);
  g = der(weight);
  if (!is_finite(f_value))
    throw error("The objective function generated non-finite outputs");
  if (!is_finite(g))
    throw error("The objective function generated non-finite outputs");
  clock_t begin_time = clock();
  mem(tr_res, f_value, 0.);
  for (int iter = 0; (epochs==0 || iter<epochs) && f_value > goal; iter++)
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
      if (iter % show == 0)
	{
	  mem(tr_res, f_value, float( clock () - begin_time ) /  CLOCKS_PER_SEC);
	  int prg = iter*100/epochs;
	  if (cb.Handler != NULL)
		     cb.triggerEvent(prg);
	}
    }
  return  tr_res;
}

void Net::mem(train_res& tr_res, double f_value, double t)
{
	  for (int i=0; i<nc; i++)
	    {
	      tr_res["a"][i].push_back(wn[i].a);
	      tr_res["b"][i].push_back(wn[i].b);
	      tr_res["w"][i].push_back(wn[i].w);
	      tr_res["p"][i].push_back(wn[i].p);
	    }
	  tr_res["e"][0].push_back(f_value);
	  tr_res["c"][0].push_back(weight(0));
	  tr_res["t"][0].push_back(t);
	  for (int i=0; i<fc; i++)
	    {
	      tr_res["f"][i].push_back(weight(i+1));
	    }
}

