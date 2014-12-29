#ifndef NET_HPP
#define NET_HPP

#include "wavelet.hpp"
#include <vector>
#include <list>
#include <map>
#include <dlib/optimization/optimization_search_strategies_abstract.h>
#include <dlib/optimization.h>
#include <dlib/optimization/optimization_search_strategies.h>
#include <dlib/optimization/optimization_stop_strategies.h>
#include <dlib/optimization/optimization_line_search.h>
//#include "model.hpp"
class NetModel;
typedef dlib::matrix<double,0,1> column_vector;
typedef dlib::matrix<double> general_matrix;

typedef std::vector<double> std_vector;
typedef std::list<double> param_series;
typedef std::vector<param_series> train_set;
typedef std::map<std::string, train_set> train_res;

struct wavelon
{
  double a;
  double b;
  double p;
  double w;
};

struct ActivateFunc {
  enum Func {Morlet,SLOG,POLYWOG};
};
typedef ActivateFunc::Func ActFunc;

struct TrainStrategy {
  enum Func {Gradient, CG, BFGS};
};
typedef TrainStrategy::Func TrainStrat;

class Net 
{
public:
  double f (const column_vector& x) const;
  column_vector der (const column_vector& x)const;
  Net(int ncount, double xmin, double xmax, double ymin, double a0=10.,
      double w0=0.1, ActFunc f = ActivateFunc::Morlet);
  std_vector sim(const std_vector&  t);
  std_vector gradientVector(const std_vector& t, const std_vector& target);
  double energy(const std_vector& t, const std_vector& target);
  train_res train(const std_vector t, const std_vector target,
		  TrainStrat train_strategy, int epochs, double goal, int show);
  friend NetModel;
 
private:
  int nc;
  int wcount;
  column_vector weight;
  std_vector inp;
  std_vector targ;
  Wavelet* wt;
  wavelon* wn;
  column_vector gradient(const std_vector&  t, const std_vector&  target);
  template <typename search_strategy_type>
  train_res _train(const std_vector t, const std_vector target,
		   search_strategy_type train_strategy, int epochs, double goal, int show);
};

class NetF
{
  private:
   const  Net *n;

  public:
    NetF(Net *net )
    {
      n = net;
    };
    double operator() (
		       const column_vector& x
		       ) const { return n->f(x); }
  };

class NetDer
{
  private:
   const  Net *n;

  public:
    NetDer(Net *net)
    {
      n = net;
    };
    double operator() (
		       const column_vector& x
		       ) const { return n->der(x); }
  };

#endif
