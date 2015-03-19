#ifndef NET_HPP
#define NET_HPP

#include "wavelet.hpp"
#include <vector>
#include <list>
#include <map>
#include <string>
#include <dlib/optimization/optimization_search_strategies_abstract.h>
#include <dlib/optimization.h>
#include <dlib/optimization/optimization_search_strategies.h>
#include <dlib/optimization/optimization_stop_strategies.h>
#include <dlib/optimization/optimization_line_search.h>
#include <omp.h>
#include "cregister.cpp"
//#include "model.hpp"
class NetModel;
class Callback;
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
  double f (const column_vector& x);
  column_vector der (const column_vector& x);
  Net(int ncount, double tmin, double tmax, double ymin, double a0=10., double a1=10.,
      double w0=0.1, double w1=0.1, double p0=1.0, double p1=1.0, ActFunc f = ActivateFunc::Morlet, int numberOfThreads=2);
  ~Net();
  std_vector sim(const std_vector&  t, const std_vector&  inp);
  std_vector gradient(const std_vector& t, const std_vector& target, const std_vector&  inp);
  double energy(const std_vector& t, const std_vector&  inp, const std_vector& target);
  train_res train(Caller& cb, const std_vector& t,const std_vector&  inp, const std_vector& target,
		     TrainStrat train_strategy=TrainStrategy::CG,
		  int epochs=30, double goal = 0.3, int show=1, bool varc=true, bool varp=true);
  friend NetModel;
 
private:
  int nc;
  int wcount;
  column_vector weight;
  std_vector t;
  std_vector inp;
  std_vector targ;
  Wavelet* wt;
  wavelon* wn;
  bool varc;
  bool varp;
  column_vector _gradient(const std_vector&  t, const std_vector&  inp, const std_vector&  target, const column_vector& weight, bool varc, bool varp);
  double _energy(const std_vector& t, const std_vector&  inp,  const std_vector& target, const column_vector& weight);
  std_vector _sim(const std_vector&  t, const std_vector&  inp, const column_vector& weight);
  template <typename search_strategy_type>
  train_res _train(const std_vector& t, const std_vector&  inp, const std_vector& target,
		   search_strategy_type train_strategy, int epochs, double goal, int show,
		   Caller& cb);
  void mem(train_res& tr_res, double f_value);
};

class NetF
{
  private:
   Net *n;

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
   Net *n;

  public:
    NetDer(Net *net)
    {
      n = net;
    };
    column_vector operator() (
		       const column_vector& x
		       ) const { return n->der(x); }
  };
#endif
