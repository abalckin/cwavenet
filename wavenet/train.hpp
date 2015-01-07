//#include "net.hpp"
#include <dlib/optimization.h>
//#include <list>
//#include <dlib/optimization/optimization_search_strategies.h>
//#include <dlib/optimization/optimization_stop_strategies.h>
//#include <dlib/optimization/optimization_line_search.h>
//using namespace std;
using namespace dlib;

    class gr_search_strategy
    {
    public:
        gr_search_strategy(){}

        double get_wolfe_rho (
        ) const { return 0.001; }

        double get_wolfe_sigma (
        ) const { return 0.01; }

        unsigned long get_max_line_search_iterations (
        ) const { return 100; }

        template <typename T>
        const matrix<double,0,1>& get_next_direction (
            const T& ,
            const double ,
            const T& funct_derivative
        )
        {
	  prev_derivative = funct_derivative;
                prev_direction = -funct_derivative;
		return prev_direction;
        }

    private:
        matrix<double,0,1> prev_derivative;
        matrix<double,0,1> prev_direction;
    };
