#include "train.hpp"

double Trainer::f (const column_vector& m)
{
  return 0;
}

const column_vector Trainer::der (const column_vector& m)
{
  return 0
}

list_array Trainer::learn (
        search_strategy_type search_strategy,
        stop_strategy_type stop_strategy,
	int epoch,
        double goal,
	int show
    )
    {

        column_vector g, s;

        double f_value = f(x);
        g = der(x);

        if (!is_finite(f_value))
            throw error("The objective function generated non-finite outputs");
        if (!is_finite(g))
            throw error("The objective function generated non-finite outputs");

        while(stop_strategy.should_continue_search(x, f_value, g) && f_value > min_f)
        {
            s = search_strategy.get_next_direction(x, f_value, g);

            double alpha = line_search(
                        make_line_search_function(f,x,s, f_value),
                        f_value,
                        make_line_search_function(der,x,s, g),
                        dot(g,s), // compute initial gradient for the line search
                        search_strategy.get_wolfe_rho(), search_strategy.get_wolfe_sigma(), min_f,
                        search_strategy.get_max_line_search_iterations());

            // Take the search step indicated by the above line search
            x += alpha*s;

            if (!is_finite(f_value))
                throw error("The objective function generated non-finite outputs");
            if (!is_finite(g))
                throw error("The objective function generated non-finite outputs");
        }

        return f_value;
    }
