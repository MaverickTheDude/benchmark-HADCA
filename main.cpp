#include "include/input.h"
#include "include/utils.h"
#include "include/adjoint.h"
#include "include/task/Phi.h"
#include "Eigen/Dense"
#include <iostream>
#include <time.h>
#include <math.h>
#include <nlopt.hpp>
#include <iomanip>
#include <Eigen/StdVector> // zamiast <vector> (to chyba to samo z dodatkowym paddingiem dla pamieci?)
#include <omp.h>
// export OMP_NUM_THREADS=4

// .natvis (Eigen visualization on debug): broken
// https://stackoverflow.com/questions/58624914/using-natvis-file-to-visualise-c-objects-in-vs-code
// https://github.com/cdcseacave/Visual-Studio-Visualizers

using namespace Eigen;
using std::cout;
using std::endl;
static double costFunction(unsigned int n, const double *x, double *grad, void *my_func_data);

int main(int argc, char* argv[]) {
# ifdef _OPENMP
#pragma omp parallel
{
    if (omp_get_thread_num() == 0)
        cout << "OpenMP test executed in parallel on " << omp_get_num_threads() << " threads." << endl;
} //end pragma omp
# else
    cout << "Caution: Your sourcecode was compiled without switching OpenMP on" << endl;
# endif

    /* SETTINGS */
    const int Nbodies = 4;
    double inputSignal = 2.0;
    double w_hq = 1, w_hdq = 0.0; double w_hsig = 0.0001;
    double lb = -HUGE_VAL, ub = HUGE_VAL;

    /* initialize */
    _input_* input = new _input_(Nbodies);
    VectorXd u_zero = VectorXd::Constant(input->Nsamples, inputSignal);
    _solution_ solutionFwd = RK_solver(u_zero, *input);
    solutionFwd.show_xStatus(u_zero, *input);
    input->w_hq   = w_hq;
    input->w_hdq  = w_hdq;
    input->w_hsig = w_hsig;

#if true // check adjoint equations or initial setup
    solutionFwd.print(); // dla porownania
	{
		_solutionAdj_ solution = RK_AdjointSolver(u_zero, solutionFwd, *input, _solutionAdj_::HDCA);
		solution.print();
	}
	_solutionAdj_ solutionG = RK_AdjointSolver(u_zero, solutionFwd, *input, _solutionAdj_::GLOBAL);
	solutionG.print();
#endif

#if false // optimize
    nlopt::opt opt(nlopt::LD_MMA, input->Nsamples); // LD_SLSQP  LD_MMA  LD_CCSAQ  AUGLAG  G_MLSL_LDS (useless: GN_DIRECT_L, GN_ISRES)
    opt.set_xtol_rel(1e-4);
    opt.set_maxeval(100);
    
    /* opt. globalna + lokalna G_MLSL_LDS */
/*     nlopt::opt opt_aux(nlopt::LD_MMA, input->Nsamples);
    opt_aux.set_xtol_rel(1e-4);
    opt_aux.set_maxeval(100);
    opt.set_local_optimizer(opt_aux); */

    cout << "---optimization parameters:---"
         << "\nstop value  = " << opt.get_stopval()  // Stop when an objective value of at least stopval is found
         << "\nrel. tol.   = " << opt.get_ftol_rel() // Relative tolerance on function value.
         << "\nabs. tol.   = " << opt.get_ftol_abs() // Absolute tolerance on function value.
         << "\nx tol. rel. = " << opt.get_xtol_rel() // Relative tolerance on optimization parameters.
         << std::endl;

    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);
    opt.set_min_objective(costFunction, input);
    std::vector<double> signal(input->Nsamples);
    for (unsigned int i = 0; i < signal.size(); i++) signal[i] = u_zero(i);
    double minf;
    VectorXd u(input->Nsamples);

    try{
        /* nlopt::result result = */ opt.optimize(signal, minf);
        std::cout << "found minimum!\nf = " << std::setprecision(10) << minf << std::endl;
        for (int i = 0; i < input->Nsamples; i++)
            u(i) = signal[i];
        _solution_ solutionFwd = RK_solver(u, *input);
        solutionFwd.print(u);

        /* post processing */
        _solution_ solutionPP = RK_solver(u, *input);
        solutionPP.show_xStatus(u, *input);
    }
    catch(std::exception &e) {
        std::cout << "nlopt failed: " << e.what() << std::endl;
    }
#endif

    delete input;
	cout << "done\n";
	return 0;
}


static double costFunction(unsigned int n, const double *x, double *grad, void *my_func_data)
{
    _input_* input = static_cast<_input_*>(my_func_data);
    Map<const VectorXd> u(x, input->Nsamples);
	_solution_ solutionFwd = RK_solver(u, *input);

    const double& x_tf = solutionFwd.alpha(0, input->Nsamples-1);
    double S_tf  = input->w_Sq * x_tf * x_tf;
    double alpha = input->w_hq, beta = input->w_hdq; double gama = input->w_hsig;

    if (grad) {
    	_solutionAdj_ solution = RK_AdjointSolver(u, solutionFwd, *input, _solutionAdj_::HDCA);
        for (int i = 0; i < input->Nsamples; i++)
            grad[i] =( 2*gama*u(i) - solution.c(0,i) ) * input->dt;
        cout << "grad evaluation " << endl;
    }
    
    /* (1) manual convertion to array (2) automtic conv. to Vector via operator= */
    VectorXd expr = alpha * solutionFwd.alpha.row(0).array().square()
                  + beta  * solutionFwd.dalpha.row(0).array().square();
                  + gama  * u.array().square();
    double f = trapz(expr, *input) + S_tf;

    static int cnt = 1;
    cout << "fun evaluation #" << cnt++ << ", value: " << f << endl;
    return f;
}
