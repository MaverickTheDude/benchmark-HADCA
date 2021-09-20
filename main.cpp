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

using namespace Eigen;
using std::cout;
using std::endl;
static double costFunction(unsigned int n, const double *x, double *grad, void *my_func_data);

int main(int argc, char* argv[]) {
    const int Nbodies = 4;
    _input_* input = new _input_(Nbodies);

#if false // check adjoint equations or initial setup
    VectorXd u_zero = VectorXd::Zero(input->Nsamples);
    _solution_ solutionFwd = RK_solver(u_zero, *input);
    // solutionFwd.print(); // dla porownania
	{
		_solutionAdj_ solution = RK_AdjointSolver(u_zero, solutionFwd, *input, _solutionAdj_::HDCA);
		solution.print();
	}
	_solutionAdj_ solutionG = RK_AdjointSolver(u_zero, solutionFwd, *input, _solutionAdj_::GLOBAL);
	solutionG.print();
#endif

#if true // optimize
    nlopt::opt opt(nlopt::LD_SLSQP, input->Nsamples);
    opt.set_min_objective(costFunction, input);
    opt.set_xtol_rel(1e-4);
    std::vector<double> signal(input->Nsamples);
    for (auto& x : signal) x = 0.0;
    double minf;

    try{
        /* nlopt::result result = */ opt.optimize(signal, minf);
        std::cout << "found minimum!\nf = " << std::setprecision(10) << minf << std::endl;
        VectorXd u(input->Nsamples);
        for (int i = 0; i < input->Nsamples; i++)
            u(i) = signal[i];
        _solution_ solutionFwd = RK_solver(u, *input);
        solutionFwd.print(u);
    }
    catch(std::exception &e) {
        std::cout << "nlopt failed: " << e.what() << std::endl;
    }
#endif

	cout << "done\n";
	return 0;
}


static double costFunction(unsigned int n, const double *x, double *grad, void *my_func_data)
{
    _input_* input = static_cast<_input_*>(my_func_data);
	VectorXd u(input->Nsamples);
    for (int i = 0; i < input->Nsamples; i++) { // const double *x => std::vector x => cos sprytniejszego ? https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html
        u(i) = x[i]; }
	_solution_ solutionFwd = RK_solver(u, *input);

    if (grad) {
    	_solutionAdj_ solution = RK_AdjointSolver(u, solutionFwd, *input, _solutionAdj_::HDCA);
        for (int i = 0; i < input->Nsamples; i++)
            grad[i] = -1.0*input->dt * solution.c(0,i);
        cout << "grad evaluation " << endl;
    }
    
    const double& x_tf = solutionFwd.alpha(0, input->Nsamples-1);
    double S_tf = x_tf * x_tf;
    /* (1) manual convertion to array (2) automtic conv. to Vector via operator= */
    VectorXd expr = solutionFwd.alpha.row(0).array().square();
    // double f = trapz(expr, *input) + S_tf; // = int(x0*x0) dt + S_tf (S_q bookmarked)
    double f = trapz(expr, *input); // = int(x0*x0) dt

    static int cnt = 1;
    cout << "fun evaluation #" << cnt++ << ", value: " << f << endl;
    return f;
}
