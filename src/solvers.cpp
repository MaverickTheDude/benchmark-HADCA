#include "../include/utils.h"
#include "../include/input.h"
#include "../Eigen/Dense"

using namespace Eigen;

VectorXd RK_AdjointSolver(const _input_& input, _solution_& solution) {
    // to do
}

_solution_ RK_solver(const _input_& input) {
	const double dt = input.dt;
	const double Tk = input.Tk;
	const int Nbodies = input.Nbodies;
    _solution_ solution(input); // TODO: przy wyjsciu z funkcji solution zostanie przekopiowane. Zrobic lepszy menege pamieci

	VectorXd T = VectorXd::LinSpaced(input.Nsamples, 0, Tk);
    solution.setT(T);
	VectorXd y_m1(2*Nbodies);
	y_m1.head(Nbodies) = input.pjoint0;
	y_m1.tail(Nbodies) = input.alpha0;

//	double t = omp_get_wtime(); //tic
	for (int i = 1; i < input.Nsamples; i++) {
        const double t = T(i-1);
		VectorXd k1 = RHS_HDCA(t         , y_m1,               input, solution); //RHS_HDCA appends the solution at i-1
        VectorXd k2 = RHS_HDCA(t + dt/2.0, y_m1 + dt/2.0 * k1, input);
		VectorXd k3 = RHS_HDCA(t + dt/2.0, y_m1 + dt/2.0 * k2, input);
		VectorXd k4 = RHS_HDCA(t + dt,     y_m1 + dt     * k3, input);

		VectorXd y = y_m1 +  dt/6 * (k1 + 2*k2 + 2*k3 + k4);
		y_m1 = y;

        if (input.logEnergy)
            logTotalEnergy(t, y, input);
	}
    RHS_HDCA(input.Tk, y_m1, input, solution); // save last entry
//	t =  omp_get_wtime() - t; //toc
//	std::cout << "calkowity czas: " << t << std::endl << std::endl;

	return solution;
};