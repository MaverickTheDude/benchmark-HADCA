#include "../include/utils.h"
#include "../include/input.h"
#include "../Eigen/Dense"

using namespace Eigen;

static VectorXd (*RHS_FUN_S)(const double& tau, const VectorXd& y, const VectorXd& uVec,
					const _solution_& solutionFwd, const _input_& input, _solutionAdj_& solution);
static VectorXd (*RHS_FUN__)(const double& tau, const VectorXd& y, const VectorXd& uVec,
					const _solution_& solutionFwd, const _input_& input);

_solutionAdj_ RK_AdjointSolver(const VectorXd& uVec, const _solution_& solutionFwd, const _input_& input, int formulation) {
	const double dt = input.dt;
	const int Nbodies = input.Nbodies;
	_solutionAdj_ solution(input);
	RHS_FUN__ = &RHS_ADJOINT;
	RHS_FUN_S = &RHS_ADJOINT;
	if (formulation == _solutionAdj_::GLOBAL) {
		solution.switchToGlobalFormulation(input);
		RHS_FUN__ = &RHS_ADJOINT_GLOBAL;
		RHS_FUN_S = &RHS_ADJOINT_GLOBAL;
	}

	const VectorXd& T = solutionFwd.T;
	VectorXd y_m1(2*Nbodies);
	y_m1 = boundaryConditions(solutionFwd, input, formulation);

	// backwards integration
	for (int i = 1; i < input.Nsamples; i++) {
        const double t = T(i-1);
		VectorXd k1 = RHS_FUN_S(t		  , y_m1              , uVec, solutionFwd, input, solution);
		VectorXd k2 = RHS_FUN__(t + dt/2.0, y_m1 + dt/2.0 * k1, uVec, solutionFwd, input);
		VectorXd k3 = RHS_FUN__(t + dt/2.0, y_m1 + dt/2.0 * k2, uVec, solutionFwd, input);
		VectorXd k4 = RHS_FUN__(t + dt    , y_m1 + dt     * k3, uVec, solutionFwd, input);

		VectorXd y = y_m1 +  dt/6 * (k1 + 2*k2 + 2*k3 + k4);
		y_m1 = y;
	}

	RHS_FUN_S(input.Tk, y_m1, uVec, solutionFwd, input, solution); // save last entry

	return solution;
}

_solution_ RK_solver(const _input_& input) {
	VectorXd U_ZERO(input.Nsamples);
	U_ZERO.setZero();
	return RK_solver(U_ZERO, input);
}

_solution_ RK_solver(const VectorXd& uVec, const _input_& input) {
	const double dt = input.dt;
	const double Tk = input.Tk;
	const int Nbodies = input.Nbodies;
    _solution_ solution(input); // note: Return Value Optimization (RVO) guarantees copy elison (details in solution.h)

	VectorXd T = VectorXd::LinSpaced(input.Nsamples, 0, Tk);
    solution.setT(T);
	VectorXd y_m1(2*Nbodies);
	y_m1.head(Nbodies) = input.pjoint0;
	y_m1.tail(Nbodies) = input.alpha0;

//	double t = omp_get_wtime(); //tic
	for (int i = 1; i < input.Nsamples; i++) {
        const double t = T(i-1);
		VectorXd k1 = RHS_HDCA(t         , y_m1              , uVec, input, solution); //RHS_HDCA appends the solution at i-1
        VectorXd k2 = RHS_HDCA(t + dt/2.0, y_m1 + dt/2.0 * k1, uVec, input);
		VectorXd k3 = RHS_HDCA(t + dt/2.0, y_m1 + dt/2.0 * k2, uVec, input);
		VectorXd k4 = RHS_HDCA(t + dt,     y_m1 + dt     * k3, uVec, input);

		VectorXd y = y_m1 +  dt/6 * (k1 + 2*k2 + 2*k3 + k4);
		y_m1 = y;

        if (input.logEnergy)
            logTotalEnergy(t, y, input);
	}
    RHS_HDCA(input.Tk, y_m1, uVec, input, solution); // save last entry
//	t =  omp_get_wtime() - t; //toc
//	std::cout << "calkowity czas: " << t << std::endl << std::endl;

	return solution;
};
