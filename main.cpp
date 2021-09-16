#include "include/input.h"
#include "include/utils.h"
#include "include/adjoint.h"
#include "include/task/Phi.h"
#include "Eigen/Dense"
#include <time.h>

#include <iostream>
using namespace Eigen;
using std::cout;
using std::endl;


int main(int argc, char* argv[]) {
    const int Nbodies = 4;
    _input_ input = _input_(Nbodies);
	VectorXd u0(input.Nsamples);
	u0.setZero();

	_solution_ solutionFwd = RK_solver(u0, input);

	// solutionFwd.print();

	_solutionAdj_ solution = RK_AdjointSolver(u0, solutionFwd,  input, _solutionAdj_::HDCA);
	solution.print();

	cout << "done\n";
	return 0;
}