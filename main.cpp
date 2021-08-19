#include "include/input.h"
#include "include/utils.h"
#include "include/assembly.h"
#include "include/solution.h"
#include "Eigen/Dense"
#include <time.h>

#include <iostream>
using namespace Eigen;
using std::cout;
using std::endl;


int main(int argc, char* argv[]) {
    const int Nbodies = 4;
    _input_ input = _input_(Nbodies);

    _solution_ sol = RK_solver(input);

	sol.print();

	// MatrixXd x = jointToAbsoluteCoords(sol.alpha.col(1), sol.dalpha.col(1), sol.d2alpha.col(1), input);
	VectorXd q = jointToAbsolutePosition(sol.alpha.col(1), input);
	VectorXd v = jointToAbsoluteVelocity(sol.alpha.col(1), sol.dalpha.col(1), input);

	// cout << q << endl;
	// cout << v << endl;

    cout << "done" << endl;
    return 0;
}

	// clock_t start, end;
	// double time;
	// start = clock();
	// for (int i = 0; i < 9999; i++)
	// 	atTime(0.01, T);
	// end = clock();
	// time = double(end - start) / double(CLOCKS_PER_SEC);
	// cout << time << endl;