#include "include/input.h"
#include "include/utils.h"
#include "include/adjoint.h"
#include "Eigen/Dense"
#include <time.h>

#include <iostream>
using namespace Eigen;
using std::cout;
using std::endl;


int main(int argc, char* argv[]) {
    const int Nbodies = 4;
    _input_ input = _input_(Nbodies);

	VectorXd q = jointToAbsolutePosition(input.alpha0, input);
	VectorXd dq = jointToAbsoluteVelocity(input.alpha0, input.dalpha0, input);

	std::cout<<Adjoint::RHS(input, q, dq, VectorXd::Ones(input.Nconstr), (VectorXd(1) << 1).finished(),
		VectorXd::Ones(3 * input.Nbodies), VectorXd::Ones(3 * input.Nbodies))<<std::endl;
}