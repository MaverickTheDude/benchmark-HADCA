#pragma once

#include "input.h"
#include "../Eigen/Dense"

using namespace Eigen;

VectorXd Phi(const VectorXd& q, const _input_& input);
MatrixXd Jacobian(const VectorXd& q, const _input_& input);