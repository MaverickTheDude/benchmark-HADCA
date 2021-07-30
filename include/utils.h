#pragma once

#include "../Eigen/Dense"
#include "input.h"

using namespace Eigen;

Matrix2d Rot(double fi);
const Matrix2d Om = Rot(M_PI_2);
MatrixXd jacobianReal(VectorXd (*fun)(const VectorXd&, const _input_&), VectorXd alfa0, _input_ input);
VectorXd jointToAbsolutePosition(const VectorXd& alpha, const _input_& input);
VectorXd jointToAbsoluteVelocity(const VectorXd &alpha, const VectorXd &dalpha, const _input_ &input);
Matrix3d SAB(const std::string& _sAB_, const int id, const VectorXd& alphaAbs, const _input_& input);
VectorXd joint2AbsAngles(const VectorXd& alpha);
Matrix3d massMatrix(const int id, const _input_ input);
Vector3d Q1_init(int id, const VectorXd& alphaAbs, const _input_& input);