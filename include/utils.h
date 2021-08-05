#pragma once

#include "../Eigen/Dense"
#include "input.h"

using namespace Eigen;

Matrix2d Rot(double fi);
MatrixXd jacobianReal(VectorXd (*fun)(const VectorXd&, const _input_&), VectorXd alfa0, const _input_& input);
VectorXd jointToAbsolutePosition(const VectorXd& alpha, const _input_& input);
VectorXd jointToAbsoluteVelocity(const VectorXd& alpha, const VectorXd& dalpha, const _input_& input);
Matrix3d SAB(const std::string& _sAB_, const int id, const VectorXd& alphaAbs, const _input_& input);
Matrix3d dSAB(const std::string& _sAB_, const int id, const VectorXd& alphaAbs, const VectorXd& dAlphaAbs, const _input_& input);
VectorXd joint2AbsAngles(const VectorXd& alpha);
Matrix3d massMatrix(const int id, const _input_& input);
Vector3d Q1_init(int id, const VectorXd& alphaAbs, const _input_& input);
MatrixXd RK_solver(const _input_& input);
VectorXd RHS_HDCA(const double& t, const VectorXd& y, const _input_& input);
double calculateTotalEnergy(const double& t, const VectorXd &y, const _input_ &input);