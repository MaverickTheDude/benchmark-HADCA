#pragma once

#include "../Eigen/Dense"
#include "input.h"
#include "../include/solution.h"

using namespace Eigen;

// utils.cpp
Matrix2d Rot(double fi);
MatrixXd jacobianReal(VectorXd (*fun)(const VectorXd&, const _input_&), VectorXd alfa0, const _input_& input);
VectorXd jointToAbsolutePosition(const VectorXd& alpha, const _input_& input);
VectorXd jointToAbsoluteVelocity(const VectorXd& alpha, const VectorXd& dalpha, const _input_& input);
MatrixXd jointToAbsoluteCoords(const VectorXd &alpha, const VectorXd &dalpha, const VectorXd &d2alpha, const _input_ &input);
Matrix3d SAB(const std::string& _sAB_, const int id, const VectorXd& alphaAbs, const _input_& input);
Matrix3d dSAB(const std::string& _sAB_, const int id, const VectorXd& alphaAbs, const VectorXd& dAlphaAbs, const _input_& input);
VectorXd joint2AbsAngles(const VectorXd& alpha);
Matrix3d massMatrix(const int id, const _input_& input);
Vector3d Q1_init(int id, const VectorXd& alphaAbs, const _input_& input);
double calculateTotalEnergy(const double& t, const VectorXd& y, const _input_& input);
void logTotalEnergy(const double& t, const VectorXd& y, const _input_& input);

// solvers.cpp
_solution_ RK_solver(const _input_& input);
VectorXd RK_AdjointSolver(const _input_& input, _solution_& solution);

// RHS.cpp
VectorXd RHS_HDCA(const double& t, const VectorXd& y, const _input_& input);
VectorXd RHS_HDCA(const double& t, const VectorXd& y, const _input_& input, _solution_& solution);