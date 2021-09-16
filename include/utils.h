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
VectorXd absolutePositionToAbsoluteAlpha(const VectorXd& q);
Matrix3d SAB(const std::string &_sAB_, const int id, const double alphaAbs, const _input_ &input);
Matrix3d SAB(const std::string& _sAB_, const int id, const VectorXd& alphaAbs, const _input_& input);
Matrix3d dSAB(const std::string& _sAB_, const int id, const VectorXd& alphaAbs, const VectorXd& dAlphaAbs, const _input_& input);
Matrix3d dSAB(const std::string& _sAB_, const int id, const double& alphaAbs, const double& dAlphaAbs, const _input_& input);
Matrix3d d2SAB(const std::string& _sAB_, const int id, const double& alphaAbs, const double& dAlphaAbs, const double& d2AlphaAbs, const _input_& input);
Matrix3d dSABdAlpha(const Vector2d& translation, const double absoluteAlpha);
VectorXd joint2AbsAngles(const VectorXd& alpha);
Matrix3d massMatrix(const int id, const _input_& input);
Vector3d Q1_init(int id, const VectorXd& alphaAbs, const double& u, const _input_& input);
double calculateTotalEnergy(const double& t, const VectorXd& y, const _input_& input);
void logTotalEnergy(const double& t, const VectorXd& y, const _input_& input);
dataJoint interpolate(const double& t, const _solution_& solutionFwd, const _input_& input);
dataJoint interpolateLinear(const double& t, const _solution_& solutionFwd, const _input_& input);
std::pair<int, const bool> atTime(const double& t, const VectorXd& T, const _input_& input);
double interpolateControl(const double& t, const VectorXd& uVec, const _input_& input);

// solvers.cpp
_solution_ RK_solver(const _input_& input);
_solution_ RK_solver(const VectorXd& u, const _input_& input);
_solutionAdj_ RK_AdjointSolver(const VectorXd& uVec, const _solution_& solutionFwd, const _input_& input, int formulation);

// RHS.cpp
VectorXd RHS_HDCA(const double& t, const VectorXd& y, const VectorXd& uVec, const _input_& input);
VectorXd RHS_HDCA(const double& t, const VectorXd& y, const VectorXd& uVec, const _input_& input, _solution_& solution);
VectorXd RHS_ADJOINT(const double& t, const VectorXd& y, const VectorXd& uVec, const _solution_& solutionFwd, const _input_& input);
VectorXd RHS_ADJOINT(const double& t, const VectorXd& y, const VectorXd& uVec, const _solution_& solutionFwd, const _input_& input, _solutionAdj_& solution);
VectorXd RHS_ADJOINT_GLOBAL(const double& t, const VectorXd& y, const VectorXd& uVec, const _solution_& solutionFwd, const _input_& input);
VectorXd RHS_ADJOINT_GLOBAL(const double& t, const VectorXd& y, const VectorXd& uVec, const _solution_& solutionFwd, const _input_& input, _solutionAdj_& solution);
VectorXd boundaryConditions(const _solution_& solutionFwd, const _input_& input, int formulation);
