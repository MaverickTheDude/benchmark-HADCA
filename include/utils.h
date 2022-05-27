#pragma once

#include "../Eigen/Dense"
#include "../include/assembly.h"
#include "../include/input.h"
#include "../include/solution.h"

using namespace Eigen;
using aaA = aligned_allocator<Assembly>;

struct taskTimes {double wt=0.0, wt_adj=0.0, t=0.0, t_adj=0.0;
    void update(const taskTimes&); void divide(const int mean);};

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
Vector3d Q1_init(int id, const VectorXd &alphaAbs, const VectorXd &dAlphaJoint, const double& u, const _input_ &input);
VectorXd Q1_global(const VectorXd &q, const VectorXd &dq, const double& u, const _input_ &input);
void logTotalEnergy(const double& t, const VectorXd& y, const VectorXd& dy, const VectorXd& uVec, const _input_& input);
dataJoint interpolate(const double& t, const _solution_& solutionFwd, const _input_& input);
dataJoint interpolateLinear(const double& t, const _solution_& solutionFwd, const _input_& input);
std::pair<int, const bool> atTime(const double& t, const VectorXd& T, const _input_& input);
double interpolateControl(const double& t, const VectorXd& uVec, const _input_& input);
double trapz(const VectorXd& x, const _input_& input);
double trapz(const int ind, const VectorXd& x, const _input_& input);
void print_checkGrad(const _solution_& solFwd, const _solutionAdj_& solAdj, const VectorXd& uVec, const _input_& input);
void write_vector_to_file(const std::vector<double>& myVector, std::string filename);
std::vector<double> read_vector_from_file(std::string filename);


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