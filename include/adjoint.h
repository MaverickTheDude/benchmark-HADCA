#pragma once

#include "../Eigen/Dense"
#include "../include/task/h.h"
#include "../include/task/F.h"
#include "../include/task/M.h"
#include "../include/task/Phi.h"
#include "../include/input.h"
#include "../include/utils.h"

using namespace Eigen;

class Adjoint
{
friend VectorXd boundaryConditions(const _solution_& solutionFwd, const _input_& input, int formulation);
friend VectorXd RHS_ADJOINT_GLOBAL(const double& tau, const VectorXd& y, const VectorXd& uVec,
                     const _solution_& solutionFwd, const _input_& input, _solutionAdj_& solution);
friend _solutionGlobal_ RK_GlobalSolver_odeInt(const VectorXd& uVec, const _input_& input);
friend class RHS_GLOBAL_ODE;
friend class odeint_globalObserver;
public:
    Adjoint(const _input_& i);
    Adjoint() = delete;
    ~Adjoint();
    Adjoint(const Adjoint&) = delete;
    Adjoint& operator=(const Adjoint&) = delete;

    static VectorXd RHS(const _input_ &i, const VectorXd& q, const VectorXd& dq, const VectorXd& d2q, const VectorXd& lambda,
        const VectorXd& u, const VectorXd& eta, const VectorXd& ksi);
    VectorXd RHS(const VectorXd& q, const VectorXd& dq, const VectorXd& d2q, const VectorXd& lambda,
        const VectorXd& u, const VectorXd& eta, const VectorXd& ksi) const;

    static VectorXd RHS(const int bodyNumber, const _input_ &i, const VectorXd& q, const VectorXd& dq, const VectorXd& d2q,
        const VectorXd& lambda, const VectorXd& u, const VectorXd& eta, const VectorXd& ksi);
    VectorXd RHS(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& d2q, const VectorXd& lambda,
        const VectorXd& u, const VectorXd& eta, const VectorXd& ksi) const;

    static VectorXd RHS3d(const int bodyNumber, const _input_ &i, const VectorXd& q, const VectorXd& dq, const VectorXd& d2q,
        const VectorXd& lambda, const VectorXd& u, const VectorXd& eta, const VectorXd& ksi);

    VectorXd RHS3d(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& d2q, const VectorXd& lambda,
        const VectorXd& u, const VectorXd& eta, const VectorXd& ksi) const;

private:
    const _input_& input;
    const task::h* h;
    const task::F* F;
    const task::M* M;
    const task::Phi* Phi;

    VectorXd __RHS(const _input_ &i, const VectorXd& q, const VectorXd& dq, const VectorXd& d2q, const VectorXd& lambda,
        const VectorXd& u, const VectorXd& eta, const VectorXd& ksi) const;
    VectorXd __RHS(const int bodyNumber, const _input_ &i, const VectorXd& q, const VectorXd& dq, const VectorXd& d2q,
        const VectorXd& lambda, const VectorXd& u, const VectorXd& eta, const VectorXd& ksi) const;
    VectorXd __RHS3d(const int bodyNumber, const _input_ &i, const VectorXd& q, const VectorXd& dq, const VectorXd& d2q,
        const VectorXd& lambda, const VectorXd& u, const VectorXd& eta, const VectorXd& ksi) const;
};