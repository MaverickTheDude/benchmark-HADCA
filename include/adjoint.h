#pragma once

#include "../Eigen/Dense"
#include "../include/task/h.h"
#include "../include/task/F.h"
#include "../include/task/M.h"
#include "../include/task/Phi.h"
#include "../include/input.h"
#include "input.h"

using namespace Eigen;

class Adjoint
{
public:
    Adjoint(const _input_& i);
    Adjoint() = delete;
    ~Adjoint();
    Adjoint(const Adjoint&) = delete;
    Adjoint& operator=(const Adjoint&) = delete;

    static VectorXd RHS(const _input_ &i, const VectorXd& q, const VectorXd& dq, const VectorXd& lambda,
        const VectorXd& u, const VectorXd& eta, const VectorXd& ksi);
    VectorXd RHS(const VectorXd& q, const VectorXd& dq, const VectorXd& lambda,
        const VectorXd& u, const VectorXd& eta, const VectorXd& ksi) const;

private:
    const _input_& input;
    const task::h* h;
    const task::F* F;
    const task::M* M;
    const task::Phi* Phi;

    VectorXd __RHS(const _input_ &i, const VectorXd& q, const VectorXd& dq, const VectorXd& lambda,
        const VectorXd& u, const VectorXd& eta, const VectorXd& ksi) const;
};