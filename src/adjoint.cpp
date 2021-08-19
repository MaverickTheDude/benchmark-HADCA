#include "../include/adjoint.h"

#include "../Eigen/Dense"
#include "../include/task/h.h"
#include "../include/task/F.h"
#include "../include/task/M.h"
#include "../include/task/Phi.h"
#include "../include/input.h"

using namespace Eigen;

Adjoint::Adjoint(const _input_& i) : input(i), h(new task::h_1(i)), F(new task::F_1(i)),
    M(new task::M(i)), Phi(new task::Phi(i)) {}

Adjoint::~Adjoint()
{
    delete h;
    delete F;
    delete M;
    delete Phi;
}

VectorXd Adjoint::RHS(const _input_ &i, const VectorXd& q, const VectorXd& dq, const VectorXd& lambda,
        const VectorXd& u, const VectorXd& eta, const VectorXd& ksi)
{
    return Adjoint(i).RHS(q, dq, lambda, u, eta, ksi);
}

VectorXd Adjoint::RHS(const VectorXd& q, const VectorXd& dq, const VectorXd& lambda,
        const VectorXd& u, const VectorXd& eta, const VectorXd& ksi) const
{
    return __RHS(input, q, dq, lambda, u, eta, ksi);
}

VectorXd Adjoint::__RHS(const _input_ &i, const VectorXd& q, const VectorXd& dq, const VectorXd& lambda,
        const VectorXd& u, const VectorXd& eta, const VectorXd& ksi) const
{
    /**
     * Current implementation looks more clearly, but a loop over pieces of
     * particular matrices here would be probably more efficient.
     */

    return (h->q(q, dq) - h->ddtdq(q, dq)) - 
        (M->ddqdq(q, dq, eta) - M->ddt(q, eta) - F->dq(q, dq, u, eta)) -
        (F->q(q, dq, u, ksi) - F->ddtdq(q, dq, u, ksi) - Phi->ddqddqlambda(q, lambda, ksi));
}