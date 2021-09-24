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

VectorXd Adjoint::RHS(const _input_ &i, const VectorXd& q, const VectorXd& dq, const VectorXd& d2q, const VectorXd& lambda,
    const VectorXd& u, const VectorXd& eta, const VectorXd& ksi)
{
    return Adjoint(i).RHS(q, dq, d2q, lambda, u, eta, ksi);
}

VectorXd Adjoint::RHS(const VectorXd& q, const VectorXd& dq, const VectorXd& d2q, const VectorXd& lambda,
    const VectorXd& u, const VectorXd& eta, const VectorXd& ksi) const
{
    return __RHS(input, q, dq, d2q, lambda, u, eta, ksi);
}

VectorXd Adjoint::RHS(const int bodyNumber, const _input_ &i, const VectorXd& q, const VectorXd& dq, const VectorXd& d2q, 
    const VectorXd& lambda, const VectorXd& u, const VectorXd& eta, const VectorXd& ksi)
{
    return Adjoint(i).RHS(bodyNumber, q, dq, d2q, lambda, u, eta, ksi);
}

VectorXd Adjoint::RHS(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& d2q, const VectorXd& lambda,
    const VectorXd& u, const VectorXd& eta, const VectorXd& ksi) const
{
    return __RHS(bodyNumber, input, q, dq, d2q, lambda, u, eta, ksi);
}

VectorXd Adjoint::RHS3d(const int bodyNumber, const _input_ &i, const VectorXd& q, const VectorXd& dq, const VectorXd& d2q,
    const VectorXd& lambda, const VectorXd& u, const Vector3d& eta, const Vector3d& ksi)
{
    return Adjoint(i).RHS3d(bodyNumber, q, dq, d2q, lambda, u, eta, ksi);
}
VectorXd Adjoint::RHS3d(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& d2q, const VectorXd& lambda,
    const VectorXd& u, const Vector3d& eta, const Vector3d& ksi) const
{
    return __RHS3d(bodyNumber, input, q, dq, d2q, lambda, u, eta, ksi);
}

VectorXd Adjoint::__RHS(const _input_ &i, const VectorXd& q, const VectorXd& dq, const VectorXd& d2q, const VectorXd& lambda,
        const VectorXd& u, const VectorXd& eta, const VectorXd& ksi) const
{
    /**
     * Current implementation looks more clearly, but a loop over pieces of
     * particular matrices here would be probably more efficient.
     */

    return (h->q(q, dq) - h->ddtdq(q, dq, d2q)) - 
        (M->ddqdq(q, dq, eta) - M->ddt(q, dq, eta) - F->dq(q, dq, u, eta)) -
        (F->q(q, dq, u, ksi) - F->ddtdq(q, dq, u, ksi) - Phi->ddqddqlambda(q, lambda, ksi));
}

VectorXd Adjoint::__RHS(const int bodyNumber, const _input_ &i, const VectorXd& q, const VectorXd& dq, const VectorXd& d2q,
    const VectorXd& lambda, const VectorXd& u, const VectorXd& eta, const VectorXd& ksi) const
{
    return (h->q(bodyNumber, q, dq) - h->ddtdq(bodyNumber, q, dq, d2q)) - 
        (M->ddqdq(bodyNumber, q, dq) - M->ddt(bodyNumber, q, dq) - F->dq(bodyNumber, q, dq, u)) * eta.segment(3 * bodyNumber, 3) -
        ((F->q(bodyNumber, q, dq, u) - F->ddtdq(bodyNumber, q, dq, u)) * ksi.segment(3 * bodyNumber, 3) - 
        Phi->ddqddqlambda(bodyNumber, q, lambda, ksi));
}

VectorXd Adjoint::__RHS3d(const int bodyNumber, const _input_ &i, const VectorXd& q, const VectorXd& dq, const VectorXd& d2q,
    const VectorXd& lambda, const VectorXd& u, const Vector3d& eta, const Vector3d& ksi) const
{   
    return (h->q(bodyNumber, q, dq) - h->ddtdq(bodyNumber, q, dq, d2q)) - 
        (M->ddqdq(bodyNumber, q, dq) - M->ddt(bodyNumber, q, dq) - F->dq(bodyNumber, q, dq, u)) * eta -
        ((F->q(bodyNumber, q, dq, u) - F->ddtdq(bodyNumber, q, dq, u)) * ksi - 
        Phi->ddqddqlambda3d(bodyNumber, q, lambda, ksi));
}