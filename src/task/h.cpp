#include "../../include/task/h.h"

#include "../../Eigen/Dense"

using namespace task;
using namespace Eigen;

h::h(const _input_& i) : input(i) {}

h_1::h_1(const _input_& i) : h(i), alpha(i.w_hq), beta(i.w_hdq) {}

// h_1::h_1(const _input_& i, double _alpha) : h(i), alpha(_alpha) {}

// void h_1::setAlpha(double _alpha) { alpha = _alpha; }

double h_1::operator()(const VectorXd& q, const VectorXd& dq) const
{
    return alpha * q(0) * q(0) + beta * dq(0) * dq(0);
}

VectorXd h_1::q(const VectorXd& q, const VectorXd& dq) const
{
    VectorXd h_1_q = VectorXd::Zero(3 * input.Nbodies);
    h_1_q(0) = 2 * alpha * q(0);

    return h_1_q;
}

VectorXd h_1::dq(const VectorXd& q, const VectorXd& dq) const
{
    VectorXd h_1_dq = VectorXd::Zero(3 * input.Nbodies);
    h_1_dq(0) = 2 * beta * dq(0);

    return h_1_dq;
}

VectorXd h_1::ddtdq(const VectorXd& q, const VectorXd& dq, const VectorXd& d2q) const
{
    VectorXd h_1_ddtdq = VectorXd::Zero(3 * input.Nbodies);  
    h_1_ddtdq(0) = 2 * beta * d2q(0);  // WARNING: tutaj powinno byc przyspieszenie

    return h_1_ddtdq;
}

double h_1::operator()(const int bodyNumber, const VectorXd& q, const VectorXd& dq) const
{
    return h_1::operator()(q, dq);
}

VectorXd h_1::q(const int bodyNumber, const VectorXd& q, const VectorXd& dq) const
{
    Vector3d h_1_q = Vector3d::Zero();
    if(bodyNumber == 0)
        h_1_q(0) = 2 * alpha * q(0);

    return h_1_q;
}

VectorXd h_1::dq(const int bodyNumber, const VectorXd& q, const VectorXd& dq) const
{
    Vector3d h_1_dq = Vector3d::Zero();
    h_1_dq(0) = 2 * beta * dq(0);

    return h_1_dq;
}

VectorXd h_1::ddtdq(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& d2q) const
{
    Vector3d h_1_ddtdq = Vector3d::Zero();
    if(bodyNumber == 0)
        h_1_ddtdq(0) = 2 * beta * d2q(0); // WARNING: tutaj powinno byc przyspieszenie

    return h_1_ddtdq;
}