#include "../../include/task/M.h"

#include "../../Eigen/Dense"
#include "../../include/input.h"
#include "../../include/utils.h"

using namespace task;
using namespace Eigen;

M::M(const _input_& i) : input(i) {}

Matrix3d M::local(const int bodyNumber, const _input_& input)
{
    Matrix3d m = Matrix3d::Zero();
    m(0, 0) = input.pickBodyType(bodyNumber).m;
    m(1, 1) = input.pickBodyType(bodyNumber).m;
    m(2, 2) = input.pickBodyType(bodyNumber).J;

    return m;
}

Matrix3d M::local(const int bodyNumer) const
{
    return local(bodyNumer, input);
}

MatrixXd M::operator()(const VectorXd& q) const
{
    MatrixXd m = MatrixXd::Zero(3 * input.Nbodies, 3 * input.Nbodies);
    const VectorXd absoluteAlpha = absolutePositionToAbsoluteAlpha(q);

    for(int i = 0; i < input.Nbodies; i++)
    {
        const MatrixXd S1C = SAB("s1C", i, absoluteAlpha, input);
        m.block(3 * i, 3 * i, 3, 3) = S1C * local(i) * S1C.transpose();
    }

    return m;
}

MatrixXd M::ddt(const VectorXd& q) const
{
    MatrixXd m = MatrixXd(3 * input.Nbodies, 3 * input.Nbodies);

    return m;
}

MatrixXd M::ddqdq(const VectorXd& q, const VectorXd& dq) const
{
    /**
     * The current name of this member function is not self-explaining. I do
     * not have idea how to improve that at the moment.
     * 
     * Returns d/dq (M * dq) (transposed).
     */

    MatrixXd m = MatrixXd::Zero(3 * input.Nbodies, 3 * input.Nbodies);
     const VectorXd absoluteAlpha = absolutePositionToAbsoluteAlpha(q);

    for(int i = 0; i < input.Nbodies; i++)
    {
        const MatrixXd S = SAB("s1C", i, absoluteAlpha(i), input);
        const MatrixXd dS = dSABdAlpha(input.pickBodyType(i).s1C,
            absoluteAlpha(i));
        const MatrixXd M = local(i);

        m.block(3 * i + 2, 3 * i, 1, 3) = ((dS * M * S.transpose() + 
            S * M * dS.transpose()) * dq.segment(3 * i, 3)).transpose();
    }

    return m;
}

MatrixXd M::operator()(const int bodyNumber, const VectorXd& q) const
{
    MatrixXd m = MatrixXd::Zero(3, 3);
    
    const MatrixXd S1C = SAB("s1C", bodyNumber, q(3 * bodyNumber + 2), input);
    m = S1C * local(bodyNumber) * S1C.transpose();

    return m;
}

MatrixXd M::ddt(const int bodyNumber, const VectorXd& q) const
{
    MatrixXd m = MatrixXd::Zero(3, 3);

    return m;
}

MatrixXd M::ddqdq(const int bodyNumber, const VectorXd& q, const VectorXd& dq) const
{
    /**
     * Returns part of d/dq (M * dq) (transposed).
     */

    MatrixXd m = MatrixXd::Zero(3, 3);

    const MatrixXd S = SAB("s1C", bodyNumber, q(3 * bodyNumber + 2), input);
    const MatrixXd dS = dSABdAlpha(input.pickBodyType(bodyNumber).s1C,
        q(3 * bodyNumber + 2));
    const MatrixXd M = local(bodyNumber);

    m.block(2, 0, 1, 3) = ((dS * M * S.transpose() +
        S * M * dS.transpose()) * dq.segment(3 * bodyNumber, 3)).transpose();

    return m;
}

VectorXd M::ddt(const VectorXd& q, const VectorXd& eta) const
{
    return VectorXd::Zero(3 * input.Nbodies);
}

VectorXd M::ddqdq(const VectorXd& q, const VectorXd& dq, const VectorXd& eta ) const
{
    VectorXd rhs = VectorXd::Zero(3 * input.Nbodies);

    for(int i = 0; i < input.Nbodies; i++)
    {
        rhs.segment(3 * i, 3) = ddqdq(i, q, dq) * eta.segment(3 * i , 3);
    }

    return rhs;
}