#include "../../include/task/F.h"

#include "../../Eigen/Dense"
#include "../../include/input.h"
#include "../../include/utils.h"
#include "../../include/constants.h"
#include "../../include/body.h"

using namespace task;
using namespace Eigen;

F::F(const _input_& i) : input(i) {}

F_1::F_1(const _input_& i) : F(i) {}

VectorXd F_1::operator()(const VectorXd &q, const VectorXd &dq, const VectorXd &u) const
{
    /**
     * Returns forces column-vector F (it is not transposed).
     */

    VectorXd forces = VectorXd::Zero(3 * input.Nbodies);
    const VectorXd absoluteAlpha = absolutePositionToAbsoluteAlpha(q);

    forces(0) = u(0);
    for (int i = 0; i < input.Nbodies; i++)
        forces.segment(3 * i, 3) += SAB("s1C", i, absoluteAlpha, input) * g;

    return forces;
}

MatrixXd F_1::q(const VectorXd &q, const VectorXd &dq, const VectorXd &u) const
{
    MatrixXd F_1_q = MatrixXd::Zero(3 * input.Nbodies, 3 * input.Nbodies);
    const VectorXd absoluteAlpha = absolutePositionToAbsoluteAlpha(q);

    for(int i = 0; i < input.Nbodies; i++)
        F_1_q.block(3 * i + 2, 3 * i, 1, 3) = (dSABdAlpha(input.pickBodyType(i).s1C,
            absoluteAlpha(i)) * g).transpose();

    return F_1_q;
}

MatrixXd F_1::dq(const VectorXd &q, const VectorXd &dq, const VectorXd &u) const
{
    MatrixXd F_1_dq = MatrixXd::Zero(3 * input.Nbodies, 3 * input.Nbodies);

    return F_1_dq;
}

MatrixXd F_1::ddtdq(const VectorXd& q, const VectorXd& dq, const VectorXd& u) const
{
    MatrixXd F_1_ddtdq = MatrixXd::Zero(3 * input.Nbodies, 3 * input.Nbodies);

    return F_1_ddtdq;
}

VectorXd F_1::operator()(const int bodyNumber, const VectorXd &q, const VectorXd &dq, const VectorXd &u) const
{
    /**
     * Returns part of forces column-vector F (it is not transposed).
     */

    Vector3d forces = Vector3d::Zero();

    if(bodyNumber == 0)
        forces(0) = u(0);

    forces += SAB("s1C", bodyNumber, q(3 * bodyNumber + 2), input) * g;

    return forces;
}

MatrixXd F_1::q(const int bodyNumber, const VectorXd &q, const VectorXd &dq, const VectorXd &u) const
{
    Matrix3d F_1_q = Matrix3d::Zero();
    F_1_q.block(2, 0, 1, 3) = (dSABdAlpha(input.pickBodyType(bodyNumber).s1C,
        q(3 * bodyNumber + 2)) * g).transpose();

    return F_1_q;
}

MatrixXd F_1::dq(const int bodyNumber, const VectorXd &q, const VectorXd &dq, const VectorXd &u) const
{
    Matrix3d F_1_dq = Matrix3d::Zero();

    return F_1_dq;
}

MatrixXd F_1::ddtdq(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& u) const
{
    Matrix3d F_1_ddtdq = Matrix3d::Zero();

    return F_1_ddtdq; 
}

VectorXd F_1::q(const VectorXd& q, const VectorXd& dq, const VectorXd& u, const VectorXd& ksi) const
{
    VectorXd rhs = VectorXd::Zero(3 * input.Nbodies);

    for(int i = 0; i < input.Nbodies; i++)
    {
        rhs.segment(3 * i, 3) = F_1::q(i, q, dq, u) * ksi.segment(3 * i , 3);
    }

    return rhs;
}

VectorXd F_1::dq(const VectorXd& q, const VectorXd& dq, const VectorXd& u, const VectorXd& eta) const
{
    return VectorXd::Zero(3 * input.Nbodies);
}

VectorXd F_1::ddtdq(const VectorXd& q, const VectorXd& dq, const VectorXd& u, const VectorXd& ksi) const
{
    return VectorXd::Zero(3 * input.Nbodies);
}