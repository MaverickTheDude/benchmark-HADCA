#include "../../include/task/F.h"

#include "../../Eigen/Dense"

using namespace task;
using namespace Eigen;

VectorXd F_1::operator()(const VectorXd &q, const VectorXd &dq) const
{
    return VectorXd(3);
}

MatrixXd F_1::q(const VectorXd &q, const VectorXd &dq) const
{
    return MatrixXd(3, 3);
}

MatrixXd F_1::dq(const VectorXd &q, const VectorXd &dq) const
{
    return MatrixXd(3, 3);
}