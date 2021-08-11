#include "../../include/task/h.h"

#include "../../Eigen/Dense"

using namespace task;
using namespace Eigen;

double h_1::operator()(const VectorXd& q, const VectorXd& dq) const
{
    return 0;
}

VectorXd h_1::q(const VectorXd& q, const VectorXd& dq) const
{
    return VectorXd(3);
}

VectorXd h_1::dq(const VectorXd& q, const VectorXd& dq) const
{
    return VectorXd(3);
}