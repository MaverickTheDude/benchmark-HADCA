#include "../include/derivatives.h"

#include "../Eigen/Dense"
#include "../include/task/Phi.h"

using namespace Eigen;

VectorXd Phi(const VectorXd& q, const _input_& input) {
    /**
     * Warning! This function is kept in this version for backward
     * compatibility reasons. It might be inefficient due to additional
     * copying method's results over return.
     */
    task::Phi Phi(input);
    return Phi(q);
}

MatrixXd Jacobian(const VectorXd& q, const _input_& input) {
    /**
     * Warning! This function is kept in this version for backward
     * compatibility reasons. It might be inefficient due to additional
     * copying method's results over return.
     */
    task::Phi Phi(input);
    return Phi.q(q);
}