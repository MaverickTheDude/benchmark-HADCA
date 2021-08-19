#pragma once

#include "../../Eigen/Dense"
#include "../../Eigen/Sparse"
#include "../input.h"

using namespace Eigen;

namespace task
{
    class Phi
    {
    public:
        Phi(const _input_& i);
        ~Phi() = default;
        Phi(const Phi&) = delete;
        Phi& operator=(const Phi&) = delete;

        VectorXd operator()(const VectorXd& q) const;
        MatrixXd q(const VectorXd& q) const;
        MatrixXd ddqddqlambda(const VectorXd& q, const VectorXd& lambda) const;
        VectorXd ddqddqlambda(const VectorXd& q, const VectorXd& lambda,
            const VectorXd& ksi) const;

        MatrixXd qDenseMatrix(const VectorXd& q) const;;
        MatrixXd qDenseChunked(const VectorXd& q, const int jointNumber) const;
        SparseMatrix<double> qSparseMatrix(const VectorXd& q) const;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    private:
        const _input_& input;
        const Vector2d v; // A constant vector for the prismatic pair definition
        const int jointsNumber;
        const int constrDim;
        const int bodiesNumber;
    };
}