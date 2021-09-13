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
        MatrixXd ddtq(const VectorXd& q, const VectorXd& dq) const;
        MatrixXd d2dt2q(const VectorXd& q, const VectorXd& dq, const VectorXd& ddq) const;
        MatrixXd ddqddqlambda(const VectorXd& q, const VectorXd& lambda) const;
        VectorXd ddqddqlambda(const VectorXd& q, const VectorXd& lambda,
            const VectorXd& ksi) const;
        VectorXd ddqddqlambda(const int bodyNumber, const VectorXd& q,
            const VectorXd& lambda, const VectorXd& ksi) const;
        /**
         * Why you cannot overload in this manner:
         * https://stackoverflow.com/questions/23129741/c-method-overloading-base-and-derived-parameters
         * https://stackoverflow.com/questions/9818132/difference-betwen-visitor-pattern-double-dispatch
         * 
         * Vector3d ddqddqlambda(const int bodyNumber, const VectorXd& q,
         *    const VectorXd& lambda, const Vector3d& ksi) const;
         */
        Vector3d ddqddqlambda3d(const int bodyNumber, const VectorXd& q,
            const VectorXd& lambda, const Vector3d& ksi) const;

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