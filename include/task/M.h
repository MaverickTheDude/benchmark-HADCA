#pragma once

#include "../../Eigen/Dense"
#include "../input.h"

using namespace Eigen;

namespace task
{
    class M
    {
    public:
        M() = delete;
        M(const _input_& i);
        virtual ~M() = default;
        M(const M&) = delete;
        M& operator=(const M&) = delete;

        static Matrix3d local(const int bodyNumber, const _input_& i);
        Matrix3d local(const int bodyNumber) const;

        MatrixXd operator()(const VectorXd& q) const;
        MatrixXd ddt(const VectorXd& q) const;
        MatrixXd ddqdq(const VectorXd& q, const VectorXd& dq) const;

        MatrixXd operator()(const int bodyNumber, const VectorXd& q) const;
        MatrixXd ddt(const int bodyNumber, const VectorXd& q) const;
        MatrixXd ddqdq(const int bodyNumber, const VectorXd& q, const VectorXd& dq) const;

        VectorXd ddt(const VectorXd& q, const VectorXd& eta) const;
        VectorXd ddqdq(const VectorXd& q, const VectorXd& dq, const VectorXd& eta ) const;

    protected:
        const _input_& input;
    };
}
