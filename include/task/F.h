#pragma once

#include "../../Eigen/Dense"

using namespace Eigen;

namespace task
{
    class F
    {
    public:
        /**
         * C.21: If you define or =delete any copy, move, or destructor function,
         * define or =delete them all. Probably, move functions should be defined
         * also in classes F, h, Phi nad their derivatives.
         */
        F() = default;
        virtual ~F() = default;
        F(const F&) = delete;
        F& operator=(const F&) = delete;

        virtual VectorXd operator()(const VectorXd& q, const VectorXd& dq) const = 0;
        virtual MatrixXd q(const VectorXd& q, const VectorXd& dq) const = 0;
        virtual MatrixXd dq(const VectorXd& q, const VectorXd& dq) const = 0;
    };

    class F_1 : public F
    {
    public:
        F_1() = default;
        ~F_1() = default;
        F_1(const F_1&) = delete;
        F_1& operator=(const F_1&) = delete;

        VectorXd operator()(const VectorXd& q, const VectorXd& dq) const;
        MatrixXd q(const VectorXd& q, const VectorXd& dq) const;
        MatrixXd dq(const VectorXd& q, const VectorXd& dq) const;
    };
}