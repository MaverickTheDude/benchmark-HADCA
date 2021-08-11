#pragma once

#include "../../Eigen/Dense"

using namespace Eigen;

namespace task
{
    class h
    {
    public:
        h() = default;
        virtual ~h() = default;
        h(const h&) = delete;
        h& operator=(const h&) = delete;

        virtual double operator()(const VectorXd& q, const VectorXd& dq) const = 0;
        virtual VectorXd q(const VectorXd& q, const VectorXd& dq) const = 0;
        virtual VectorXd dq(const VectorXd& q, const VectorXd& dq) const = 0;
    };

    class h_1 : public h
    {
    public:
        h_1() = default;
        ~h_1() = default;
        h_1(const h_1&) = delete;
        h_1& operator=(const h_1&) = delete;

        double operator()(const VectorXd& q, const VectorXd& dq) const;
        VectorXd q(const VectorXd& q, const VectorXd& dq) const;
        VectorXd dq(const VectorXd& q, const VectorXd& dq) const;
    };
}