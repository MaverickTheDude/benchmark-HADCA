#pragma once

#include "../../Eigen/Dense"
#include "../input.h"

using namespace Eigen;

namespace task
{
    /**
     * Objects returned by this class are already transposed so to comply with (B.7).
     */
    class h
    {
    public:
        h() = delete;
        h(const _input_& i);
        virtual ~h() = default;
        h(const h&) = delete;
        h& operator=(const h&) = delete;

        virtual double operator()(const VectorXd& q, const VectorXd& dq) const = 0;
        virtual VectorXd q(const VectorXd& q, const VectorXd& dq) const = 0;
        virtual VectorXd dq(const VectorXd& q, const VectorXd& dq) const = 0;
        virtual VectorXd ddtdq(const VectorXd& q, const VectorXd& dq) const = 0;

        virtual double operator()(const int bodyNumber, const VectorXd& q, const VectorXd& dq) const = 0;
        virtual VectorXd q(const int bodyNumber, const VectorXd& q, const VectorXd& dq) const = 0;
        virtual VectorXd dq(const int bodyNumber, const VectorXd& q, const VectorXd& dq) const = 0;
        virtual VectorXd ddtdq(const int bodyNumber, const VectorXd& q, const VectorXd& dq) const = 0;

    protected:
        const _input_& input;
    };


    class h_1 : public h
    {
    public:
        h_1() = delete;
        h_1(const _input_& i);
        h_1(const _input_& i, double _alpha);
        ~h_1() = default;
        h_1(const h_1&) = delete;
        h_1& operator=(const h_1&) = delete;

        void setAlpha(double _alpha);

        double operator()(const VectorXd& q, const VectorXd& dq) const;
        VectorXd q(const VectorXd& q, const VectorXd& dq) const;
        VectorXd dq(const VectorXd& q, const VectorXd& dq) const;
        VectorXd ddtdq(const VectorXd& q, const VectorXd& dq) const;

        double operator()(const int bodyNumber, const VectorXd& q, const VectorXd& dq) const;
        VectorXd q(const int bodyNumber, const VectorXd& q, const VectorXd& dq) const;
        VectorXd dq(const int bodyNumber, const VectorXd& q, const VectorXd& dq) const;
        VectorXd ddtdq(const int bodyNumber, const VectorXd& q, const VectorXd& dq) const;

    private:
        double alpha;
    };
}