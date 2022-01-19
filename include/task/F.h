#pragma once

#include "../../Eigen/Dense"
#include "../input.h"

using namespace Eigen;

namespace task
{
    /**
     * Objects returned by this class are already transposed so to comply with (B.7).
     */
    class F
    {
    public:
        /**
         * C.21: If you define or =delete any copy, move, or destructor function,
         * define or =delete them all. Probably, move functions should be defined
         * also in classes F, h, Phi nad their derivatives.
         */
        F() = delete;
        F(const _input_& i);
        virtual ~F() = default;
        F(const F&) = delete;
        F& operator=(const F&) = delete;

        virtual VectorXd operator()(const VectorXd& q, const VectorXd& dq, const VectorXd& u) const = 0;
        virtual MatrixXd q(const VectorXd& q, const VectorXd& dq, const VectorXd& u) const = 0;
        virtual MatrixXd dq(const VectorXd& q, const VectorXd& dq, const VectorXd& u) const = 0;
        virtual MatrixXd ddtdq(const VectorXd& q, const VectorXd& dq, const VectorXd& u) const = 0;

        virtual VectorXd operator()(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& u) const = 0;
        virtual MatrixXd q(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& u) const = 0;
        virtual MatrixXd dq(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& u) const = 0;
        virtual MatrixXd ddtdq(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& u) const = 0;

        virtual Vector3d q(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& u, const VectorXd& ksi) const = 0;
        virtual Vector3d dq(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& u, const VectorXd& eta) const = 0;
        virtual Vector3d ddtdq(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& u, const VectorXd& ksi) const = 0;

        virtual VectorXd q(const VectorXd& q, const VectorXd& dq, const VectorXd& u, const VectorXd& ksi) const = 0;
        virtual VectorXd dq(const VectorXd& q, const VectorXd& dq, const VectorXd& u, const VectorXd& eta) const = 0;
        virtual VectorXd ddtdq(const VectorXd& q, const VectorXd& dq, const VectorXd& u, const VectorXd& ksi) const = 0;
    
    protected:
        const _input_& input;
        const Vector3d g = Vector3d(0.0, -9.81, 0);
    };

    class F_1 : public F
    {
    public:
        F_1() = delete;
        F_1(const _input_& i);
        ~F_1() = default;
        F_1(const F_1&) = delete;
        F_1& operator=(const F_1&) = delete;

        VectorXd operator()(const VectorXd& q, const VectorXd& dq, const VectorXd& u) const;
        MatrixXd q(const VectorXd& q, const VectorXd& dq, const VectorXd& u) const;
        MatrixXd dq(const VectorXd& q, const VectorXd& dq, const VectorXd& u) const;
        MatrixXd ddtdq(const VectorXd& q, const VectorXd& dq, const VectorXd& u) const;

        VectorXd operator()(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& u) const;
        MatrixXd q(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& u) const;
        MatrixXd dq(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& u) const;
        MatrixXd ddtdq(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& u) const;

        Vector3d q(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& u, const VectorXd& ksi) const ;
        Vector3d dq(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& u, const VectorXd& eta) const;
        Vector3d ddtdq(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& u, const VectorXd& ksi) const;

        VectorXd q(const VectorXd& q, const VectorXd& dq, const VectorXd& u, const VectorXd& ksi) const;
        VectorXd dq(const VectorXd& q, const VectorXd& dq, const VectorXd& u, const VectorXd& eta) const;
        VectorXd ddtdq(const VectorXd& q, const VectorXd& dq, const VectorXd& u, const VectorXd& ksi) const;
    };

    class F_2 : public F_1
    {
        /**
         * Class F_2 adds viscous friction in joints to F_1. As for the approach
         * of dealing with joints it follows the one used in task::Phi class.
         */
    public:
        F_2() = delete;
        F_2(const _input_& i);
        ~F_2() = default;
        F_2(const F_2&) = delete;
        F_2& operator=(const F_2&) = delete;

        VectorXd operator()(const VectorXd& q, const VectorXd& dq, const VectorXd& u) const;
        MatrixXd dq(const VectorXd &q, const VectorXd &dq, const VectorXd &u) const;

        VectorXd operator()(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& u) const;
        MatrixXd dq(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& u) const;

        Vector3d dq(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& u, const VectorXd& eta) const;

        VectorXd dq(const VectorXd& q, const VectorXd& dq, const VectorXd& u, const VectorXd& eta) const;

    private:
        const int jointsNumber;
    };
}
