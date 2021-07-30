#include "../include/input.h"

#include "../include/utils.h"
#include "../Eigen/Dense"

#include <vector>

using namespace Eigen;

_input_::_input_(int _Nbodies_) : Nbodies(_Nbodies_), alpha0(_Nbodies_),
                                  dalpha0(_Nbodies_), pjoint0(_Nbodies_),
                                  sigma0(2 * _Nbodies_)
{
    bodyTypes.emplace_back("box");
    bodyTypes.emplace_back("link");
    alpha0(0) = 0.0;
    alpha0.tail(_Nbodies_ - 1).setConstant(M_PI_4);
    if (Nbodies >= 3) alpha0(2) = M_PI_2;
    dalpha0(0) = 0.5;
    dalpha0(1) = 0.2;
    dalpha0.tail(_Nbodies_-2).setZero();
    setPJointAndSigma();
}

void  _input_::setPJointAndSigma(void)
{
    /**
     * Returns pJoint momentum and sigma coefficient.
     * Assumption: D^{T} * D = H^{T} * H = 1
     */

    const VectorXd alphaAbs = joint2AbsAngles(alpha0);
    const VectorXd dq = jointToAbsoluteVelocity(alpha0, dalpha0, *this);

    Vector3d PP = Vector3d::Zero();
    for(int bodyId = Nbodies - 1; bodyId >= 0; bodyId--)
    {
        const Matrix3d S1C = SAB("s1C", bodyId, alphaAbs, *this);
        const Matrix3d M1 = S1C * massMatrix(bodyId, *this) * S1C.transpose();
        const Vector3d V1 = dq.segment(3 * bodyId + 0, 3);
        const Matrix3d S12 = SAB("s12", bodyId, alphaAbs, *this);    
        const MatrixXd H1T = pickBodyType(bodyId).H.transpose();
        const MatrixXd D1T = pickBodyType(bodyId).D.transpose();

        PP = M1 * V1 + S12 * PP;
        pjoint0.segment(bodyId, 1) = H1T * PP;
        sigma0.segment(2 * bodyId + 0, 2) = D1T * PP;
    }

    return;
}