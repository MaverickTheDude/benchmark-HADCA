#include "../include/input.h"
#include "../include/utils.h"
#include "../include/constants.h"
#include "../Eigen/Dense"

using namespace Eigen;

#include <iostream>

#define SIMULATION_TIME 1
#define TIME_STEP 0.01
#define LOG_ENERGY false
#define LOG_CONSTR true
#define CALCULATE_SIGNAL true // (dotyczy tylko sformulowania globalnego)

_input_::_input_(int _Nbodies_) : Nbodies(_Nbodies_), alpha0(Nbodies),
                                  dalpha0(Nbodies),   pjoint0(Nbodies),
                                  sigma0(2 * Nbodies),
                                  Tk(SIMULATION_TIME), dt(TIME_STEP), Nsamples(Tk/dt+1),
                                  Ntiers(ceil(log2(Nbodies))+1),
                                  logEnergy(LOG_ENERGY), logConstr(LOG_CONSTR), calculateSignal(CALCULATE_SIGNAL)
{
    bodyTypes.emplace_back("box");
    bodyTypes.emplace_back("link");
    alpha0(0) = 0.0;
    alpha0(1) = M_PI_4;
    alpha0.tail(Nbodies - 2).setConstant(0);
    dalpha0.setZero();
    setPJointAndSigma();
    
    int Ntmp = Nbodies;
	tiersInfo = new int[Ntiers];
	for (int i = 0; i < Ntiers; i++){
		tiersInfo[i] = Ntmp;
		Ntmp = static_cast<int>( ceil(static_cast<double>(Ntmp) / 2.0) );
	}
}

double _input_::pickBodyFriction(int bodyId) const {
    return bodyId == 0 ? _c_cart_ : _c_pend_;
}

void _input_::setTk(double tk) { Tk = tk; }

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

_input_::~_input_() {
	delete [] tiersInfo;
}