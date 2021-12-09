#pragma once

#include "body.h"
#include "../Eigen/Dense"
#include "../Eigen/StdVector"

using namespace Eigen;

/**
 * Forward declarations for tests.cpp compatibility.
 */
class _input_;

namespace PhiTimeDerivatives
{
    namespace constraints
    {
        void setStartConditions(_input_&);
    }
}

class _input_ {
public:
friend void PhiTimeDerivatives::constraints::setStartConditions(_input_ &input);
    _input_(int Nbodies);
    ~_input_();
    body pickBodyType(int bodyId) const {
        return (bodyId == 0) ? bodyTypes[0] : bodyTypes[1];
    }
    void setTk(double tk);

public:
    const int Nbodies;
    const int Nconstr = 2*Nbodies;
    VectorXd alpha0, dalpha0, pjoint0, sigma0;
    std::vector<body, aligned_allocator<body> > bodyTypes;
    double Tk;
    const double dt;
    const int Nsamples, Ntiers;
    int* tiersInfo;
    const bool logEnergy, logConstr;
    double w_hq  = 1.0;
    double w_hdq = 0.0;
    double w_hsig= 0.0;
    double w_Sq  = 0.0;
    double w_Sdq = 0.0;
private:
    void setPJointAndSigma(void);
};