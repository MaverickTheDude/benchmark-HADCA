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
    // _input_() {
    //     sA1(2.0, 0.0); << runtime error bo sA1 juz zdazylo powstac. Czy to nie powinno rzucac bledu przy kompilacji?
    //     sA2(3.0, 0.0);
    // }
    _input_(int Nbodies);
    ~_input_();
    body pickBodyType(int bodyId) const {
        return (bodyId == 0) ? bodyTypes[0] : bodyTypes[1];
    }

public:
    const int Nbodies;
    const int Nconstr = 2*Nbodies;
    VectorXd alpha0, dalpha0, pjoint0, sigma0;
    std::vector<body, aligned_allocator<body> > bodyTypes;
    const double Tk, dt;
    const int Nsamples, Ntiers;
    int* tiersInfo;
    const bool logEnergy;

private:
    void setPJointAndSigma(void);
};