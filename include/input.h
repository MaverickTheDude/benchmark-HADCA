#pragma once

#include "body.h"
#include "../Eigen/Dense"

#include <vector>

using namespace Eigen;

class _input_ {
public:
    // _input_() {
    //     sA1(2.0, 0.0); << runtime error bo sA1 juz zdazylo powstac. Czy to nie powinno rzucac bledu przy kompilacji?
    //     sA2(3.0, 0.0);
    // }
    _input_(int Nbodies);
    body pickBodyType(int bodyId) const {
        return (bodyId == 0) ? bodyTypes[0] : bodyTypes[1];
    }

public:
    const int Nbodies;
    const int Nconstr = 2*Nbodies;
    VectorXd alpha0, dalpha0, pjoint0, sigma0;
    std::vector<body> bodyTypes;

private:
    void setPJointAndSigma(void);
};