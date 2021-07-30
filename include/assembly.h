#pragma once

#include "input.h"
#include "ksiCoefs.h"
#include "../Eigen/Dense"

using namespace Eigen;

class Assembly {
public:
    Assembly(const int id, const VectorXd& alphaAbs, const VectorXd& pjoint, const _input_& input);
    Assembly(Assembly& AsmA, Assembly& AsmB);

public: // to do: set private
    const ksi_coefs ksi;
    const Vector3d Q1Acc;
    const Matrix3d S12;
    Assembly * const ptrAsmA, * const ptrAsmB; // const jest tentatywny
    Vector3d T1, T2, Q1Art, Q2Art;
};