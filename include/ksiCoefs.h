#pragma once

#include "input.h"
#include "../Eigen/Dense"

using namespace Eigen;

class ksi_coefs {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    ksi_coefs(const int id, const VectorXd& alphaAbs, const VectorXd& pjoint, const _input_& input);
    ksi_coefs(const ksi_coefs& ksiA, const ksi_coefs& ksiB);

	Matrix3d k11() const {return i11;}
	Matrix3d k12() const {return i12;}
	Matrix3d k21() const {return i21;}
	Matrix3d k22() const {return i22;}
	Vector3d k10() const {return i10;}
	Vector3d k20() const {return i20;}

private:
    Matrix3d i11, i12, i21, i22;
    Vector3d i10, i20;
};