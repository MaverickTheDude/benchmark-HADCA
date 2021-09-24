#pragma once

#include "../Eigen/Dense"
#include "utils.h"

using namespace Eigen;

const double _L_ = 0.25;
const double _m_box_ = 0.8;
const double _J_box_ = 0.0833;
const double _m_link_ = 0.4;
const double _J_link_ = 0.0333;
const Matrix2d I = Matrix2d::Identity();
const Matrix2d Om = Rot(M_PI_2);
const double M_GRAV = 9.80665;