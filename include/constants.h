#pragma once

#include "../Eigen/Dense"
#include "../include/utils.h"
using namespace Eigen;

const double _L_ = 0.25;
const double _m_box_  = 0.5;
const double _J_box_  = 0.0833;  // 25.0*_m_box_ * _L_*_L_ * 5.0/48.0;
const double _m_link_ = 0.25;
const double _J_link_ = 0.03333; //25.0*_m_link_ * _L_*_L_ / 12.0;
const double _c_cart_ = 0.3;
const double _c_pend_ = 0.5;
const Matrix2d I = Matrix2d::Identity();
const Matrix2d Om = Rot(M_PI_2);
const double M_GRAV = 9.80665;