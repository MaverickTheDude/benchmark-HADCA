#pragma once

#include "input.h"
#include "ksiCoefs.h"
#include "../Eigen/Dense"

using namespace Eigen;

class Assembly {
    friend VectorXd RHS_HDCA(const double& t, const VectorXd& y, const _input_& input);
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Assembly(const int id, const VectorXd& alphaAbs, const VectorXd& pjoint, const _input_& input);
    Assembly(Assembly& AsmA, Assembly& AsmB);
    void connect_base_body();
    void disassembleAll();
    Vector3d calculate_V1() const;
    Vector3d calculate_V2() const;
    void setAll(const Assembly&);

    /* acceleration level functions */
    void setKsiAcc(const int id, const VectorXd& alphaAbs, const VectorXd& dAlphaAbs, const MatrixXd& P1art, const _input_& input);
    void assembleAcc(const Assembly& AsmA, const Assembly& AsmB);
    void assembleAcc(const Assembly& AsmA);
    void connect_base_bodyAcc();
    void disassembleAcc();
    void setAcc(const Assembly& Asm);
    Vector3d calculate_dV1() const;
    Vector3d calculate_dV2() const;

public: // to do: set private
    const ksi_coefs ksi;
    const Vector3d Q1Acc;
    const Matrix3d S12;
    Assembly * const ptrAsmA, * const ptrAsmB; // const jest tentatywny
    Vector3d T1, T2, Q1Art, Q2Art;

    /* acceleration level attributes */
    Vector3d L1, L2, ksiAcc_10, ksiAcc_20;
};