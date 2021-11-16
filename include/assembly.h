#pragma once

#include "input.h"
#include "ksiCoefs.h"
#include "../Eigen/Dense"

using namespace Eigen;

class AssemblyAdj {
    friend VectorXd RHS_ADJOINT(const double& t, const VectorXd& y, const _input_& input);
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    AssemblyAdj(const int id, const dataAbsolute& data, const double& u, const _input_& input);
    AssemblyAdj(AssemblyAdj& AsmA, AssemblyAdj& AsmB);
    void connect_base_body();
    void disassemble();
    Vector3d calculate_dETA1() const;
    Vector3d calculate_dETA2() const;
    void setAll(const AssemblyAdj&);

private:
    /*const*/ ksi_coefs ksi;
    AssemblyAdj * /*const*/ ptrAsmA, * /*const*/ ptrAsmB;
    Vector3d T1, T2;
};

class Assembly {
    friend VectorXd RHS_HDCA(const double& t, const VectorXd& y, const _input_& input);
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Assembly(const int id, const VectorXd& alphaAbs, const VectorXd& pjoint, const double& u, const _input_& input);
    Assembly(Assembly& AsmA, Assembly& AsmB);
    Assembly();
    Assembly(const Assembly&) = default; // note: shallow copy with default ctor is exactly what we need
    Assembly& operator=(const Assembly&);
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

friend VectorXd RHS_HDCA(const double& t, const VectorXd& y, const VectorXd& uVec, const _input_&, _solution_&);
friend void test_SetAssembly(void);
private:
    /* const correctness fails when we want to use std::copy inside std::vector */
    /*const*/ ksi_coefs ksi;
    /*const*/ Vector3d Q1Acc;
    /*const*/ Matrix3d S12;
    Assembly * /*const*/ ptrAsmA, * /*const*/ ptrAsmB;
    Vector3d T1, T2, Q1Art, Q2Art;

    /* acceleration level attributes */
    Vector3d L1, L2, ksiAcc_10, ksiAcc_20;
};