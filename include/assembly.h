#pragma once

#include "../include/input.h"
#include "../include/ksiCoefs.h"
#include "../Eigen/Dense"

using namespace Eigen;

class AssemblyAdj {
    friend VectorXd RHS_ADJOINT(const double& t, const VectorXd& y, const _input_& input);
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    AssemblyAdj& operator=(const AssemblyAdj& A);
    AssemblyAdj();
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
    friend void test_SetAssembly(void);
    friend VectorXd RHS_HDCA(const double& t, const VectorXd& y, const VectorXd& uVec, const _input_&, _solution_&);
    friend VectorXd RHS_HDCA(const double& t, const VectorXd& y, const _input_& input);
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Assembly(const int id, const VectorXd& alphaAbs, const VectorXd& pjoint, const double& u, const _input_& input);
    Assembly(Assembly& AsmA, Assembly& AsmB);
    Assembly();
    void assembleForce(Assembly& AsmA, Assembly& AsmB);
    void setArtForces(const Assembly& Asm);
    Assembly(const Assembly&) = default; // note: shallow copy with default ctor is exactly what we need
    Assembly& operator=(const Assembly&);
    void connect_base_body();
    void disassembleAll();
    void disassembleVel();
    void disassembleForce();
    Vector3d calculate_V1() const;
    Vector3d calculate_V2() const;
    void setVel(const Assembly&);
    void setAll(const Assembly&);
    void setCoefArtForces(int i, const VectorXd& alphaAbs, const VectorXd& dalpha, const double& u_ctrl, const _input_& input);
    void setAccForces(const Assembly& Asm);
    void connect_base_artForces();


    /* acceleration level functions */
    void setKsiAcc(const int id, const VectorXd& alphaAbs, const VectorXd& dAlphaAbs, const MatrixXd& P1art, const _input_& input);
    void assembleAcc(const Assembly& AsmA, const Assembly& AsmB);
    void assembleAcc(const Assembly& AsmA);
    void connect_base_bodyAcc();
    void disassembleAcc();
    void setAcc(const Assembly& Asm);
    Vector3d calculate_dV1() const;
    Vector3d calculate_dV2() const;

    /* getters for RHS-odeint */
    const Vector3d T1_() const { return T1; }
    const Vector3d T2_() const { return T2; }
    const Vector3d L1_() const { return L1; }
    const Vector3d L2_() const { return L2; }
    const Vector3d Q1Art_() const { return Q1Art; }
    const Vector3d Q2Art_() const { return Q2Art; }

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