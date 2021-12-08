#include "../include/assembly.h"

#include "../include/input.h"
#include "../include/ksiCoefs.h"
#include "../include/utils.h"
#include "../include/constants.h"
#include "../Eigen/Dense"

#include <iostream>

using namespace Eigen;

static Matrix3d massMatrixDot(const std::string &_sAB_, const int id, 
							  const VectorXd& alphaAbs, const VectorXd& dAlphaAbs, const _input_& input) {
	/* note: wynik taki sam jak dla: m * (dS1c + dS1c.transpose())
	 * zostawiamy, bo juz sie przy tym narobilismy... */

	Matrix3d dM = Matrix3d::Zero();
    const Vector2d s1C = Rot(alphaAbs(id)) * input.pickBodyType(id).dimensions.at(_sAB_);
	const double om = dAlphaAbs(id);
	const double m  = input.pickBodyType(id).m;

	dM.block(2, 0, 1, 2) = m * (-1) * s1C.transpose() * om;
	dM.block(0, 2, 2, 1) = m * (-1) * s1C * om;

	return dM;
}

/* default constructor needed to call vector<Assembly>.resize() */
Assembly::Assembly() :
    ksi(ksi_coefs()) ,
    Q1Acc(Vector3d()) ,
    S12(Matrix3d()) ,
    ptrAsmA(nullptr), ptrAsmB(nullptr)  { }
	
/* copy assignment needed to call std::copy(vectors in a tree). The default is 'delete' */
Assembly& Assembly::operator=(const Assembly& A) {
	ksi = A.ksi;
	Q1Acc = A.Q1Acc;
	S12 = A.S12;
	ptrAsmA = A.ptrAsmA;
	ptrAsmB = A.ptrAsmB;
	return *this;
}

Assembly::Assembly(const int id, const VectorXd& alphaAbs, 
            const VectorXd& pjoint, const double& u, const _input_& input) :
    ksi(ksi_coefs(id, alphaAbs, pjoint, input)) ,
    Q1Acc(Q1_init(id, alphaAbs, u, input)) ,
    S12(SAB("s12", id, alphaAbs, input)) ,
    ptrAsmA(nullptr), ptrAsmB(nullptr)  { }

Assembly::Assembly(Assembly& AsmA, Assembly& AsmB) :
    ksi(ksi_coefs(AsmA.ksi, AsmB.ksi)) ,
    Q1Acc(AsmA.Q1Acc + AsmA.S12 * AsmB.Q1Acc) ,
    S12(AsmA.S12 * AsmB.S12) ,
    ptrAsmA(&AsmA), ptrAsmB(&AsmB)  { }

void Assembly::connect_base_body() {
    Matrix<double, 3, 2> D;
    D << 0, 0, 1, 0, 0, 1;  // D_box is hardcoded
	Matrix2d c = - D.transpose() * ksi.k11() * D;
	T1 = D * c.ldlt().solve(D.transpose()) * ksi.k10();
	T2 << 0.0, 0.0, 0.0;
	Q1Art = Q1Acc;
	Q2Art << 0.0, 0.0, 0.0;
}

void Assembly::connect_base_bodyAcc() {
    Matrix<double, 3, 2> D;
    D << 0, 0, 1, 0, 0, 1;  // D_box is hardcoded
	Matrix2d c = - D.transpose() * ksi.k11() * D;
	L1 = D * c.ldlt().solve(D.transpose()) * ksiAcc_10;
	L2 << 0.0, 0.0, 0.0;
}

void Assembly::disassembleAll() {
    Matrix<double, 3,2> D;
    D << 1, 0, 0, 1, 0, 0; // D_rot is hardcoded
	Matrix2d C = -D.transpose() * (ptrAsmB->ksi.k11() + ptrAsmA->ksi.k22()) * D;
	Matrix3d W =  D * C.ldlt().solve(D.transpose());
	Vector2d b =  D.transpose() * (ptrAsmB->ksi.k10() - ptrAsmA->ksi.k20());
	Vector3d beta = D * C.ldlt().solve(b);

	ptrAsmB->T1 = W * ptrAsmB->ksi.k12() * T2 - W * ptrAsmA->ksi.k21() * T1 + beta;
	ptrAsmA->T2 = (-1) * ptrAsmB->T1;
	ptrAsmA->T1 = T1;
	ptrAsmB->T2 = T2;

    ptrAsmB->Q1Art =  ptrAsmB->Q1Acc - ptrAsmB->S12 * Q2Art;
	ptrAsmA->Q2Art = -ptrAsmB->Q1Art;
	ptrAsmA->Q1Art =  Q1Art;
	ptrAsmB->Q2Art =  Q2Art;
}

Vector3d Assembly::calculate_V1() const {
	return ksi.k11()*T1 + ksi.k12()*T2 + ksi.k10();
}

Vector3d Assembly::calculate_V2() const {
	return ksi.k21()*T1 + ksi.k22()*T2 + ksi.k20();
}

Vector3d Assembly::calculate_dV1() const {
	return ksi.k11()*L1 + ksi.k12()*L2 + ksiAcc_10;
}

Vector3d Assembly::calculate_dV2() const {
	return ksi.k21()*L1 + ksi.k22()*L2 + ksiAcc_20;
}

void Assembly::setAll(const Assembly& Asm) {
	T1 = Asm.T1;
	T2 = Asm.T2;
	Q1Art = Asm.Q1Art;
	Q2Art = Asm.Q2Art;
}

void Assembly::setKsiAcc(const int id, const VectorXd& alphaAbs, const VectorXd& dAlphaAbs, const MatrixXd& P1art, const _input_& input) {
	const Matrix3d S12 = SAB("s12", id, alphaAbs, input);
	/* note: P1art.cols() = Nbodies+1, where last column are zeros */
	const Vector3d P1 = P1art.col(id) - S12 * P1art.col(id+1); // = S12 * P2 (mnozymy przez dS wiec wychodzi na to samo)
	const Vector3d Q1 = Q1Art + S12 * Q2Art;
	const Vector3d Q2 = SAB("s21", id, alphaAbs, input) * Q1;
	const Matrix3d dSc1 = -1.0 * dSAB("s1C", id, alphaAbs, dAlphaAbs, input);
	const Matrix3d dSc2 = -1.0 * dSAB("s2C", id, alphaAbs, dAlphaAbs, input);

	const Matrix3d dM1 = massMatrixDot("s1C", id, alphaAbs, dAlphaAbs, input);
	const Matrix3d dM2 = massMatrixDot("s2C", id, alphaAbs, dAlphaAbs, input);
	const Matrix3d& M1_inv = ksi.i11;	// slight performance hack: no need to define M1 and it's inverse 
	const Matrix3d& M2_inv = ksi.i22;

	ksiAcc_10 =  M1_inv * (Q1 - dSc1 * P1 - dM1 * calculate_V1());
	ksiAcc_20 =  M2_inv * (Q2 - dSc2 * P1 - dM2 * calculate_V2());
}

void Assembly::assembleAcc(const Assembly& AsmA, const Assembly& AsmB) {
    Matrix<double, 3,2> D; // hardcoded, poniewaz D_trans wystepuje tylko w BBC
    D << 1, 0, 0, 1, 0, 0; // D_rot
    Matrix2d C  = - D.transpose() * (AsmB.ksi.k11() + AsmA.ksi.k22()) * D;
    Vector2d b  =   D.transpose() * (AsmB.ksiAcc_10 - AsmA.ksiAcc_20);
    Vector3d beta = D * C.ldlt().solve(b);

    ksiAcc_10 =  AsmA.ksiAcc_10 - AsmA.ksi.k12() * beta;
    ksiAcc_20 =  AsmB.ksiAcc_20 + AsmB.ksi.k21() * beta;
}

void Assembly::assembleAcc(const Assembly& AsmA) {
	ksiAcc_10 = AsmA.ksiAcc_10;
	ksiAcc_20 = AsmA.ksiAcc_20;
}

void Assembly::disassembleAcc() {
    Matrix<double, 3,2> D;
    D << 1, 0, 0, 1, 0, 0; // D_rot is hardcoded
	Matrix2d C = -D.transpose() * (ptrAsmB->ksi.k11() + ptrAsmA->ksi.k22()) * D;
	Matrix3d W =  D * C.ldlt().solve(D.transpose());
	Vector2d b =  D.transpose() * (ptrAsmB->ksiAcc_10 - ptrAsmA->ksiAcc_20);
	Vector3d beta = D * C.ldlt().solve(b);

	ptrAsmB->L1 = W * ptrAsmB->ksi.k12() * L2 - W * ptrAsmA->ksi.k21() * L1 + beta;
	ptrAsmA->L2 = (-1.0) * ptrAsmB->L1;
	ptrAsmA->L1 = L1;
	ptrAsmB->L2 = L2;
}

void Assembly::setAcc(const Assembly& Asm) {
	L1 = Asm.L1;
	L2 = Asm.L2;
}

/* ADJOINT ASSEMBLY */
/* copy assignment needed to call std::copy(vectors in a tree). The default is 'delete' */
AssemblyAdj& AssemblyAdj::operator=(const AssemblyAdj& A) {
	ksi = A.ksi;
	ptrAsmA = A.ptrAsmA;
	ptrAsmB = A.ptrAsmB;
	return *this;
}

/* default constructor needed to call vector<AssemblyAdj>.resize() */
AssemblyAdj::AssemblyAdj() : 
	ksi(ksi_coefs()) ,
    ptrAsmA(nullptr), ptrAsmB(nullptr)  { }

AssemblyAdj::AssemblyAdj(const int id, const dataAbsolute& data, const double& u, const _input_& input) : 
	ksi(ksi_coefs(id, data, u, input)) ,
    ptrAsmA(nullptr), ptrAsmB(nullptr)  { }

AssemblyAdj::AssemblyAdj(AssemblyAdj& AsmA, AssemblyAdj& AsmB) :
    ksi(ksi_coefs(AsmA.ksi, AsmB.ksi)) ,
    ptrAsmA(&AsmA), ptrAsmB(&AsmB)  { }

void AssemblyAdj::connect_base_body() {
    Matrix<double, 3, 2> D;
    D << 0, 0, 1, 0, 0, 1;  // D_box is hardcoded
	Matrix2d c = - D.transpose() * ksi.k11() * D;
	T1 = D * c.ldlt().solve(D.transpose()) * ksi.k10();
	T2 << 0.0, 0.0, 0.0;
}

void AssemblyAdj::disassemble() {
    Matrix<double, 3,2> D;
    D << 1, 0, 0, 1, 0, 0; // D_rot is hardcoded
	Matrix2d C = -D.transpose() * (ptrAsmB->ksi.k11() + ptrAsmA->ksi.k22()) * D;
	Matrix3d W =  D * C.ldlt().solve(D.transpose());
	Vector2d b =  D.transpose() * (ptrAsmB->ksi.k10() - ptrAsmA->ksi.k20());
	Vector3d beta = D * C.ldlt().solve(b);

	ptrAsmB->T1 = W * ptrAsmB->ksi.k12() * T2 - W * ptrAsmA->ksi.k21() * T1 + beta;
	ptrAsmA->T2 = (-1) * ptrAsmB->T1;
	ptrAsmA->T1 = T1;
	ptrAsmB->T2 = T2;
}

void AssemblyAdj::setAll(const AssemblyAdj& Asm) {
	T1 = Asm.T1;
	T2 = Asm.T2;
}

Vector3d AssemblyAdj::calculate_dETA1() const {
	return ksi.k11()*T1 + ksi.k12()*T2 + ksi.k10();
}

Vector3d AssemblyAdj::calculate_dETA2() const {
	return ksi.k21()*T1 + ksi.k22()*T2 + ksi.k20();
}