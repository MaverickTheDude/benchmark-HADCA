#include "../include/assembly.h"

#include "../include/input.h"
#include "../include/ksiCoefs.h"
#include "../include/utils.h"
#include "../Eigen/Dense"

using namespace Eigen;

Assembly::Assembly(const int id, const VectorXd& alphaAbs, 
            const VectorXd& pjoint, const _input_& input) :
    ksi(ksi_coefs(id, alphaAbs, pjoint, input)) ,
    Q1Acc(Q1_init(id, alphaAbs, input)) ,
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

void Assembly::setAll(const Assembly& Asm) {
	T1 = Asm.T1;
	T2 = Asm.T2;
	Q1Art = Asm.Q1Art;
	Q2Art = Asm.Q2Art;
}