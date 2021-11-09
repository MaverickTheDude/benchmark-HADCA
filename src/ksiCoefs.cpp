#include "../include/ksiCoefs.h"
#include "../include/solution.h"
#include "../include/input.h"
#include "../include/utils.h"
#include "../Eigen/Dense"
#include "../include/adjoint.h"

#include <iostream>

using namespace Eigen;

ksi_coefs::ksi_coefs(const int id, const VectorXd& alphaAbs, const VectorXd& pjoint, const _input_& input) {
    /* konstruktor tworzy wsp. ksi dla kazdego assembly odpowiadajacego 
        * fizycznemu czlonowi na poziomie lisci. Note: alphaAbs(alphaJoint) 
        * obliczyne w caller function, pjoint zawiera ZLACZOWE pedy  */
    Matrix3d S12 = SAB("s12", id, alphaAbs, input);
    const Matrix3d S21 = SAB("s21", id, alphaAbs, input);
    const Matrix3d S1C = SAB("s1C", id, alphaAbs, input);
    const Matrix3d S2C = SAB("s2C", id, alphaAbs, input);
    const Matrix3d Mc = massMatrix(id, input);
    const Matrix3d M1 = S1C * Mc * S1C.transpose();
    const Matrix3d M2 = S2C * Mc * S2C.transpose();
    i11 = M1.inverse();
    i22 = M2.inverse();
    i12 = M1.ldlt().solve(S12);
    i21 = M2.ldlt().solve(S21);

    Vector3d rhs10, rhs20;
    rhs10 =       input.pickBodyType(id).H * pjoint(id);
    rhs20 = S21 * input.pickBodyType(id).H * pjoint(id);
    if (id < input.Nbodies-1) {
        // S12 = SAB("s12", id+1, alphaAbs, input); // wtf: tak jest w kodzie porr, co wyglada na spory blad. W Matlabie juz S12 jest S12(i), co zreszta wynika ze wzorow
        // note - aliasing below should be harmless: https://eigen.tuxfamily.org/dox/group__TopicAliasing.html
        rhs10 -= S12*input.pickBodyType(id+1).H*pjoint(id+1);
        rhs20 -= input.pickBodyType(id+1).H*pjoint(id+1);
    }
    i10 = M1.ldlt().solve(rhs10);
    i20 = M2.ldlt().solve(rhs20);
}

ksi_coefs& ksi_coefs::operator=(const ksi_coefs& ksi) {
    i11 = ksi.i11;
    i12 = ksi.i12;
    i21 = ksi.i21;
    i22 = ksi.i22;
    i10 = ksi.i10;
    i20 = ksi.i20;
    return *this;
}

ksi_coefs::ksi_coefs(const ksi_coefs& ksiA, const ksi_coefs& ksiB) {
    Matrix<double, 3,2> D; // hardcoded, poniewaz D_trans wystepuje tylko w BBC
    D << 1, 0, 0, 1, 0, 0; // D_rot
    Matrix2d C  = - D.transpose() * (ksiB.i11 + ksiA.i22) * D;
    Vector2d b  =   D.transpose() * (ksiB.i10 - ksiA.i20);
    Matrix3d W  =   D * C.ldlt().solve(D.transpose());
    Vector3d beta = D * C.ldlt().solve(b);

    i11 =  ksiA.i11 + ksiA.i12 * W * ksiA.i21;
    i22 =  ksiB.i22 + ksiB.i21 * W * ksiB.i12;
    i12 = -ksiA.i12 * W * ksiB.i12;
    i21 = -ksiB.i21 * W * ksiA.i21;
    i10 =  ksiA.i10 - ksiA.i12 * beta;
    i20 =  ksiB.i20 + ksiB.i21 * beta;
}

ksi_coefs::ksi_coefs(const int id, const dataAbsolute& data, const double& u, const _input_& input) {
    /* konstruktor tworzy wsp. ksi dla kazdego assembly odpowiadajacego 
     * fizycznemu czlonowi na poziomie lisci.   */
    const double alphaAbs = data.alphaAbs(id);
    const double dalphaAbs = data.dAlphaAbs(id);
    const double d2alphaAbs = data.d2AlphaAbs(id);
    Matrix3d S12 = SAB("s12", id, alphaAbs, input);
    const Matrix3d S21 = SAB("s21", id, alphaAbs, input);
    const Matrix3d S1C = SAB("s1C", id, alphaAbs, input);
    const Matrix3d S2C = SAB("s2C", id, alphaAbs, input);
    const Matrix3d dS21 = dSAB("s21", id, alphaAbs, dalphaAbs, input);
    const Matrix3d d2S21 = d2SAB("s21", id, alphaAbs, dalphaAbs, d2alphaAbs, input);
    const Matrix3d Mc = massMatrix(id, input);
    const Matrix3d M1 = S1C * Mc * S1C.transpose();
    const Matrix3d M2 = S2C * Mc * S2C.transpose();
    i11 = M1.inverse();
    i22 = M2.inverse();
    i12 = M1.ldlt().solve(S12);
    i21 = M2.ldlt().solve(S21);

    Vector3d rhs10, rhs20;
    Vector3d eta = data.eta.segment(3*id, 3);
    Vector3d xi  = data.ksi.segment(3*id, 3);

    const double sign = -1.0; // d/d(t) = - d/d(tau) ==> the time goes backwards
    const Vector3d rhs = Adjoint::RHS3d(id, input, data.q, data.dq, data.d2q, data.lambda, (VectorXd(1) << u).finished(), eta, xi);
    rhs10 =       rhs;
    rhs20 = S21 * rhs;

    i10 = M1.ldlt().solve(rhs10);
    i20 = M2.ldlt().solve(rhs20) + sign*(-2*dS21.transpose()*eta - d2S21.transpose()*xi);
}