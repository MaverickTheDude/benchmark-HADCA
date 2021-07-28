/*
Build with: meson test
check individual tests:
$ ./test_example                # Runs all tests in the suite
$ ./test_example test1 test2    # Runs only tests specified
$ ./test_example --skip test3   # Runs all tests but those specified
*/

#include "acutest.h"
#include "project.h"
#include <iostream>
static const double eps = 1e-7;


void test_Phi(void) {
    _input_ input = _input_(2);
    VectorXd q(6);
    q = jointToAbsolutePosition(input.alpha0, input);

    Vector4d phi  = Phi(q, input);

    TEST_CHECK_(phi.norm() <= eps, "max error = %f", phi.norm());
}

void test_Fq(void) {
    _input_ input = _input_(2);
    VectorXd q(6);
    q = jointToAbsolutePosition(input.alpha0, input);

    MatrixXd fd   = jacobianReal(Phi, q, input);
    MatrixXd Jac  = Jacobian(q, input);
    MatrixXd diff = fd - Jac;

    TEST_CHECK_(diff.norm() <= eps, "max error = %f", diff.norm());
}

void test_jointToAbsolutePosition(void) {
    _input_ input = _input_(5);

    VectorXd absoluteCoords = jointToAbsolutePosition(input.alpha0, input);

    VectorXd absoluteCoords_ideal(3 * 5);
    absoluteCoords_ideal << 0.0, 0.0, 0.0,
        0.0, 0.0, M_PI_4,
        _L_*cos(M_PI_4), _L_*sin(M_PI_4), M_PI_4 + M_PI_4,
        2*_L_*cos(M_PI_4), 2*_L_*sin(M_PI_4), M_PI_4 + M_PI_4 + M_PI_4,
        3*_L_*cos(M_PI_4), 3*_L_*sin(M_PI_4), M_PI_4 + M_PI_4 + M_PI_4 + M_PI_4;

    VectorXd diff = absoluteCoords_ideal - absoluteCoords;
    TEST_CHECK_(diff.norm() <= eps, "max error = %f", diff.norm());
}

void test_jointToAbsoluteVelocity1(void) {
    _input_ input = _input_(5);
    VectorXd alpha = VectorXd::Zero(5);
    VectorXd dalpha(5);
    dalpha << 1.0, M_PI_2, -M_PI_2, M_PI_2, -M_PI_2;

    VectorXd dq = jointToAbsoluteVelocity(alpha, dalpha, input);

    VectorXd dq_ideal(3 * 5);
    dq_ideal << 1.0, 0.0, 0.0,
        1.0, 0.0, M_PI_2,
        1.0, M_PI_2, 0.0,
        1.0, 0, M_PI_2,
        1.0, M_PI_2, 0.0;

    TEST_CHECK_((dq - dq_ideal).norm() <= eps, "max error = %f", (dq - dq_ideal).norm());
}


void test_jointToAbsoluteVelocity2(void) {
    _input_ input = _input_(4);
    VectorXd dq = jointToAbsoluteVelocity(input.alpha0, input.dalpha0, input);
    VectorXd alphaAbs = joint2AbsAngles(input.alpha0);
    const Matrix3d S12_0 = SAB("s12", 0, alphaAbs, input);
    const Matrix3d S12_1 = SAB("s12", 1, alphaAbs, input);
    const Matrix3d S12_2 = SAB("s12", 2, alphaAbs, input);
    const Matrix3d s12[] = {S12_0, S12_1, S12_2};
    
// jak test sie powiedzie, mozna wsadzic do petli for
    int i = 0;
    Vector3d Vref;
    Vref << input.dalpha0(i), 0.0, 0.0; // == H0 * dalfa0
    Vector3d test = dq.segment(3*i, 3);
    Vector3d diff = test - Vref; 
    TEST_CHECK_(diff.norm() <= eps, "error = %f, value = [%f, %f, %f]", diff.norm(), Vref(0), Vref(1), Vref(2));

    for(int i = 1; i < 4; i++)
    {
        test = dq.segment(3*i, 3);
        Vref = s12[i-1].transpose() * Vref + input.pickBodyType(i).H * input.dalpha0(i);
        diff = test - Vref;
        TEST_CHECK_(diff.norm() <= eps, "error = %f, value = [%f, %f, %f]", diff.norm(), Vref(0), Vref(1), Vref(2));
    }
}


// void test_SetPJoint(void) {
//     int id = 2; // mozna testowac od id = 0 do id = Nbodies - 2
//     _input_ input = _input_(4);
//     VectorXd sigmaStacked = input.getPJointAndSigma().second;
//     VectorXd dq = jointToAbsoluteVelocity(input.alpha0, input.dalpha0, input);
//     VectorXd alphaAbs = joint2AbsAngles(input.alpha0);
//     const Matrix3d S12 = SAB("s12", id, alphaAbs, input);
//     const Matrix3d S21 = SAB("s21", id, alphaAbs, input);
//     const Matrix3d S1C = SAB("s1C", id, alphaAbs, input);
//     const Matrix3d S2C = SAB("s2C", id, alphaAbs, input);
//     const Matrix3d Mc = massMatrix(id, input);
//     const Matrix3d M1 = S1C * Mc * S1C.transpose();
//     const Matrix3d M2 = S2C * Mc * S2C.transpose();

//     Vector3d V1 = dq.segment(3*id, 3);
//     Vector3d V2 = S12.transpose() * V1;
//     Vector3d H = input.pickBodyType(id).H;
//     Vector3d D = input.pickBodyType(id).D;
//     Vector3d Hnext = input.pickBodyType(id+1).H; // note: upewnic sie, ze sprawdzamy dla nie-ostatniego czlonu
//     Vector3d Dnext = input.pickBodyType(id+1).D;
//     Vector2d sigma = sigmaStacked.segment(2*id, 2);
//     Vector2d sigmaNext = sigmaStacked.segment(2*(id+1), 2);
    
//     Vector3d RHS1 = H * input.pjoint0(id) + D * sigma - S12 * (Hnext * input.pjoint0(id+1) + Dnext * sigmaNext);
//     Vector3d RHS2 = S21 * (H * input.pjoint0(id) + D * sigma) - (Hnext * input.pjoint0(id+1) + Dnext * sigmaNext);
//     VectorXd diff1 = M1 * V1 - RHS1;
//     VectorXd diff2 = M2 * V2 - RHS2;
//     TEST_CHECK_(diff1.norm() <= eps, "error = %f, value = [%f, %f, %f]", diff1.norm(), RHS1(1), RHS1(2), RHS1(3));
//     TEST_CHECK_(diff2.norm() <= eps, "error = %f, value = [%f, %f, %f]", diff2.norm(), RHS2(2), RHS2(2), RHS2(3));
// }

// void test_SetAssembly(void) {
//     int id = 2; // mozna testowac od id = 0 do id = Nbodies - 2
//     _input_ input = _input_(4);
//     VectorXd sigmaStacked = input.getPJointAndSigma().second;
//     VectorXd dq = jointToAbsoluteVelocity(input.alpha0, input.dalpha0, input);
//     VectorXd alphaAbs = joint2AbsAngles(input.alpha0);
//     Assembly body(0, alphaAbs, input.pjoint0, input);
//     Vector3d T1A = input.pickBodyType(id).D   * sigmaStacked.segment(2*id, 2);
//     // minus, poniewaz sigma ma znak + w interfejsie 1
//     Vector3d T2A = - input.pickBodyType(id+1).D * sigmaStacked.segment(2*(id+1), 2); // note: upewnic sie, ze sprawdzamy dla nie-ostatniego czlonu

//     VectorXd RHS1 = body.ksi.k11() * T1A + body.ksi.k12() * T2A + body.ksi.k10();
//     VectorXd RHS2 = body.ksi.k21() * T1A + body.ksi.k22() * T2A + body.ksi.k10();

//     const Matrix3d S12 = SAB("s12", id, alphaAbs, input);
//     Vector3d V1 = dq.segment(3*id, 3);
//     Vector3d V2 = S12.transpose() * V1;

//     Vector3d diff1 = V1 - RHS1;
//     Vector3d diff2 = V2 - RHS2;

//     TEST_CHECK_(diff1.norm() <= eps, "error = %f, value = [%f, %f, %f]", diff1.norm(), RHS1(1), RHS1(2), RHS1(3));
//     TEST_CHECK_(diff2.norm() <= eps, "error = %f, value = [%f, %f, %f]", diff2.norm(), RHS2(1), RHS2(2), RHS2(3));
// }


TEST_LIST = {
   { "phi", test_Phi },
   { "jacobian", test_Fq },
   { "joint2AbsPosition", test_jointToAbsolutePosition },
   { "joint2AbsVelocity (Test 1)", test_jointToAbsoluteVelocity1 },
   { "joint2AbsVelocity (Test 2)", test_jointToAbsoluteVelocity2 },
   { NULL, NULL }     /* zeroed record marking the end of the list */
};