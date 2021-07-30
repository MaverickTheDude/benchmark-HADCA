/*
Build with: meson test
check individual tests:
$ ./test_example                # Runs all tests in the suite
$ ./test_example test1 test2    # Runs only tests specified
$ ./test_example --skip test3   # Runs all tests but those specified
*/

/**
 * We put global includes at the top:
 *  - acutest.h - acutest library
 *  - iostream - in case a verbose debugging is needed
 * 
 * Other libraries / headers are included at the top of each test function
 * as tailored to its needs. This way it is easier to maintain these test
 * functions into other files, just in case.
 */
#include "Acutest/acutest.h"

#include <iostream>
static const double eps = 1e-7;


#include "include/input.h"
#include "Eigen/Dense"
#include "include/derivatives.h"
#include "include/utils.h"

using namespace Eigen;

void test_Phi(void) {
    _input_ input = _input_(2);
    VectorXd q(6);
    q = jointToAbsolutePosition(input.alpha0, input);

    Vector4d phi  = Phi(q, input);

    TEST_CHECK_(phi.norm() <= eps, "max error = %f", phi.norm());
}


#include "include/input.h"
#include "Eigen/Dense"
#include "include/utils.h"
#include "include/derivatives.h"

using namespace Eigen;

void test_Fq(void) {
    _input_ input = _input_(2);
    VectorXd q(6);
    q = jointToAbsolutePosition(input.alpha0, input);

    MatrixXd fd   = jacobianReal(Phi, q, input);
    MatrixXd Jac  = Jacobian(q, input);
    MatrixXd diff = fd - Jac;

    TEST_CHECK_(diff.norm() <= eps, "max error = %f", diff.norm());
}


#include "include/input.h"
#include "Eigen/Dense"
#include "include/utils.h"
#include "include/constants.h"

using namespace Eigen;

void test_jointToAbsolutePosition(void) {
    _input_ input = _input_(5);

    VectorXd alpha(input.Nbodies);
    alpha.tail(input.Nbodies - 1).setConstant(0);
    alpha(1) = M_PI_4;
    VectorXd absoluteCoords = jointToAbsolutePosition(alpha, input);

    VectorXd absoluteCoords_ideal(3 * 5);
    absoluteCoords_ideal << 0.0, 0.0, 0.0,
        0.0, 0.0, M_PI_4,
        _L_*cos(M_PI_4), _L_*sin(M_PI_4), M_PI_4,
        2*_L_*cos(M_PI_4), 2*_L_*sin(M_PI_4), M_PI_4,
        3*_L_*cos(M_PI_4), 3*_L_*sin(M_PI_4), M_PI_4;

    VectorXd diff = absoluteCoords_ideal - absoluteCoords;
    TEST_CHECK_(diff.norm() <= eps, "max error = %f", diff.norm());
}


#include "include/input.h"
#include "Eigen/Dense"
#include "include/utils.h"

using namespace Eigen;

void test_jointToAbsoluteVelocity1(void) {
    _input_ input = _input_(5);
    VectorXd alpha = VectorXd::Zero(5);
    VectorXd dalpha(5);
    dalpha << 1.0, M_PI_2, 0, 0, 0;

    VectorXd dq = jointToAbsoluteVelocity(alpha, dalpha, input);

    VectorXd dq_ideal(3 * 5);
    dq_ideal << 1.0, 0.0, 0.0,
        1.0, 0.0, M_PI_2,
        1.0, M_PI_2, M_PI_2,
        1.0, 2*M_PI_2, M_PI_2,
        1.0, 3*M_PI_2, M_PI_2;

    TEST_CHECK_((dq - dq_ideal).norm() <= eps, "max error = %f", (dq - dq_ideal).norm());
}


#include "include/input.h"
#include "Eigen/Dense"
#include "include/utils.h"

using namespace Eigen;

void test_jointToAbsoluteVelocity2(void) {
    _input_ input = _input_(4);
    VectorXd dq = jointToAbsoluteVelocity(input.alpha0, input.dalpha0, input);
    VectorXd alphaAbs = joint2AbsAngles(input.alpha0);
    const Matrix3d S12_0 = SAB("s12", 0, alphaAbs, input);
    const Matrix3d S12_1 = SAB("s12", 1, alphaAbs, input);
    const Matrix3d S12_2 = SAB("s12", 2, alphaAbs, input);
    const Matrix3d s12[] = {S12_0, S12_1, S12_2};
    
    int i = 0;
    Vector3d Vref;
    Vref << input.dalpha0(i), 0.0, 0.0; // == H0 * dalfa0
    Vector3d test = dq.segment(3*i, 3);
    Vector3d diff = test - Vref; 
    TEST_CHECK_(diff.norm() <= eps, "result = [%f, %f, %f], ref. result = [%f, %f, %f] in bodyId = %d",
        test(0), test(1), test(2), 
        Vref(0), Vref(1), Vref(2),
        i);

    for(int i = 1; i < 4; i++)
    {
        test = dq.segment(3*i, 3);
        Vref = s12[i-1].transpose() * Vref + input.pickBodyType(i).H * input.dalpha0(i);
        diff = test - Vref;
        TEST_CHECK_(diff.norm() <= eps, "result = [%f, %f, %f], ref. result = [%f, %f, %f] in bodyId = %d",
            test(0), test(1), test(2), 
            Vref(0), Vref(1), Vref(2),
            i);
    }
}


#include "include/input.h"
#include "Eigen/Dense"
#include "include/utils.h"

using namespace Eigen;

void test_SetPJoint(void) {
    _input_ input = _input_(4);

    for( int id = input.Nbodies - 2; id >= 0; id--)
    {
        VectorXd sigmaStacked = input.sigma0;
        VectorXd dq = jointToAbsoluteVelocity(input.alpha0, input.dalpha0, input);
        VectorXd alphaAbs = joint2AbsAngles(input.alpha0);
        const Matrix3d S12 = SAB("s12", id, alphaAbs, input);
        const Matrix3d S21 = SAB("s21", id, alphaAbs, input);
        const Matrix3d S1C = SAB("s1C", id, alphaAbs, input);
        const Matrix3d S2C = SAB("s2C", id, alphaAbs, input);
        const Matrix3d Mc = massMatrix(id, input);
        const Matrix3d M1 = S1C * Mc * S1C.transpose();
        const Matrix3d M2 = S2C * Mc * S2C.transpose();

        Vector3d V1 = dq.segment(3*id, 3);
        Vector3d V2 = S12.transpose() * V1;
        Vector3d H = input.pickBodyType(id).H;
        MatrixXd D = input.pickBodyType(id).D;
        Vector3d Hnext = input.pickBodyType(id+1).H; // note: upewnic sie, ze sprawdzamy dla nie-ostatniego czlonu
        MatrixXd Dnext = input.pickBodyType(id+1).D;
        Vector2d sigma = sigmaStacked.segment(2*id, 2);
        Vector2d sigmaNext = sigmaStacked.segment(2*(id+1), 2);
        
        Vector3d RHS1 = H * input.pjoint0(id) + D * sigma - S12 * (Hnext * input.pjoint0(id+1) + Dnext * sigmaNext);
        Vector3d RHS2 = S21 * (H * input.pjoint0(id) + D * sigma) - (Hnext * input.pjoint0(id+1) + Dnext * sigmaNext);
        Vector3d M1V1 = M1 * V1;
        Vector3d M2V2 = M2 * V2;
        VectorXd diff1 = M1V1 - RHS1;
        VectorXd diff2 = M2V2 - RHS2;
        TEST_CHECK_(diff1.norm() <= eps, "result = [%f, %f, %f], ref. result = [%f, %f, %f] in bodyId = %d",
            RHS1(0), RHS1(1), RHS1(2),
            M1V1(0), M1V1(1), M1V1(2),
            id);
        TEST_CHECK_(diff2.norm() <= eps, "result = [%f, %f, %f], ref. result = [%f, %f, %f] in bodyId = %d",
            RHS2(0), RHS2(1), RHS2(2),
            M2V2(0), M2V2(1), M2V2(2),
            id);
    }
}

#include "include/input.h"
#include "Eigen/Dense"
#include "include/utils.h"
#include "include/assembly.h"

using namespace Eigen;

void test_SetAssembly(void) {
    _input_ input = _input_(4);

    for( int id = 0; id <= input.Nbodies - 1; id++)
    {
        VectorXd sigmaStacked = input.sigma0;
        VectorXd dq = jointToAbsoluteVelocity(input.alpha0, input.dalpha0, input);
        VectorXd alphaAbs = joint2AbsAngles(input.alpha0);
        Assembly body(id, alphaAbs, input.pjoint0, input);
        Vector3d T1A = input.pickBodyType(id).D   * sigmaStacked.segment(2*id, 2);
        Vector3d T2A;
        if (id == input.Nbodies - 1)
            T2A = Vector3d::Zero(); // dla ostatniego czlonu
        else
            T2A = - input.pickBodyType(id+1).D * sigmaStacked.segment(2*(id+1), 2); // minus, poniewaz sigma ma znak + w interfejsie 1

        VectorXd RHS1 = body.ksi.k11() * T1A + body.ksi.k12() * T2A + body.ksi.k10();
        VectorXd RHS2 = body.ksi.k21() * T1A + body.ksi.k22() * T2A + body.ksi.k20();

        const Matrix3d S12 = SAB("s12", id, alphaAbs, input);
        Vector3d V1 = dq.segment(3*id, 3);
        Vector3d V2 = S12.transpose() * V1;

        Vector3d diff1 = V1 - RHS1;
        Vector3d diff2 = V2 - RHS2;

        TEST_CHECK_(diff1.norm() <= eps, "error1 = %f, \n Rvalue1 = [%f, %f, %f], \n Tvalue1 = [%f, %f, %f]", 
                    diff1.norm(), V1(0), V1(1), V1(2), RHS1(0), RHS1(1), RHS1(2));
        TEST_CHECK_(diff2.norm() <= eps, "error2 = %f, \n Rvalue2 = [%f, %f, %f], \n Tvalue2 = [%f, %f, %f]", 
                    diff2.norm(), V2(0), V2(1), V2(2), RHS2(0), RHS2(1), RHS2(2));
    }
}


TEST_LIST = {
   { "phi", test_Phi },
   { "jacobian", test_Fq },
   { "joint2AbsPosition", test_jointToAbsolutePosition },
   { "joint2AbsVelocity (Test 1)", test_jointToAbsoluteVelocity1 },
   { "joint2AbsVelocity (Test 2)", test_jointToAbsoluteVelocity2 },
   { "_input_.setPJointAndSigma", test_SetPJoint },
   { "setAssembly", test_SetAssembly },
   { NULL, NULL }     /* zeroed record marking the end of the list */
};