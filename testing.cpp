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

void test_jointToAbsoluteVelocity(void) {
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


TEST_LIST = {
   { "phi", test_Phi },
   { "jacobian", test_Fq },
   { "joint2AbsPosition", test_jointToAbsolutePosition },
   { "joint2AbsVelocity", test_jointToAbsoluteVelocity },
   { NULL, NULL }     /* zeroed record marking the end of the list */
};