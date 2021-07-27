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
    q = jointToAbsoluteCoordinates(input.alpha0, input);

    Vector4d phi  = Phi(q, input);

    TEST_CHECK_(phi.norm() <= eps, "max error = %f", phi.norm());
}

void test_Fq(void) {
    _input_ input = _input_(2);
    VectorXd q(6);
    q = jointToAbsoluteCoordinates(input.alpha0, input);

    MatrixXd fd   = jacobianReal(Phi, q, input);
    MatrixXd Jac  = Jacobian(q, input);
    MatrixXd diff = fd - Jac;

    TEST_CHECK_(diff.norm() <= eps, "max error = %f", diff.norm());
}

void test_jointToAbsoluteCoordinates(void) {
    _input_ input = _input_(5);
    Vector3d jointCoordsAlpha(0.0, M_PI_4, M_PI_4);

    VectorXd absoluteCoords = jointToAbsoluteCoordinates(input.alpha0, input);

    VectorXd absoluteCoords_ideal(3 * 5);
    absoluteCoords_ideal << 0.0, 0.0, 0.0,
        0.0, 0.0, M_PI_4,
        _L_*cos(M_PI_4), _L_*sin(M_PI_4), M_PI_4 + M_PI_4,
        2*_L_*cos(M_PI_4), 2*_L_*sin(M_PI_4), M_PI_4 + M_PI_4 + M_PI_4,
        3*_L_*cos(M_PI_4), 3*_L_*sin(M_PI_4), M_PI_4 + M_PI_4 + M_PI_4 + M_PI_4;

    VectorXd diff = absoluteCoords_ideal - absoluteCoords;
    TEST_CHECK_(diff.norm() <= eps, "max error = %f", diff.norm());
}


TEST_LIST = {
   { "phi", test_Phi },
   { "jacobian", test_Fq },
   { "joint2Abs", test_jointToAbsoluteCoordinates },
   { NULL, NULL }     /* zeroed record marking the end of the list */
};