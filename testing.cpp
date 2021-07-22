/*
Build with: meson test
check individual tests:
$ ./test_example                # Runs all tests in the suite
$ ./test_example test1 test2    # Runs only tests specified
$ ./test_example --skip test3   # Runs all tests but those specified
*/

#include "acutest.h"
#include "project.h"
static const double eps = 1e-14;


void test_Phi(void) {
    _input_ input = _input_();
    Vector2d r1(2.0, 2.0), r2(4.0, -1.0);
    double fi1(0.0), fi2(M_PI/2);
    VectorXd q(6);
    q << r1, fi1, r2, fi2;

    Vector4d phi  = Phi(q, input);
    TEST_CHECK_(phi.norm() <= eps, "max error = %f", phi.norm());
}

void test_Fq(void) {
    _input_ input = _input_();
    Vector2d r1(2.0, 2.0), r2(4.0, -1.0);
    double fi1(0.0), fi2(M_PI/2);
    VectorXd q(6);
    q << r1, fi1, r2, fi2;

    MatrixXd fd   = jacobianComplex(Phi, q, input);
    MatrixXd Jac  = Jacobian(q, input);
    MatrixXd diff = fd - Jac;

    TEST_CHECK_(diff.norm() <= eps, "max error = %f", diff.norm());
}

TEST_LIST = {
   { "test1", test_Phi },
   { "test2", test_Fq },
   { NULL, NULL }     /* zeroed record marking the end of the list */
};