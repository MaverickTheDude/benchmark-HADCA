#include<iostream>
#include<project.h>
using namespace Eigen;
using std::cout;
using std::endl;


int main(int argc, char* argv[]) {
    _input_ input = _input_();
    Vector2d r1(2.0, 2.0), r2(4.0, -1.0);
    double fi1(0.0), fi2(M_PI/2);
    VectorXd q(6);
    q << r1, fi1, r2, fi2;

    Vector4d phi = Phi(q, input);

    MatrixXd fd = jacobianComplex(Phi, q, input);

    MatrixXd Jac = Jacobian(q, input);

    MatrixXd diff = fd - Jac;

    cout << "q = " << endl << q << endl << endl;
    cout << "phi = " << endl << phi << endl << endl;
    cout << "Phi_qFD = " << endl << fd << endl << endl;
    cout << "Phi_q = " << endl << Jac << endl << endl;
    cout << "diff = " << endl << diff << endl << endl;

    return 0;
}