#include<iostream>
#include<project.h>
using namespace Eigen;
using std::cout;
using std::endl;



int main(int argc, char* argv[]) {
    // const int Nbodies = argv[1];
    const int Nbodies = 2;
    _input_ input = _input_(Nbodies);
    Vector2d r1(0.0, 0.0), r2(_L_/sqrt(2), _L_/sqrt(2));
    double fi1(0.0), fi2(M_PI/4);
    VectorXd q(6);
    q << r1, fi1, r2, fi2;

    MatrixXd fd = jacobianReal(Phi, q, input);

    MatrixXd Jac = Jacobian(q, input);

    MatrixXd diff = fd - Jac;

    body b1("box");
    body b2("link");

//     cout << b1.sCM << endl << b1.H << endl << b1.D << b1.m;
//     cout << endl << endl;
//     cout << b2.sCM << endl << b2.H << endl << b2.D << b2.m; 

    // cout << "q = " << endl << q << endl << endl;
    // cout << "phi = " << endl << phi << endl << endl;
    cout << "Phi_qFD = " << endl << fd << endl << endl;
    // cout << "Phi_q = " << endl << Jac << endl << endl;
    // cout << "diff = " << endl << diff << endl << endl;

    return 0;
}