#include "include/input.h"
#include "include/utils.h"
#include "include/assembly.h"
#include "Eigen/Dense"

#include <iostream>

using namespace Eigen;
using std::cout;
using std::endl;


int main(int argc, char* argv[]) {
    // const int Nbodies = argv[1];
    const int Nbodies = 4;
    _input_ input = _input_(Nbodies);
    
    VectorXd alphaAbs = joint2AbsAngles(input.alpha0);
    Assembly A(0, alphaAbs, input.pjoint0, input);
    Assembly B(1, alphaAbs, input.pjoint0, input);
    Assembly C(A, B);

    cout << " ksi: "     << endl << C.ksi.k12() << endl;
    cout << " Q1Acc: "   << endl << C.Q1Acc << endl;
    cout << " S12 link " << endl << C.ptrAsmB->S12 << endl;
    cout << " Q2box  "   << endl << C.ptrAsmA->Q2Art << endl;

    return 0;
}