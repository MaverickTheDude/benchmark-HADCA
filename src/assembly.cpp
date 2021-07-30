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