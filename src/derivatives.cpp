#include "../include/derivatives.h"

#include "../include/input.h"
#include "../include/utils.h"
#include "../include/constants.h"
#include "../Eigen/Dense"

using namespace Eigen;

VectorXd Phi(const VectorXd& q, const _input_& input) {
    const int Nf = input.Nconstr;
    Vector2d v; v << 0.0, 1.0;
    VectorXd phi(Nf);
    Vector2d r1, r2;
    double fi1; //, fi2;
    r1 = q.segment(0,2);    fi1 = q(2);
    r2 = q.segment(3,2);    //fi2 = q(5);
    Vector2d s1CB =  input.pickBodyType(0).s1C;
    Vector2d s12B =  input.pickBodyType(0).s12;

    // para postepowa
    phi(0) = v.transpose() * (r1 + Rot(fi1) * s1CB);
    phi(1) = fi1;
    // para obrotowa (to do: petla for) ... na razie dzialamy na samych zerach
    phi.tail(2) = r1 + Rot(fi1) * s12B - r2;
    return phi;
}

MatrixXd Jacobian(const VectorXd& q, const _input_& input) {
    Vector2d v; v << 0.0, 1.0;
    const int n = q.size();
    const int Nf = input.Nconstr;
    MatrixXd Fq = MatrixXd::Zero(Nf, n);
    double fi1 = q(2);
    // double fi2 = q(5);
    Vector2d s1CB = input.pickBodyType(0).s1C;
    Vector2d s12B = input.pickBodyType(0).s12;

    Fq.block(0,0,1,2) = v.transpose();
    Fq(0,2) = v.transpose() * Om*Rot(fi1)*s1CB;
    Fq(1,2) = 1;

    Fq.block(2,0,2,2) =  I;
    Fq.block(2,2,2,1) =  Om*Rot(fi1)*s12B;
    Fq.block(2,3,2,2) = -I;
    // Fq.block(2,5,2,1) = -Om*Rot(fi2)*sA2;
    return Fq;
}