#include<project.h>

VectorXd Phi(const VectorXd& q, const _input_& input) {
    VectorXd phi(4);
    phi.head(2) << 0.0, 0.0;
    Vector2d r1, r2;
    double fi1, fi2;
    r1 = q.segment(0,2);    fi1 = q(2);
    r2 = q.segment(3,2);    fi2 = q(5);
    phi.tail(2) = r1 + Rot(fi1)*input.sA1 - r2 - Rot(fi2)*input.sA2;
    return phi;
}

MatrixXd Jacobian(const VectorXd& q, const _input_& input) {
    const int n = q.size();
    const int Nf = input.Nconstr;
    MatrixXd Fq = MatrixXd::Zero(Nf, n);
    double fi1 = q(2);
    double fi2 = q(5);
    Fq.block(2,0,2,2) =  I;
    Fq.block(2,2,2,1) =  Om*Rot(fi1)*input.sA1;
    Fq.block(2,3,2,2) = -I;
    Fq.block(2,5,2,1) = -Om*Rot(fi2)*input.sA2;
    return Fq;
}

// Test zespolonych roznic skonczonych. Ok! Ale usuwamy, bo RealFD wygodniejsze
VectorXcd Phi(const VectorXcd& q, const _input_& input) {
    VectorXcd phi(4);
    phi.head(2) << 0.0, 0.0;
    Vector2cd r1, r2;
    std::complex<double> fi1(q(2));
    std::complex<double> fi2(q(5));
    r1 = q.segment(0,2);
    r2 = q.segment(3,2);
    phi.tail(2) = r1 + Rot(fi1)*input.sA1 - r2 - Rot(fi2)*input.sA2;
    return phi;
}