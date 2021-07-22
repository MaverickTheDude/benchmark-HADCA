#include<project.h>

_input_::_input_() : sA1(2.0, 0.0), sA2(3.0, 0.0) {    }

Matrix2d Rot(double fi) {
	Matrix2d R;
	R << cos(fi), -sin(fi), sin(fi), cos(fi);
	return R;
}

Matrix2cd Rot(std::complex<double> fi) {
	Matrix2cd R;
	R << cos(fi), -sin(fi), sin(fi), cos(fi);
	return R;
}

MatrixXd jacobianReal(VectorXd (*fun)(const VectorXd&, const _input_&), VectorXd q0, _input_ input) {
    const double h = 1e-8;
    const int n = q0.size();
    const int Nf = input.Nconstr;

    MatrixXd Fun_q(Nf, n);
    for (int i = 0; i < n; i++) {
        VectorXd delta = ArrayXd::Zero(n);
        delta(i) = h;
        VectorXd qFF = q0 + delta;
        VectorXd qRV = q0 - delta;
        MatrixXd funForward = fun(qFF, input);
        MatrixXd funRev     = fun(qRV, input);
        Fun_q.col(i) = (funForward - funRev) / (2*h);
    }

    return Fun_q;
}

// Test zespolonych roznic skonczonych. Ok! Ale usuwamy, bo RealFD wygodniejsze
MatrixXd jacobianComplex(VectorXcd (*fun)(const VectorXcd&, const _input_&), VectorXd q0, _input_ input) {
    std::complex<double> j(0,1);
    const double h = 1e-8;
    const int n = q0.size();
    const int Nf = input.Nconstr;

    MatrixXd Fun_q = MatrixXd::Zero(Nf, n);
    for (int i = 0; i < n; i++) {
        
        VectorXcd delta = ArrayXcd::Zero(n);
        delta(i) = h*j;
        VectorXcd qFF = q0 + delta;
        MatrixXcd funForward = fun(qFF, input);
        Fun_q.col(i) = funForward.imag() / h;
    }

    return Fun_q;
}