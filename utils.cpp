#include<project.h>

_input_::_input_(int _Nbodies_) : Nbodies(_Nbodies_), alfa0(_Nbodies_), dalfa0(_Nbodies_) {
    bodies.emplace_back("box");
    bodies.emplace_back("link");
    alfa0(0) = 0.0;
    alfa0.tail(_Nbodies_-1).setConstant(M_PI_4);
    dalfa0.setZero();

    cartInitialPosition_absolute << 0.0, 0.0, 0.0;
}

Vector3d _input_::getCartInitialPosition_absolute() {
    /**
     * Returns initial position of the cart (absolute coordinates)
     */
    return cartInitialPosition_absolute;
}

body::body(std::string type) {
    m = _m_;
    J = _J_;
    if ( ! type.compare("box") ) {
        s1C.setZero();
        s12.setZero();
        H << 1, 0, 0;
        D << 0, 0, 1, 0, 0, 1;
    }
    else if ( ! type.compare("link") ) {
        s1C << _L_/2, 0;
        s12 << _L_  , 0;
        H << 0, 0, 1;
        D << 1, 0, 0, 1, 0, 0;
    }
    else
        throw std::runtime_error("not supported body / joint");
}

Matrix2d Rot(double fi) {
	Matrix2d R;
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

VectorXd jointToAbsoluteCoordinates(const VectorXd jointCoordsAlpha, _input_ input) {
    /**
     * Converts joint coordinates to absolute coordinates (3 * number_of_bodies)
     */
    VectorXd absoluteCoordinates(3 * input.Nbodies);
    absoluteCoordinates.block(0, 0, 3, 1) = input.getCartInitialPosition_absolute();

    for(int i = 1; i < input.Nbodies; i++)
    {
        absoluteCoordinates.block((3*i + 0), 0, 2, 1) = absoluteCoordinates.block((3*(i-1) + 0), 0, 2, 1) + Rot(jointCoordsAlpha(i)) * input.bodies[i-1].s12;
        absoluteCoordinates(3*i + 2) = absoluteCoordinates(3*(i-1) + 2) + jointCoordsAlpha(i);
    }

    return absoluteCoordinates;
}