#include<Eigen/Dense>
#include<map>
#include<vector>
using namespace Eigen;

const double _L_ = 1.0;
const double _m_ = 0.2;
const double _J_ = 0.1;
const Matrix2d I = Matrix2d::Identity();

class body {
public:
    body(std::string type);

public:
    Vector2d s1C;
    Vector2d s12;
    double m, J;
    Vector3d H;
    Matrix<double, 3, 2> D;
    std::map<std::string, Vector2d> dimensions;
};

class _input_ {
public:
    // _input_() {
    //     sA1(2.0, 0.0); << runtime error bo sA1 juz zdazylo powstac. Czy to nie powinno rzucac bledu przy kompilacji?
    //     sA2(3.0, 0.0);
    // }
    _input_(int Nbodies);
    body pickBodyType(int bodyId) const {
        return (bodyId == 0) ? bodyTypes[0] : bodyTypes[1];
    }

public:
    const int Nbodies;
    const int Nconstr = 2*Nbodies;
    VectorXd alpha0, dalpha0, pjoint0;
    std::vector<body> bodyTypes;
};


// derivatives.cpp
VectorXd Phi(const VectorXd& q, const _input_& input);
MatrixXd Jacobian(const VectorXd& q, const _input_& input);

// utils.cpp
Matrix2d Rot(double fi);
const Matrix2d Om = Rot(M_PI_2);
MatrixXd jacobianReal(VectorXd (*fun)(const VectorXd&, const _input_&), VectorXd alfa0, _input_ input);
VectorXd jointToAbsoluteCoordinates(const VectorXd& jointCoordsAlpha, const _input_& input);
Matrix3d SAB(const std::string& _sAB_, const int id, const VectorXd& alphaAbs, const _input_& input);
VectorXd joint2AbsAngles(const VectorXd& alpha);
Matrix3d massMatrix(const int id, const _input_ input);
Vector3d Q1_init(int id, const VectorXd& alphaAbs, const _input_& input);


// HDCA
class ksi_coefs {
public:
    ksi_coefs(const int id, const VectorXd& alphaAbs, const VectorXd& pjoint, const _input_& input);
    ksi_coefs(const ksi_coefs& ksiA, const ksi_coefs& ksiB);

	Matrix3d k11() const {return i11;}
	Matrix3d k12() const {return i12;}
	Matrix3d k21() const {return i21;}
	Matrix3d k22() const {return i22;}
	Vector3d k10() const {return i10;}
	Vector3d k20() const {return i20;}

private:
    Matrix3d i11, i12, i21, i22;
    Vector3d i10, i20;
};

class Assembly {
public:
    Assembly(const int id, const VectorXd& alphaAbs, const VectorXd& pjoint, const _input_& input);
    Assembly(Assembly& AsmA, Assembly& AsmB);

public: // to do: set private
    const ksi_coefs ksi;
    const Vector3d Q1Acc;
    const Matrix3d S12;
    Assembly * const ptrAsmA, *const ptrAsmB; // const jest tentatywny
    Vector3d T1, T2, Q1Art, Q2Art;
};