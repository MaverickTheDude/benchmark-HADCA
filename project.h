#include<Eigen/Dense>
#include<vector>
using namespace Eigen;

const double _L_ = 1.0;
const double _m_ = 0.2;
const double _J_ = 0.1;

class body {
public:
    body(std::string type);

public:
    Vector2d s1C;
    Vector2d s12;
    double m, J;
    Vector3d H;
    Matrix<double, 3, 2> D;
};

class _input_ {
public:
    // _input_() {
    //     sA1(2.0, 0.0); << runtime error bo sA1 juz zdazylo powstac. Czy to nie powinno rzucac bledu przy kompilacji?
    //     sA2(3.0, 0.0);
    // }
    _input_(int Nbodies);
    body pick(int bodyId) const {
        return (bodyId == 0) ? bodies[0] : bodies[1];
    }

public:
    const int Nbodies;
    const int Nconstr = 2*Nbodies;
    VectorXd alfa0, dalfa0;
    std::vector<body> bodies;
};
Matrix2d Rot(double fi);
MatrixXd jacobianReal(VectorXd (*fun)(const VectorXd&, const _input_&), VectorXd alfa0, _input_ input);
VectorXd Phi(const VectorXd& q, const _input_& input);
MatrixXd Jacobian(const VectorXd& q, const _input_& input);

const Matrix2d I = Matrix2d::Identity();
const Matrix2d Om = Rot(M_PI_2);

