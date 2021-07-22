#include<Eigen/Dense>
using namespace Eigen;


class _input_ {
public:
    // _input_() {
    //     sA1(2.0, 0.0); << runtime error bo sA1 juz zdazylo powstac. Czy to nie powinno rzucac bledu przy kompilacji?
    //     sA2(3.0, 0.0);
    // }
    _input_();
    Vector2d sA1;
    Vector2d sA2;
    const int Nbodies = 2;
    const int Nconstr = 4;
};
Matrix2d Rot(double fi);
MatrixXd jacobianReal(VectorXd (*fun)(const VectorXd&, const _input_&), VectorXd q0, _input_ input);
VectorXd Phi(const VectorXd& q, const _input_& input);
MatrixXd Jacobian(const VectorXd& q, const _input_& input);

const Matrix2d I = Matrix2d::Identity();
const Matrix2d Om = Rot(M_PI_2);

// do usuniecia:
Matrix2cd Rot(std::complex<double> fi);
VectorXcd Phi(const VectorXcd& q, const _input_& input);
MatrixXd jacobianComplex(VectorXcd (*fun)(const VectorXcd&, const _input_&), VectorXd q0, _input_ input);