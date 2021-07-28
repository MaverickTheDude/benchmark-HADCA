#include <project.h>

_input_::_input_(int _Nbodies_) : Nbodies(_Nbodies_), alpha0(_Nbodies_),
                                  dalpha0(_Nbodies_), pjoint0(_Nbodies_)
{
    bodyTypes.emplace_back("box");
    bodyTypes.emplace_back("link");
    alpha0(0) = 0.0;
    alpha0.tail(_Nbodies_ - 1).setConstant(M_PI_4);
    dalpha0.setZero(); // to do: niezerowa predkosc / ped do testowania funkcji
    pjoint0.setZero();
}

body::body(std::string type)
{
    m = _m_;
    J = _J_;
    if (!type.compare("box"))
    {
        s1C.setZero();
        s12.setZero();
        H << 1, 0, 0;
        D << 0, 0, 1, 0, 0, 1;
    }
    else if (!type.compare("link"))
    {
        s1C << _L_ / 2, 0;
        s12 << _L_, 0;
        H << 0, 0, 1;
        D << 1, 0, 0, 1, 0, 0;
    }
    else
        throw std::runtime_error("not supported body / joint");
    dimensions["s12"] = s12;
    dimensions["s21"] = -s12;
    dimensions["s1C"] = s1C;
    dimensions["s2C"] = s1C - s12;
}

Matrix2d Rot(double fi)
{
    Matrix2d R;
    R << cos(fi), -sin(fi), sin(fi), cos(fi);
    return R;
}

MatrixXd jacobianReal(VectorXd (*fun)(const VectorXd &, const _input_ &), VectorXd q0, _input_ input)
{
    const double h = 1e-8;
    const int n = q0.size();
    const int Nf = input.Nconstr;

    MatrixXd Fun_q(Nf, n);
    for (int i = 0; i < n; i++)
    {
        VectorXd delta = ArrayXd::Zero(n);
        delta(i) = h;
        VectorXd qFF = q0 + delta;
        VectorXd qRV = q0 - delta;
        MatrixXd funForward = fun(qFF, input);
        MatrixXd funRev = fun(qRV, input);
        Fun_q.col(i) = (funForward - funRev) / (2 * h);
    }

    return Fun_q;
}

VectorXd jointToAbsolutePosition(const VectorXd &alpha, const _input_ &input)
{
    /**
     * Converts joint coordinates alpha to absolute coordinates q (3 * number_of_bodies)
     */
    VectorXd q(3 * input.Nbodies);

    q.segment(0, 3) << alpha(0), 0.0, 0.0;      // box: (x0, y0, fi0)
    q.segment(3, 3) << alpha(0), 0.0, alpha(1); // first link: (x1=x0, y1, fi1)

    for (int i = 2; i < input.Nbodies; i++)
    {
        const int prev = i - 1;
        q.segment(3 * i, 2) = q.segment(3 * prev, 2) +
                              Rot(alpha(prev)) * input.pickBodyType(prev).s12;
        q(3 * i + 2) = q(3 * prev + 2) + alpha(i);
    }

    return q;
}

VectorXd jointToAbsoluteVelocity(const VectorXd &alpha, const VectorXd &dalpha, const _input_ &input)
{
    /**
     * Converts joint velocity dalpha to absolute velocity dq (3 * number_of_bodies)
     */
    VectorXd dq(3 * input.Nbodies);

    dq.segment(0, 3) << dalpha(0), 0.0, 0.0;
    dq.segment(3, 3) << dalpha(0), 0.0, dalpha(1);

    for (int i = 2; i < input.Nbodies; i++)
    {
        const int prev = i - 1;
        dq.segment(3 * i, 2) = dq.segment(3 * prev, 2) +
                    Om*Rot(alpha(prev)) * dalpha(prev) * input.pickBodyType(prev).s12;
        dq(3 * i + 2) = dq(3 * prev + 2) + dalpha(i);
    }

    return dq;
}

/* opcje dla vec : {s12, s21, s1C, s2C} */
Matrix3d SAB(const std::string &_sAB_, const int id, const VectorXd &alphaAbs, const _input_ &input)
{
    Vector2d sAB = input.pickBodyType(id).dimensions.at(_sAB_);
    Matrix3d out = Matrix3d::Identity();
    out.block(2, 0, 1, 2) = (Om * alphaAbs(id) * sAB).transpose();
    return out;
}

VectorXd joint2AbsAngles(const VectorXd &alpha)
{
    VectorXd alphaAbs(alpha.size());
    alphaAbs(0) = alpha(0);
    alphaAbs(1) = alpha(1);

    // zrownoleglic tutaj?
    for (int i = 2; i < alphaAbs.size(); i++)
    {
        alphaAbs(i) = alphaAbs(i - 1) + alpha(i);
    }
    return alphaAbs;
}

Matrix3d massMatrix(const int id, const _input_ input)
{
    Matrix3d out = Matrix3d::Zero();
    out(0, 0) = input.pickBodyType(id).m;
    out(1, 1) = input.pickBodyType(id).m;
    out(2, 2) = input.pickBodyType(id).J;
    return out;
}

Vector3d Q1_init(int id, const VectorXd &alphaAbs, const _input_ &input)
{
    Vector3d Q_out = Vector3d::Zero();
    Q_out(1) = input.pickBodyType(id).m;
    Q_out = SAB("s1C", id, alphaAbs, input) * Q_out;
    return Q_out;
}
