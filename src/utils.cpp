#include "../include/utils.h"
#include "../include/input.h"
#include "../Eigen/Dense"
#include "../include/constants.h"

    #include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>

using namespace Eigen;

Matrix2d Rot(double fi)
{
    Matrix2d R;
    R << cos(fi), -sin(fi), sin(fi), cos(fi);
    return R;
}

MatrixXd jacobianReal(VectorXd (*fun)(const VectorXd &, const _input_ &), VectorXd q0, const _input_& input)
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
     * Converts joint coordinates alpha to absolute coordinates q (3 * number_of_bodies).
     */
    VectorXd q(3 * input.Nbodies);

    q.segment(0, 3) << alpha(0), 0.0, 0.0;      // box: (x0, y0, fi0)
    q.segment(3, 3) << alpha(0), 0.0, alpha(1); // first link: (x1=x0, y1, fi1)

    for (int i = 2; i < input.Nbodies; i++)
    {
        const int prev = i - 1;
        q.segment(3 * i, 2) = q.segment(3 * prev, 2) +
                              Rot(q(3 * prev + 2)) * input.pickBodyType(prev).s12;
        q(3 * i + 2) = q(3 * prev + 2) + alpha(i);
    }

    return q;
}

VectorXd jointToAbsoluteVelocity(const VectorXd &alpha, const VectorXd &dalpha, const _input_ &input)
{
    /**
     * Converts joint velocity dalpha to absolute velocity dq (3 * number_of_bodies).
     */
    VectorXd dq(3 * input.Nbodies);
    VectorXd alphaAbsolute(input.Nbodies);

    dq.segment(0, 3) << dalpha(0), 0.0, 0.0;
    dq.segment(3, 3) << dalpha(0), 0.0, dalpha(1);

    alphaAbsolute.segment(0, 2) << 0.0, alpha(1);

    for (int i = 2; i < input.Nbodies; i++)
    {
        const int prev = i - 1;
        dq.segment(3 * i, 2) = dq.segment(3 * prev, 2) +
                    Om*Rot(alphaAbsolute(prev)) * dq(3 * prev + 2) * input.pickBodyType(prev).s12;
        dq(3 * i + 2) = dq(3 * prev + 2) + dalpha(i);

        alphaAbsolute(i) = alphaAbsolute(prev) + alpha(i);
    }

    return dq;
}

// struct coordinates {

// }

MatrixXd jointToAbsoluteCoords(const VectorXd &alpha, const VectorXd &dalpha, 
                               const VectorXd &d2alpha, const _input_ &input)
{
    /**
     * Converts joint alpha, dalpha, and d2alpha to their absolute counterparts [q, dq, d2q].
     * note: 1) the caller function can use: const VectorXd& q = x.col(0);
     *       2) alphaAbs and derivatives can be obtained from slicing
     */
    enum place {q, v, a};
    MatrixXd x(3 * input.Nbodies, 3);
    VectorXd alphaAbsolute(input.Nbodies);
    x.block(0, 0, 6, 3) << alpha(0), dalpha(0), d2alpha(0), 
                                0.0,       0.0,       0.0, 
                                0.0,       0.0,       0.0,
                            alpha(0), dalpha(0), d2alpha(0),
                                0.0,       0.0,       0.0,
                            alpha(1), dalpha(1), d2alpha(1);

    alphaAbsolute.segment(0, 2) << alpha(0), alpha(1);

// zrownoleglamy wszystkie obliczenia na raz
    for (int i = 2; i < input.Nbodies; i++)
    {
        const int prev = i - 1;
        const double phi = x(3*prev + 2, q);
        const double om  = x(3*prev + 2, v);
        const double eps = x(3*prev + 2, a);
        const Vector2d& s12 = input.pickBodyType(prev).s12;

        x.block(3*i, q, 2, 1) = x.block(3*prev, q, 2, 1) +  Rot(phi) * s12;
        x(3*i + 2, q) = phi + alpha(i);


        x.block(3*i, v, 2, 1) = x.block(3*prev, v, 2, 1) +  Om*Rot(phi) * om * s12;
        x(3*i + 2, v) = om + dalpha(i);

        x.block(3*i, a, 2, 1) = x.block(3*prev, a, 2, 1) +  (Om*Rot(phi) * eps - Rot(phi) * om*om) * s12;
        x(3*i + 2, a) = eps + d2alpha(i);
    }

    return x;
}

/* opcje dla vec : {s12, s21, s1C, s2C} */
Matrix3d SAB(const std::string &_sAB_, const int id, const VectorXd &alphaAbs, const _input_ &input)
{
    Vector2d sAB = input.pickBodyType(id).dimensions.at(_sAB_);
    Matrix3d out = Matrix3d::Identity();
    out.block(2, 0, 1, 2) = (Om * Rot(alphaAbs(id)) * sAB).transpose();
    return out;
}

Matrix3d dSAB(const std::string& _sAB_, const int id, const VectorXd& alphaAbs, 
              const VectorXd& dAlphaAbs, const _input_& input) {
    Vector2d sAB = input.pickBodyType(id).dimensions.at(_sAB_);
    Matrix3d out = Matrix3d::Zero();
    out.block(2, 0, 1, 2) = (-1.0) * (Rot(alphaAbs(id)) * sAB).transpose() * dAlphaAbs(id);
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

Matrix3d massMatrix(const int id, const _input_& input)
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
    Q_out(1) = - M_GRAV * input.pickBodyType(id).m;
    Q_out = SAB("s1C", id, alphaAbs, input) * Q_out;
    return Q_out;
}

double calculateTotalEnergy(const double& t, const VectorXd& y, const _input_& input) {
    const unsigned int n = input.alpha0.size();
    VectorXd dy = RHS_HDCA(t , y, input);
    VectorXd alpha  = y.tail(n);
    VectorXd dalpha = dy.tail(n);
    VectorXd alphaAbs = joint2AbsAngles(alpha);
    VectorXd q  = jointToAbsolutePosition(alpha, input);
    VectorXd dq = jointToAbsoluteVelocity(alpha, dalpha, input);
    double energy = 0.0;

    for (int i = 0; i < input.Nbodies; ++i) {
        double m = input.pickBodyType(i).m;
        Matrix3d M = massMatrix(i, input);
        Matrix3d S1c = SAB("s1C", i, alphaAbs, input);
        Vector3d V  = dq.segment(3 * i, 3);
        Vector2d qi = q.segment(3*i, 2) + Rot(q(3*i+2)) * input.pickBodyType(i).s1C;
        M = S1c * M * S1c.transpose(); // note: aliasing nie zachodzi

        energy += 0.5 * V.transpose() * M * V + 
                  m * M_GRAV * qi(1);
    }

    return energy;
}

void logTotalEnergy(const double& t, const VectorXd& y, const _input_& input) {
    /*  
     * Funkcjia loguje calkowita energie w ukladzie i sluzy do testowania obliczen.
     * */
    static bool cleanFile = true;
    if (cleanFile) {
        std::ofstream outFile;
        outFile.open("../output/energy.txt");
        outFile << "";
        outFile.close();
        cleanFile = false;
    }
    
    double energy = calculateTotalEnergy(t, y, input);

	std::ofstream outFile;
	outFile.open("../output/energy.txt", std::ios_base::app);

	if (outFile.fail() ) {
		std::cerr << "nie udalo sie otworzyc pliku.";
        throw std::runtime_error("logTotalEnergy: nie udalo sie otworzyc pliku");
	}
	outFile << std::setw(4)         << t      << std::setfill('0') << "\t" 
            << std::setprecision(5) << energy << std::endl;
	outFile.close();
}

dataJoint interpolate(const double& t, const _solution_& solutionFwd, const _input_& input) {
    /* return values at node index */
    std::pair<int,bool> indStruct = solutionFwd.atTime(t, input);
    if (indStruct.second == NODE_VALUE )
        return solutionFwd.getDynamicValues(indStruct.first, input);
    
    /* interpolate via 3rd order polynomial */
    const int Nvars = input.Nbodies;
    int baseInd = indStruct.first;
	dataJoint dataInterp = dataJoint(Nvars);

    if (baseInd == 0)
        baseInd++;
    else if (baseInd == input.Nsamples-2) // pre-last node
        baseInd--;
    dataJoint sr1 = solutionFwd.getDynamicValues(baseInd-1, input);
    dataJoint s00 = solutionFwd.getDynamicValues(baseInd  , input);
    dataJoint sf1 = solutionFwd.getDynamicValues(baseInd+1, input);
    dataJoint sf2 = solutionFwd.getDynamicValues(baseInd+2, input);

// to do: zrownoleglic
    for (int i = 0; i < Nvars; i++) {
        Matrix4d nodeVals;
        nodeVals << pow(sr1.t, 3), pow(sr1.t, 2), sr1.t, 1.0,
                    pow(s00.t, 3), pow(s00.t, 2), s00.t, 1.0,
                    pow(sf1.t, 3), pow(sf1.t, 2), sf1.t, 1.0,
                    pow(sf2.t, 3), pow(sf2.t, 2), sf2.t, 1.0;

        Matrix<double, 4,5> RHS, c;
        RHS.col(0) << sr1.alpha(i),       s00.alpha(i),      sf1.alpha(i),      sf2.alpha(i);
        RHS.col(1) << sr1.dalpha(i),      s00.dalpha(i),     sf1.dalpha(i),     sf2.dalpha(i);
        RHS.col(2) << sr1.d2alpha(i),     s00.d2alpha(i),    sf1.d2alpha(i),    sf2.d2alpha(i);
        RHS.col(3) << sr1.lambda(2*i),    s00.lambda(2*i),   sf1.lambda(2*i),   sf2.lambda(2*i);
        RHS.col(4) << sr1.lambda(2*i+1),  s00.lambda(2*i+1), sf1.lambda(2*i+1), sf2.lambda(2*i+1);
        c = nodeVals.partialPivLu().solve(RHS);

        dataInterp.alpha(i)   = c(0,0)*pow(t,3) + c(1,0)*pow(t,2) + c(2,0)*t + c(3,0);
        dataInterp.dalpha(i)  = c(0,1)*pow(t,3) + c(1,1)*pow(t,2) + c(2,1)*t + c(3,1);
        dataInterp.d2alpha(i) = c(0,2)*pow(t,3) + c(1,2)*pow(t,2) + c(2,2)*t + c(3,2);
        dataInterp.lambda.segment(2*i,2) << c(0,3)*pow(t,3) + c(1,3)*pow(t,2) + c(2,3)*t + c(3,3),
                                            c(0,4)*pow(t,3) + c(1,4)*pow(t,2) + c(2,4)*t + c(3,4);
/*         if (i == 4 &&  abs(t-0.98)<0.01 ) {
            enum vars {a, da, d2a, l1, l2};
            std::cout << std::endl << nodeVals << std::endl << RHS << std::endl;
            std::cout << "t = " << t << std::endl;
            std::cout << "value " << dataInterp.d2alpha(i) << std::endl;
            std::cout << "a = " << c(0,d2a) << " b = " << c(1,d2a) << " c = " << c(2,d2a) << " d = " << c(3,d2a) << std::endl;
        } */
    }

    return dataInterp;
}

dataJoint interpolateLinear(const double& t, const _solution_& solutionFwd, const _input_& input) {
    /* return values at node index */
    std::pair<int,bool> indStruct = solutionFwd.atTime(t, input);
    if (indStruct.second == NODE_VALUE )
        return solutionFwd.getDynamicValues(indStruct.first, input);

    /* interpolate linearly */
    const int Nvars = input.Nbodies;
    const int& baseInd = indStruct.first;
	dataJoint dataInterp = dataJoint(Nvars);

    dataJoint s00 = solutionFwd.getDynamicValues(baseInd  , input);
    dataJoint sf1 = solutionFwd.getDynamicValues(baseInd+1, input);

    dataInterp.alpha   = (s00.alpha   + sf1.alpha)   / 2.0;
    dataInterp.dalpha  = (s00.dalpha  + sf1.dalpha)  / 2.0;
    dataInterp.d2alpha = (s00.d2alpha + sf1.d2alpha) / 2.0;
    dataInterp.lambda  = (s00.lambda  + sf1.lambda)  / 2.0;

    return dataInterp;
}