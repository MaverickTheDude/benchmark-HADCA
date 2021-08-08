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

static void logTotalEnergy(const double& t, const VectorXd &y, const _input_ &input);
_solution_ RK_solver(const _input_ &input) {
	const double dt = input.dt;
	const double Tk = input.Tk;
	const int Nbodies = input.Nbodies;
    _solution_ solution(input); // pytanie: czy przy wyjsciu z funkcji solution zostanie przekopiowane, czy obiekt pozostanie w pamieci a funkcjia zwroci referencje?

	VectorXd T = VectorXd::LinSpaced(input.Nsamples, 0, Tk);
    solution.setT(T);
	VectorXd y_m1(2*Nbodies);
	y_m1.head(Nbodies) = input.pjoint0;
	y_m1.tail(Nbodies) = input.alpha0;

//	double t = omp_get_wtime(); //tic
	for (int i = 1; i < input.Nsamples; i++) {
        const double t = T(i-1);
		VectorXd k1 = RHS_HDCA(t         , y_m1,               input, solution); //RHS_HDCA appends the solution at i-1
        VectorXd k2 = RHS_HDCA(t + dt/2.0, y_m1 + dt/2.0 * k1, input);
		VectorXd k3 = RHS_HDCA(t + dt/2.0, y_m1 + dt/2.0 * k2, input);
		VectorXd k4 = RHS_HDCA(t + dt,     y_m1 + dt     * k3, input);

		VectorXd y = y_m1 +  dt/6 * (k1 + 2*k2 + 2*k3 + k4);
		y_m1 = y;

        if (input.logEnergy)
            logTotalEnergy(t, y, input);
	}
    RHS_HDCA(input.Tk, y_m1, input, solution); // save last entry
//	t =  omp_get_wtime() - t; //toc
//	std::cout << "calkowity czas: " << t << std::endl << std::endl;

	return solution;
};

double calculateTotalEnergy(const double& t, const VectorXd &y, const _input_ &input) {
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

static void logTotalEnergy(const double& t, const VectorXd &y, const _input_ &input) {
    /*  
     * Funkcjia oblicza calkowita energie w ukladzie i sluzy do testowania obliczen.
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