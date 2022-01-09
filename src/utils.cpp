#include "../include/utils.h"
#include "../include/input.h"
#include "../Eigen/Dense"
#include "../include/constants.h"
#include "../include/task/M.h"
#include "omp.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>

using namespace Eigen;

#define PARALLEL_UTILS false

#if PARALLEL_UTILS

VectorXd joint2AbsAngles(const VectorXd &alpha)
{
    const int Nbodies = alpha.size();
    VectorXd alphaAbs(alpha.size());
    alphaAbs(0) = alpha(0);
    alphaAbs(1) = alpha(1);
    double sum  = 0.0;
    
	//--- phi term calculation (cummulative sum) -------
    int Nthr;
#pragma omp parallel
    {
        if (omp_get_thread_num() == 0)
            Nthr = omp_get_num_threads();
    }
	VectorXd sumThr = VectorXd::Zero(Nthr + 1);

    // Compute cumsum for each thread
    int chunk = (Nbodies - 1 + Nthr - 1) / Nthr;
#pragma omp parallel for schedule(static, chunk) firstprivate(sum)
	for (int i = 1; i < Nbodies; i++) {
		sum += alpha(i);
		alphaAbs(i) = sum;
		sumThr(omp_get_thread_num() + 1) = sum;
	}

    // Compute offsets for consecutive threads (phi)
	for (int i=1; i < Nthr; i++)
		sumThr(i) += sumThr(i - 1);

	// Apply offsets and compute cumsums for (r)
#pragma omp parallel for schedule(static, chunk)
	for (int i = 1; i < Nbodies; i++)
        alphaAbs(i) += sumThr(omp_get_thread_num());

    return alphaAbs;
}

VectorXd jointToAbsolutePosition(const VectorXd &alpha, const _input_ &input)
{
    /**
     * Converts joint coordinates alpha to absolute coordinates q (3 * number_of_bodies).
     */
    const int Nbodies = input.Nbodies;

    VectorXd q(3 * Nbodies);
    q.segment(0, 3) << alpha(0), 0.0, 0.0;      // box: (x0, y0, fi0)
    q.segment(3, 3) << alpha(0), 0.0, alpha(1); // first link: (x1=x0, y1, fi1)

	VectorXd phi   = VectorXd::Zero(Nbodies);
    MatrixXd r_mat = MatrixXd::Zero(2, Nbodies + 1);

    double sum      = 0.0;
    Vector2d r_sum  = Vector2d::Zero();
    Vector2d r_bias = q.head(2);

	//--- phi term calculation (cummulative sum) -------
    int Nthr;
#pragma omp parallel
    {
        if (omp_get_thread_num() == 0)
            Nthr = omp_get_num_threads();
    }
	VectorXd sumThr_phi = VectorXd::Zero(Nthr + 1);
	MatrixXd sumThr_r   = MatrixXd::Zero(2, Nthr + 1);

    // Compute cumsum for each thread (phi)
    // Save the results in sumThr_phi(thread_id + 1)
    int chunk = (Nbodies - 1 + Nthr - 1) / Nthr;
#pragma omp parallel for schedule(static, chunk) firstprivate(sum)
	for (int i = 1; i < Nbodies; i++) {
		sum += alpha(i);
		phi(i) = sum;
		sumThr_phi(omp_get_thread_num() + 1) = sum;
	}

	// Compute offsets for consecutive threads (phi)
	for (int i=1; i < Nthr; i++)
		sumThr_phi(i) += sumThr_phi(i - 1);

	// Apply offsets and compute cumsums for (r)
#pragma omp parallel for schedule(static, chunk) firstprivate(r_sum)
	for (int i = 1; i < Nbodies; i++) {
        phi(i) += sumThr_phi(omp_get_thread_num());
        r_sum += Rot(phi(i)) * input.pickBodyType(i).s12;       
        r_mat.col(i + 1) = r_sum;
        sumThr_r.col(omp_get_thread_num() + 1) = r_sum;
	}

	// Compute offsets for consecutive threads (r)
	for (int i=1; i < Nthr; i++)
		sumThr_r.col(i) += sumThr_r.col(i - 1);

	// Apply offsets (r) and build q vactor
#pragma omp parallel for schedule(static, chunk)
	for (int i = 1; i < Nbodies - 1; i++) {
        const int next = i + 1;
        r_mat.col(next) += sumThr_r.col(omp_get_thread_num());
        q.segment(3*next, 2) = r_mat.col(next) + r_bias;
        q(3*next+2) = phi(next);
    }

    return q;
}

VectorXd jointToAbsoluteVelocity(const VectorXd &alpha, const VectorXd &dalpha, const _input_ &input)
{
    /**
     * Converts joint velocity dalpha to absolute velocity dq (3 * number_of_bodies).
     */

    const int Nbodies = input.Nbodies;

    VectorXd dq(3 * input.Nbodies);
    dq.segment(0, 3) << dalpha(0), 0.0, 0.0;
    dq.segment(3, 3) << dalpha(0), 0.0, dalpha(1);

    VectorXd phi = VectorXd::Zero(Nbodies);
    VectorXd phi_prim = VectorXd::Zero(Nbodies);
    MatrixXd r_prim = MatrixXd::Zero(2, Nbodies + 1);

    double phi_sum = 0, phi_prim_sum = 0;
    Vector2d r_prim_sum = Vector2d::Zero();
    Vector2d r_prim_bias = dq.head(2);

    int Nthr;
#pragma omp parallel
    {
        if(omp_get_thread_num() == 0)
            Nthr = omp_get_num_threads();
    }
    VectorXd sumThr_phi = VectorXd::Zero(Nthr + 1);
    VectorXd sumThr_phi_prim = VectorXd::Zero(Nthr + 1);
    MatrixXd sumThr_r_prim = MatrixXd::Zero(2, Nthr + 1);

    // Compute cumsum for each thread (phi, phi_prim)
    // Save the results in sumThr_phi and sumThr_phi_prmi (indexed thread_id + 1)
    int chunk = (Nbodies - 1 + Nthr - 1) / Nthr;
#pragma omp parallel for schedule(static, chunk) firstprivate(phi_sum, phi_prim_sum)
	for (int i = 1; i < Nbodies; i++) {
        phi_sum += alpha(i);
        phi_prim_sum += dalpha(i);
        phi(i) = phi_sum;
        phi_prim(i) = phi_prim_sum;
        sumThr_phi(omp_get_thread_num() + 1) = phi_sum;
        sumThr_phi_prim(omp_get_thread_num() + 1) = phi_prim_sum;
    }

    // Compute offsets for consecutive thread (phi, phi_prim)
    for (int i=1; i < Nthr; i++)
    {
		sumThr_phi(i) += sumThr_phi(i - 1);
        sumThr_phi_prim(i) += sumThr_phi_prim(i - 1);
    }

    //Apply offsets and compute cumsums for (r_prim)
#pragma omp parallel for schedule(static, chunk) firstprivate(r_prim_sum)
	for (int i = 1; i < Nbodies; i++) {
        phi(i) += sumThr_phi(omp_get_thread_num());
        phi_prim(i) += sumThr_phi_prim(omp_get_thread_num());
        r_prim_sum += Om * Rot(phi(i)) * phi_prim(i) * input.pickBodyType(i).s12;
        r_prim.col(i + 1) = r_prim_sum;
        sumThr_r_prim.col(omp_get_thread_num() + 1) = r_prim_sum;
    }

    // Compute offsets for consecutive threads (r_prim)
    for (int i=1; i < Nthr; i++)
		sumThr_r_prim.col(i) += sumThr_r_prim.col(i - 1);

    // Apply offsets (r) and build q vactor
#pragma omp parallel for schedule(static, chunk)
	for (int i = 1; i < Nbodies - 1; i++) {
        const int next = i + 1;
        r_prim.col(next) += sumThr_r_prim.col(omp_get_thread_num());
        dq.segment(3 * next, 2) = r_prim.col(next) + r_prim_bias;
        dq(3 * next + 2) = phi_prim(next);
    }

    return dq;
}

#else

VectorXd joint2AbsAngles(const VectorXd &alpha)
{
    VectorXd alphaAbs(alpha.size());
    alphaAbs(0) = alpha(0);
    alphaAbs(1) = alpha(1);

    for (int i = 2; i < alphaAbs.size(); i++)
    {
        alphaAbs(i) = alphaAbs(i - 1) + alpha(i);
    }
    return alphaAbs;
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


#endif

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
    const int Nf = fun(q0, input).size();

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

VectorXd absolutePositionToAbsoluteAlpha(const VectorXd& q)
{
    // https://eigen.tuxfamily.org/dox-devel/group__TutorialSlicingIndexing.html
    // const int qSize = q.size();
    // VectorXd absoluteAlpha = VectorXd::Zero(q.size() / 3);
    // absoluteAlpha = q(seqN())
// to do: parallelize
    const int qSize = q.size();
    VectorXd absoluteAlpha = VectorXd::Zero(qSize / 3);
    for(int i = 2; i < qSize; i += 3)
        absoluteAlpha((i - 2) / 3) = q(i);
    
    return absoluteAlpha;
}

/* opcje dla vec : {s12, s21, s1C, s2C} */
Matrix3d SAB(const std::string &_sAB_, const int id, const double alphaAbs, const _input_ &input)
{
    Vector2d sAB = input.pickBodyType(id).dimensions.at(_sAB_);
    Matrix3d out = Matrix3d::Identity();
    out.block(2, 0, 1, 2) = (Om * Rot(alphaAbs) * sAB).transpose();
    return out;
}

Matrix3d SAB(const std::string &_sAB_, const int id, const VectorXd &alphaAbs, const _input_ &input)
{
    return SAB(_sAB_, id, alphaAbs(id), input);
}

Matrix3d dSAB(const std::string& _sAB_, const int id, const double& alphaAbs, 
              const double& dAlphaAbs, const _input_& input) {
    Vector2d sAB = input.pickBodyType(id).dimensions.at(_sAB_);
    Matrix3d out = Matrix3d::Zero();
    out.block(2, 0, 1, 2) = (-1.0) * (Rot(alphaAbs) * sAB).transpose() * dAlphaAbs;
    return out;
}

Matrix3d dSAB(const std::string& _sAB_, const int id, const VectorXd& alphaAbs, 
              const VectorXd& dAlphaAbs, const _input_& input) 
{
    return dSAB(_sAB_, id, alphaAbs(id), dAlphaAbs(id), input);
}

Matrix3d d2SAB(const std::string& _sAB_, const int id, const double& alphaAbs, 
              const double& dAlphaAbs, const double& d2AlphaAbs, const _input_& input) {
    Vector2d sAB = input.pickBodyType(id).dimensions.at(_sAB_);
    Matrix3d out = Matrix3d::Zero();
    out.block(2, 0, 1, 2) = (-1.0) * (Rot(alphaAbs) * sAB).transpose() * d2AlphaAbs - 
                                  (Om*Rot(alphaAbs) * sAB).transpose() * dAlphaAbs*dAlphaAbs;
    return out;
}

Matrix3d dSABdAlpha(const Vector2d& translation, const double absoluteAlpha)
{
    /** 
     * Returns derivative dSAB / dAbsoluteAlpha (3 x 3).
     */

    Matrix3d S = Matrix3d::Zero();
    S.block(2, 0, 1, 2) = (- Rot(absoluteAlpha) * translation).transpose();
    return S;
}

Matrix3d massMatrix(const int id, const _input_& input)
{
    return task::M::local(id, input);
}

Vector3d Q1_init(int id, const VectorXd &alphaAbs, const double& u, const _input_ &input)
{
    Vector3d Q_out = Vector3d::Zero();
    /* gravity */
    Q_out(1) = - M_GRAV * input.pickBodyType(id).m;
    Q_out = SAB("s1C", id, alphaAbs, input) * Q_out;
    /* control force */
    if (id == 0) Q_out(0) = u;
    return Q_out;
}

Vector3d Q1_init(int id, const VectorXd &alphaAbs, const VectorXd &dAlphaJoint, const double& u, const _input_ &input)
{
    Vector3d Q_out = Vector3d::Zero();
    /* gravity */
    Q_out(1) = - M_GRAV * input.pickBodyType(id).m;
    Q_out = SAB("s1C", id, alphaAbs, input) * Q_out;
    const double c_cart = input.pickBodyFriction(0);
    const double c_pend = input.pickBodyFriction(1);
    const double& dx = dAlphaJoint(0);
    if (id == 0) {
        Q_out(0) = u - c_cart * dx;    /* control + friction force */
    }
    else {
        /* note: Q_out uwzglednia tylko tlumienie *swojego* zlacza, ale juz nie sasiada
         * prawdopodobnie przez zlaczowy opis wspolrzednych. Blad: Q_out(2) += c_pend * (-dphi_i + dphi_j) */
        const double& dphi_i = dAlphaJoint(id);
        Q_out(2) += -c_pend * dphi_i;   /* friction torque */
    }
    return Q_out;
}

static double calculateTotalEnergy(const double& t, const VectorXd& y, const VectorXd& dy, 
                                   const VectorXd& uVec, const _input_& input) {
    const unsigned int n = input.alpha0.size();
    VectorXd alpha  = y.tail(n);
    VectorXd dalpha = dy.tail(n);
    VectorXd alphaAbs = joint2AbsAngles(alpha);
    VectorXd omega    = joint2AbsAngles(dalpha);
    VectorXd q  = jointToAbsolutePosition(alpha, input);
    VectorXd dq = jointToAbsoluteVelocity(alpha, dalpha, input);
    double energy = 0.0;

    static MatrixXd powerDamp = MatrixXd::Zero(n, input.Nsamples);
    int ind = atTime(t, VectorXd::LinSpaced(input.Nsamples, 0, input.Tk), input).first;

    for (int i = 0; i < input.Nbodies; ++i) {
        double m      = input.pickBodyType(i).m;
        double c_fric = input.pickBodyFriction(i);
        Matrix3d M = massMatrix(i, input);
        Matrix3d S1c = SAB("s1C", i, alphaAbs, input);
        Vector3d V  = dq.segment(3 * i, 3);
        Vector2d qi = q.segment(3*i, 2) + Rot(q(3*i+2)) * input.pickBodyType(i).s1C;
        M = S1c * M * S1c.transpose(); // note: aliasing nie zachodzi

        energy += 0.5 * V.transpose() * M * V + 
                  m * M_GRAV * qi(1);

        /* damping */
        powerDamp(i, ind) = - c_fric * dalpha(i) * omega(i);
        energy -= trapz(ind, powerDamp.row(i), input);
    }
    /* control signal */
    double x  = alpha(0);
    double u = interpolateControl(t, uVec, input);
    energy -= u * x;

    return energy;
}

void logTotalEnergy(const double& t, const VectorXd& y, const VectorXd& dy, const VectorXd& uVec, const _input_& input) {
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
    
    double energy = calculateTotalEnergy(t, y, dy, uVec, input);

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
    const double eps = 1e-10;
    assert(t >= 0.0 && t <= input.Tk + eps);

    /* return values at node index */
    std::pair<int,bool> indStruct = solutionFwd.atTime(t, input);
    if (indStruct.second == _solution_::NODE_VALUE )
        return solutionFwd.getDynamicValues(indStruct.first, input);
    
    /* interpolate via 3rd order polynomial */
    const int Nvars = input.Nbodies;
    int baseInd = indStruct.first;
	dataJoint dataInterp = dataJoint(Nvars);

    if (baseInd == 0)
        baseInd++;
    else if (baseInd == input.Nsamples-2)   // pre-last node
        baseInd--;
    else if (baseInd == input.Nsamples-1) { // last node
        const int last = Nvars-1;
        dataInterp.alpha   = solutionFwd.alpha.col(last);
        dataInterp.dalpha  = solutionFwd.dalpha.col(last);
        dataInterp.d2alpha = solutionFwd.d2alpha.col(last);
        dataInterp.lambda  = solutionFwd.lambda.col(last);
        return dataInterp;
    }
    dataJoint sr1 = solutionFwd.getDynamicValues(baseInd-1, input);
    dataJoint s00 = solutionFwd.getDynamicValues(baseInd  , input);
    dataJoint sf1 = solutionFwd.getDynamicValues(baseInd+1, input);
    dataJoint sf2 = solutionFwd.getDynamicValues(baseInd+2, input);

#pragma omp parallel for schedule(static)
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
    const double eps = 1e-10;
    assert(t >= 0.0 && t <= input.Tk + eps);

    /* return values at node index */
    std::pair<int,bool> indStruct = solutionFwd.atTime(t, input);
    if (indStruct.second == _solution_::NODE_VALUE )
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

static int compare_min(const int i, const int j, const VectorXd& T, const double& t) {
    return abs(T[i] - t) < abs(T[j]-t) ? i : j;
}
std::pair<int, const bool> atTime(const double& t, const VectorXd& T, const _input_& input) {
	/* 
	* overload metody klasy _solution_ (to do: uogolnic do statycznej metody)
	*/
    const double eps = 1e-10;
    assert(t >= 0.0 && t <= input.Tk + eps);
	const double& dt = input.dt;
    int begin = 0;
    int end = input.Nsamples-1;

	const int linearSearchRegion = 5;
	while (end - begin > linearSearchRegion) {
		const int mid = (begin + end) / 2;
		if ( abs(t - T(mid)) < 1e-10 ) return std::make_pair(mid, _solution_::NODE_VALUE);
		if (t > T(mid))		begin = mid;
		else				end   = mid;
	}

    int min_ind = begin;
	for (int i = begin+1; i <= end; i++)
        min_ind = compare_min(min_ind, i, T, t);

    if (abs(T(min_ind) - t) > dt)
	    throw std::runtime_error("atTime: index not found");
    else if ( abs(T(min_ind) 	 - t) < 1e-10 ) 
        return std::make_pair(min_ind, _solution_::NODE_VALUE);
    else
        return std::make_pair(min_ind, _solution_::INTERMEDIATE_VALUE);
}

double interpolateControl(const double& t, const VectorXd& uVec, const _input_& input) {
    const double eps = 1e-10;
    assert(t >= 0.0 && t <= input.Tk + eps);
     /* return values at node index */
    VectorXd T = VectorXd::LinSpaced(input.Nsamples, 0, input.Tk);
    std::pair<int,bool> indStruct = atTime(t, T, input);
    if (indStruct.second == _solution_::NODE_VALUE )
        return uVec(indStruct.first);

    /* interpolate via 3rd order polynomial */
    int baseInd = indStruct.first;

    if (baseInd == 0)
        baseInd++;
    else if (baseInd == input.Nsamples-2) // pre-last node
        baseInd--;
    else if (baseInd == input.Nsamples-1) // last node
        return uVec.tail(1).value();

    double tr1 = T(baseInd-1);      double ur1 = uVec(baseInd-1);
    double t00 = T(baseInd  );      double u00 = uVec(baseInd  );
    double tf1 = T(baseInd+1);      double uf1 = uVec(baseInd+1);
    double tf2 = T(baseInd+2);      double uf2 = uVec(baseInd+2);

    Matrix4d nodeVals;
    nodeVals << pow(tr1, 3), pow(tr1, 2), tr1, 1.0,
                pow(t00, 3), pow(t00, 2), t00, 1.0,
                pow(tf1, 3), pow(tf1, 2), tf1, 1.0,
                pow(tf2, 3), pow(tf2, 2), tf2, 1.0;

    Vector4d RHS, c;
    RHS << ur1, u00, uf1, uf2;
    c = nodeVals.partialPivLu().solve(RHS);

    return c(0)*pow(t,3) + c(1)*pow(t,2) + c(2)*t + c(3);
}

double trapz(const VectorXd& x, const _input_& input) {
    /* Numeric trapezoidal integration: * * * * *  ...  *     * 
     *                                  0 1 2 3 4     (N-2) (N-1)
     * x1 to x(N-2) times dt. Border nodes times dt/2  */
    const int N = x.size();
    const double& dt = input.dt;
    double out = dt * x.segment(1, N-2).sum();
    out += (x(0)+x(N-1))*dt/2;
    return out;
}

double trapz(const int ind, const VectorXd& x, const _input_& input) {
    /* Numeric trapezoidal integration: * * * * *  ...  *     * 
     *                                  0 1 2 3 4     (N-2) (N-1)
     * x1 to x(N-2) times dt. Border nodes times dt/2  */
    const int Nsize = ind+1;
    const double& dt = input.dt;
    if (Nsize == 1) return 0.0;
    if (Nsize == 2) return (x(0)+x(1))*dt/2;
    double out = dt * x.segment(1, Nsize-2).sum();
    out += (x(0)+x(Nsize-1))*dt/2;
    return out;
}

void print_checkGrad(const _solution_& solFwd, const _solutionAdj_& solAdj, 
					 const VectorXd& uVec, const _input_& input) {
	IOFormat exportFmt(FullPrecision, 0, " ", "\n", "", "", "", "");
	std::ofstream outFile;
	outFile.open("../output/checkGrad.txt");

	if (outFile.fail() )
		throw std::runtime_error("print_checkGrad(...): nie udalo sie otworzyc pliku.");

	double gama = input.w_hsig;
	const int N = solAdj.e.cols();
	// const int n = solAdj.e.rows();

	MatrixXd sol(5, N);
	sol.row(0) = solFwd.T;
	sol.row(1) = solAdj.c.row(0);									// c_adj
	for (int i = 0; i < input.Nsamples; i++)						// 
		sol(2,i) = ( 2*gama*uVec(i) - solAdj.c(0,i) ) * input.dt;	// gradient
	sol.row(3) = solFwd.alpha.row(0);								// x
	sol.row(4) = solFwd.dalpha.row(0);								// dx

	outFile << sol;
	outFile.close();
}