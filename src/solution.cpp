#include "../include/solution.h"
#include "../include/constants.h"
#include <fstream>

const bool _solution_::INTERMEDIATE_VALUE = false;
const bool _solution_::NODE_VALUE = true;

static int compare_min(const int i, const int j, const VectorXd& T, const double& t) {
    return abs(T[i] - t) < abs(T[j]-t) ? i : j;
}
std::pair<int, const bool> _solution_::atTime(const double& t, const _input_& input) const {
	/* 
	* Znajduje indeks wartosci t w wektorze T metoda bisekcji. 
	* Ostatnie kilka krokow odbywa sie liniowo, zeby nie bawic sie corner case'y
	* Zwraca pare <indeks, NodeValue>
	*/
	const double eps = 1e-10;
    assert(t >= 0.0 && t <= input.Tk + eps);
	const double& dt = input.dt;
    int begin = 0;
    int end = T.size()-1;

	const int linearSearchRegion = 5;
	while (end - begin > linearSearchRegion) {
		const int mid = (begin + end) / 2;
		if ( abs(t - T(mid)) < 1e-10 ) return std::make_pair(mid, _solution_::NODE_VALUE);
		if (t > T(mid))		begin = mid;
		else				end   = mid;
	}

	int  min_ind = begin;
	for (int i = begin+1; i <= end; i++)
        min_ind = compare_min(min_ind, i, T, t);

    if (abs(T(min_ind) - t) > dt)
	    throw std::runtime_error("atTime: index not found");
    else if ( abs(T(min_ind) 	 - t) < 1e-10 ) 
        return std::make_pair(min_ind, _solution_::NODE_VALUE);
    else
        return std::make_pair(min_ind, _solution_::INTERMEDIATE_VALUE);
}

dataJoint _solution_::getDynamicValues(const int index, const _input_& input) const {
	const int Nvars = alpha.rows();
	dataJoint data = dataJoint(Nvars);
	data.t       = T(index);
	data.alpha   = alpha.col(index);
	data.dalpha  = dalpha.col(index);
	data.d2alpha = d2alpha.col(index);
	data.lambda  = lambda.col(index);
	data.pjoint  = pjoint.col(index);

	return data;
}

void _solution_::show_xStatus(const VectorXd& u, const _input_& input) const {
	double mean, rms, min, max, max_u;
	const VectorXd& x0 = alpha.row(0);
	mean = x0.mean();
	rms = sqrt(x0.array().square().sum() / input.Nsamples);
	min = x0.minCoeff();
	max = x0.maxCoeff();
	max_u = u.array().abs().maxCoeff();
	std::cout 	<< "---results:---"
				<< "\nmean  = " << mean  << "\nRMS   = " << rms
				<< "\nmin   = " << min   << "\nmax   = " << max 
				<< "\nmax u = " << max_u << std::endl;
}

void _solution_::print() const {

	IOFormat exportFmt(FullPrecision, 0, " ", "\n", "", "", "", "");
	std::ofstream outFile;
	outFile.open("../output/results.txt");

	if (outFile.fail() )
		throw std::runtime_error("nie udalo sie otworzyc pliku.");

	const int N = alpha.cols();
	const int n = alpha.rows();
	MatrixXd sol(2*n+1, N);
	// MatrixXd sol(3*n+1, N);

	sol.row(0) = T;
	sol.block(1, 0, n, N) = dalpha;
	sol.block(1+n, 0, n, N) = alpha;
	// sol.block(1+n, 0, 2*n, N) = lambda;

	outFile << sol;
	outFile.close();
}

void _solution_::print(const VectorXd& u) const {

	IOFormat exportFmt(FullPrecision, 0, " ", "\n", "", "", "", "");
	std::ofstream outFile;
	outFile.open("../output/resultsForward.txt");

	if (outFile.fail() )
		throw std::runtime_error("_solution_::print(const VectorXd&): nie udalo sie otworzyc pliku.");

	const int N = alpha.cols();
	const int n = alpha.rows();
	MatrixXd sol(2*n+2, N);

	sol.row(0) = T;
	sol.block(1, 0, n, N) = dalpha;
	sol.block(1+n, 0, n, N) = alpha;
	sol.row(2*n+1) = u;

	outFile << sol;
	outFile.close();

	outFile.open("../output/checkOpt.txt");
	if (outFile.fail() )
		throw std::runtime_error("_solution_::print(const VectorXd&): nie udalo sie otworzyc pliku (2).");

	MatrixXd solCheck(8, N);
	solCheck.row(0) = T;
	solCheck.row(1) = u;
	solCheck.row(2) = alpha.row(0);    // x
	solCheck.row(3) = alpha.row(1);    // phi1
	solCheck.row(4) = alpha.row(n-1);  // phi_n
	solCheck.row(5) = dalpha.row(0);   // dx
	solCheck.row(6) = dalpha.row(1);   // dphi1
	solCheck.row(7) = dalpha.row(n-1); // dphi_n

	outFile << sol;
	outFile.close();
}

void _solutionAdj_::print() const {
	if (HDCA_formulation)	printHDCA();
	else					printGlobal();
}

void _solutionAdj_::printHDCA() const {

	IOFormat exportFmt(FullPrecision, 0, " ", "\n", "", "", "", "");
	std::ofstream outFile;
	outFile.open("../output/resultsAdjoint.txt");

	if (outFile.fail() )
		throw std::runtime_error("_solutionAdj_::printHDCA(): nie udalo sie otworzyc pliku.");

	const int N = e.cols();
	const int n = e.rows();
	MatrixXd sol(2*n+1 + 3, N);
	sol.row(0) = T;
	sol.block(1, 0, n, N) = c; // zmieniamy kolejnosc, zeby latwiej bylo znalezc w pliku .txt gradient
	sol.block(1+n, 0, n, N) = e;
	sol.block(1+2*n, 0, 3, N) = norms;
	outFile << sol;
	outFile.close();
}

void _solutionAdj_::printGlobal() const {

	IOFormat exportFmt(FullPrecision, 0, " ", "\n", "", "", "", "");
	std::ofstream outFile;
	outFile.open("../output/resultsAdjointGlobal.txt");

	if (outFile.fail() )
		throw std::runtime_error("_solutionAdj_::printGlobal: nie udalo sie otworzyc pliku.");

	const int N = eta.cols();
	const int n = eta.rows();
	MatrixXd sol(2*n+1 + 3, N);
	sol.row(0) = T;
	sol.block(1, 0, n, N) = eta;
	sol.block(1+n, 0, n, N) = ksi;
	sol.block(1+2*n, 0, 3, N) = norms;
	outFile << sol;
	outFile.close();
}

dataAbsolute::dataAbsolute(const VectorXd& e, const VectorXd& c, const dataJoint& data, const _input_ &input) :
							t(data.t), q(3*input.Nbodies), dq(3*input.Nbodies), d2q(3*input.Nbodies), 
							eta(3*input.Nbodies), ksi(3*input.Nbodies),  lambda(data.lambda) // to do: move() lambda
{
    /**
     * Converts joint alpha, dalpha, and d2alpha to their absolute counterparts [q, dq, d2q].
     */
	const VectorXd& alpha   = data.alpha;
	const VectorXd& dalpha  = data.dalpha;
	const VectorXd& d2alpha = data.d2alpha;

    q.segment(0,6)   << alpha(0),   0.0, 0.0,  alpha(0),   0.0,  alpha(1);
	dq.segment(0,6)  << dalpha(0),  0.0, 0.0,  dalpha(0),  0.0,  dalpha(1);
	d2q.segment(0,6) << d2alpha(0), 0.0, 0.0,  d2alpha(0), 0.0,  d2alpha(1);
	eta.segment(0,6) << e(0),  	    0.0, 0.0,  e(0),       0.0,  e(1);
	ksi.segment(0,6) << c(0),  	    0.0, 0.0,  c(0),       0.0,  c(1);

// zrownoleglamy wszystkie obliczenia na raz
    for (int i = 2; i < input.Nbodies; i++)
    {
        const int prev = i - 1;
        const double phi = q(3*prev + 2);
        const double om  = dq(3*prev + 2);
        const double eps = d2q(3*prev + 2);
        const Vector2d& s12 = input.pickBodyType(prev).s12;
        const double ksiRot = ksi(3*prev + 2);
        const double etaRot = eta(3*prev + 2);

        q.segment(3*i, 2) = q.segment(3*prev, 2) +  Rot(phi) * s12;
        q(3*i + 2) = phi + alpha(i);

        dq.segment(3*i, 2) = dq.segment(3*prev, 2) +  Om*Rot(phi) * s12 * om;
        dq(3*i + 2) = om + dalpha(i);

        d2q.segment(3*i, 2) = d2q.segment(3*prev, 2) +  (Om*Rot(phi) * eps - Rot(phi) * om*om) * s12;
        d2q(3*i + 2) = eps + d2alpha(i);

		/* Absolute adjoint variables are computed identically as their dynamic counterparts (vel and acc) */
		ksi.segment(3*i, 2) = ksi.segment(3*prev, 2) +  Om*Rot(phi) * s12 * ksiRot;
        ksi(3*i + 2) = ksiRot + c(i);

        eta.segment(3*i, 2) = eta.segment(3*prev, 2) +  (Om*Rot(phi) * etaRot - Rot(phi)*om * ksiRot) * s12;
        eta(3*i + 2) = etaRot + e(i);
	}
}