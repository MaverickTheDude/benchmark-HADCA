#include "../include/utils.h"
#include "../include/constants.h"
#include "../Eigen/Dense"

#include <iomanip>
#include <iostream>
#include <omp.h>
#include <time.h>

/* cummulative sum example: https://stackoverflow.com/a/65681828/4283100 */

VectorXd jointToAbsolutePositionParallel(const VectorXd &alpha, const _input_& input);
VectorXd jointToAbsoluteVelocityParallel(const VectorXd &alpha, const VectorXd &dalpha, const _input_ &input);
VectorXd absolutePositionToAbsoluteAlphaParallel(const VectorXd& q);
VectorXd joint2AbsAnglesParallel(const VectorXd &alpha);
 
using namespace Eigen;
using std::cout; using std::endl;

int main(int argc, char* argv[]) {
    int size = 5, Nthreads = 4;
    if (argc > 1) {
        Nthreads = atoi(argv[1]);
        size     = atoi(argv[2]);
    }
# ifdef _OPENMP
    omp_set_num_threads(Nthreads);
#pragma omp parallel
{
    if (omp_get_thread_num() == 0)
        cout << "OpenMP test executed in parallel on " << omp_get_num_threads() << " threads with " << size << " bodies.\n" << endl;
}
# else
    cout << "Caution: Your sourcecode was compiled without switching OpenMP on.\n" << endl;
# endif 

    double wt1, wt, ct;
	clock_t start, end;

    srand((unsigned int) time(0));
    VectorXd alpha  = VectorXd::Random(size);
    VectorXd dalpha = VectorXd::Random(size);
    VectorXd q      = VectorXd::Random(3*size);

    _input_ input = _input_(size);
    // input.setTk(0.1);
    // _solution_ solutionFwd = RK_solver(input);

    start = clock();
    wt1 = omp_get_wtime();    
        
    VectorXd t1 = jointToAbsolutePositionParallel(alpha, input);
    // VectorXd t1 = jointToAbsoluteVelocityParallel(alpha, dalpha, input);
    // VectorXd t1 = joint2AbsAnglesParallel(alpha);
    // VectorXd t1 = absolutePositionToAbsoluteAlphaParallel(q);

    end = clock();
    ct = double(end - start) / double(CLOCKS_PER_SEC);
    wt = omp_get_wtime() - wt1;
    std::cout << "parallel clock time:\t" << std::setprecision(6) << ct << "\twall time:\t" << std::setprecision(6) << wt << "\n";

    start = clock();
    wt1 = omp_get_wtime();

    VectorXd t2 = jointToAbsolutePosition(alpha, input);
    // VectorXd t2 = jointToAbsoluteVelocity(alpha, dalpha, input);
    // VectorXd t2 = joint2AbsAngles(alpha);
    // VectorXd t2 = absolutePositionToAbsoluteAlpha(q);

    end = clock();
    ct = double(end - start) / double(CLOCKS_PER_SEC);
    wt = omp_get_wtime() - wt1;
    std::cout << "Position: sequential clock time:\t" << std::setprecision(6) << ct << "\twall time:\t" << std::setprecision(6) << wt << "\n";

    double diff = (t1 - t2).norm();
    cout << "diff = " << diff << '\n' << endl;
    if (diff > 1e-5)
    {
        MatrixXd x(2, 3*size);
        x.row(0) = t1.transpose();
        x.row(1) = t2.transpose();
        cout << x << endl;
    }

    return 0;
}

/* tutaj zysk jest ledwo widoczny dla size > 1024 */
VectorXd absolutePositionToAbsoluteAlphaParallel(const VectorXd& q)
{
    // https://eigen.tuxfamily.org/dox-devel/group__TutorialSlicingIndexing.html
    const int qSize = q.size();
    VectorXd absoluteAlpha = VectorXd::Zero(qSize / 3);
#pragma omp parallel for schedule(static)
    for(int i = 2; i < qSize; i += 3)
        absoluteAlpha((i - 2) / 3) = q(i);
    
    return absoluteAlpha;
}

VectorXd joint2AbsAnglesParallel(const VectorXd &alpha)
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

VectorXd jointToAbsolutePositionParallel(const VectorXd &alpha, const _input_ &input)
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

VectorXd jointToAbsoluteVelocityParallel(const VectorXd &alpha, const VectorXd &dalpha, const _input_ &input)
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