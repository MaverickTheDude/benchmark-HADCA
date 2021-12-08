#include <iomanip>
#include <iostream>
#include <omp.h>
#include <time.h>
#include "../include/utils.h"
#include "../Eigen/Dense"

/* cummulative sum example: https://stackoverflow.com/a/65681828/4283100 */

VectorXd jointToAbsolutePosition(const VectorXd &alpha);
VectorXd jointToAbsolutePositionParallel(const VectorXd &alpha);
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
        cout << "OpenMP test executed in parallel on " << omp_get_num_threads() << " threads with " << size << " bodies." << endl;
}
# else
    cout << "Caution: Your sourcecode was compiled without switching OpenMP on." << endl;
# endif 

    double wt1, wt, ct;
	clock_t start, end;

    srand((unsigned int) time(0));
    VectorXd alpha = VectorXd::Random(size);

    start = clock();
    wt1 = omp_get_wtime();    
        
    VectorXd t1 = jointToAbsolutePositionParallel(alpha);

    end = clock();
    ct = double(end - start) / double(CLOCKS_PER_SEC);
    wt = omp_get_wtime() - wt1;
    std::cout << "parallel clock time:\t" << std::setprecision(6) << ct << "\twall time:\t" << std::setprecision(6) << wt << "\n";

    start = clock();
    wt1 = omp_get_wtime();

    VectorXd t2 = jointToAbsolutePosition(alpha);

    end = clock();
    ct = double(end - start) / double(CLOCKS_PER_SEC);
    wt = omp_get_wtime() - wt1;
    std::cout << "sequential clock time:\t" << std::setprecision(6) << ct << "\twall time:\t" << std::setprecision(6) << wt << "\n";

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

VectorXd jointToAbsolutePositionParallel(const VectorXd &alpha)
{
    /**
     * Converts joint coordinates alpha to absolute coordinates q (3 * number_of_bodies).
     */
    const int size = alpha.size();

    VectorXd q(3 * size);
    q.segment(0, 3) << alpha(0), 0.0, 0.0;      // box: (x0, y0, fi0)
    q.segment(3, 3) << alpha(0), 0.0, alpha(1); // first link: (x1=x0, y1, fi1)

    Vector2d s12;
    // s12 << 1, 0;
    s12 << .2, .5;

	VectorXd phi   = VectorXd::Zero(size);
    MatrixXd r_mat = MatrixXd::Zero(2, size + 1);

    double sum     = 0.0;
    Vector2d r_sum = Vector2d::Zero();
    
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
    int chunk = (size - 1 + Nthr - 1) / Nthr;
#pragma omp parallel for schedule(static, chunk) firstprivate(sum)
	for (int i = 1; i < size; i++) {
		sum += alpha(i);
		phi(i) = sum;
		sumThr_phi(omp_get_thread_num() + 1) = sum;
	}

	// Compute offsets for consecutive threads (phi)
	for (int i=1; i < Nthr; i++)
		sumThr_phi(i) += sumThr_phi(i - 1);

	// Apply offsets and compute cumsums for (r)
#pragma omp parallel for schedule(static, chunk) firstprivate(r_sum)
	for (int i = 1; i < size; i++) {
        phi(i) += sumThr_phi(omp_get_thread_num());
        r_sum += Rot(phi(i)) * s12;       
        r_mat.col(i + 1) = r_sum;
        sumThr_r.col(omp_get_thread_num() + 1) = r_sum;
	}

	// Compute offsets for consecutive threads (r)
	for (int i=1; i < Nthr; i++)
		sumThr_r.col(i) += sumThr_r.col(i - 1);

	// Apply offsets (r) and build q vactor
#pragma omp parallel for schedule(static, chunk)
	for (int i = 1; i < size - 1; i++) {
        const int next = i + 1;
        r_mat.col(next) += sumThr_r.col(omp_get_thread_num());
        q.segment(3*next, 2) = r_mat.col(next) + r_bias;
        q(3*next+2) = phi(next);
    }

    return q;
}

VectorXd jointToAbsolutePosition(const VectorXd &alpha)
{
    /**
     * Converts joint coordinates alpha to absolute coordinates q (3 * number_of_bodies).
     */

    const int size = alpha.size();
    VectorXd q(3 * size);
    Vector2d s12;
    // s12 << 1, 0;
    s12 << .2, .5;

    q.segment(0, 3) << alpha(0), 0.0, 0.0;      // box: (x0, y0, fi0)
    q.segment(3, 3) << alpha(0), 0.0, alpha(1); // first link: (x1=x0, y1, fi1)

    for (int i = 2; i < size; i++)
    {
        const int prev = i - 1;
        q.segment(3 * i, 2) = q.segment(3 * prev, 2) +
                              Rot(q(3 * prev + 2)) * s12;
        q(3 * i + 2) = q(3 * prev + 2) + alpha(i);
    }

    return q;
}

#if false
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


VectorXd joint2AbsAngles(const VectorXd &alpha)
{
    VectorXd alphaAbs(alpha.size());
    alphaAbs(0) = alpha(0);
    alphaAbs(1) = alpha(1);

// to do: parallelize, note: cumulative sum 
    for (int i = 2; i < alphaAbs.size(); i++)
    {
        alphaAbs(i) = alphaAbs(i - 1) + alpha(i);
    }
    return alphaAbs;
}


dataJoint interpolate(const double& t, const _solution_& solutionFwd, const _input_& input) {
    assert(t >= 0.0 && t <= input.Tk);

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
    else if (baseInd == input.Nsamples-2) // pre-last node
        baseInd--;
    dataJoint sr1 = solutionFwd.getDynamicValues(baseInd-1, input);
    dataJoint s00 = solutionFwd.getDynamicValues(baseInd  , input);
    dataJoint sf1 = solutionFwd.getDynamicValues(baseInd+1, input);
    dataJoint sf2 = solutionFwd.getDynamicValues(baseInd+2, input);

// to do: parallelize
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

#endif