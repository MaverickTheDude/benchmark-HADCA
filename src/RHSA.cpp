#include "../include/utils.h"
#include "../include/input.h"
#include "../Eigen/Dense"
#include "../include/assembly.h"
#include "../include/solution.h"
#include "../include/task/Phi.h"
#include "../include/adjoint.h"
    #include <iostream>
using std::vector;

VectorXd RHS_ADJOINT(const double& tau, const VectorXd& y, const VectorXd& uVec, const _solution_& solutionFwd, const _input_& input) {
    _solutionAdj_ solution;
    return RHS_ADJOINT(tau, y, uVec, solutionFwd, input, solution);
}

VectorXd RHS_ADJOINT(const double& tau, const VectorXd& y, const VectorXd& uVec,
                     const _solution_& solutionFwd, const _input_& input, _solutionAdj_& solution) {
    const double t = input.Tk - tau;
    const unsigned int n = y.size()/2;
    VectorXd e = y.head(n);
    VectorXd c = y.tail(n);
    dataJoint state = interpolate(t, solutionFwd, input);
    dataAbsolute stateAbs = dataAbsolute(e, c, state, input);
    VectorXd de(n);
    const double u = interpolateControl(t, uVec, input);

    using aaA = aligned_allocator<AssemblyAdj>;
    vector<vector<AssemblyAdj, aaA >, aaA > tree;
    tree.resize(input.Ntiers);
    vector<AssemblyAdj, aaA >& leafBodies = tree[0];
    leafBodies.resize(input.tiersInfo[0]);

    /* Initialize leaf bodies (baza indukcyjna) */
    // https://eigen.tuxfamily.org/dox/group__TopicStlContainers.html --> The case of std::vector
	size_t *prefix;
#pragma omp parallel
{
    int ithread = 0, nthreads = 1;
# ifdef _OPENMP
    ithread  = omp_get_thread_num();
    nthreads = omp_get_num_threads();
# endif
#pragma omp single
    {
        prefix = new size_t[nthreads+1];
        prefix[0] = 0;
    }
    vector<AssemblyAdj, aaA > vec_private;
    vec_private.reserve(input.Nbodies / nthreads);
#pragma omp for schedule(static) nowait
    for (int i = 0; i < input.Nbodies; i++) {
        vec_private.emplace_back(i, stateAbs, u, input);
    }
    prefix[ithread+1] = vec_private.size();
#pragma omp barrier
#pragma omp single
{
    for(int i=1; i<(nthreads+1); i++) prefix[i] += prefix[i-1];
} // implicit barrier
    auto dest_iter = std::next(leafBodies.begin(), prefix[ithread]);
    std::copy(vec_private.begin(), vec_private.end(), dest_iter );
}
    delete[] prefix;

    for (int i = 1; i < input.Ntiers; i++) {
        vector<AssemblyAdj, aaA >& branch = tree[i];
        vector<AssemblyAdj, aaA >& upperBranch = tree[i-1];
        branch.resize(input.tiersInfo[i]);
        const int endOfBranch = input.tiersInfo[i-1] - 1;

        /* core loop of DCA algorithm */
        // for (int j = 0; j < endOfBranch; j+=2) {
        //     branch.emplace_back(upperBranch[j], upperBranch[j+1]);
        // }
#pragma omp parallel
{
        int ithread = 0, nthreads = 1;
    # ifdef _OPENMP
        ithread  = omp_get_thread_num();
        nthreads = omp_get_num_threads();
    # endif
    #pragma omp single
    {
        prefix = new size_t[nthreads+1];
        prefix[0] = 0;
    }
        vector<AssemblyAdj, aaA > vec_private;
        vec_private.reserve(input.tiersInfo[i] / nthreads);
    #pragma omp for schedule(static) nowait
        for (int j = 0; j < endOfBranch; j+=2) {
            vec_private.emplace_back(upperBranch[j], upperBranch[j+1]);
        }
        prefix[ithread+1] = vec_private.size();
    #pragma omp barrier
    #pragma omp single
    {
        for(int i=1; i<(nthreads+1); i++) prefix[i] += prefix[i-1];
    } // implicit barrier
        auto dest_iter = std::next(branch.begin(), prefix[ithread]);
        std::copy(vec_private.begin(), vec_private.end(), dest_iter );
}
        delete[] prefix;
        
        /* case: odd # elements */
		if (input.tiersInfo[i-1] % 2 == 1)
            branch.back() = upperBranch.back(); // note: shallow copy is exactly what we need
    }
    
    /* base body connection */
    AssemblyAdj& AssemblyS = tree[input.Ntiers-1][0];
	AssemblyS.connect_base_body();
	AssemblyS.disassemble();

    const int prelastTier = input.Ntiers-2;
	for (int i = prelastTier; i > 0; i--) {
		const int end_of_branch = input.tiersInfo[i-1] % 2 == 0 ?
				input.tiersInfo[i] : input.tiersInfo[i]-1;

#pragma omp parallel for schedule(static)
		for (int j = 0; j < end_of_branch; j++) {
			tree[i].at(j).disassemble();
		}

        /* case: odd # elements */
		if (input.tiersInfo[i-1] % 2 == 1) {
			tree[i-1].back().setAll(tree[i].back());
		}
	}

    /* joint space projection */
    MatrixXd P1art(3, input.Nbodies+1);
    const Vector3d Hground = input.pickBodyType(0).H;
	de(0) = Hground.transpose() * leafBodies[0].calculate_dETA1();
    VectorXd deta(3*n);
    deta.segment(0, 3) = leafBodies[0].calculate_dETA1();

#pragma omp parallel for schedule(static)
	for (int i = 1; i < input.Nbodies; i++) {
        const Vector3d H = input.pickBodyType(i).H;
		Vector3d dE1B = leafBodies[i].calculate_dETA1();
		Vector3d dE2A = leafBodies[i-1].calculate_dETA2();
		de(i) = H.transpose() * (dE1B - dE2A);
        deta.segment(3*i, 3) = dE1B;
	}

    VectorXd dy(2*n);
    dy.head(n) = de;
    dy.tail(n) = -e; // == dc

    if (solution.dummySolution())
        return dy;

    const int ind = atTime(t, solutionFwd.T, input).first;

    solution.set_c(ind, c);
    solution.set_e(ind, e);

    if (input.logConstr) {
        task::Phi Phi(input);
        Vector3d norms;
        norms(0) = (Phi.q(stateAbs.q) * stateAbs.ksi).norm();
        norms(1) = (Phi.q(stateAbs.q) * stateAbs.eta + Phi.ddtq(stateAbs.q, stateAbs.dq) * stateAbs.ksi).norm();
        norms(2) = ( Phi.q(stateAbs.q) * deta - 2 * Phi.ddtq(stateAbs.q, stateAbs.dq) * stateAbs.eta  -
                    Phi.d2dt2q(stateAbs.q, stateAbs.dq, stateAbs.d2q) * stateAbs.ksi ).norm();
        solution.setNorms(ind, norms);
    }

    return dy;
}


VectorXd boundaryConditions(const _solution_& solutionFwd, const _input_& input, int formulation) {
	const int& Nbodies = input.Nbodies;
	const int& Nconstr = input.Nconstr;
	MatrixXd A(3*Nbodies + Nconstr, 3*Nbodies + Nconstr);
	dataJoint state = interpolate(input.Tk, solutionFwd, input);
	dataAbsolute stateAbs = dataAbsolute(VectorXd::Zero(Nbodies), VectorXd::Zero(Nbodies), state, input);
	Adjoint derivatives(input);
	A.block(0, 0, 3*Nbodies, 3*Nbodies) = derivatives.M->operator()(stateAbs.q);
	A.block(0, 3*Nbodies, 3*Nbodies, Nconstr) = derivatives.Phi->q(stateAbs.q).transpose();
	A.block(3*Nbodies, 0, Nconstr, 3*Nbodies) = derivatives.Phi->q(stateAbs.q);
	
	// velocity condition
	VectorXd RHS = VectorXd::Zero(3*Nbodies+Nconstr);
	RHS(0) = input.w_Sdq * 2.0 * stateAbs.dq(0);
	VectorXd adjointAbs = A.partialPivLu().solve(RHS);
	VectorXd xi = -1.0 * adjointAbs.head(3*Nbodies);
	VectorXd nu = adjointAbs.tail(Nconstr);
	// components equal to zero if no vel. condition
	VectorXd added = derivatives.M->ddqdq(stateAbs.q, stateAbs.dq, xi) - derivatives.Phi->ddtq(stateAbs.q, stateAbs.dq).transpose() * nu;
	VectorXd added2 = - derivatives.Phi->ddtq(stateAbs.q, stateAbs.dq) * xi;

	// position condition
	VectorXd Sq  = VectorXd::Zero(3*Nbodies);
	VectorXd h_dq = derivatives.h->dq(stateAbs.q, stateAbs.dq);
	// h_1::dq(const VectorXd& q, const VectorXd& dq)
	Sq(0) = input.w_Sq * 2.0 * stateAbs.q(0);
	RHS.head(3*Nbodies) = Sq + h_dq + added;
	RHS.tail(Nconstr) = added2;
	adjointAbs = A.partialPivLu().solve(RHS);
	VectorXd eta = adjointAbs.head(3*Nbodies);

	if (formulation == _solutionAdj_::GLOBAL) {
		VectorXd y(6*Nbodies);
		y.head(3*Nbodies) = eta;
		y.tail(3*Nbodies) = xi;
		return y;
	}

	VectorXd e(Nbodies), c(Nbodies);
    const Vector3d Hground = input.pickBodyType(0).H;
	e(0) = Hground.transpose() * eta.segment(0, 3);
	c(0) = Hground.transpose() * xi.segment(0, 3);
	for (int i = 1; i < Nbodies; i++) {
		const int prev = i-1;
        const Vector3d H = input.pickBodyType(i).H;
		Matrix3d S12  =  SAB("s12", prev, stateAbs.q(3*prev+2), 					   input);
		Matrix3d dS12 = dSAB("s12", prev, stateAbs.q(3*prev+2), stateAbs.dq(3*prev+2), input);
		Vector3d X1B = xi.segment(3*i, 3);
		Vector3d X2A = S12.transpose() * xi.segment(3*prev, 3);
		Vector3d E1B = eta.segment(3*i, 3);
		Vector3d E2A = S12.transpose() * eta.segment(3*prev, 3) + dS12.transpose() * xi.segment(3*prev, 3);
		e(i) = H.transpose() * (E1B - E2A);
		c(i) = H.transpose() * (X1B - X2A);
	}

	VectorXd y(2*Nbodies);
	y.head(Nbodies) = e;
	y.tail(Nbodies) = c;

	return y;
}


VectorXd RHS_ADJOINT_GLOBAL(const double& tau, const VectorXd& y, const VectorXd& uVec, const _solution_& solutionFwd, const _input_& input) {
    _solutionAdj_ solution;
    return RHS_ADJOINT_GLOBAL(tau, y, uVec, solutionFwd, input, solution);
}

VectorXd RHS_ADJOINT_GLOBAL(const double& tau, const VectorXd& y, const VectorXd& uVec,
                     const _solution_& solutionFwd, const _input_& input, _solutionAdj_& solution) {
	const int& Nbodies = input.Nbodies;
	const int& Nconstr = input.Nconstr;
	const double t = input.Tk - tau;
    const unsigned int n = y.size()/2; // = 3*Nbodies
    VectorXd eta = y.head(n);
    VectorXd ksi = y.tail(n);
    dataJoint state = interpolate(t, solutionFwd, input);
    dataAbsolute stateAbs = dataAbsolute(VectorXd::Zero(Nbodies), VectorXd::Zero(Nbodies), state, input);
    const double u = interpolateControl(t, uVec, input);
    
	Adjoint derivatives(input); 
	MatrixXd A = MatrixXd::Zero(n + Nconstr, n + Nconstr);
	A.block(0, 0, n, n) = derivatives.M->operator()(stateAbs.q);
	A.block(0, n, n, Nconstr) = derivatives.Phi->q(stateAbs.q).transpose();
	A.block(n, 0, Nconstr, n) = derivatives.Phi->q(stateAbs.q);
	VectorXd RHS = VectorXd::Zero(n + Nconstr);
	RHS.head(n) = derivatives.RHS(stateAbs.q, stateAbs.dq, stateAbs.d2q, stateAbs.lambda, (VectorXd(1) << u).finished(), eta, ksi);
	RHS.tail(Nconstr) = 2 * derivatives.Phi->ddtq(stateAbs.q, stateAbs.dq) * eta +
                			derivatives.Phi->d2dt2q(stateAbs.q, stateAbs.dq, stateAbs.d2q) * ksi;
	VectorXd adjointAbs = A.partialPivLu().solve(RHS);
	VectorXd deta = adjointAbs.head(n);

	VectorXd dy(2*n);
	dy.head(n) = deta;
	dy.tail(n) = -eta;

    if (solution.dummySolution())
        return dy;

    const int ind = atTime(t, solutionFwd.T, input).first;

    task::Phi Phi(input);
    Vector3d norms;
    norms(0) = (Phi.q(stateAbs.q) * ksi).norm();
    norms(1) = (Phi.q(stateAbs.q) * eta + Phi.ddtq(stateAbs.q, stateAbs.dq) * ksi).norm();
    norms(2) = (Phi.q(stateAbs.q) * deta - 2 * Phi.ddtq(stateAbs.q, stateAbs.dq) * eta  -
                Phi.d2dt2q(stateAbs.q, stateAbs.dq, stateAbs.d2q) * ksi ).norm();

    solution.set_ksi(ind, ksi);
    solution.set_eta(ind, eta);
    solution.setNorms(ind, norms);
    return dy;
}