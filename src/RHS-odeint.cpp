#include "../Eigen/Dense"
#include "../include/utils.h"
#include "../include/input.h"
#include "../include/assembly.h"
#include "../include/solution.h"
#include "../include/task/Phi.h"
#include "../include/odeint.h"
// #include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/stepper/controlled_runge_kutta.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_cash_karp54.hpp>
#include <boost/numeric/odeint/integrate/integrate_times.hpp>
#include <omp.h>
    #include <iostream>
using std::vector;


_solution_ RK_solver_odeInt(const VectorXd& uVec, const _input_& input) {
    using namespace boost::numeric;

	const double dt = input.dt;
	const int Nbodies = input.Nbodies;
    _solution_ solution(input);

    std::vector<double> y0(2*Nbodies);
    std::vector<double> T(input.Nsamples);
    std::generate(T.begin(), T.end(), [n = 0, &dt]() mutable { return n++ * dt; });
    // przestroga: w ten sposob dwa wywolania RK_solver_odeInt() generuja rozna dziedzine (statyczne t)
    // for_each(T.begin(), T.end(), [&](double& v) {static double t = 0.0; v = t; t+=dt;});

    solution.setT(T);
    auto dest_iter = std::next(y0.begin(), Nbodies);
    std::copy(input.pjoint0.data(), input.pjoint0.data() + Nbodies, y0.begin());
    std::copy(input.alpha0.data(),  input.alpha0.data()  + Nbodies, dest_iter);

    VectorXd b_alphaAbs(Nbodies);
    VectorXd b_dAlphaAbs(Nbodies);
    MatrixXd b_P1art(3, Nbodies+1);
    std::vector<vector<Assembly, aaA >, aaA > b_tree;
    b_tree.resize(input.Ntiers);
    for (int i = 0; i < input.Ntiers; ++i)
        b_tree[i].resize(input.tiersInfo[i]);


    //[ integration_class
    RHS_HDCA_ODE rho(input, uVec, b_alphaAbs, b_dAlphaAbs, b_P1art, b_tree);

    // define_adaptive_stepper
    typedef odeint::runge_kutta_cash_karp54< state_type > error_stepper_type;
    typedef odeint::controlled_runge_kutta< error_stepper_type > controlled_stepper_type;
    controlled_stepper_type controlled_stepper;

    size_t steps = odeint::integrate_times(controlled_stepper , rho , 
                    y0, T.begin() , T.end(), input.dt, 
                    odeint_observer(input, rho, solution, uVec, b_alphaAbs, b_dAlphaAbs, b_P1art, b_tree) );

	return solution;
};

_solutionAdj_ RK_AdjointSolver_odeInt(const VectorXd& uVec, const _solution_& solutionFwd, 
                                      const _input_& input, const int& formulation) {
    using namespace boost::numeric;
    if (formulation == _solutionAdj_::GLOBAL)
        throw std::runtime_error("RK_AdjointSolver_odeInt: unsuported formulation (GLOBAL)");
	const int Nbodies = input.Nbodies;
    _solutionAdj_ solution(input);

	VectorXd y0eigen = boundaryConditions(solutionFwd, input, formulation);

    std::vector<double> y0(2*Nbodies);
    std::vector<double> T(input.Nsamples);
    std::generate(T.begin(), T.end(), [n = 0, &input]() mutable { return n++ * input.dt; });
    std::copy(y0eigen.data(), y0eigen.data() + 2*Nbodies, y0.begin());

    // integration_class
    RHS_ADJ_ODE rho(input, uVec, solutionFwd);

    // define_adaptive_stepper
    typedef odeint::runge_kutta_cash_karp54< state_type > error_stepper_type;
    typedef odeint::controlled_runge_kutta< error_stepper_type > controlled_stepper_type;
    controlled_stepper_type controlled_stepper;

    size_t steps = odeint::integrate_times(controlled_stepper, rho, 
                    y0, T.begin(), T.end(), input.dt, 
                    odeint_observer_adj(input, rho, solutionFwd, solution) );

	return solution;
};

// ======== ADJOINT ========

void RHS_ADJ_ODE::operator() (const state_type &y, state_type &dy , const double tau) {
    const double t = input.Tk - tau;
    const unsigned int n = y.size()/2;
    VectorXd e = VectorXd::Map(y.data(),   n);
    VectorXd c = VectorXd::Map(y.data()+n, n);
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

    VectorXd dc = -e;
    dy.resize(2*input.Nbodies);
    auto dest_iter = std::next(dy.begin(), n);
    std::copy(de.data(), de.data()+n, dy.begin());
    std::copy(dc.data(), dc.data()+n,  dest_iter);
}

void odeint_observer_adj::operator()( const state_type &y , double tau )
{
    const double t = input.Tk - tau;
	const unsigned int n = y.size() / 2;
    VectorXd e = VectorXd::Map(y.data(),    n);
    VectorXd c = VectorXd::Map(y.data()+n,  n);

    const int ind = atTime(t, solutionFwd.T, input).first;

    solution.set_c(ind, c);
    solution.set_e(ind, e);

    if (input.logConstr) {
	    state_type dy;
	    dca_adj_obj.operator()(y, dy, t);
        VectorXd deta = VectorXd::Map(dy.data(),   n);
        dataJoint state = interpolate(t, solutionFwd, input);
        dataAbsolute stateAbs = dataAbsolute(e, c, state, input);


        task::Phi Phi(input);
        Vector3d norms;
        norms(0) = (Phi.q(stateAbs.q) * stateAbs.ksi).norm();
        norms(1) = (Phi.q(stateAbs.q) * stateAbs.eta + Phi.ddtq(stateAbs.q, stateAbs.dq) * stateAbs.ksi).norm();
        // to do: deta_joint -> deta_abs...
        // norms(2) = (Phi.q(stateAbs.q) * deta - 2 * Phi.ddtq(stateAbs.q, stateAbs.dq) * stateAbs.eta  -
                    // Phi.d2dt2q(stateAbs.q, stateAbs.dq, stateAbs.d2q) * stateAbs.ksi ).norm();
        solution.setNorms(ind, norms);
    }
}

// ======== FORWARD ========

void RHS_HDCA_ODE::operator() (const state_type &y, state_type &dy , const double t) {
// VectorXd RHS_HDCA(const double& t, const VectorXd& y, const VectorXd& uVec, const _input_& input, _solution_& solution) {
	const unsigned int n = y.size()/2;
    /* https://stackoverflow.com/a/40853254/4283100 - opcja wykorzystania zasobow y in-place
       nie da sie ze wzgledu na const correctness (y jest const referencja) 
       https://forum.kde.org/viewtopic.php?t=94839#p331301 - this works */
    VectorXd pjoint = VectorXd::Map(y.data(),   n);
    VectorXd alpha  = VectorXd::Map(y.data()+n, n);
    this->alphaAbs  = joint2AbsAngles(alpha);
    VectorXd dalpha(n), dpjoint(n);
    double u;
    if (input.Nsamples < 4)
        u = 0;     // we simulate only how fast RHS gets evaluated, so we don't need that
    else
        u = interpolateControl(t, uVec, input);
    

    // vector<vector<Assembly, aaA >, aaA > tree;
    // tree.resize(input.Ntiers);
    vector<Assembly, aaA >& leafBodies = this->tree[0];
    // leafBodies.resize(input.tiersInfo[0]);

    /* Initialize leaf bodies (baza indukcyjna) */
    // https://eigen.tuxfamily.org/dox/group__TopicStlContainers.html --> The case of std::vector
    // https://stackoverflow.com/a/18671256/4283100
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
    vector<Assembly, aaA > vec_private;
    vec_private.reserve(input.Nbodies / nthreads);
#pragma omp for schedule(static) nowait
    for(int i = 0; i < input.Nbodies; i++) {
        vec_private.emplace_back(i, alphaAbs, pjoint, u, input);
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
        vector<Assembly, aaA >& branch = tree[i];
        vector<Assembly, aaA >& upperBranch = tree[i-1];
        branch.resize(input.tiersInfo[i]);
        const int endOfBranch = input.tiersInfo[i-1] - 1;

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
        vector<Assembly, aaA > vec_private;
        vec_private.reserve(input.tiersInfo[i] / nthreads);
#pragma omp for schedule(static) nowait
        for(int j = 0; j < endOfBranch; j += 2) {
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
    Assembly& AssemblyS = tree[input.Ntiers-1][0];
	AssemblyS.connect_base_body();
	AssemblyS.disassembleAll();

    const int prelastTier = input.Ntiers-2;
	for (int i = prelastTier; i > 0; i--) {
		const int end_of_branch = input.tiersInfo[i-1] % 2 == 0 ?
				input.tiersInfo[i] : input.tiersInfo[i]-1;

#pragma omp parallel for schedule(static)
        for (int j = 0; j < end_of_branch; j++)
            tree[i].at(j).disassembleAll();

        /* case: odd # elements */
		if (input.tiersInfo[i-1] % 2 == 1) {
			tree[i-1].back().setAll(tree[i].back());
		}
	}

    /* velocity calculation */
    const Vector3d Hground = input.pickBodyType(0).H;
	dalpha(0) = Hground.transpose() * leafBodies[0].calculate_V1();
	this->P1art.col(0) = leafBodies[0].T1_() + Hground*pjoint(0);
    this->P1art.col(input.Nbodies).setZero();

#pragma omp parallel for schedule(static)
	for (int i = 1; i < input.tiersInfo[0]; i++) {
        const Vector3d H = input.pickBodyType(i).H;
		const Vector3d V1B = leafBodies[i].calculate_V1();
		const Vector3d V2A = leafBodies[i-1].calculate_V2();
		dalpha(i) = H.transpose() * (V1B - V2A);
		P1art.col(i) = leafBodies[i].T1_() + H*pjoint(i);
	}
    this->dAlphaAbs = joint2AbsAngles(dalpha);

// TODO: Parallelize (???)
    /* descendants term calculation */
    MatrixXd des = MatrixXd::Zero(3, input.Nbodies);
    Vector3d sum = Vector3d::Zero();
    const int preLastBody = input.Nbodies-2;
    for (int i = preLastBody; i >= 0; i--) {
        const Matrix3d dSc2 = (-1.0) * dSAB("s2C", i,   alphaAbs, dAlphaAbs, input);
        const Matrix3d dS1c =          dSAB("s1C", i+1, alphaAbs, dAlphaAbs, input);
        sum += (dSc2 + dS1c) * P1art.col(i+1);
		des.col(i) = sum;
    }

    /* joint dp from the articulated quantities */
#pragma omp parallel for schedule(static)
    for (int i = 0; i < input.Nbodies; i++) {
        const Vector3d H = input.pickBodyType(i).H;
        const Matrix3d dS1c = dSAB("s1C", i, alphaAbs, dAlphaAbs, input);
		dpjoint(i) = H.transpose() * (des.col(i) + leafBodies[i].Q1Art_() + dS1c*P1art.col(i) );
	}

    dy.resize(2*input.Nbodies);
    auto dest_iter = std::next(dy.begin(), n);
    std::copy(dpjoint.data(), dpjoint.data()+n, dy.begin());
    std::copy(dalpha.data(),  dalpha.data()+n,  dest_iter);

    // test me: move tree -> m_tree : jak sprawdzicz, czy robimy move, czy kopie? Pointery na data() sugeruja, ze to kopia...
    /* https://stackoverflow.com/a/22995765/4283100 */
/*     auto ptr = tree.data();
    auto m_ptr = m_tree.data();
    std::cout << " m_tree : " << m_ptr << ", tree: " << ptr << std::endl;
    m_tree = std::move(tree);
    // std::copy(tree.begin(), tree.end(), m_tree.begin());
    std::cout << " m_tree : " << m_ptr << ", tree: " << ptr << std::endl; */
}


void odeint_observer::operator()( const state_type &y , double t )
{
	const unsigned int n = y.size() / 2;
	state_type dy;
	hdca_obj.operator()(y, dy, t);
    VectorXd pjoint = VectorXd::Map(y.data(),    n);
    VectorXd alpha  = VectorXd::Map(y.data()+n,  n);
    VectorXd dalpha = VectorXd::Map(dy.data()+n, n);

    if (input.logEnergy) {
        VectorXd y2  = VectorXd::Map(y.data(),  2*n);
        VectorXd dy2 = VectorXd::Map(dy.data(), 2*n);
        logTotalEnergy(t, y2, dy2, uVec, input);
    }

    const int index = solution.atTime(t, input).first;
    solution.setAlpha(index, alpha);
    solution.setDalpha(index, dalpha);
    solution.setPjoint(index, pjoint);
    VectorXd d2alpha(n);
    VectorXd lambda(input.Nconstr);

    /* acceleration analysis */
    std::vector<Assembly, aaA >& leafBodies = tree[0];
#pragma omp parallel for schedule(static)
    for (int i = 0; i < input.Nbodies; i++)
        leafBodies[i].setKsiAcc(i, alphaAbs, dAlphaAbs, P1art, input);
        

    for (int i = 1; i < input.Ntiers; i++) {
        std::vector<Assembly, aaA >& branch = tree[i];
        std::vector<Assembly, aaA >& upperBranch = tree[i-1];
        const int Nparents = input.tiersInfo[i-1];
        const int Nparents_even = Nparents / 2; // # of nodes above current branch (-1 if they're odd)

        /* core loop of HDCA algorithm */
#pragma omp parallel for schedule(static)
        for (int j = 0; j < Nparents_even; j++)
            branch[j].assembleAcc(upperBranch[2*j], upperBranch[2*j+1]);
        
        /* case: odd # elements */
        if (Nparents % 2 == 1)
			branch.back().assembleAcc(upperBranch.back());
    }


    /* base body connection */
    Assembly& AssemblyS = tree[input.Ntiers-1][0];
    AssemblyS.connect_base_bodyAcc();
	AssemblyS.disassembleAcc();

    const int prelastTier = input.Ntiers-2;
	for (int i = prelastTier; i > 0; i--) {
        const int Nparents_even = input.tiersInfo[i-1] / 2;

#pragma omp parallel for schedule(static)
		for (int j = 0; j < Nparents_even; j++)
			tree[i].at(j).disassembleAcc();

        /* case: odd # elements */
		if (input.tiersInfo[i-1] % 2 == 1)
			tree[i-1].back().setAcc(tree[i].back());
	}

    /* joint acceleration calculation */
    const Vector3d Hground = input.pickBodyType(0).H;
    const Matrix<double, 3,2> Dground = input.pickBodyType(0).D;
	d2alpha(0) = Hground.transpose() * leafBodies[0].calculate_dV1();
    lambda.segment(0,2) = Dground.transpose() * leafBodies[0].L1_();

#pragma omp parallel for schedule(static)
	for (int i = 1; i < input.tiersInfo[0]; i++) {
        const Vector3d H = input.pickBodyType(i).H;
        const Matrix<double, 3,2> D = input.pickBodyType(i).D;
		const Vector3d dV1B = leafBodies[i].calculate_dV1();
		const Vector3d dV2A = leafBodies[i-1].calculate_dV2();
		d2alpha(i) = H.transpose() * (dV1B - dV2A);
        lambda.segment(2*i, 2) = D.transpose() * leafBodies[i].L1_();
	}
    
    solution.setD2alpha(index, d2alpha);
    solution.setLambda( index, lambda);
}