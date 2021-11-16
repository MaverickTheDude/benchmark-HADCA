#include "../include/utils.h"
#include "../include/input.h"
#include "../Eigen/Dense"
#include "../include/assembly.h"
#include "../include/solution.h"
#include <omp.h>
    #include <iostream>
using std::vector;

VectorXd RHS_HDCA(const double& t, const VectorXd& y, const VectorXd& u, const _input_& input) {
    _solution_ solution;
    return RHS_HDCA(t, y, u, input, solution);
}

VectorXd RHS_HDCA(const double& t, const VectorXd& y, const VectorXd& uVec, const _input_& input, _solution_& solution) {
	const unsigned int n = y.size()/2;
    VectorXd pjoint = y.head(n);
    VectorXd alpha  = y.tail(n);
    VectorXd alphaAbs = joint2AbsAngles(alpha);
    VectorXd dalpha(n), dpjoint(n);
    const double u = interpolateControl(t, uVec, input);

    using aaA = aligned_allocator<Assembly>;
    vector<vector<Assembly, aaA >, aaA > tree;
    tree.resize(input.Ntiers);
    vector<Assembly, aaA >& leafBodies = tree[0];
    leafBodies.resize(input.tiersInfo[0]);

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
    MatrixXd P1art(3, input.Nbodies+1);
    const Vector3d Hground = input.pickBodyType(0).H;
	dalpha(0) = Hground.transpose() * leafBodies[0].calculate_V1();
	P1art.col(0) = leafBodies[0].T1 + Hground*pjoint(0);
    P1art.col(input.Nbodies).setZero();

#pragma omp parallel for schedule(static)
	for (int i = 1; i < input.tiersInfo[0]; i++) {
        const Vector3d H = input.pickBodyType(i).H;
		const Vector3d V1B = leafBodies[i].calculate_V1();
		const Vector3d V2A = leafBodies[i-1].calculate_V2();
		dalpha(i) = H.transpose() * (V1B - V2A);
		P1art.col(i) = leafBodies[i].T1 + H*pjoint(i);
	}
    VectorXd dAlphaAbs = joint2AbsAngles(dalpha);

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
		dpjoint(i) = H.transpose() * (des.col(i) + leafBodies[i].Q1Art + dS1c*P1art.col(i) );
	}

    VectorXd dy(y.size());
    dy.head(n) = dpjoint;
    dy.tail(n) = dalpha;

    if (solution.dummySolution())
        return dy;
    
    const int index = solution.atTime(t, input).first;
    solution.setAlpha(index, alpha);
    solution.setDalpha(index, dalpha);
    solution.setPjoint(index, pjoint);
    VectorXd d2alpha(n);
    VectorXd lambda(input.Nconstr);


    /* acceleration analysis */
#pragma omp parallel for schedule(static)
    for (int i = 0; i < input.Nbodies; i++)
        leafBodies[i].setKsiAcc(i, alphaAbs, dAlphaAbs, P1art, input);
        

    for (int i = 1; i < input.Ntiers; i++) {
        vector<Assembly, aaA >& branch = tree[i];
        vector<Assembly, aaA >& upperBranch = tree[i-1];
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
    AssemblyS.connect_base_bodyAcc();
	AssemblyS.disassembleAcc();

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
    const Matrix<double, 3,2> Dground = input.pickBodyType(0).D;
	d2alpha(0) = Hground.transpose() * leafBodies[0].calculate_dV1();
    lambda.segment(0,2) = Dground.transpose() * leafBodies[0].L1;

#pragma omp parallel for schedule(static)
	for (int i = 1; i < input.tiersInfo[0]; i++) {
        const Vector3d H = input.pickBodyType(i).H;
        const Matrix<double, 3,2> D = input.pickBodyType(i).D;
		const Vector3d dV1B = leafBodies[i].calculate_dV1();
		const Vector3d dV2A = leafBodies[i-1].calculate_dV2();
		d2alpha(i) = H.transpose() * (dV1B - dV2A);
        lambda.segment(2*i, 2) = D.transpose() * leafBodies[0].L1;
	}
    
    solution.setD2alpha(index, d2alpha);
    solution.setLambda( index, lambda);
    
    return dy;
}