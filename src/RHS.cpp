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
// for (int i = 0; i < input.Nbodies; i++)
// leafBodies.emplace_back(i, alphaAbs, pjoint, u, input);

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
        branch.reserve(input.tiersInfo[i]);
        const int endOfBranch = input.tiersInfo[i-1] - 1;

// TODO: parallelize
        /* core loop of HDCA algorithm */
        for (int j = 0; j < endOfBranch; j+=2) {
            branch.emplace_back(upperBranch[j], upperBranch[j+1]);
        }
        
        /* case: odd # elements */
		if (input.tiersInfo[i-1] % 2 == 1)
			branch.emplace_back(upperBranch.back()); // fixme: konstruktor kopiujacy ?
    }
    
    /* base body connection */
    Assembly& AssemblyS = tree[input.Ntiers-1][0];
	AssemblyS.connect_base_body();
	AssemblyS.disassembleAll();

    const int prelastTier = input.Ntiers-2;
	for (int i = prelastTier; i > 0; i--) {
		const int end_of_branch = input.tiersInfo[i-1] % 2 == 0 ?
				input.tiersInfo[i] : input.tiersInfo[i]-1;

// TODO: parallelize
		for (int j = 0; j < end_of_branch; j++) {
			tree[i].at(j).disassembleAll();
		}

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

// TODO: parallelize
	for (int i = 1; i < input.tiersInfo[0]; i++) {
        const Vector3d H = input.pickBodyType(i).H;
		Vector3d V1B = leafBodies[i].calculate_V1();
		Vector3d V2A = leafBodies[i-1].calculate_V2();
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
        Matrix3d dSc2 = (-1.0) * dSAB("s2C", i,   alphaAbs, dAlphaAbs, input);
        Matrix3d dS1c =          dSAB("s1C", i+1, alphaAbs, dAlphaAbs, input);
        sum += (dSc2 + dS1c) * P1art.col(i+1);
		des.col(i) = sum;
    }

// TODO: Parallelize
    /* joint dp from the articulated quantities */
    for (int i = 0; i < input.Nbodies; i++) {
        const Vector3d H = input.pickBodyType(i).H;
        Matrix3d dS1c = dSAB("s1C", i, alphaAbs, dAlphaAbs, input);
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
// TODO: Parallelize
    for (int i = 0; i < input.Nbodies; i++)
        leafBodies[i].setKsiAcc(i, alphaAbs, dAlphaAbs, P1art, input);
        

    for (int i = 1; i < input.Ntiers; i++) {
        vector<Assembly, aaA >& branch = tree[i];
        vector<Assembly, aaA >& upperBranch = tree[i-1];
        const int endOfBranch = input.tiersInfo[i-1] - 1;
        int branchIndex = 0;

// TODO: parallelize
        /* core loop of HDCA algorithm */
        for (int j = 0; j < endOfBranch; j+=2) {
            branch[branchIndex].assembleAcc(upperBranch[j], upperBranch[j+1]);
            ++branchIndex;
        }
        
        /* case: odd # elements */
		if (input.tiersInfo[i-1] % 2 == 1)
			branch[branchIndex].assembleAcc(upperBranch.back());
    }


    /* base body connection */
    AssemblyS.connect_base_bodyAcc();
	AssemblyS.disassembleAcc();

	for (int i = prelastTier; i > 0; i--) {
		const int end_of_branch = input.tiersInfo[i-1] % 2 == 0 ?
				input.tiersInfo[i] : input.tiersInfo[i]-1;

// TODO: parallelize
		for (int j = 0; j < end_of_branch; j++) {
			tree[i].at(j).disassembleAcc();
		}

        /* case: odd # elements */
		if (input.tiersInfo[i-1] % 2 == 1) {
			tree[i-1].back().setAcc(tree[i].back());
		}
	}

    /* joint acceleration calculation */
    const Matrix<double, 3,2> Dground = input.pickBodyType(0).D;
	d2alpha(0) = Hground.transpose() * leafBodies[0].calculate_dV1();
    lambda.segment(0,2) = Dground.transpose() * leafBodies[0].L1;

// TODO: parallelize
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