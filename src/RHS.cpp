#include "../include/utils.h"
#include "../include/input.h"
#include "../Eigen/Dense"
#include "../include/assembly.h"
#include "../include/solution.h"
    #include <iostream>
using std::vector;

VectorXd RHS_HDCA(const double& t, const VectorXd& y, const _input_& input) {
    _solution_ solution;
    return RHS_HDCA(t, y, input, solution);
}

VectorXd RHS_HDCA(const double& t, const VectorXd& y, const _input_& input, _solution_& solution) {
	const unsigned int n = y.size()/2;
    VectorXd pjoint = y.head(n);
    VectorXd alpha  = y.tail(n);
    VectorXd alphaAbs = joint2AbsAngles(alpha);
    VectorXd dalpha(n);
    VectorXd dpjoint(n);

    // vector<vector<Assembly>> tree;
    vector<vector<Assembly, aligned_allocator<Assembly> >, aligned_allocator<Assembly> > tree;
    tree.resize(input.Ntiers); // note: resize() zawiera juz w sobie reserve (?)
    vector<Assembly, aligned_allocator<Assembly> >& leafBodies = tree[0];
    leafBodies.reserve(input.tiersInfo[0]);

// TODO: parallelize
    /* Initialize leaf bodies (baza indukcyjna) */
    // https://eigen.tuxfamily.org/dox/group__TopicStlContainers.html --> The case of std::vector
    for (int i = 0; i < input.Nbodies; i++)
        leafBodies.emplace_back(i, alphaAbs, pjoint, input);

    for (int i = 1; i < input.Ntiers; i++) {
        vector<Assembly, aligned_allocator<Assembly> >& branch = tree[i];
        branch.reserve(input.tiersInfo[i]);
        const int endOfBranch = input.tiersInfo[i-1] - 1;

// TODO: parallelize
        /* core loop of HDCA algorithm */
        for (int j = 0; j < endOfBranch; j+=2) {
            branch.emplace_back(tree[i-1][j], tree[i-1][j+1]);
        }
        
        /* case: odd # elements */
		if (input.tiersInfo[i-1] % 2 == 1)
			branch.emplace_back(tree[i-1].back()); // fixme: konstruktor kopiujacy ?
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
    MatrixXd P1art(3, input.Nbodies);
    Vector3d H = input.pickBodyType(0).H;
	dalpha(0) = H.transpose() * leafBodies[0].calculate_V1();
	P1art.col(0) = leafBodies[0].T1 + H*pjoint(0);

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

    // std::cout << "des = " << std::endl << des << std::endl;
    // std::cout << "dalpha0 = " << std::endl << input.dalpha0 << std::endl;
    // std::cout << "dalpha = " << std::endl << dalpha << std::endl;
    // std::cout << "Part = " << std::endl << P1art << std::endl;
    // std::cout << "dp = " << std::endl << dpjoint << std::endl;

    VectorXd dy(y.size());
    dy.head(n) = dpjoint;
    dy.tail(n) = dalpha;

    if (solution.dummySolution())
        return dy;
    
    int index = solution.atTime(t);
    solution.setAlpha(index, alpha);
    solution.setDalpha(index, dalpha);
    solution.setPjoint(index, pjoint);


    /* acceleration analysis */
    
    return dy;
}