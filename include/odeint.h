#pragma once

#include "../include/assembly.h"
#include "../include/utils.h"
#include "../Eigen/Dense"

using namespace Eigen;


_solution_ RK_solver_odeInt(const VectorXd& uVec, const _input_& input);
_solutionAdj_ RK_AdjointSolver_odeInt(const VectorXd& uVec, const _solution_& solutionFwd, 
                                      const _input_& input, const int& formulation);
_solutionGlobal_ RK_GlobalSolver_odeInt(const VectorXd& uVec, const _input_& input) ;

// RHS-odeint.cpp
typedef std::vector< double > state_type;
class RHS_HDCA_ODE {
    
    const _input_& input;
    const VectorXd& uVec;
	VectorXd &alphaAbs, &dAlphaAbs;
    MatrixXd &P1art;
	std::vector<std::vector<Assembly, aaA >, aaA > &tree; // check me: czy move dziala poprawnie ?

public:
    RHS_HDCA_ODE(const _input_& c_input, const VectorXd& c_uVec, VectorXd& c_alphaAbs, 
                 VectorXd& c_dAlphaAbs, MatrixXd& c_P1art, 
                 std::vector<std::vector<Assembly, aaA >, aaA >& c_tree) :
    input(c_input), uVec(c_uVec), alphaAbs(c_alphaAbs), 
    dAlphaAbs(c_dAlphaAbs), P1art(c_P1art), tree(c_tree) { }
    void operator() ( const state_type &x , state_type &dxdt , const double t);
};

struct odeint_observer
{
	const _input_& input;
	RHS_HDCA_ODE& hdca_obj;
	_solution_& solution;
	const VectorXd& uVec;
	const VectorXd &alphaAbs, &dAlphaAbs;
    const MatrixXd &P1art;
	std::vector<std::vector<Assembly, aaA >, aaA > &tree;

    odeint_observer(const _input_& c_input, RHS_HDCA_ODE& c_hdca_obj,  _solution_& c_solution, 
                    const VectorXd& c_uVec, const VectorXd& c_alphaAbs, const VectorXd& c_dAlphaAbs,
                    const MatrixXd& c_P1art, std::vector<std::vector<Assembly, aaA >, aaA >& c_tree) :
    input(c_input), hdca_obj(c_hdca_obj),  solution(c_solution),  uVec(c_uVec), alphaAbs(c_alphaAbs),
    dAlphaAbs(c_dAlphaAbs), P1art(c_P1art), tree(c_tree) { }

    void operator()(const state_type &y , double t);
};

// ======= FORWARD GLOBAL =======

class RHS_GLOBAL_ODE {
    
    const _input_& input;
    const VectorXd& uVec;

public:
    RHS_GLOBAL_ODE(const _input_& c_input, const VectorXd& c_uVec) :
    input(c_input), uVec(c_uVec) { }
    void operator() ( const state_type &x , state_type &dxdt , const double t);
};

struct odeint_globalObserver
{
	const _input_& input;
	RHS_GLOBAL_ODE& global_obj;
	_solutionGlobal_& solution;
	const VectorXd& uVec;

    odeint_globalObserver(const _input_& c_input, RHS_GLOBAL_ODE& c_global_obj,  
                    _solutionGlobal_& c_solution, const VectorXd& c_uVec) :
    input(c_input), global_obj(c_global_obj),  solution(c_solution),  uVec(c_uVec) { }

    void operator()(const state_type &y , double t);
};

// ======= ADJOINT =======

class RHS_ADJ_ODE {
    
    const _input_& input;
    const VectorXd& uVec;
    const _solution_& solutionFwd;

public:
    RHS_ADJ_ODE(const _input_& c_input, const VectorXd& c_uVec, const _solution_& c_solutionFwd) :
    input(c_input), uVec(c_uVec), solutionFwd(c_solutionFwd) { }
    void operator() ( const state_type &y, state_type &dy, const double t);
};


struct odeint_observer_adj
{
	const _input_& input;
	RHS_ADJ_ODE& dca_adj_obj;
    const _solution_& solutionFwd;
	_solutionAdj_& solution;


    odeint_observer_adj(const _input_& c_input, RHS_ADJ_ODE& c_adj_obj, 
                        const _solution_& c_solutionFwd,  _solutionAdj_& c_solution) :
    input(c_input), dca_adj_obj(c_adj_obj), solutionFwd(c_solutionFwd), solution(c_solution) { }

    void operator()(const state_type &y , double t);
};