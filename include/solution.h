#pragma once
#include "../Eigen/Dense"
#include "../include/input.h"

using namespace Eigen;

struct dataJoint {
    dataJoint(int n) : alpha(n), dalpha(n), d2alpha(n), lambda(2*n), pjoint(n) { }

    double t;
    VectorXd alpha;
    VectorXd dalpha;
    VectorXd d2alpha;
    VectorXd lambda;
    VectorXd pjoint;
};

class _solution_{
public:
    _solution_(const _input_& input) : T(input.Nsamples), 
                                     alpha(input.Nbodies, input.Nsamples),
                                     dalpha(input.Nbodies, input.Nsamples), 
                                     d2alpha(input.Nbodies, input.Nsamples), 
                                     lambda(input.Nconstr, input.Nsamples),
                                     pjoint(input.Nbodies, input.Nsamples), 
                                     flag(active)
    {    }
    _solution_() : flag(dummy) { /* dummy structure to be passed as an argument in RHS_HDCA */ }
    enum flags {active, dummy};
    int atTime(const double& t);
    int atTimeRev(const double& tau);
    dataJoint getDynamicValuesRev(const double& tau);


    void setT(VectorXd _T_) {T = _T_;}
    void setAlpha(  int ind, VectorXd _alpha_)   { alpha.col(ind)   = _alpha_  ; }
    void setDalpha( int ind, VectorXd _dalpha_)  { dalpha.col(ind)  = _dalpha_ ; }
    void setD2alpha(int ind, VectorXd _d2alpha_) { d2alpha.col(ind) = _d2alpha_; }
    void setPjoint( int ind, VectorXd _pjoint_)  { pjoint.col(ind)  = _pjoint_ ; }
    void setLambda( int ind, VectorXd _lambda_)  { lambda.col(ind)  = _lambda_ ; }
    bool dummySolution() const { return (flag == dummy) ? true : false; }
    void print() const;

public:
    VectorXd T;
    MatrixXd alpha;
    MatrixXd dalpha;
    MatrixXd d2alpha;
    MatrixXd lambda;
    MatrixXd pjoint;
    int flag;
    // MatrixXd sigma;  niepotrzebne do adjointa
};