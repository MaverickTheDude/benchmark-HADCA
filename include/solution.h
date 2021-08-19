#pragma once
#include "../Eigen/Dense"
#include "../include/input.h"
#include <iostream>

using namespace Eigen;

struct dataJoint {
    dataJoint(int n) : alpha(n), dalpha(n), d2alpha(n), lambda(2*n), pjoint(n) { }

    dataJoint() = default;
    dataJoint(const dataJoint& o) = default; // copy Ctor
    dataJoint(dataJoint&& o)      = default; // move Ctro
    ~dataJoint() { }

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
    {   }
    _solution_() : flag(dummy) { 
    /* dummy structure to be passed as an argument in RHS_HDCA */
        //std::cout<<this<<'\t'<<"Ctor\n"; 
    }

    // https://coders-corner.net/2018/02/18/fast-way-to-return-a-large-object/
    // https://stackoverflow.com/a/27916892/4283100
    // copy elison: ponizsze konstruktory nie sa wywolywane, ale gwarantuja poprawna prace. 
    // W przypadku wywolania, prawdopodobnie RVO nie zadzialalo. (note: RVO dziala na poziomie -O0, czy to ok ?)
  _solution_(const _solution_& o) : T(o.T), alpha(o.alpha), dalpha(o.dalpha), d2alpha(o.d2alpha), lambda(o.lambda), pjoint(o.pjoint), flag(o.flag)
    { std::cout<<this<<'\t'<<"CCtor\n"; }
  _solution_(_solution_&& o)      : T(o.T), alpha(o.alpha), dalpha(o.dalpha), d2alpha(o.d2alpha), lambda(o.lambda), pjoint(o.pjoint), flag(o.flag)
    { std::cout<<this<<'\t'<<"MCtor\n"; }
  ~_solution_() 
    {  /* std::cout<<this<<'\t'<<"Dtor\n"; */  }

    enum flags {active, dummy};
    std::pair<int, const bool> atTime(const double& t, const _input_& input) const;
    dataJoint getDynamicValues(const int index, const _input_& input) const;

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