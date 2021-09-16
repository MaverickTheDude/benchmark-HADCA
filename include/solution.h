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

struct dataAbsolute {
	dataAbsolute(const VectorXd& e, const VectorXd& c, const dataJoint& data, const _input_ &input);
	inline double alphaAbs(int id)   const { return q(3*id+2); }
	inline double dAlphaAbs(int id)  const { return dq(3*id+2); }
	inline double d2AlphaAbs(int id) const { return dq(3*id+2); }

	dataAbsolute() = default;
	dataAbsolute(const dataAbsolute& o) = default; // copy Ctor
	dataAbsolute(dataAbsolute&& o)      = default; // move Ctro
	~dataAbsolute() { }

	double t;
	VectorXd q;
	VectorXd dq;
	VectorXd d2q;
    VectorXd eta;
    VectorXd ksi;
	VectorXd lambda;
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

    static const bool NODE_VALUE; // wtf: zmienne statyczne definiujemy w pliku .cpp nawet jesli sa const
    static const bool INTERMEDIATE_VALUE;
    // MatrixXd sigma;  niepotrzebne do adjointa
};

class _solutionAdj_ {
public:
	enum flags {active, dummy, HDCA, GLOBAL};
    /* default constructor for HDCA formulation */
    _solutionAdj_(const _input_& input) : T(VectorXd::LinSpaced(input.Nsamples, 0, input.Tk)), 
                                    e(input.Nbodies, input.Nsamples),
                                    c(input.Nbodies, input.Nsamples), 
                                    norms(3, input.Nsamples), flag(active),
                                    eta(0, 0), ksi(0, 0) // this instance doesn't use eta and ksi
    {   }

    _solutionAdj_() : flag(dummy) { /* dummy structure to be passed as an argument in RHS_ADJOINT */ }

	_solutionAdj_(const _solutionAdj_& o) : T(o.T), e(o.e), c(o.c), norms(o.norms), flag(o.flag), eta(o.eta), ksi(o.ksi)
	{ std::cout<<this<<'\t'<<"CCtor\n"; }
	_solutionAdj_(_solutionAdj_&& o)      : T(o.T), e(o.e), c(o.c), norms(o.norms), flag(o.flag), eta(o.eta), ksi(o.ksi)
	{ std::cout<<this<<'\t'<<"MCtor\n"; }
	~_solutionAdj_() {	}

    void switchToGlobalFormulation(const _input_& input) {
        e.resize(0, 0);
        c.resize(0, 0);
        eta.resize(3*input.Nbodies, input.Nsamples);
        ksi.resize(3*input.Nbodies, input.Nsamples);
        HDCA_formulation = false;
    }

    void set_e(  int ind, VectorXd _e_)   { e.col(ind)   = _e_  ; }
    void set_c(  int ind, VectorXd _c_)   { c.col(ind)   = _c_  ; }
    void set_eta(  int ind, VectorXd _eta_)   { eta.col(ind)   = _eta_  ; }
    void set_ksi(  int ind, VectorXd _ksi_)   { ksi.col(ind)   = _ksi_  ; }
    void setNorms(  int ind, Vector3d _norms_)   { norms.col(ind)   = _norms_  ; }
	bool dummySolution() const { return (flag == dummy) ? true : false; }
    void print() const;

public:
    VectorXd T;
    MatrixXd e;
    MatrixXd c;
    Matrix<double, 3, Eigen::Dynamic> norms; // vs. MatrixXd norms: czy teraz konstruktor tez przyjmuje 2 argumenty?
    int flag;
    MatrixXd eta;
    MatrixXd ksi;
    bool HDCA_formulation = true;
};