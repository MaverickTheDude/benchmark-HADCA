/*
Build with: meson test
check individual tests:
$ ./test_example                # Runs all tests in the suite
$ ./test_example test1 test2    # Runs only tests specified
$ ./test_example --skip test3   # Runs all tests but those specified
*/

/**
 * We put global includes at the top:
 *  - acutest.h - acutest library
 *  - iostream - in case a verbose debugging is needed
 * 
 * Other libraries / headers are included at the top of each test function
 * as tailored to its needs. This way it is easier to maintain these test
 * functions into other files, just in case.
 */
#include "Acutest/acutest.h"

#include <iostream>
static const double eps = 1e-7;


#include "include/input.h"
#include "Eigen/Dense"
#include "include/derivatives.h"
#include "include/utils.h"

using namespace Eigen;

void test_Phi(void) {
    _input_ input = _input_(10);
    VectorXd q(30);
    q = jointToAbsolutePosition(input.alpha0, input);

    VectorXd phi  = Phi(q, input);

    TEST_CHECK_(phi.norm() <= eps, "max error = %f", phi.norm());
}


#include "include/input.h"
#include "Eigen/Dense"
#include "include/utils.h"
#include "include/derivatives.h"

using namespace Eigen;

void test_Fq(void) {
    _input_ input = _input_(10);
    VectorXd q(30);
    q = jointToAbsolutePosition(input.alpha0, input);

    MatrixXd fd   = jacobianReal(Phi, q, input);
    MatrixXd Jac  = Jacobian(q, input);
    MatrixXd diff = fd - Jac;

    TEST_CHECK_(diff.norm() <= eps, "max error = %f", diff.norm());
}


#include "include/input.h"
#include "Eigen/Dense"
#include "include/utils.h"
#include "include/constants.h"

using namespace Eigen;

void test_jointToAbsolutePosition(void) {
    _input_ input = _input_(5);

    VectorXd alpha(input.Nbodies);
    alpha.tail(input.Nbodies - 1).setConstant(0);
    alpha(1) = M_PI_4;
    VectorXd absoluteCoords = jointToAbsolutePosition(alpha, input);

    VectorXd absoluteCoords_ideal(3 * 5);
    absoluteCoords_ideal << 0.0, 0.0, 0.0,
        0.0, 0.0, M_PI_4,
        _L_*cos(M_PI_4), _L_*sin(M_PI_4), M_PI_4,
        2*_L_*cos(M_PI_4), 2*_L_*sin(M_PI_4), M_PI_4,
        3*_L_*cos(M_PI_4), 3*_L_*sin(M_PI_4), M_PI_4;

    VectorXd diff = absoluteCoords_ideal - absoluteCoords;
    TEST_CHECK_(diff.norm() <= eps, "max error = %f", diff.norm());
}


#include "include/input.h"
#include "Eigen/Dense"
#include "include/utils.h"

using namespace Eigen;

void test_jointToAbsoluteVelocity1(void) {
    _input_ input = _input_(5);
    VectorXd alpha = VectorXd::Zero(5);
    VectorXd dalpha(5);
    dalpha << 1.0, M_PI_2, 0, 0, 0;

    VectorXd dq = jointToAbsoluteVelocity(alpha, dalpha, input);

    VectorXd dq_ideal(3 * 5);
    dq_ideal << 1.0, 0.0, 0.0,
        1.0, 0.0, M_PI_2,
        1.0, M_PI_2, M_PI_2,
        1.0, 2*M_PI_2, M_PI_2,
        1.0, 3*M_PI_2, M_PI_2;

    TEST_CHECK_((dq - dq_ideal).norm() <= eps, "max error = %f", (dq - dq_ideal).norm());
}


#include "include/input.h"
#include "Eigen/Dense"
#include "include/utils.h"

using namespace Eigen;

void test_jointToAbsoluteVelocity2(void) {
    _input_ input = _input_(4);
    VectorXd dq = jointToAbsoluteVelocity(input.alpha0, input.dalpha0, input);
    VectorXd alphaAbs = joint2AbsAngles(input.alpha0);
    const Matrix3d S12_0 = SAB("s12", 0, alphaAbs, input);
    const Matrix3d S12_1 = SAB("s12", 1, alphaAbs, input);
    const Matrix3d S12_2 = SAB("s12", 2, alphaAbs, input);
    const Matrix3d s12[] = {S12_0, S12_1, S12_2};
    
    int i = 0;
    Vector3d Vref;
    Vref << input.dalpha0(i), 0.0, 0.0; // == H0 * dalfa0
    Vector3d test = dq.segment(3*i, 3);
    Vector3d diff = test - Vref; 
    TEST_CHECK_(diff.norm() <= eps, "result = [%f, %f, %f], ref. result = [%f, %f, %f] in bodyId = %d",
        test(0), test(1), test(2), 
        Vref(0), Vref(1), Vref(2),
        i);

    for(int i = 1; i < 4; i++)
    {
        test = dq.segment(3*i, 3);
        Vref = s12[i-1].transpose() * Vref + input.pickBodyType(i).H * input.dalpha0(i);
        diff = test - Vref;
        TEST_CHECK_(diff.norm() <= eps, "result = [%f, %f, %f], ref. result = [%f, %f, %f] in bodyId = %d",
            test(0), test(1), test(2), 
            Vref(0), Vref(1), Vref(2),
            i);
    }
}


#include "include/input.h"
#include "Eigen/Dense"
#include "include/utils.h"

using namespace Eigen;

void test_SetPJoint(void) {
    _input_ input = _input_(4);

    for( int id = input.Nbodies - 2; id >= 0; id--)
    {
        VectorXd sigmaStacked = input.sigma0;
        VectorXd dq = jointToAbsoluteVelocity(input.alpha0, input.dalpha0, input);
        VectorXd alphaAbs = joint2AbsAngles(input.alpha0);
        const Matrix3d S12 = SAB("s12", id, alphaAbs, input);
        const Matrix3d S21 = SAB("s21", id, alphaAbs, input);
        const Matrix3d S1C = SAB("s1C", id, alphaAbs, input);
        const Matrix3d S2C = SAB("s2C", id, alphaAbs, input);
        const Matrix3d Mc = massMatrix(id, input);
        const Matrix3d M1 = S1C * Mc * S1C.transpose();
        const Matrix3d M2 = S2C * Mc * S2C.transpose();

        Vector3d V1 = dq.segment(3*id, 3);
        Vector3d V2 = S12.transpose() * V1;
        Vector3d H = input.pickBodyType(id).H;
        MatrixXd D = input.pickBodyType(id).D;
        Vector3d Hnext = input.pickBodyType(id+1).H; // note: upewnic sie, ze sprawdzamy dla nie-ostatniego czlonu
        MatrixXd Dnext = input.pickBodyType(id+1).D;
        Vector2d sigma = sigmaStacked.segment(2*id, 2);
        Vector2d sigmaNext = sigmaStacked.segment(2*(id+1), 2);
        
        Vector3d RHS1 = H * input.pjoint0(id) + D * sigma - S12 * (Hnext * input.pjoint0(id+1) + Dnext * sigmaNext);
        Vector3d RHS2 = S21 * (H * input.pjoint0(id) + D * sigma) - (Hnext * input.pjoint0(id+1) + Dnext * sigmaNext);
        Vector3d M1V1 = M1 * V1;
        Vector3d M2V2 = M2 * V2;
        VectorXd diff1 = M1V1 - RHS1;
        VectorXd diff2 = M2V2 - RHS2;
        TEST_CHECK_(diff1.norm() <= eps, "result = [%f, %f, %f], ref. result = [%f, %f, %f] in bodyId = %d",
            RHS1(0), RHS1(1), RHS1(2),
            M1V1(0), M1V1(1), M1V1(2),
            id);
        TEST_CHECK_(diff2.norm() <= eps, "result = [%f, %f, %f], ref. result = [%f, %f, %f] in bodyId = %d",
            RHS2(0), RHS2(1), RHS2(2),
            M2V2(0), M2V2(1), M2V2(2),
            id);
    }
}


#include "include/input.h"
#include "Eigen/Dense"
#include "include/utils.h"
#include "include/assembly.h"

using namespace Eigen;

void test_SetAssembly(void) {
    _input_ input = _input_(4);
	const double U_ZERO = 0.0; // quick hack

    for( int id = 0; id <= input.Nbodies - 1; id++)
    {
        VectorXd sigmaStacked = input.sigma0;
        VectorXd dq = jointToAbsoluteVelocity(input.alpha0, input.dalpha0, input);
        VectorXd alphaAbs = joint2AbsAngles(input.alpha0);
        Assembly body(id, alphaAbs, input.pjoint0, U_ZERO, input);
        Vector3d T1A = input.pickBodyType(id).D   * sigmaStacked.segment(2*id, 2);
        Vector3d T2A;
        if (id == input.Nbodies - 1)
            T2A = Vector3d::Zero(); // dla ostatniego czlonu
        else
            T2A = - input.pickBodyType(id+1).D * sigmaStacked.segment(2*(id+1), 2); // minus, poniewaz sigma ma znak + w interfejsie 1

        VectorXd RHS1 = body.ksi.k11() * T1A + body.ksi.k12() * T2A + body.ksi.k10();
        VectorXd RHS2 = body.ksi.k21() * T1A + body.ksi.k22() * T2A + body.ksi.k20();

        const Matrix3d S12 = SAB("s12", id, alphaAbs, input);
        Vector3d V1 = dq.segment(3*id, 3);
        Vector3d V2 = S12.transpose() * V1;

        Vector3d diff1 = V1 - RHS1;
        Vector3d diff2 = V2 - RHS2;

        TEST_CHECK_(diff1.norm() <= eps, "error1 = %f, \n Rvalue1 = [%f, %f, %f], \n Tvalue1 = [%f, %f, %f]", 
                    diff1.norm(), V1(0), V1(1), V1(2), RHS1(0), RHS1(1), RHS1(2));
        TEST_CHECK_(diff2.norm() <= eps, "error2 = %f, \n Rvalue2 = [%f, %f, %f], \n Tvalue2 = [%f, %f, %f]", 
                    diff2.norm(), V2(0), V2(1), V2(2), RHS2(0), RHS2(1), RHS2(2));
    }
}


void test_solveHDCA(void) {
/*
 * Test na podstawie zachowania energii w ukladzie. Dla dluzszych czasow symulacji konieczne jest obnizenie tolerancji.
 * Prawdopodobnie w pewnej chwili ruchu zachodzi "dziwny" rodzaj ruchu (np. jerk jednego z czlonow) i psuje zachowanie energii
 * Pass dla: Tk = 2 sek, Nbodies = 4 i Rtol = 6e-4
 */
    int Nbodies = 9;
    _input_ input = _input_(Nbodies);
    _solution_ sol = RK_solver(input);

	VectorXd y1(2*Nbodies);
    VectorXd y2(2*Nbodies);
    y1.head(Nbodies) = sol.pjoint.col(0);
    y1.tail(Nbodies) = sol.alpha.col(0);

    y2.head(Nbodies) = sol.pjoint.col(input.Nsamples-1);
    y2.tail(Nbodies) = sol.alpha.col(input.Nsamples-1);


    double e1 = calculateTotalEnergy(0,        y1, input);
    double e2 = calculateTotalEnergy(input.Tk, y2, input);

    double Rtol = 1e-5;
    double diff = abs(e1-e2);
    TEST_CHECK_(diff <= e1*Rtol, "\ndiff = %f = %f %% \ntol = %f = %f %% (Rtol) \ne1 =  %f \ne2 =  %f \nNbodies = %d", 
                diff, diff/e1*100, e1*Rtol, Rtol*100, e1, e2, Nbodies);
}


#include "include/task/F.h"

namespace F_1_q
{
    struct data_struct
    {
        task::F* F;
        VectorXd* dq;
        VectorXd* u;
    } data;

    VectorXd wrapper(const VectorXd& q, const _input_& input)
    {
        return data.F->operator()(q, *data.dq, *data.u);
    }

    void test(void)
    {
        int NBodies = 10;
        _input_ input = _input_(NBodies);
        task::F_1 F(input);

        VectorXd q = jointToAbsolutePosition(input.alpha0, input);
        VectorXd dq = jointToAbsolutePosition(input.dalpha0, input);
        VectorXd u = (VectorXd(1) << 0).finished();

        data = (data_struct) {.F = &F, .dq = &dq, .u = &u};

        MatrixXd F_q_FDM = jacobianReal(wrapper, q, input).transpose();
        MatrixXd F_q_explicit = F.q(q, dq, u);
        MatrixXd diff = F_q_FDM - F_q_explicit;
        
        TEST_CHECK_(diff.norm() <= eps, "Error: %f (threshold: %f)\n", diff.norm(), eps);
    }
}


#include "include/task/F.h"

using namespace Eigen;

namespace F_1_dq
{
    struct data_struct
    {
        task::F* F;
        VectorXd* q;
        VectorXd* u;
    } data;

    VectorXd wrapper(const VectorXd& dq, const _input_& input)
    {
        return data.F->operator()(*data.q, dq, *data.u);
    }

    void test(void)
    {
        int NBodies = 10;
        _input_ input = _input_(NBodies);
        task::F_1 F(input);

        VectorXd q = jointToAbsolutePosition(input.alpha0, input);
        VectorXd dq = jointToAbsolutePosition(input.dalpha0, input);
        VectorXd u = (VectorXd(1) << 0).finished();

        data = (data_struct) {.F = &F, .q = &q, .u = &u};

        MatrixXd F_dq_FDM = jacobianReal(wrapper, dq, input).transpose();
        MatrixXd F_dq_explicit = F.dq(q, dq, u);
        MatrixXd diff = F_dq_FDM - F_dq_explicit;
        
        TEST_CHECK_(diff.norm() <= eps, "Error: %f (threshold: %f)\n", diff.norm(), eps);
    }
}

#include "include/task/M.h"

using namespace Eigen;

namespace M_ddqdq
{
    struct data_struct
    {
        task::M* M;
        VectorXd* dq;
    } data;

    VectorXd wrapper(const VectorXd& q, const _input_& input)
    {
        return data.M->operator()(q) * (*data.dq);
    }

    void test(void)
    {
        int NBodies = 10;
        _input_ input = _input_(NBodies);
        task::M M(input);
        VectorXd q = jointToAbsolutePosition(input.alpha0, input);
        VectorXd dq = VectorXd::Ones(3 * NBodies);

        data = (data_struct) {.M = &M, .dq = &dq};

        MatrixXd Mdq_q_FDM = jacobianReal(wrapper, q, input).transpose();
        MatrixXd Mdq_q_explicit = M.ddqdq(q, dq);
        MatrixXd diff = Mdq_q_FDM - Mdq_q_explicit;

        TEST_CHECK_(diff.norm() <= eps, "Error: %f (threshold: %f)\n", diff.norm(), eps);
    }
}

#include "include/task/Phi.h"

namespace Phi_ddqddqlambda
{
    struct data_struct
    {
        task::Phi* Phi;
        VectorXd* lambda;
    } data;

    VectorXd wrapper(const VectorXd& q, const _input_& input)
    {
        return data.Phi->q(q).transpose() * (*data.lambda);
    }

    void test(void)
    {
        int NBodies = 10;
        _input_ input = _input_(NBodies);
        task::Phi Phi(input);
        VectorXd q = jointToAbsolutePosition(input.alpha0, input);
        VectorXd lambda = VectorXd::Ones(input.Nconstr);

        data = (data_struct) {.Phi = &Phi, .lambda = &lambda};

        MatrixXd Phi_FDM = jacobianReal(wrapper, q, input).transpose();
        MatrixXd Phi_explicit = Phi.ddqddqlambda(q, lambda);
        MatrixXd diff = Phi_FDM - Phi_explicit;

        TEST_CHECK_(diff.norm() <= eps, "Error: %f (threshold: %f)\n", diff.norm(), eps);
    }
}

void test_atTime(void) {
    const int eps = 1e-10;
    const int Nbodies = 5;
    _input_ input = _input_(Nbodies);
    _solution_ sol = RK_solver(input);
    const double& dt = input.dt;

    double t_tab[] = {dt, 2*dt     , 0.33     , 0.5      , 0.69     , input.Tk-dt    , input.Tk,
                 dt+dt/2, 2*dt+dt/2, 0.33+dt/2, 0.51+dt/2, 0.71+dt/2, input.Tk-3*dt/2, input.Tk-dt/2};
    const int size = 14;

    for (int i = 0; i < size; i++) {
        double t = t_tab[i], t_test;
        std::pair<int,bool> indStruct = sol.atTime(t, input);
        int ind = indStruct.first;
        if (indStruct.second == true) // node value
            t_test = sol.T(ind);
        else 
            t_test = sol.T(ind) + input.dt/2;

        const double diff = t - t_test;
        TEST_CHECK_(diff <= eps, "\ndiff = %f\t t = %f, \t ind = %d, \t t_test = %f\n",
                    diff, t, ind, t_test);
    }
}

void test_interpolate(void) {
    /* 
     * test interpolacji wielomianem 3. stopnia parametrow kinematycznych.
     * Sprawdzamy czy obliczona wartosc (c) znajduje sie w przedziale miedzy wartosciami brzegowymi [a,b]
     * note: test moze nie przejsc jesli trafimy na eskremum lokalne, np. dla t=0.33, alpha(3)
     * dla t=0.85, d2alpha(4) niezle widac przewage interpolacji wzgledem opcji liniowej (test oczywiscie failed)  */
    const int Nbodies = 5;
    _input_ input = _input_(Nbodies);
    _solution_ sol = RK_solver(input);
    const double& dt = input.dt;
	
    for (double t : {0.0, dt, 0.34, 0.57, input.Tk-2*dt, input.Tk-dt}) {

        auto ind1 = sol.atTime(t, input);
        auto ind2 = sol.atTime(t+dt, input);
        dataJoint s1 = sol.getDynamicValues(ind1.first, input);
        dataJoint s2 = sol.getDynamicValues(ind2.first, input);
        dataJoint sI = interpolate(t+dt/2, sol, input);
        dataJoint sL = interpolateLinear(t+dt/2, sol, input);


        for (int i = 0; i < Nbodies; i++) {
            double& a  = s1.alpha(i);
            double& c  = sI.alpha(i);
            double& cL = sL.alpha(i);
            double& b  = s2.alpha(i);
            int sign = (b > a) ? 1 : -1;

            TEST_CHECK_(sign*(c - a) > 0, "alpha(%d) at t=%f : (c-a) > 0 fail \n%f (a)\n%f (c)\n%f (c linear)\n%f (b)",
                                                  i,        t,                    a,      c,      cL,           b);
            TEST_CHECK_(sign*(c - b) < 0, "alpha(%d) at t=%f : (c-b) < 0 fail \n%f (a)\n%f (c)\n%f (c linear)\n%f (b)",
                                                  i,        t,                    a,      c,      cL,           b);
            a  = s1.dalpha(i);
            c  = sI.dalpha(i);
            cL = sL.dalpha(i);
            b  = s2.dalpha(i);
            sign = (b > a) ? 1 : -1;

            TEST_CHECK_(sign*(c - a) > 0, "dalpha(%d) at t=%f : (c-a) > 0 fail \n%f (a)\n%f (c)\n%f (c linear)\n%f (b)",
                                                  i,        t,                    a,      c,      cL,           b);
            TEST_CHECK_(sign*(c - b) < 0, "dalpha(%d) at t=%f : (c-b) < 0 fail \n%f (a)\n%f (c)\n%f (c linear)\n%f (b)",
                                                  i,        t,                    a,      c,      cL,           b);
            a  = s1.d2alpha(i);
            c  = sI.d2alpha(i);
            cL = sL.d2alpha(i);
            b  = s2.d2alpha(i);
            sign = (b > a) ? 1 : -1;

            const bool specialCase_d2a_ekstremum = abs(t-0.98)<1e-5 && i == 4;
        if ( !specialCase_d2a_ekstremum )
            TEST_CHECK_(sign*(c - a) > 0, "d2alpha(%d) at t=%f : (c-a) > 0 fail \n%f (a)\n%f (c)\n%f (c linear)\n%f (b)",
                                                  i,        t,                    a,      c,      cL,           b);
            TEST_CHECK_(sign*(c - b) < 0, "d2alpha(%d) at t=%f : (c-b) < 0 fail \n%f (a)\n%f (c)\n%f (c linear)\n%f (b)",
                                                  i,        t,                    a,      c,      cL,           b);
            a  = s1.lambda(2*i);
            c  = sI.lambda(2*i);
            cL = sL.lambda(2*i);
            b  = s2.lambda(2*i);
            sign = (b > a) ? 1 : -1;

            if (abs(a) < eps && abs(b) < eps)
                continue;

            TEST_CHECK_(sign*(c - a) > 0, "lambda(%d) at t=%f : (c-a) > 0 fail \n%f (a)\n%f (c)\n%f (c linear)\n%f (b)",
                                                i,           t,                    a,      c,      cL,           b);
            TEST_CHECK_(sign*(c - b) < 0, "lambda(%d) at t=%f : (c-b) < 0 fail \n%f (a)\n%f (c)\n%f (c linear)\n%f (b)",
                                                i,           t,                    a,      c,      cL,           b);
            a  = s1.lambda(2*i+1);
            c  = sI.lambda(2*i+1);
            cL = sL.lambda(2*i+1);
            b  = s2.lambda(2*i+1);
            sign = (b > a) ? 1 : -1;

            if (abs(a) < eps && abs(b) < eps)
                continue;

            TEST_CHECK_(sign*(c - a) > 0, "lambda(%d) at t=%f : (c-a) > 0 fail \n%f (a)\n%f (c)\n%f (c linear)\n%f (b)",
                                                i,           t,                    a,      c,      cL,           b);
            TEST_CHECK_(sign*(c - b) < 0, "lambda(%d) at t=%f : (c-b) < 0 fail \n%f (a)\n%f (c)\n%f (c linear)\n%f (b)",
                                                i,           t,                    a,      c,      cL,           b);
        }
    }
}

namespace PhiTimeDerivatives
{
    double test_time = 0.9;
    double eps = 1e-6;

    struct data_struct
    {
        _input_* input;
        task::Phi* Phi;
        _solution_* solution;
        double t = 0.0;

        data_struct() : input(NULL), Phi(NULL), solution(NULL) {}

        data_struct(_input_* i, task::Phi* P, _solution_* s, double test_time) :
            input(i), Phi(P), solution(s)
        {
            t = floor(test_time / input->dt) * input->dt;
        }

    } data;

    VectorXd q(double t)
    {
        dataJoint solution_t = interpolate(t, *data.solution, *data.input);
        dataAbsolute absoluteSolution_t(VectorXd::Zero(3 * data.input->Nbodies), VectorXd::Zero(3 * data.input->Nbodies),
            solution_t, *data.input);
        return absoluteSolution_t.q;
    }

    VectorXd dq(double t)
    {
        dataJoint solution_t = interpolate(t, *data.solution, *data.input);
        dataAbsolute absoluteSolution_t(VectorXd::Zero(3 * data.input->Nbodies), VectorXd::Zero(3 * data.input->Nbodies),
            solution_t, *data.input);
        return absoluteSolution_t.dq;
    }

    VectorXd ddq(double t)
    {
        dataJoint solution_t = interpolate(t, *data.solution, *data.input);
        dataAbsolute absoluteSolution_t(VectorXd::Zero(3 * data.input->Nbodies), VectorXd::Zero(3 * data.input->Nbodies),
            solution_t, *data.input);
        return absoluteSolution_t.d2q;
    }

    namespace ddt
    {
        VectorXd wrapper(const VectorXd& q, const _input_& input)
        {
            return data.Phi->q(q) * dq(data.t);
        }

        void test(void)
        {
            int NBodies = 10;
            _input_ input = _input_(NBodies);
            task::Phi Phi(input);
            _solution_ solution = RK_solver(input);

            data = data_struct(&input, &Phi, &solution, test_time);

            MatrixXd Phi_ddt_explicit = Phi.ddtq(q(data.t), dq(data.t));
            MatrixXd Phi_ddt_FDM = jacobianReal(wrapper, q(data.t), input);
            MatrixXd diff = Phi_ddt_FDM - Phi_ddt_explicit;
            TEST_CHECK_(diff.norm() <= eps, "Error: %.10lf (threshold: %.10lf)\n", diff.norm(), eps);
        }
    }

    namespace d2dt2
    {
        VectorXd wrapper(const VectorXd& q, const _input_& input)
        {
            return data.Phi->ddtq(q, dq(data.t)) * dq(data.t) + 
                data.Phi->q(q) * ddq(data.t);
        }

        void test(void)
        {
            int NBodies = 10;
            _input_ input = _input_(NBodies);
            task::Phi Phi(input);
            _solution_ solution = RK_solver(input);

            data = data_struct(&input, &Phi, &solution, test_time);

            MatrixXd Phi_d2dt2_explicit = Phi.d2dt2q(q(data.t), dq(data.t), ddq(data.t));
            MatrixXd Phi_d2dt2_FDM = jacobianReal(wrapper, q(data.t), input);
            MatrixXd diff = Phi_d2dt2_FDM - Phi_d2dt2_explicit;

            TEST_CHECK_(diff.norm() <= eps, "Error: %.10lf (threshold: %.10lf)\n", diff.norm(), eps);
        }
    }
}

TEST_LIST = {
   { "phi", test_Phi },
   { "jacobian", test_Fq },
   { "joint2AbsPosition", test_jointToAbsolutePosition },
   { "joint2AbsVelocity (Test 1)", test_jointToAbsoluteVelocity1 },
   { "joint2AbsVelocity (Test 2)", test_jointToAbsoluteVelocity2 },
   { "_input_.setPJointAndSigma", test_SetPJoint },
   { "setAssembly", test_SetAssembly },
   { "solveHDCA", test_solveHDCA},
   { "F_1_q", F_1_q::test},
   { "F_1_dq", F_1_dq::test},
   { "M_ddqdq", M_ddqdq::test},
   { "Phi_ddqddqlambda", Phi_ddqddqlambda::test},
   { "atTime", test_atTime},
   { "interpolation", test_interpolate},
   { "Phi_ddtq", PhiTimeDerivatives::ddt::test},
   { "Phi_d2dt2q", PhiTimeDerivatives::d2dt2::test},
   { NULL, NULL }     /* zeroed record marking the end of the list */
};