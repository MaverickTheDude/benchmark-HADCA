/**
 * Standalone test to research performance of different approaches
 * to building the Jacobi matrix of the multibody system's constraints.
 */

#include "../include/input.h"
#include "../include/utils.h"
#include "../include/task/Phi.h"
#include "../Eigen/Dense"
#include "../Eigen/Sparse"

#include <iostream>
#include <vector>
#include <cstdlib>

using namespace Eigen;

/**
 * Time measurement functions borrowed from MCT classes (Gabriel Wlazłowski, WUT)
 */
#include <sys/time.h>
#include <time.h> /* for ctime() */

static double t_gettimeofday ;
static struct timeval s ;

void b_t( void );
double e_t();

int main(int argc, char* argv[]) {
    std::vector<int> testsVector;

    if(argc == 2)
        testsVector.push_back(atoi(argv[1]));
    else
    {
        for(int i = 10000; i > 4; i /= 2)
            testsVector.push_back(i);
    }

    /**
     * Now, I have an idea that it would be more meaningful to repeat
     * these tests in A-B-C-A-B-C-A-B-C-... pattern, contrary to the current
     * implementation (A-A-A-...-B-B-B-...-C-C-C-...).
     */
    int repetitionsNumber = 10;

    bool testChunked = true;

    /**
     * Vectors agregating results of the test.
     */
    std::vector<int> r_bodies;
    std::vector<double> r_dense, r_chunked, r_sparse;

    while(!testsVector.empty())
    {
        double t_dense = 0, t_chunked = 0, t_sparse = 0;

        const int Nbodies = testsVector.back();
        testsVector.pop_back();

        r_bodies.push_back(Nbodies);

        /**
         * Meh, setting precision of the cpp std::cout is a way too much.
         */
        printf("Bodies: %d (dense matrix has ~%.3f GB)\n", Nbodies,
            (3 * Nbodies / 1024.0) * (2 * Nbodies / 1024.0) * 8 / 1024.0);

        _input_ input = _input_(Nbodies);

        task::Phi* Phi = new task::Phi(input);

        VectorXd sigma = VectorXd::Random(2 * Nbodies);
        VectorXd product = VectorXd::Zero(3 * Nbodies);

        // Dense matrix
        std::cout<<"Dense matrix: ";

        for(int i = 0; i < repetitionsNumber; i++)
        {
            product = VectorXd::Zero(3 * Nbodies);

            b_t();
            product = Phi->qDenseMatrix(jointToAbsolutePosition(input.alpha0, input)).transpose() * sigma;
            t_dense += e_t();
        }
        t_dense /= (double) repetitionsNumber;

        std::cout<<t_dense<<" s"<<std::endl;
        r_dense.push_back(t_dense);

        // Dense chunked matrix
        if(testChunked)
        {
            std::cout<<"Chunked matrix: ";

            for(int i = 0; i < repetitionsNumber; i++)
            {
                product = VectorXd::Zero(3 * Nbodies);
                const VectorXd q = jointToAbsolutePosition(input.alpha0, input);

                b_t();
                product.segment(0, 3) = Phi->qDenseChunked(q, 0) *
                    sigma.segment(0, 2);
                for(int i = 1; i < Nbodies; ++i)
                    product.segment(3 * (i - 1), 6) += Phi->qDenseChunked(q, i) * sigma.segment(2 * i, 2);
                t_chunked += e_t();
            }
            t_chunked /= (double) repetitionsNumber;

            std::cout<<t_chunked<<" s"<<std::endl;
            r_chunked.push_back(t_chunked);
        }

        // Sparse matrix
        std::cout<<"Sparse matrix: ";

        for(int i = 0; i < repetitionsNumber; i++)
        {
            product = VectorXd::Zero(3 * Nbodies);

            b_t();
            product = Phi->qSparseMatrix(jointToAbsolutePosition(input.alpha0, input)) * sigma;
            t_sparse += e_t();
        }
        t_sparse /= (double) repetitionsNumber;

        std::cout<<t_sparse<<" s"<<std::endl;
        r_sparse.push_back(t_sparse);

        delete Phi;
    }

    /**
     * Print summary of the test in Matlab format.
     */
    int n = r_bodies.size();
    for(int i = 0; i < n; i++)
    {
        std::cout<<r_bodies[i]<<" "<<
            r_dense[i]<<" "<<
            (testChunked ? r_chunked[i] : 0)<<" "<<
            r_sparse[i]<<";"<<
            std::endl;
    }

    return 0;
}

void b_t(void)
{
    /* hack together a clock w/ microsecond resolution */
    gettimeofday(&s, NULL);
    t_gettimeofday = s.tv_sec + 1e-6 * s.tv_usec;
}

double e_t()
{
    gettimeofday(&s, NULL);
    return s.tv_sec + 1e-6 * s.tv_usec - t_gettimeofday;
}