#include "../../include/task/Phi.h"

#include "../../Eigen/Dense"
#include "../../Eigen/Sparse"
#include "../../Eigen/StdVector"
#include "../../include/input.h"
#include "../../include/utils.h"
#include "../../include/constants.h"
#include "../../include/body.h"

#include <stdexcept>
#include <vector>

using namespace Eigen;
using namespace task;

task::Phi::Phi(const _input_& i) :
    input(i), v(0.0, 1.0), jointsNumber(i.Nbodies),
    constrDim(i.Nconstr),bodiesNumber(i.Nbodies) {}

VectorXd task::Phi::operator()(const VectorXd& q) const
{
    VectorXd phi(constrDim);
    int phiPile = 0; // Saves index of the last entry in Phi vector modified by the recent joint
    
    for(int jN = 0; jN < jointsNumber; jN++)
    {
        // Prepare joint-specific data.
        const body::JOINT_TYPE jointType = input.pickBodyType(jN).jointType;
        const int jointClass = 2;
        const int bodyA = jN - 1;
        const int bodyB = jN;

        /**
         * In future, in a developed version a list of joints which maps to pointers
         * of bodies can be utilized. Then, a ground element should be created also
         * so the ternary operator below is not needed.
         */
        const Vector2d rA = bodyA < 0 ? Vector2d(0.0, 0.0) : q.segment(3 * bodyA + 0, 2);
        const double phiA = bodyA < 0 ? 0 : q.segment(3 * bodyA + 2, 1).value();
        const Vector2d rB = q.segment(3 * bodyB + 0, 2);
        const double phiB = q.segment(3 * bodyB + 2, 1).value();
        
        switch (jointType)
        {
            case body::JOINT_TYPE::PRISMATIC:
            {
                /** 
                 *  I am not sure whether sA should be considered a body's feature
                 *  or joint's feature. Anyway, in our current implementation all
                 *  the joint- and body- specific data is acquired from the body
                 *  class.
                 */
                const Vector2d sA = bodyA < 0 ? Vector2d(0.0, 0.0) : input.pickBodyType(bodyA).s1C;
                const Vector2d sB = input.pickBodyType(bodyB).s1C;
                phi(phiPile) = (Rot(phiB) * v).transpose() * 
                    (rB + Rot(phiB) * sB   - (rA + Rot(phiA) * sA));
                phi(phiPile + 1) = phiA - phiB;
                break;
            }
            case body::JOINT_TYPE::REVOLUTE:
            {
                const Vector2d sA = bodyA < 0 ? Vector2d(0.0, 0.0) : input.pickBodyType(bodyA).s12;
                const Vector2d sB = Vector2d::Zero();
                phi.segment(phiPile, 2) = rA + Rot(phiA) * sA - (rB + Rot(phiB) * sB);
                break;
            }
            default:
            {
                std::runtime_error("Unknown joint type.");
                break;
            }
        }

        phiPile += jointClass;
    }

    return phi;
}

MatrixXd Phi::q(const VectorXd& q) const
{
    MatrixXd phiQ = MatrixXd::Zero(constrDim, 3 * bodiesNumber);
    int phiPile = 0; // analogously as in ()
    
    for(int jN = 0; jN < jointsNumber; jN++)
    {
        // Prepare joint-specific data.
        const body::JOINT_TYPE jointType = input.pickBodyType(jN).jointType;
        const int jointClass = 2;
        const int bodyA = jN - 1;
        const int bodyB = jN;

        const Vector2d rA = bodyA < 0 ? Vector2d(0.0, 0.0) : q.segment(3 * bodyA + 0, 2);
        const double phiA = bodyA < 0 ? 0 : q.segment(3 * bodyA + 2, 1).value();
        const Vector2d rB = q.segment(3 * bodyB + 0, 2);
        const double phiB = q.segment(3 * bodyB + 2, 1).value();

        switch (jointType)
        {
            case body::JOINT_TYPE::PRISMATIC:
            {
                const Vector2d sA = bodyA < 0 ? Vector2d(0.0, 0.0) : input.pickBodyType(bodyA).s1C;

                if(bodyA >= 0)
                {
                    phiQ.block(phiPile + 0, 3 * bodyA + 0, 1, 2) = -(Rot(phiB) * v).transpose();
                    phiQ(phiPile + 0, 3 * bodyA + 2) = -(Rot(phiB) * v).transpose() * Om * Rot(phiA) * sA;
                    phiQ(phiPile + 1, 3 * bodyA + 2) = 1;
                }

                if(bodyB >= 0)
                {
                    phiQ.block(phiPile + 0, 3 * bodyB + 0, 1, 2) = (Rot(phiB) * v).transpose();
                    phiQ(phiPile + 0, 3 * bodyB + 2) = -(Rot(phiB) * v).transpose() * Om *
                        (rB - rA - Rot(phiA) * sA);
                    phiQ(phiPile + 1, 3 * bodyB + 2) = -1;
                }

                break;
            }
            case body::JOINT_TYPE::REVOLUTE:
            {
                const Vector2d sA = bodyA < 0 ? Vector2d(0.0, 0.0) : input.pickBodyType(bodyA).s12;
                const Vector2d sB = Vector2d::Zero();

                if(bodyA >= 0)
                {
                    phiQ.block(phiPile + 0, 3 * bodyA + 0, 2, 2) = Matrix2d::Identity();
                    phiQ.block(phiPile + 0, 3 * bodyA + 2, 2, 1) = Om * Rot(phiA) * sA;
                }

                if(bodyB >= 0)
                {
                    phiQ.block(phiPile + 0, 3 * bodyB + 0, 2, 2) = -Matrix2d::Identity();
                    phiQ.block(phiPile + 0, 3 * bodyB + 2, 2, 1) = - Om * Rot(phiB) * sB;
                }

                break;
            }
            default:
            {
                std::runtime_error("Unknown joint type.");
                break;
            }
        }

        phiPile += jointClass;
    }

    return phiQ;
}

MatrixXd Phi::qDenseMatrix(const VectorXd& q) const
{
    return Phi::q(q);
}

MatrixXd Phi::qDenseChunked(const VectorXd& q, const int jointNumber) const
{
    /**
     * Returns part of the matrix associeted with a given constraint, i.e.
     * the returned chunk of the Phi_q matrix is transposed and ready for
     * multiplication by sigma related to the jointNumber joint.
     */
    const int jN = jointNumber;
    const body::JOINT_TYPE jointType = input.pickBodyType(jN).jointType;
    const int jointClass = 2;
    /**
     * In this chunked matrix version it must be assured that bodyA < bodyB
     * as we promise to deliver the return matrix in asceding rows fashion.
     */
    const int bodyA = jN - 1;
    const int bodyB = jN;

    const Vector2d rA = bodyA < 0 ? Vector2d(0.0, 0.0) : q.segment(3 * bodyA + 0, 2);
    const double phiA = bodyA < 0 ? 0 : q.segment(3 * bodyA + 2, 1).value();
    const Vector2d rB = q.segment(3 * bodyB + 0, 2);
    const double phiB = q.segment(3 * bodyB + 2, 1).value();

    /**
     * Returned matrix is of size:
     *    a) 1 * 3 x jointClass - one of the bodies is the ground,
     *    b) 2 * 3 x jointClass - joint connecting two bodies.
     * We assume that considered constraints involves at most two bodies.
     * 
     * Also, already stated (in Phi()) assumption of allowing only bodyA
     * to be a ground element is used - ternary operator once again.
     * 
     * To be honest, I still wonder wheter it is a good idea to serve different
     * sizes of return matrix for these cases (bodyInternal counter is needed).
     */
    const int rowsNumber = bodyA < 0 ? 1 * 3 : 2 * 3;
    int bodyInternal = 0;
    MatrixXd qChunked = MatrixXd::Zero(rowsNumber, jointClass);

    switch (jointType)
    {
        case body::JOINT_TYPE::PRISMATIC:
        {
            const Vector2d sA = bodyA < 0 ? Vector2d(0.0, 0.0) : input.pickBodyType(bodyA).s1C;

            if (bodyA >= 0)
            {
                qChunked.block(3 * bodyInternal + 0, 0, 2, 1) = -Rot(phiB) * v;
                qChunked(3 * bodyInternal + 2, 0) = -(Rot(phiB) * v).transpose() * Om * Rot(phiA) * sA;
                qChunked(3 * bodyInternal + 2, 1) = 1;

                bodyInternal++;
            }

            if (bodyB >= 0)
            {
                qChunked.block(3 * bodyInternal + 0, 0, 2, 1) = Rot(phiB) * v;
                qChunked(3 * bodyInternal + 2, 0) = -(Rot(phiB) * v).transpose() * Om *
                    (rB - rA - Rot(phiA) * sA);
                qChunked(3 * bodyInternal + 2, 1) = -1;
            }

            break;
        }
        case body::JOINT_TYPE::REVOLUTE:
        {
            const Vector2d sA = bodyA < 0 ? Vector2d(0.0, 0.0) : input.pickBodyType(bodyA).s12;
            const Vector2d sB = Vector2d::Zero();

            if (bodyA >= 0)
            {
                qChunked.block(3 * bodyInternal + 0, 0, 2, 2) = Matrix2d::Identity();
                qChunked.block(3 * bodyInternal + 2, 0, 1, 2) = (Om * Rot(phiA) * sA).transpose();

                bodyInternal++;
            }

            if (bodyB >= 0)
            {
                qChunked.block(3 * bodyInternal + 0, 0, 2, 2) = -Matrix2d::Identity();
                qChunked.block(3 * bodyInternal + 2, 0, 1, 2) = (-Om * Rot(phiB) * sB).transpose();
            }

            break;
        }
        default:
        {
            std::runtime_error("Unknown joint type.");
            break;
        }
    }

    return qChunked;
}

SparseMatrix<double> Phi::qSparseMatrix(const VectorXd& q) const
{
    /**
     * Returns transposed Jacobian matrix (in form of a sparse matrix).
     */
    typedef Triplet<double> T;
    std::vector<T> matrixEntries;
    matrixEntries.reserve(6 * jointsNumber);
    int phiPile = 0; // analogously as in ()
    
    for(int jN = 0; jN < jointsNumber; jN++)
    {
        // Prepare joint-specific data.
        const body::JOINT_TYPE jointType = input.pickBodyType(jN).jointType;
        const int jointClass = 2;
        const int bodyA = jN - 1;
        const int bodyB = jN;

        const Vector2d rA = bodyA < 0 ? Vector2d(0.0, 0.0) : q.segment(3 * bodyA + 0, 2);
        const double phiA = bodyA < 0 ? 0 : q.segment(3 * bodyA + 2, 1).value();
        const Vector2d rB = q.segment(3 * bodyB + 0, 2);
        const double phiB = q.segment(3 * bodyB + 2, 1).value();

        switch (jointType)
        {
            case body::JOINT_TYPE::PRISMATIC:
            {
                const Vector2d sA = bodyA < 0 ? Vector2d(0.0, 0.0) : input.pickBodyType(bodyA).s1C;

                if(bodyA >= 0)
                {
                    const VectorXd tempV = -(Rot(phiB) * v);
                    matrixEntries.emplace_back(3 * bodyA + 0, phiPile + 0, tempV(0));
                    matrixEntries.emplace_back(3 * bodyA + 1, phiPile + 0, tempV(1));

                    matrixEntries.emplace_back(3 * bodyA + 2, phiPile + 0, -(Rot(phiB) * v).transpose() * Om * Rot(phiA) * sA);
                    matrixEntries.emplace_back(3 * bodyA + 2, phiPile + 1, 1);
                }

                if(bodyB >= 0)
                {
                    const VectorXd tempV = (Rot(phiB) * v);
                    matrixEntries.emplace_back(3 * bodyB + 0, phiPile + 0, tempV(0));
                    matrixEntries.emplace_back(3 * bodyB + 1, phiPile + 0, tempV(1));

                    matrixEntries.emplace_back(3 * bodyB + 2, phiPile + 0, -(Rot(phiB) * v).transpose() * Om *
                        (rB - rA - Rot(phiA) * sA));
                    matrixEntries.emplace_back(3 * bodyB + 2, phiPile + 1, -1);
                }

                break;
            }
            case body::JOINT_TYPE::REVOLUTE:
            {
                const Vector2d sA = bodyA < 0 ? Vector2d(0.0, 0.0) : input.pickBodyType(bodyA).s12;
                const Vector2d sB = Vector2d::Zero();

                if(bodyA >= 0)
                {
                    matrixEntries.emplace_back(3 * bodyA + 0, phiPile + 0, 1);
                    matrixEntries.emplace_back(3 * bodyA + 1, phiPile + 1, 1);

                    const VectorXd tempV = Om * Rot(phiA) * sA;
                    matrixEntries.emplace_back(3 * bodyA + 2, phiPile + 0, tempV(0));
                    matrixEntries.emplace_back(3 * bodyA + 2, phiPile + 1, tempV(1));
                }

                if(bodyB >= 0)
                {
                    matrixEntries.emplace_back(3 * bodyB + 0, phiPile + 0, -1);
                    matrixEntries.emplace_back(3 * bodyB + 1, phiPile + 1, -1);

                    const VectorXd tempV = - Om * Rot(phiB) * sB;
                    matrixEntries.emplace_back(3 * bodyB + 2, phiPile + 0, tempV(0));
                    matrixEntries.emplace_back(3 * bodyB + 2, phiPile + 1, tempV(1));
                }

                break;
            }
            default:
            {
                std::runtime_error("Unknown joint type.");
                break;
            }
        }

        phiPile += jointClass;
    }

    SparseMatrix<double> qSparse(3 * bodiesNumber, constrDim);

    qSparse.setFromTriplets(matrixEntries.begin(), matrixEntries.end());
    return qSparse;
}