#include "../../include/task/F.h"

#include "../../Eigen/Dense"
#include "../../include/input.h"
#include "../../include/utils.h"
#include "../../include/constants.h"
#include "../../include/body.h"
#include "../../include/task/M.h"

using namespace task;
using namespace Eigen;

F::F(const _input_& i) : input(i) {}

F_1::F_1(const _input_& i) : F(i) {}


/**
 * Arguments: (const VectorXd &q, const VectorXd &dq, const VectorXd &u)
 */

VectorXd F_1::operator()(const VectorXd &q, const VectorXd &dq, const VectorXd &u) const
{
    /**
     * Returns forces column-vector F (it is not transposed).
     */

    VectorXd forces = VectorXd::Zero(3 * input.Nbodies);
    const VectorXd absoluteAlpha = absolutePositionToAbsoluteAlpha(q);

    forces(0) = u(0);
    for (int i = 0; i < input.Nbodies; i++)
        forces.segment(3 * i, 3) += SAB("s1C", i, absoluteAlpha, input) * g;

    return forces;
}

MatrixXd F_1::q(const VectorXd &q, const VectorXd &dq, const VectorXd &u) const
{
    MatrixXd F_1_q = MatrixXd::Zero(3 * input.Nbodies, 3 * input.Nbodies);
    const VectorXd absoluteAlpha = absolutePositionToAbsoluteAlpha(q);

    for(int i = 0; i < input.Nbodies; i++)
        F_1_q.block(3 * i + 2, 3 * i, 1, 3) = (dSABdAlpha(input.pickBodyType(i).s1C,
            absoluteAlpha(i)) * g).transpose();

    return F_1_q;
}

MatrixXd F_1::dq(const VectorXd &q, const VectorXd &dq, const VectorXd &u) const
{
    MatrixXd F_1_dq = MatrixXd::Zero(3 * input.Nbodies, 3 * input.Nbodies);

    return F_1_dq;
}

MatrixXd F_1::ddtdq(const VectorXd& q, const VectorXd& dq, const VectorXd& u) const
{
    MatrixXd F_1_ddtdq = MatrixXd::Zero(3 * input.Nbodies, 3 * input.Nbodies);

    return F_1_ddtdq;
}

/**
 * Arguments: (const int bodyNumber, const VectorXd &q, const VectorXd &dq, const VectorXd &u)
 */

VectorXd F_1::operator()(const int bodyNumber, const VectorXd &q, const VectorXd &dq, const VectorXd &u) const
{
    /**
     * Returns part of forces column-vector F (it is not transposed).
     */

    Vector3d forces = Vector3d::Zero();

    if(bodyNumber == 0)
        forces(0) = u(0);

    forces += SAB("s1C", bodyNumber, q(3 * bodyNumber + 2), input) * g;

    return forces;
}

MatrixXd F_1::q(const int bodyNumber, const VectorXd &q, const VectorXd &dq, const VectorXd &u) const
{
    MatrixXd F_1_q = MatrixXd::Zero(3, 3 * input.Nbodies);
    Matrix3d dS_dAlpha = dSABdAlpha(input.pickBodyType(bodyNumber).s1C, q(3 * bodyNumber + 2));
    F_1_q.block(2, 3 * bodyNumber, 1, 3) = ( dS_dAlpha * M::local(bodyNumber, input) * g ).transpose();

    return F_1_q;
}

MatrixXd F_1::dq(const int bodyNumber, const VectorXd &q, const VectorXd &dq, const VectorXd &u) const
{
    MatrixXd F_1_dq = MatrixXd::Zero(3, 3 * input.Nbodies);

    return F_1_dq;
}

MatrixXd F_1::ddtdq(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& u) const
{
    MatrixXd F_1_ddtdq = MatrixXd::Zero(3, 3 * input.Nbodies);

    return F_1_ddtdq; 
}

/**
 * Arguments: (const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& u, VectorXd& eta)
 */

Vector3d F_1::q(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& u, const VectorXd& ksi) const
{
    /**
     * Arguments:
     *    VectorXd ksi [3]
     */

     return F_1::q(bodyNumber, q, dq, u).block(0, 3 * bodyNumber, 3, 3) * ksi;
}

 Vector3d F_1::dq(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& u, const VectorXd& eta) const
 {
     /**
      * Arguments:
      *    VectorXd eta [3 * input.Nbodies]
      */

     // komentuje, zeby nie generowac trywialnych iloczynow
    //  return F_1::dq(bodyNumber, q, dq, u) * eta;
    return Vector3d::Zero();
 }

 Vector3d F_1::ddtdq(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& u, const VectorXd& ksi) const
{
    /**
     * Arguments:
     *    VectorXd ksi [3]
     */

     return F_1::ddtdq(bodyNumber, q, dq, u).block(0, 3 * bodyNumber, 3, 3) * ksi;
}

/**
 * Arguments: (const VectorXd& q, const VectorXd& dq, const VectorXd& u, const VectorXd& ksi)
 */

VectorXd F_1::q(const VectorXd& q, const VectorXd& dq, const VectorXd& u, const VectorXd& ksi) const
{
    VectorXd rhs = VectorXd::Zero(3 * input.Nbodies);

    for(int i = 0; i < input.Nbodies; i++)
        rhs.segment(3 * i, 3) = F_1::q(i, q, dq, u) * ksi; 

    return rhs;
}

VectorXd F_1::dq(const VectorXd& q, const VectorXd& dq, const VectorXd& u, const VectorXd& eta) const
{
    return VectorXd::Zero(3 * input.Nbodies);
}

VectorXd F_1::ddtdq(const VectorXd& q, const VectorXd& dq, const VectorXd& u, const VectorXd& ksi) const
{
    return VectorXd::Zero(3 * input.Nbodies);
}

F_2::F_2(const _input_& i) : F_1(i), jointsNumber(i.Nbodies) {}

/**
 * Arguments: (const VectorXd &q, const VectorXd &dq, const VectorXd &u)
 */

VectorXd F_2::operator()(const VectorXd &q, const VectorXd &dq, const VectorXd &u) const
{
    VectorXd F_2 = F_1::operator()(q, dq, u);

    for(int jN = 0; jN < jointsNumber; jN++)
    {
        const body::JOINT_TYPE jointType = input.pickBodyType(jN).jointType;
        const int bodyA = jN - 1;
        const int bodyB = jN;

        switch (jointType)
        {
            case body::JOINT_TYPE::PRISMATIC:
            {
                /** 
                 * I leave the friction force hard-coded in center of the body 0 (cart) as
                 * currently the code does not need to be any more general - ceratinly it
                 * is #TODO for more general version of the code.
                 */
                F_2(0) += - input.pickBodyFriction(0) * dq(0); 
                break;
            }
            case body::JOINT_TYPE::REVOLUTE:
            {
                /**
                 * It might be a bit unclear whether picked body friction is the body's
                 * feature or the joint's feature.
                 */ 
                F_2(3 * bodyB + 2) += - input.pickBodyFriction(bodyB) *
                    (dq(3 * bodyB + 2) - dq(3 * bodyA + 2));
                F_2(3 * bodyA + 2) += - F_2(3 * bodyB + 2);
                break;
            }
            default:
            {
                std::runtime_error("Unknown joint type.");
                break;
            }
        }
    }

    return F_2;
}

MatrixXd F_2::dq(const VectorXd &q, const VectorXd &dq, const VectorXd &u) const
{
    MatrixXd F_2_dq = F_1::dq(q, dq, u);

    for(int jN = 0; jN < jointsNumber; jN++)
    {
        const body::JOINT_TYPE jointType = input.pickBodyType(jN).jointType;
        const int bodyA = jN - 1;
        const int bodyB = jN;

        switch (jointType)
        {
            case body::JOINT_TYPE::PRISMATIC:
            {
                /** 
                 * I leave the friction force hard-coded in center of the body 0 (cart) as
                 * currently the code does not need to be any more general - ceratinly it
                 * is #TODO for more general version of the code.
                 */
                F_2_dq(0, 0) += - input.pickBodyFriction(0); 
                break;
            }
            case body::JOINT_TYPE::REVOLUTE:
            {
                /**
                 * We add to the transposed matrix so F_transposed(2, 1) -> F_(1, 2)
                 */

                F_2_dq(3 * bodyB + 2, 3 * bodyB + 2) += - input.pickBodyFriction(bodyB);
                F_2_dq(3 * bodyA + 2, 3 * bodyB + 2) += input.pickBodyFriction(bodyB);
                F_2_dq(3 * bodyB + 2, 3 * bodyA + 2) += - F_2_dq(3 * bodyB + 2, 3 * bodyB + 2);
                F_2_dq(3 * bodyA + 2, 3 * bodyA + 2) += - F_2_dq(3 * bodyA + 2, 3 * bodyB + 2);
                break;
            }
            default:
            {
                std::runtime_error("Unknown joint type.");
                break;
            }
        }
    }

    return F_2_dq;
}

/**
 * Arguments: (const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& u)
 */

VectorXd F_2::operator()(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& u) const
{
    Vector3d F_2 = F_1::operator()(bodyNumber, q, dq, u);

    struct jointData {
        int bodyA;
        int bodyB;
        int jN;
        int phiPile;

        jointData(int _bodyA, int _bodyB, int _jN, int _phiPile) :
            bodyA(_bodyA), bodyB(_bodyB), jN(_jN), phiPile(_phiPile) {} 
    };

    std::vector<struct jointData> bodyAdjacents;
    bodyAdjacents.emplace_back(bodyNumber - 1, bodyNumber, bodyNumber, 2 * bodyNumber);
    if(bodyNumber < input.Nbodies - 1)
        bodyAdjacents.emplace_back(bodyNumber, bodyNumber + 1, bodyNumber + 1, 2 * (bodyNumber + 1));

    for(std::vector<struct jointData>::iterator joint = bodyAdjacents.begin(); 
        joint < bodyAdjacents.end(); joint++)
    {
        const int jN = joint->jN;

        const body::JOINT_TYPE jointType = input.pickBodyType(jN).jointType;

        const int bodyA = joint->bodyA;
        const int bodyB = joint->bodyB;

        switch (jointType)
        {
            case body::JOINT_TYPE::PRISMATIC:
            {
                F_2(0) += - input.pickBodyFriction(0); 
                break;
            }
            case body::JOINT_TYPE::REVOLUTE:
            {
                if(bodyNumber == bodyB)
                    F_2(3 * bodyB + 2) += - input.pickBodyFriction(bodyB) *
                        (dq(3 * bodyB + 2) - dq(3 * bodyA + 2));
                else if(bodyNumber == bodyA)
                    F_2(3 * bodyA + 2) += input.pickBodyFriction(bodyB) *
                        (dq(3 * bodyB + 2) - dq(3 * bodyA + 2));
                break;
            }
            default:
            {
                std::runtime_error("Unknown joint type.");
                break;
            }
        }
    }

    return F_2;
}

MatrixXd F_2::dq(const int bodyNumber, const VectorXd &q, const VectorXd &dq, const VectorXd &u) const
{
    MatrixXd F_2_dq = F_1::dq(bodyNumber, q, dq, u);

    struct jointData {
        int bodyA;
        int bodyB;
        int jN;
        int phiPile;

        jointData(int _bodyA, int _bodyB, int _jN, int _phiPile) :
            bodyA(_bodyA), bodyB(_bodyB), jN(_jN), phiPile(_phiPile) {} 
    };

    std::vector<struct jointData> bodyAdjacents;
    bodyAdjacents.emplace_back(bodyNumber - 1, bodyNumber, bodyNumber, 2 * bodyNumber);
    if(bodyNumber < input.Nbodies - 1)
        bodyAdjacents.emplace_back(bodyNumber, bodyNumber + 1, bodyNumber + 1, 2 * (bodyNumber + 1));

    for(std::vector<struct jointData>::iterator joint = bodyAdjacents.begin(); 
        joint < bodyAdjacents.end(); joint++)
    {
        const int jN = joint->jN;

        const body::JOINT_TYPE jointType = input.pickBodyType(jN).jointType;

        const int bodyA = joint->bodyA;
        const int bodyB = joint->bodyB;

        switch (jointType)
        {
            case body::JOINT_TYPE::PRISMATIC:
            {
                F_2_dq(0, 0) += - input.pickBodyFriction(0); 
                break;
            }
            case body::JOINT_TYPE::REVOLUTE:
            {
                if(bodyNumber == bodyB)
                {
                    // d / d (3 * bodyB + 2)
                    F_2_dq(2, 3 * bodyB + 2) += - input.pickBodyFriction(bodyB);
                    F_2_dq(2, 3 * bodyA + 2) += input.pickBodyFriction(bodyB);
                }
                else if(bodyNumber == bodyA)
                {
                    // d / d (3 * bodyA + 2)
                    F_2_dq(2, 3 * bodyB + 2) += input.pickBodyFriction(bodyB);
                    F_2_dq(2, 3 * bodyA + 2) += - input.pickBodyFriction(bodyB);
                }

                break;
            }
            default:
            {
                std::runtime_error("Unknown joint type.");
                break;
            }
        }
    }

    return F_2_dq;
}

/**
 * Arguments: (const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& u, VectorXd& eta)
 */

Vector3d F_2::dq(const int bodyNumber, const VectorXd& q, const VectorXd& dq, const VectorXd& u, const VectorXd& eta) const
{
    /**
     * Arguments:
     *    VectorXd eta [3 * input.Nbodies]
     */

    Vector3d F_2_dq = F_1::dq(bodyNumber, q, dq, u, eta);

    struct jointData {
        int bodyA;
        int bodyB;
        int jN;
        int phiPile;

        jointData(int _bodyA, int _bodyB, int _jN, int _phiPile) :
            bodyA(_bodyA), bodyB(_bodyB), jN(_jN), phiPile(_phiPile) {} 
    };

    std::vector<struct jointData> bodyAdjacents;
    bodyAdjacents.emplace_back(bodyNumber - 1, bodyNumber, bodyNumber, 2 * bodyNumber);
    if(bodyNumber < input.Nbodies - 1)
        bodyAdjacents.emplace_back(bodyNumber, bodyNumber + 1, bodyNumber + 1, 2 * (bodyNumber + 1));

    for(std::vector<struct jointData>::iterator joint = bodyAdjacents.begin(); 
        joint < bodyAdjacents.end(); joint++)
    {
        const int jN = joint->jN;

        const body::JOINT_TYPE jointType = input.pickBodyType(jN).jointType;

        const int bodyA = joint->bodyA;
        const int bodyB = joint->bodyB;

        switch (jointType)
        {
            case body::JOINT_TYPE::PRISMATIC:
            {
                F_2_dq(0) += - input.pickBodyFriction(0) * eta(0);
                break;
            }
            case body::JOINT_TYPE::REVOLUTE:
            {
                if(bodyNumber == bodyB)
                {
                    // d / d (3 * bodyB + 2)
                    F_2_dq(2) += - input.pickBodyFriction(bodyB) * eta(3 * bodyB + 2);
                    F_2_dq(2) += input.pickBodyFriction(bodyB) * eta(3 * bodyA + 2);
                }
                else if(bodyNumber == bodyA)
                {
                    // d / d (3 * bodyA + 2)
                    F_2_dq(2) += input.pickBodyFriction(bodyB) * eta(3 * bodyB + 2);
                    F_2_dq(2) += - input.pickBodyFriction(bodyB) * eta(3 * bodyA + 2);
                }

                break;
            }
            default:
            {
                throw std::runtime_error("Vector3d F_2_dq: Unknown joint type.");
                break;
            }
        }
    }

    return F_2_dq;
}

/**
 * Arguments: (const VectorXd& q, const VectorXd& dq, const VectorXd& u, const VectorXd& ksi)
 */

VectorXd F_2::dq(const VectorXd& q, const VectorXd& dq, const VectorXd& u, const VectorXd& eta) const
{
    VectorXd F_2_dq = F_1::dq(q, dq, u, eta);

    for(int i = 0; i < input.Nbodies; i++)
        F_2_dq.segment(3 * i, 3) = F_2::dq(i, q, dq, u) * eta;

    return F_2_dq;
}