#pragma once

#include "../Eigen/Dense"

#include <map>

using namespace Eigen;

class body {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
public:
    enum JOINT_TYPE {PRISMATIC, REVOLUTE};

public:
    body(std::string type);

public:
    Vector2d s1C;
    Vector2d s12;
    double m, J;
    Vector3d H;
    Matrix<double, 3, 2> D;

    // std::map<std::string, Vector2d> dimensions;
    // https://eigen.tuxfamily.org/dox/group__TopicStlContainers.html -> Using an aligned allocator
    std::map<std::string, Vector2d, std::less<std::string>, 
         Eigen::aligned_allocator<std::pair<const std::string, Vector2d> > > dimensions;
    body::JOINT_TYPE jointType;
};