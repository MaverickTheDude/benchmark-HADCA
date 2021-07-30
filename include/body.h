#pragma once

#include "../Eigen/Dense"

#include <map>

using namespace Eigen;

class body {
public:
    body(std::string type);

public:
    Vector2d s1C;
    Vector2d s12;
    double m, J;
    Vector3d H;
    Matrix<double, 3, 2> D;
    std::map<std::string, Vector2d> dimensions;
};