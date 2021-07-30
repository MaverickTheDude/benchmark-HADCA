#include "../include/body.h"

#include "../include/constants.h"
#include "../Eigen/Dense"

#include <string>
#include <stdexcept>

using namespace Eigen;

body::body(std::string type)
{
    m = _m_;
    J = _J_;
    if (!type.compare("box"))
    {
        s1C.setZero();
        s12.setZero();
        H << 1, 0, 0;
        D << 0, 0, 1, 0, 0, 1;
    }
    else if (!type.compare("link"))
    {
        s1C << _L_ / 2, 0;
        s12 << _L_, 0;
        H << 0, 0, 1;
        D << 1, 0, 0, 1, 0, 0;
    }
    else
        throw std::runtime_error("not supported body / joint");
    dimensions["s12"] = s12;
    dimensions["s21"] = -s12;
    dimensions["s1C"] = s1C;
    dimensions["s2C"] = s1C - s12;
}