#ifndef KALMAN2D_H
#define KALMAN2D_H

#pragma once
#include <stdio.h>

#include <fstream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>

//#include "utility/utility.h"

#define DATATYPE_DECIMAL 0
#define DATATYPE_FLOAT 1
#define DATATYPE_CHAR 2
#define ELEM_SWAP(a, b)         \
    {                           \
        register float t = (a); \
        (a) = (b);              \
        (b) = t;                \
    }

using namespace cv;
using namespace std;

class Kalman2D
{
private:
    Mat transiteMatrix(double dt);
    Mat matrixB();
    Mat matrixC();
    Mat xhatMatrix(double input);
    Mat matrixQ();
    Mat matrixI();
    Mat transposeMatrix(const Mat matrix);
    Mat A;
    Mat Q;
    Mat C;
    Mat I;
    Mat A_transpose;
    Mat C_transpose;
    Mat pan_xhat;
    Mat tilt_xhat;
    Mat tilt_P;
    Mat pan_P;

public:
    Kalman2D();
    virtual ~Kalman2D();
    void init(double objPan, double objTilt);
    double updateKalmanForPan(double objPan, int valueNumberPredict);
    double updateKalmanForTilt(double objTilt, int valueNumberPredict);
};

#endif // KALMAN2D_H
