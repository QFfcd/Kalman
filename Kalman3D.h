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

/*
for testing only
@brief Read data from text file
@in path Pointer to file path
@in data Pointer to save data
@param type Type of data (float, char or decimal)
*/

class Kalman3D {
   private:
    Mat hJacobian(Mat input);
    Mat hx(Mat estimate_value);
    Mat converttoxyz(Mat measurement);
    float computedistancetwopoint(Mat measurement);

    KalmanFilter KF;
    int stateDim = 9;
    int measureDim = 3;
    float madFactor = 1.482602218505602f;
    double lastValidResultTime = 0;
    float lastTime = 0;
    Mat predictValuesOneTime;
    Mat lastpoint;
    Mat predictValues;
    float dt = 1.0/ 30.0;
    double rangeEst;
    double lastrangeMeasure;
    float lastdistance;
    float lastPan;
    float lastTilt;
    double samplingFreq = 1;
    double rangeErrorEs = 5;                            // m
    double aziErrorEs = 0.3 / 1000;                     // mrad
    double eleErrorEs = 0.3 / 1000;                     // mrad
    Mat accErrorEs = (Mat_<float>(3, 1) << 10, 10, 5);  // acc_x, acc_y, acc_z
    Mat lastValidMeasurement = Mat::zeros(stateDim / 3, 1, CV_32F);
    int numValid = 0;
    Mat lastMeasurement = Mat::zeros(stateDim, 1, CV_32F);
    Mat measurementNoiseCov = Mat::zeros(measureDim, measureDim, CV_32F);
    float dLast[2048];
    int dLastCurrentIdx = 0;


   public:
    Kalman3D();
    virtual ~Kalman3D();
    const Mat& predict(Mat measurement, double currentime, int number_values_predict);
    void init(Mat measurement1, float currentTime);
    Mat update(Mat measurement, float currenttime, float objectVmax,int number_values_predict);
    void setParams(double samplingFreq, double rangeErrorEs = 0, double aziErrorEs = 0, double eleErrorEs = 0, Mat accErrorEs = cv::Mat());
    //    Mat predict(int max = 100);
};
