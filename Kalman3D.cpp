#include "Kalman3D.h"

#include <Eigen/Dense>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <string>
// predict distance laser or remove abnormal values (EKF of madam Dung)
using namespace Eigen;
using namespace std;

Kalman3D::Kalman3D() {
    KF.init(stateDim, measureDim, 0, CV_32F);
}

Kalman3D::~Kalman3D() {
}

void Kalman3D::setParams(double samplingFreq, double rangeErrorEs, double aziErrorEs, double eleErrorEs, Mat accErrorEs) {
    (void)accErrorEs;
    this->rangeErrorEs = rangeErrorEs;
    this->aziErrorEs = aziErrorEs;
    this->eleErrorEs = eleErrorEs;
    this->dt = static_cast<float>(samplingFreq);
    this->samplingFreq = samplingFreq;
}


void Kalman3D::init(Mat measurement, float currentTime) {

    Mat xState = Mat::zeros(stateDim, 1, CV_32F);
    Mat transitionMatrix(9, 9, CV_32F);
    Mat processNoiseCov = Mat::zeros(stateDim, stateDim, CV_32F);
    Mat errorCovPost = Mat::zeros(stateDim, stateDim, CV_32F);

    measurementNoiseCov.row(0).at<float>(0) = 25;
    measurementNoiseCov.row(1).at<float>(1) = static_cast<float>(pow(0.0003, 2));
    measurementNoiseCov.row(2).at<float>(2) = static_cast<float>(pow(0.0003, 2));
    xState.at<float>(0) = measurement.at<float>(0) * cos(measurement.at<float>(1)) * sin(measurement.at<float>(2));
    xState.at<float>(1) = measurement.at<float>(0) * cos(measurement.at<float>(1)) * cos(measurement.at<float>(2));
    xState.at<float>(2) = measurement.at<float>(0) * sin(measurement.at<float>(1));
    xState.at<float>(3) = 0;
    xState.at<float>(4) = 0;
    xState.at<float>(5) = 0;
    xState.at<float>(6) = 0;
    xState.at<float>(7) = 0;
    xState.at<float>(8) = 0;
    float transitionMatrix1[9][9] = {{(1), (0), (0), (dt), (0), (0), (0.5 * dt * dt), (0), (0)},
                                    {(0), (1), (0), (0), (dt), (0), (0), (0.5 * dt * dt), (0)},
                                    {(0), (0), (1), (0), (0), (dt), (0), (0), (0.5 * dt * dt)},
                                    {(0), (0), (0), (1), (0), (0), (dt), (0), (0)},
                                    {(0), (0), (0), (0), (1), (0), (0), (dt), (0)},
                                    {(0), (0), (0), (0), (0), (1), (0), (0), (dt)},
                                    {(0), (0), (0), (0), (0), (0), (1), (0), (0)},
                                    {(0), (0), (0), (0), (0), (0), (0), (1), (0)},
                                    {(0), (0), (0), (0), (0), (0), (0), (0), (1)}};

    memcpy(transitionMatrix.data, transitionMatrix1, static_cast<unsigned long>(stateDim * stateDim) * sizeof(float));

    float processNoiseCov1[9][9] = {{(0.25 * pow(dt,4)), (0), (0), (0.5*pow(dt,3)), (0), (0), (0.5 * dt * dt), (0), (0)},
                                    {(0), (0.25 * pow(dt,4)), (0), (0), (0.5*pow(dt,3)), (0), (0), (0.5 * dt * dt), (0)},
                                    {(0), (0), (0.25 * pow(dt,4)), (0), (0), (0.5*pow(dt,3)), (0), (0), (0.5 * dt * dt)},
                                    {(0.5*pow(dt,3)), (0), (0), (dt*dt), (0), (0), (dt), (0), (0)},
                                    {(0), (0.5*pow(dt,3)), (0), (0), (dt*dt), (0), (0), (dt), (0)},
                                    {(0), (0), (0.5*pow(dt,3)), (0), (0), (dt*dt), (0), (0), (dt)},
                                    {(0.5 * dt * dt), (0), (0), (0), (0), (0), (0.5 * dt * dt), (0), (0)},
                                    {(0), (0.5 * dt * dt), (0), (0), (0), (0), (0), (0.5 * dt * dt), (0)},
                                    {(0), (0), (0.5 * dt * dt), (0), (0), (0), (0), (0), (0.5 * dt * dt)}};

//    float processNoiseCov1[9][9] = {{(0.25), (0), (0), (0.5), (0), (0), (0.5), (0), (0)},
//                                    {(0), (0.25), (0), (0), (0.5), (0), (0), (0.5), (0)},
//                                    {(0), (0), (0.25), (0), (0), (0.5), (0), (0), (0.5)},
//                                    {(0.5), (0), (0), (1.0), (0), (0), (1.0), (0), (0)},
//                                    {(0), (0.5), (0), (0), (1.0), (0), (0), (1.0), (0)},
//                                    {(0), (0), (0.5), (0), (0), (1.0), (0), (0), (1.0)},
//                                    {(0.5), (0), (0), (0), (0), (0), (0.5), (0), (0)},
//                                    {(0), (0.5), (0), (0), (0), (0), (0), (0.5), (0)},
//                                    {(0), (0), (0.5), (0), (0), (0), (0), (0), (0.5)}};


    memcpy(processNoiseCov.data, processNoiseCov1, static_cast<unsigned long>(stateDim * stateDim) * sizeof(float));

    errorCovPost = 100 * Mat::eye(stateDim, stateDim, CV_32F);
    setIdentity(KF.measurementMatrix);                   // H
    transitionMatrix.copyTo(KF.transitionMatrix);        // A/F
    processNoiseCov.copyTo(KF.processNoiseCov);          // Q
    errorCovPost.copyTo(KF.errorCovPost);                // Pk
    xState.copyTo(KF.statePost);                         // xk
    measurementNoiseCov.copyTo(KF.measurementNoiseCov);  // R
    xState.copyTo(lastMeasurement);
    lastTime = currentTime;
    lastdistance = measurement.at<float>(0);
    lastTilt = measurement.at<float>(1);
    lastPan = measurement.at<float>(2);
    lastpoint = converttoxyz(measurement);
    predictValues = KF.predict();
    predictValues.copyTo(predictValuesOneTime);
    printf("end\n");
    cout << "........."<<endl;
}


Mat Kalman3D::hx(Mat estimateValue) {
    Mat hx = Mat::zeros(3, 1, CV_32F);
    float x = estimateValue.at<float>(0);
    float y = estimateValue.at<float>(1);
    float z = estimateValue.at<float>(2);
    hx.row(0).at<float>(0) = static_cast<float>(sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2)));
    hx.row(1).at<float>(0) = asin(z / hx.row(0).at<float>(0));
    float pan = atan2(x, y);
    if (pan < 0) {
        pan += static_cast<float>(2 * M_PI);
    }
    hx.row(2).at<float>(0) = pan;
    return hx;
}


Mat Kalman3D::converttoxyz(Mat measurement) {
    float rangeMeasured = measurement.at<float>(0);
    float x = rangeMeasured * cos(measurement.at<float>(1)) * sin(measurement.at<float>(2));
    float y = rangeMeasured * cos(measurement.at<float>(1)) * cos(measurement.at<float>(2));
    float z = rangeMeasured * sin(measurement.at<float>(1));
    float vx = 0;
    float vy = 0;
    float vz = 0;
    float ax = 0;
    float ay = 0;
    float az = 0;
    Mat xyz = (Mat_<float>(9, 1) << x, y, z);
    return xyz;
}

float Kalman3D::computedistancetwopoint(Mat measurement) {
    Mat currencepoint = converttoxyz(measurement);
    float xLast = lastpoint.row(0).at<float>(0);
    float yLast = lastpoint.row(1).at<float>(0);
    float zLast = lastpoint.row(2).at<float>(0);
    float xCurrence = currencepoint.row(0).at<float>(0);
    float yCurrence = currencepoint.row(1).at<float>(0);
    float zCurrence = currencepoint.row(2).at<float>(0);

    return static_cast<float>(sqrt(pow((xCurrence - xLast), 2) + pow((yCurrence - yLast), 2) + pow((zCurrence - zLast), 2)));
}

Mat Kalman3D::hJacobian(Mat input) {
    float x = float(input.at<float>(0));
    float y = float(input.at<float>(1));
    float z = float(input.at<float>(2));
    float r = static_cast<float>(sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2)));
    double Hx[3][9] = {{static_cast<double>(x / r), static_cast<double>(y / r), static_cast<double>(z / r), (0), (0), (0), (0), (0), (0)},
                       {(static_cast<double>(-x * z) / (sqrt(pow(x, 2) + pow(y, 2)) * pow(r, 2))), (static_cast<double>(-y * z) / (sqrt(pow(x, 2) + pow(y, 2)) * pow(r, 2))), (sqrt(pow(x, 2) + pow(y, 2)) / pow(r, 2)), (0), (0), (0), (0), (0), (0)},
                       {(static_cast<double>(y) / (pow(x, 2) + pow(y, 2))), (static_cast<double>(-x) / (pow(x, 2) + pow(y, 2))), (0), (0), (0), (0), (0), (0), (0)}};

    Mat hxMat(3, 9, CV_64F);  // Tạo một ma trận 3x9 với kiểu dữ liệu double
    memcpy(hxMat.data, Hx, static_cast<unsigned long>(stateDim / 3 * stateDim) * sizeof(double));
    hxMat.convertTo(hxMat, CV_32F);
    return hxMat;
}



const Mat &Kalman3D::predict(Mat measurement, double currentimem, int number_values_predict) {
    measurement.copyTo(predictValues); //KF.predict();
    cout << "input_predict_tjtr3D: " << measurement << endl;
    cout << "predictValues_before_once: " << predictValues << endl;
//    predictValues = KF.predict();
    predictValues = KF.transitionMatrix * measurement;
    predictValues.copyTo(predictValuesOneTime);
    cout << "predictValuesOneTime: " << predictValuesOneTime << endl;
    for (int count = 1; count < number_values_predict; count++) {
        predictValues = KF.transitionMatrix * predictValues;
//            predictValues = KF.predict();
        double x = static_cast<double>(predictValues.at<float>(0));
        double y = static_cast<double>(predictValues.at<float>(1));
        double z = static_cast<double>(predictValues.at<float>(2));
        double estimatedDistance = sqrt(x * x + y * y + z * z);
        double tilt = asin(z / estimatedDistance);
        double pan = atan2(x, y);
        if (pan < 0) {
            pan = pan + 2 * M_PI;
        }
        cout << "predictValues_in_loop_for " << predictValues << endl;
        cout << "pan_in_loop_for " << pan << endl;
        cout << "tilt_in_loop_for " << tilt << endl;
    }



    double x = static_cast<double>(predictValues.at<float>(0));
    double y = static_cast<double>(predictValues.at<float>(1));
    double z = static_cast<double>(predictValues.at<float>(2));
    double estimatedDistance = sqrt(x * x + y * y + z * z);
    double tilt = asin(z / estimatedDistance);
    double pan = atan2(x, y);
    if (pan < 0) {
        pan = pan + 2 * M_PI;
    }

    cout << "predictValues: " << predictValues << endl;
//    cout << "measurement: " << measurement << endl;
    return predictValues;
}

Mat Kalman3D::update(Mat measurement, float currentime, float objectVmax, int number_values_predict) {
    cout << "measurement in update 3D: " << measurement << endl;
    Mat output;
    float deltaTime = currentime - lastTime;                     // khoảng thời gian giữa lần có giá trị hợp lệ đến thời điểm hiện tại
    float maxdistance = deltaTime * objectVmax;                  // quảng đường lớn nhất mà đối tượng đi được trong khoảng thời gian delta_time
    float deltaDistance = computedistancetwopoint(measurement);  // quảng đường đi được thực tế mà đối tượng đã di chuyện trong khoảng thời gian delta_time
    float deltaTiltAbs = abs(measurement.at<float>(1) - lastTilt);
    float deltaPanAbs = abs(measurement.at<float>(2) - lastPan);
    Mat xState = predictValuesOneTime;
//    Mat xState = KF.statePre;

    cout << "xState: " << xState << endl;
    Mat hX = hx(xState);
    cout << "hX: " << hX << endl;
    double deltaPan = (measurement.at<float>(2) - hX.row(2).at<float>(0));
    if (deltaPan < - M_PI) {
        deltaPan = deltaPan + 2 * M_PI;
    }
    else if (deltaPan > M_PI) {
        deltaPan = deltaPan - 2 * M_PI;
    }
    hX.row(2).at<float>(0) = measurement.at<float>(2) - deltaPan;

    if (measurement.at<float>(0) !=0.0 &&
        (measurement.at<float>(0) != lastdistance) &&
        deltaDistance <= maxdistance && deltaTiltAbs < 2000 && deltaPanAbs < 2000) {
        printf("update_dk1\n");
        Mat deltHx = measurement.t() - hX;
        cout << "measurement.t(): " << measurement.t() << endl;
        cout << "deltHx: " << deltHx << endl;
//        Mat deltHx = measurement.t() - hx(xState);
        Mat Hk = hJacobian(xState);
        cout << "Hk: " << Hk << endl;
        Mat Sk = Hk * KF.errorCovPre * Hk.t() + KF.measurementNoiseCov;
        Mat SkInvert;
        invert(Sk, SkInvert, DECOMP_SVD);
        cout << "Sk: " << Sk << endl;
        KF.gain = KF.errorCovPre * Hk.t() * SkInvert;                      // K
        KF.errorCovPost = KF.errorCovPre - KF.gain * Hk * KF.errorCovPre;  // P
        cout << "KF.gain: " << KF.gain << endl;
        cout << "KF.errorCovPost: " << KF.errorCovPost << endl;
        KF.statePost = KF.statePost + KF.gain * deltHx;
        KF.statePost.copyTo(output);
        cout << "KF.statePost: " << KF.statePost << endl;
        lastTime = currentime;
        lastpoint = converttoxyz(measurement);
        lastdistance = measurement.at<float>(0);
        lastTilt = measurement.at<float>(1);
        lastPan = measurement.at<float>(2);
        cout << "output: " << output << endl;

    } else {
        printf("update_dk2\n");
        float distance = static_cast<float>(sqrt(pow(predictValuesOneTime.at<float>(0), 2) + pow(predictValuesOneTime.at<float>(1), 2) + pow(predictValuesOneTime.at<float>(2), 2)));
        float x = distance * cos(measurement.at<float>(1)) * sin(measurement.at<float>(2));
        float y = distance * cos(measurement.at<float>(1)) * cos(measurement.at<float>(2));
        float z = distance * sin(measurement.at<float>(1));
        output = (Mat_<float>(stateDim, 1) << x, y, z, predictValuesOneTime.at<float>(3), predictValuesOneTime.at<float>(4), predictValuesOneTime.at<float>(5));
    }
    cout << "output_in_update: " << output << endl;
    predictValues = predict(output, static_cast<double>(currentime), number_values_predict);
//    xState = KF.correct(predictValues);
    double xTrajectory = static_cast<double>(this->predictValues.at<float>(0));
    double yTrajectory = static_cast<double>(this->predictValues.at<float>(1));
    double zTrajectory = static_cast<double>(this->predictValues.at<float>(2));
    double estimatedDistanceTrajectory = sqrt(xTrajectory * xTrajectory + yTrajectory * yTrajectory + zTrajectory * zTrajectory);

    double tiltkal3d = asin(zTrajectory / estimatedDistanceTrajectory);
    double pankal3d = atan2(xTrajectory, yTrajectory);
    if (pankal3d < 0) {
        pankal3d = pankal3d + 2 * M_PI;
    }
    cout << "pankal3d: " << pankal3d << endl;
    cout << "tiltkal3d: " << tiltkal3d << endl;
    cout << "predictValues_is_in_fun_update: " << this->predictValues << endl;
    return predictValues;
}













