#include "KalmanLRF.h"
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

using namespace Eigen;
using namespace std;

void readDataFromTextFile(const char *path, void *data, const char type) {
    string line;
    ifstream dataFile(path);
    string token;
    uint32_t index = 0;
    if (type == DATATYPE_DECIMAL) {
        if (dataFile.is_open()) {
            while (getline(dataFile, line)) {
                stringstream sline(line);
                while (getline(sline, token, '\t')) {
                    ((int16_t *)data)[index] = stoi(token);
                    index++;
                }
            }
            dataFile.close();
        }
    } else if (type == DATATYPE_FLOAT) {
        if (dataFile.is_open()) {
            while (getline(dataFile, line)) {
                stringstream sline(line);
                while (getline(sline, token, '\t')) {
                    if (token.compare("\r") != 0)
                     {
                        ((float *)data)[index] = stof(token);

                        index++;
                    }

                }
            }
            dataFile.close();
        }
    } else {
        if (dataFile.is_open()) {
            while (getline(dataFile, line)) {
                stringstream sline(line);
                while (getline(sline, token, '\t')) {
                    ((char *)data)[index] = stoi(token);
                    index++;
                }
            }
            dataFile.close();
        }
    }
    return;
}

QVector<QStringList> KalmanLRF::readDataFromFile(const QString &filePath)
{
    QVector<QStringList> inputStringListVec;
    QFile inputFile(filePath);
    if (!inputFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qDebug() << "The file path is not found! " << filePath;
        return inputStringListVec;
    } else {
        qDebug() << "The file path is found! " << filePath;
    }
    QTextStream stream(&inputFile);
    stream.readLine();  // Ignore header labels
    while (!stream.atEnd()) {
        const QString line = stream.readLine();
        if (!line.startsWith('#') && line.contains(',')) {
            QStringList list = line.simplified().split(',');
            for (QString &str : list) {
                str = str.trimmed();
            }
            inputStringListVec.append(list);
        }
    }
    inputFile.close();
    return inputStringListVec;
}

KalmanLRF::KalmanLRF() {
    KF.init(stateDim, measureDim, 0, CV_32F);
}

KalmanLRF::~KalmanLRF() {
}

void KalmanLRF::setParams(double samplingFreq, double rangeErrorEs, double aziErrorEs, double eleErrorEs, Mat accErrorEs) {
    this->rangeErrorEs = rangeErrorEs;
    this->aziErrorEs = aziErrorEs;
    this->eleErrorEs = eleErrorEs;
    this->dt = 1.0 / samplingFreq;
    this->samplingFreq = samplingFreq;
    // acc_error_es.copyTo(this->acc_error_es);
}



void KalmanLRF::init(Mat measurement,  float currentTime,  double rangeErrorEs, double aziErrorEs, double eleErrorEs, double mesurementRealiability) {

    Mat xState = Mat::zeros(stateDim, 1, CV_32F);
    Mat transitionMatrix = Mat::zeros(stateDim, stateDim, CV_32F);
    Mat processNoiseCov = Mat::zeros(stateDim, stateDim, CV_32F);
    Mat errorCovPost = Mat::zeros(stateDim, stateDim, CV_32F);
    Mat temp1 = Mat::zeros(stateDim / 2, stateDim, CV_32F);
    Mat temp2 = Mat::zeros(stateDim / 2, stateDim, CV_32F);

    measurementNoiseCov.row(0).at<float>(0) = static_cast<float>(rangeErrorEs);
    measurementNoiseCov.row(1).at<float>(1) = static_cast<float>(eleErrorEs);
    measurementNoiseCov.row(2).at<float>(2) = static_cast<float>(aziErrorEs);

    xState.at<float>(0) = measurement.at<float>(0) * cos(measurement.at<float>(1)) * sin(measurement.at<float>(2));
    xState.at<float>(1) = measurement.at<float>(0) * cos(measurement.at<float>(1)) * cos(measurement.at<float>(2));
    xState.at<float>(2) = measurement.at<float>(0) * sin(measurement.at<float>(1));
    xState.at<float>(3) = 0;
    xState.at<float>(4) = 0;
    xState.at<float>(5) = 0;

    hconcat(Mat::eye(stateDim / 2, stateDim / 2, CV_32F), dt * Mat::eye(stateDim / 2, stateDim / 2, CV_32F), temp1);
    hconcat(Mat::zeros(stateDim / 2, stateDim / 2, CV_32F), Mat::eye(stateDim / 2, stateDim / 2, CV_32F), temp2);
    vconcat(temp1, temp2, transitionMatrix);
    cout << "transitionMatrix  A/F: " << transitionMatrix << endl;

    hconcat(Mat::diag(0.25 * pow(dt, 4) * accErrorEs.mul(accErrorEs)), Mat::zeros(stateDim / 2, stateDim / 2, CV_32F), temp1);
    hconcat(Mat::zeros(stateDim / 2, stateDim / 2, CV_32F), Mat::diag(dt * dt * accErrorEs.mul(accErrorEs)), temp2);
    vconcat(temp1, temp2, processNoiseCov);
    cout << "processNoiseCov Q: " << processNoiseCov << endl;
    cout << "........."<<endl;

    errorCovPost =  mesurementRealiability * cv::Mat::eye(stateDim, stateDim, CV_32F);
    setIdentity(KF.measurementMatrix);              //H
    transitionMatrix.copyTo(KF.transitionMatrix);  //A/F
    processNoiseCov.copyTo(KF.processNoiseCov);     //Q
    errorCovPost.copyTo(KF.errorCovPost);           //Pk
    xState.copyTo(KF.statePost);                    // xk
    measurementNoiseCov.copyTo(KF.measurementNoiseCov); //R
    xState.copyTo(lastMeasurement);
    lastTime = currentTime;
    lastdistance = measurement.at<float>(0);
    lastTilt = measurement.at<float>(1);
    lastPan = measurement.at<float>(2);
    lastVel = 0;
    lastAcc = 0;
    lastpoint = converttoxyz(measurement);
    predict_values = KF.predict();
}

Mat KalmanLRF::predict() {
    predict_values = KF.predict();

    return predict_values;
}

Mat KalmanLRF::HJacobian(Mat input){

    float x =  float(input.at<float>(0));
    float y =  float(input.at<float>(1));
    float z =  float(input.at<float>(2));
    float r = sqrt(pow(x,2) + pow(y,2) + pow(z,2));

    double Hx[3][6] = {{(x/r), (y/r),(z/r), (0), ( 0), ( 0)},
                    {((-x*z)/(sqrt(pow(x,2)+pow(y,2))*pow(r,2))), ( (-y*z)/(sqrt(pow(x,2)+pow(y,2))*pow(r,2))), ( sqrt(pow(x,2)+pow(y,2))/pow(r,2)), ( 0), ( 0), (0)},
                    {(y/(pow(x,2)+pow(y,2))), ( -x/(pow(x,2)+pow(y,2))), ( 0), ( 0), ( 0), ( 0)}};

    Mat Hx_mat(3, 6, CV_64F); // Tạo một ma trận 3x6 với kiểu dữ liệu double
    memcpy(Hx_mat.data, Hx, stateDim/2 * stateDim * sizeof(double));
    Hx_mat.convertTo(Hx_mat, CV_32F);
    return Hx_mat;
}

Mat KalmanLRF::hx(Mat estimate_value){
    Mat hx = Mat::zeros(3, 1, CV_32F);
    float x = estimate_value.at<float>(0);
    float y = estimate_value.at<float>(1);
    float z = estimate_value.at<float>(2);
    hx.row(0).at<float>(0) =  sqrt(pow(x,2) + pow(y,2) + pow(z,2));
    hx.row(1).at<float>(0) = asin(z/hx.row(0).at<float>(0));
    //pan = math.acos(y/(distance * math.cos(tilt)))
    float pan = atan2(x,y);
    if (pan < 0) {
        pan = pan + 2 * M_PI;
    }
    hx.row(2).at<float>(0) = pan;
    return hx;
}


Mat KalmanLRF::converttoxyz(Mat measurement){
    float rangeMeasured = measurement.at<float>(0);
    float x = rangeMeasured * cos(measurement.at<float>(1)) * sin(measurement.at<float>(2));
    float y = rangeMeasured * cos(measurement.at<float>(1)) * cos(measurement.at<float>(2));
    float z = rangeMeasured * sin(measurement.at<float>(1));
    Mat xyz = (Mat_<float>(3, 1) << x, y, z);
    return xyz;
}
// Tính khoảng cách giữa hai diểm: điểm hợp lệ gần nhất và điểm đang xét hiện tại
// mục đích: so sánh với khoảng cách lớn nhất và đối tượng đạt được trong cùng khoảng thời gian để xác định điểm bất thường
float KalmanLRF::computedistancetwopoint(Mat measurement){
//    cout << "measurement" << measurement<< endl;
//    cout << "lastpoint" << lastpoint<< endl;

    Mat currencepoint =  converttoxyz (measurement);
    cout <<"currencepoint: " << currencepoint << endl;
    float x_last = lastpoint.row(0).at<float>(0);
    float y_last = lastpoint.row(1).at<float>(0);
    float z_last = lastpoint.row(2).at<float>(0);
    float x_currence = currencepoint.row(0).at<float>(0);
    float y_currence = currencepoint.row(1).at<float>(0);
    float z_currence = currencepoint.row(2).at<float>(0);
    return sqrt(pow((x_currence - x_last),2) +pow((y_currence - y_last),2) + pow((z_currence - z_last),2));

}


Mat KalmanLRF::update(Mat measurement, float currentime, float objectVelMaxDefaul) {

    Mat output;
    this->KF.errorCovPre = KF.transitionMatrix * KF.errorCovPost * KF.transitionMatrix.t() + KF.processNoiseCov;
    float delta_time = currentime - lastTime; // khoảng thời gian giữa lần có giá trị hợp lệ đến thời điểm hiện tại
    float maxdistance = delta_time * objectVelMaxDefaul;  // quảng đường lớn nhất mà đối tượng đi được trong khoảng thời gian delta_time
    cout <<"maxdistance : " << maxdistance << endl;
    cout <<"delta_time : " << delta_time << endl;
    float delta_distance = computedistancetwopoint(measurement); // quảng đường đi được thực tế mà đối tượng đã di chuyện trong khoảng thời gian delta_time
    float objectVel = delta_distance/delta_time;
    float objectAcc = abs((objectVel- lastVel)/delta_time);
    float objectVelMaxAdap = lastVel + lastAcc;
    cout <<"lastVel : " << lastVel << endl;
    cout <<"lastAcc : " << lastAcc << endl;
    cout <<"objectAcc : " << objectAcc << endl;
    cout <<"objectVelMaxAdap : " << objectVelMaxAdap << endl;
    cout <<"objectVel : " << objectVel << endl;
    cout <<"delta_distance : " << delta_distance << endl;

    Mat xState = predict_values; // predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k)

    Mat hX = hx(xState);
    double k;
    cout <<"hX : " << hX << endl;
    double deltaPan = (measurement.at<float>(2) - hX.row(2).at<float>(0));
    if (deltaPan < - M_PI) {
        deltaPan = deltaPan + 2 * M_PI;
    }
    else if (deltaPan > M_PI) {
        deltaPan = deltaPan - 2 * M_PI;
    }
    hX.row(2).at<float>(0) = measurement.at<float>(2) - deltaPan;
    if (countRangeTrue > 6)
        {
        k = objectVelMaxAdap;
    }
    else
        {
        k = objectVelMaxDefaul;
    }
    if (measurement.at<float>(0) != 0  && objectVel <= k)
        {
        printf("update_Dk1\n");
        Mat delt_hx = measurement.t() - hX;
        cout <<"delt_hx : " << delt_hx << endl;
        Mat Hk = HJacobian(xState);
        Mat Sk = Hk * KF.errorCovPre * Hk.t() + KF.measurementNoiseCov;
        Mat Sk_invert;
        invert(Sk, Sk_invert, DECOMP_SVD); // convert of matrix Sk
        KF.gain = KF.errorCovPre * Hk.t() * Sk_invert; //K
        KF.errorCovPost = KF.errorCovPre - KF.gain * Hk* KF.errorCovPre;    //P
        KF.statePost = predict_values + KF.gain * delt_hx;
        KF.statePost.copyTo(output);
        lastTime = currentime;
        lastpoint = converttoxyz(measurement);
        lastdistance = measurement.at<float>(0);
        lastTilt = measurement.at<float>(1);
        lastPan = measurement.at<float>(2);
        lastVel = objectVel;
        lastAcc = objectAcc;
        countRangeTrue = countRangeTrue+1;
      }
    else
    {

        printf("update_Dk2\n");
//        predict_values = KF.predict();
        float distance = sqrt(pow(predict_values.at<float>(0),2) +pow(predict_values.at<float>(1),2) + pow(predict_values.at<float>(2),2));
        float x = distance * cos(measurement.at<float>(1)) * sin(measurement.at<float>(2));
        float y = distance * cos(measurement.at<float>(1)) * cos(measurement.at<float>(2));
        float z = distance * sin(measurement.at<float>(1));
        output = (Mat_<float>(stateDim, 1) << x, y, z, predict_values.at<float>(3), predict_values.at<float>(4), predict_values.at<float>(5));
//        output = (Mat_<float>(stateDim, 1) << predict_values.at<float>(0), predict_values.at<float>(1), predict_values.at<float>(2), predict_values.at<float>(3), predict_values.at<float>(4), predict_values.at<float>(5));
        }
    predict_values = KF.predict();
    return output;
}




//        CV_PROP_RW Mat statePre;           //!< predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k)
//        CV_PROP_RW Mat statePost;          //!< corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))
//        CV_PROP_RW Mat transitionMatrix;   //!< state transition matrix (A)
//        CV_PROP_RW Mat controlMatrix;      //!< control matrix (B) (not used if there is no control)
//        CV_PROP_RW Mat measurementMatrix;  //!< measurement matrix (H)
//        CV_PROP_RW Mat processNoiseCov;    //!< process noise covariance matrix (Q)
//        CV_PROP_RW Mat measurementNoiseCov;//!< measurement noise covariance matrix (R)
//        CV_PROP_RW Mat errorCovPre;        //!< priori error estimate covariance matrix (P'(k)): P'(k)=A*P(k-1)*At + Q)*/
//        CV_PROP_RW Mat gain;               //!< Kalman gain matrix (K(k)): K(k)=P'(k)*Ht*inv(H*P'(k)*Ht+R)
//        CV_PROP_RW Mat errorCovPost;       //!< posteriori error estimate covariance matrix (P(k)): P(k)=(I-K(k)*H)*P'(k)
