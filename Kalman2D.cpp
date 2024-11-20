#include "Kalman2D.h"


Kalman2D::Kalman2D()
{

}

Kalman2D::~Kalman2D() {
}
Mat Kalman2D::transiteMatrix(double dt) {
        double T[9] = {1, dt, dt * dt / 2, 0, 1, dt, 0, 0, 1};
        Mat T_mat(3, 3, CV_64F, T);  // Tạo một ma trận 3x6 với kiểu dữ liệu double
        return T_mat.clone();
}

Mat Kalman2D::matrixB() {
    double B[9] = {1, 0, 0,
                   0, 1, 0,
                   0, 0, 1};
    Mat B_mat(3, 3, CV_64F, B);  // Tạo một ma trận 3x6 với kiểu dữ liệu double
    return B_mat.clone();
}

Mat Kalman2D::matrixC() {
    double C[3] = {1, 0, 0};
    Mat C_mat(1, 3, CV_64F, C);
    return C_mat.clone();
}

Mat Kalman2D::xhatMatrix(double input) {
    double xhatInit[3] = {input, 0, 0};
    Mat xhatInit_mat(3, 1, CV_64F, xhatInit);  // Tạo một ma trận 3x6 với kiểu dữ liệu double
    return xhatInit_mat.clone();
}

Mat Kalman2D::matrixQ() {
    double Q[9] = {1.9 / 10, 0, 0, 0, 0.2, 0, 0, 0, 0.2};
    Mat Q_mat(3, 3, CV_64F, Q);  // Tạo một ma trận 3x6 với kiểu dữ liệu double
    return Q_mat.clone();
}

Mat Kalman2D::matrixI() {
    double I[9] = {1, 0, 0,
                   0, 1, 0,
                   0, 0, 1};
    Mat I_mat(3, 3, CV_64F, I);  // Tạo một ma trận 3x6 với kiểu dữ liệu double
    return I_mat.clone();
}

Mat Kalman2D::transposeMatrix(const Mat matrix) {
    int rows = matrix.rows;
    int cols = matrix.cols;
    cv::Mat transpose_mat;
    transpose(matrix, transpose_mat);
    return transpose_mat;
}

void Kalman2D::init(double objPan, double objTilt) {
//    float laserResult = static_cast<float>(lastDistance < 0 ? 0 : lastDistance);
//    double objPan;
//    double objTilt;
//    calculateObjPanTilt(objPan, objTilt);
//    laserTimeout->start();
    A = transiteMatrix(1.0f / 30.0f);
    double panInit = objPan;
    double tiltInit = objTilt;
    pan_xhat = xhatMatrix(panInit);
    tilt_xhat = xhatMatrix(tiltInit);
    Q = matrixQ();
    C = matrixC();
    pan_P = matrixB();
    tilt_P = matrixB();
    I = matrixI();
    A_transpose = transposeMatrix(A);
    C_transpose = transposeMatrix(C);

}


double Kalman2D::updateKalmanForPan(double objPan, int valueNumberPredict) {

    double measurementAnglePan = objPan;
    cout <<"measurementAnglePan " << measurementAnglePan << endl;
    Mat pan_x_hat_minus = this->A * this->pan_xhat;
    cout <<"A " << A << endl;
    Mat pan_x_hat_est = pan_x_hat_minus.clone();
    for (int j = 0; j < valueNumberPredict; j++) {
        pan_x_hat_est = this->A * pan_x_hat_est;
    }
    Mat pan_P_minus = this->A * this->pan_P * this->A_transpose + this->Q;
    Mat pan_K = pan_P_minus * this->C_transpose * (this->C * pan_P_minus *this-> C_transpose + (1 * 180 / 3.1416)).inv();
    Mat pan_X_0 = this->C * pan_x_hat_minus;
    double pan_X_1 = measurementAnglePan - pan_X_0.at<double>(0);
    this->pan_xhat = pan_x_hat_minus + pan_K.mul(pan_X_1);
    Mat pan_P = (this->I - pan_K * this->C) * pan_P_minus;
    double panPredict = pan_x_hat_est.row(0).at<double>(0);

    if (panPredict < 0) {
        panPredict= panPredict + 2 * M_PI;
    }

    return panPredict;
}



double Kalman2D::updateKalmanForTilt(double objTilt, int valueNumberPredict) {

    double measurementAngleTilt = objTilt;
    Mat tilt_x_hat_minus = this->A * this->tilt_xhat;
    Mat tilt_x_hat_est = tilt_x_hat_minus.clone();
    for (int j = 0; j < valueNumberPredict; j++) {
        tilt_x_hat_est = this->A * tilt_x_hat_est;
    }
    Mat tilt_P_minus = this->A * this->tilt_P * this->A_transpose + this->Q;
    Mat tilt_K = tilt_P_minus * this->C_transpose * (this->C * tilt_P_minus * this->C_transpose + (1 * 180 / 3.1416)).inv();
    Mat tilt_X_0 = this->C * tilt_x_hat_minus;
    double tilt_X_1 = measurementAngleTilt - tilt_X_0.at<double>(0);
    tilt_xhat = tilt_x_hat_minus + tilt_K.mul(tilt_X_1);
    this->tilt_P = (this->I - tilt_K * this->C) * tilt_P_minus;
    double tiltPredict = tilt_x_hat_est.row(0).at<double>(0);

    return tiltPredict;
}

