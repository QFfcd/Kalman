#include "resetkalman.h"
#include "KalmanLRF.h"
resetKalman::resetKalman()
{
    input1ForKalman = Mat::zeros(1, 3, CV_32F);
    if (kalmanLrf != nullptr) {
        delete kalmanLrf;
        kalmanLrf = nullptr;
    }
    if (kalman3d != nullptr) {
        delete kalman3d;
        kalman3d = nullptr;
    }

    kalmanLrf = new KalmanLRF(true);
    kalman3d = new Kalman3D(true);

//    kalmanCounter = 0;
//    isKalmanReady = false;
//    distanceSource = 0;D
//    lastDistance = -1;
//    targetHeight = -1;
//    rawLaserResult = -1;
//    currentLaserResult = -1;
//    indexHB = 0;
//    heartbeatLastTime.clear();
}
