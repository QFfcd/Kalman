#ifndef RESETKALMAN_H
#define RESETKALMAN_H
#include "KalmanLRF.h"
#include "Kalman3D.h"

class resetKalman
{
    private:
    KalmanLRF *kalmanLrf = nullptr;
    Kalman3D *kalman3d = nullptr;
    public:
        resetKalman();
};

#endif // RESETKALMAN_H
