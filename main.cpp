//#include <QCoreApplication>
#include <QApplication>
#include <iostream>
#include "KalmanLRF.h"
#include "Kalman2D.h"
#include "Kalman3D.h"
#include <QDebug>
#include <opencv2/core/core.hpp>


using namespace std;
using namespace cv;
int m_indexData;
QVector<QStringList> m_inputStringListVec;
QTimer m_radarCtrlTimer;
Mat laserEstxyzLast;

int main(int argc, char *argv[])
{
    KalmanLRF kalman_lrf;
    // input & output path

    const QString& inputFilePath("/home/ngoclth/Documents/PPK57/kalman-916/input/04-11-2024-17-49-46.csv");
    string output_path = "/home/ngoclth/Documents/PPK57/kalman-916/sd/kalman/outputPC/04-11-2024-17-49-46_1.csv";

//    const QString& inputFilePath("/home/ngoclth/Documents/PPK57/KalmanLRF/data/Datatxt/hs116-15-21.csv");
//    string output_path = "/home/ngoclth/Documents/PPK57/KalmanLRF/data/kalmanLRA/2024-06-24-17_out.csv";

    // Tham so input
    m_inputStringListVec = kalman_lrf.readDataFromFile(inputFilePath); // read input csv file
    double hz = 30.0; // tần số muốn nội suy lên
    float dt = 1.0/hz; // khoảng thời gian giữa hai dữ liệu liên tiếp (chu kì)
    int number_of_measurements = m_inputStringListVec.size();    // data size
    cout <<"number_of_measurements " << number_of_measurements << endl;
    float delta_t = 0;
    float object_vmax = 100; // the maximum speed of the considered object (m/s),
    double rangeErrorEs = 5; // Sai so range tu Laser or ADSP return(m)
    double aziErrorEs = pow(0.0003,2); // Sai số góc pan của thiết bị quay(rad)
    double eleErrorEs = pow(0.0003,2); // Sai số góc tilt của thiết bị quay (rad)
    double mesurementRealiability = 10000;

    Mat laserPanTiltEstKalman = Mat::zeros(number_of_measurements, 3, CV_32F);
    Mat laserEstxyz = Mat::zeros(number_of_measurements, 6, CV_32F);
    double x,vx;
    double y,vy;
    double z,vz;
    laserEstxyzLast = (Mat_<float>(6, 1) << 0, 0, 0, 0, 0, 0);
    // save output file
    std::ofstream output_file;
    output_file.open (output_path);
    output_file << "laserRaw,laserKalman,objTilt,tiltKalman,objPan,panKalman,vx,vy,vz" << endl;

    for (uint i = 0; i < m_inputStringListVec.size(); i++)
    {
        const QStringList& curRow = m_inputStringListVec.at(i);
            //Measurement: range (return from laser measurement), elevation, azimuth (Laser, tilt, pan)
        Mat measurement = Mat::zeros(1, 3, CV_32F);
        float laser = curRow.at(0).toFloat();
        measurement.at<float>(0) = curRow.at(0).toFloat(); // Laser (rad)
        measurement.at<float>(1) = curRow.at(2).toFloat(); // Tilt  (rad)
        measurement.at<float>(2) = curRow.at(1).toFloat(); // Pan    (rad)
        if (measurement.at<float>(2) < 0) {
            measurement.at<float>(2) = measurement.at<float>(2) + 2 * M_PI;
        }
        cout <<"measurement in " << i << " :"<< measurement << endl;
        delta_t = i* dt;
        // When tracking & islaser, use Kalman3D&KalmanLr
        if(laser!=0 && i == 0)
        {
            printf("KalmanLRF\n");
            kalman_lrf.init(measurement, delta_t,  rangeErrorEs, aziErrorEs, eleErrorEs, mesurementRealiability);
            laserPanTiltEstKalman.row(i).at<float>(0) = measurement.at<float>(0);
            laserPanTiltEstKalman.row(i).at<float>(1) = measurement.at<float>(1);
            laserPanTiltEstKalman.row(i).at<float>(2) = measurement.at<float>(2);
            vx = 0;
            vy = 0;
            vz = 0;
            laserEstxyz = (Mat_<float>(6, 1) << 0, 0, 0, 0, 0, 0);
        }
        else if (i > 0){
            if (measurement.at<float>(0) != 0)
            {
                printf("UpdateLaserKalman\n");
                delta_t = i*dt;
                laserEstxyz = kalman_lrf.update(measurement, delta_t, object_vmax);
            }

            else if (measurement.at<float>(0) == 0)
            {
                printf("PredictLaserKalman\n");
                delta_t = i*dt;
                Mat tempMat = kalman_lrf.predict();
                float distance = sqrt(pow(tempMat.at<float>(0),2) +pow(tempMat.at<float>(1),2) + pow(tempMat.at<float>(2),2));
                float x = distance * cos(measurement.at<float>(1)) * sin(measurement.at<float>(2));
                float y = distance * cos(measurement.at<float>(1)) * cos(measurement.at<float>(2));
                float z = distance * sin(measurement.at<float>(1));
                Mat tempMat2 = (Mat_<float>(6, 1) << x, y, z, tempMat.at<float>(3), tempMat.at<float>(4), tempMat.at<float>(5));
                laserEstxyz = tempMat2;
            }

            Mat laserPanTilt = kalman_lrf.hx(laserEstxyz);
            laserPanTiltEstKalman.row(i).at<float>(0) = laserPanTilt.row(0).at<float>(0);
            laserPanTiltEstKalman.row(i).at<float>(1) = laserPanTilt.row(1).at<float>(0);
            laserPanTiltEstKalman.row(i).at<float>(2) = laserPanTilt.row(2).at<float>(0);
            cout <<"laserPanTiltEstKalman " << laserPanTilt << endl;
            cout <<"laserEstxyz " << laserEstxyz << endl;
            vx = (laserEstxyz.row(0).at<float>(0) - laserEstxyzLast.row(0).at<float>(0)) / (delta_t);
            vy = (laserEstxyz.row(1).at<float>(0) - laserEstxyzLast.row(1).at<float>(0)) / (delta_t);
            vz = (laserEstxyz.row(2).at<float>(0) - laserEstxyzLast.row(2).at<float>(0)) / (delta_t);

//            vx = laserEstxyz.row(3).at<float>(0);
//            vy = laserEstxyz.row(4).at<float>(0);
//            vz = laserEstxyz.row(5).at<float>(0);
            laserEstxyzLast = laserEstxyz;
        }
        output_file <<   measurement.at<float>(0) << "," << laserPanTiltEstKalman.row(i).at<float>(0) <<"," << measurement.at<float>(1)
                      << "," << laserPanTiltEstKalman.row(i).at<float>(1) << "," << measurement.at<float>(2)
                      << "," << laserPanTiltEstKalman.row(i).at<float>(2)
                      << ","<< vx <<","<< vy << ","<< vz << endl;

}
    output_file.close();
}

