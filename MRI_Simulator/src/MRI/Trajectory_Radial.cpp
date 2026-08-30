#include "../../include/MRI/Trajectories.hpp"
#include "../../include/MRI/NUDFT.hpp"

#include <iostream>

using namespace std;

vector<vector<double>> mriRadialInterpolation(
    vector<vector<double>> img, 
    int newWidth, 
    int newHeight, 
    int numOfSpokes,  
    int numSamplesPerSpoke, 
    string reconType,
    double SNR,
    double timePercent, 
    bool debug
)
    {

    bool useNUFFT = true;
    if(reconType == "INUDFT") {
        useNUFFT = false;
    }

    double xFOV = 1.0; //changes nothing
    double yFOV = 1.0;

    int imgRows = img.size();                      // number of rows
    int imgCols = img.empty() ? 0 : img[0].size(); // number of columns (from first row)
    
    double kxMax = imgCols/(2.0*xFOV);  
    double kyMax = imgRows/(2.0*yFOV);  
    double R = max(kxMax, kyMax);

    vector<double> xSpatial(imgCols);
    vector<double> ySpatial(imgRows);

    double dx = xFOV / imgCols;
    double dy = yFOV / imgRows;

    for (int i = 0; i < imgCols; i++)
    {
        //xSpatial[i] = dx * i;
        xSpatial[i] = (i - imgCols/2.0) * dx;
    }
    for (int i = 0; i < imgRows; i++)
    {
        //ySpatial[i] = dy * i;
        ySpatial[i] = (i - imgRows/2.0) * dy;
    }

    vector<kSample> kSamples;
    double dTheta = 2*M_PI/(double)numOfSpokes;
    double thetaOffset = dTheta * 0.5;

    double dr = R / (numSamplesPerSpoke - 1);


    for (int  spoke = 0; spoke < numOfSpokes; spoke++)
    {
        int start = 0; //(spoke == 0) ? 0 : 1;
        for(int i = start; i < numSamplesPerSpoke; i++) { //might need (spoke == 0 ? 0 : 1) is to not duplicate the center value,  

            double r = i * dr;
            double theta = dTheta*spoke + thetaOffset;

            double kx = r*cos(theta) * (kxMax / R);
            double ky = r*sin(theta) * (kyMax / R);

            double weight = sqrt(kx*kx + ky*ky);
            double dcf =  max(weight / R, 1e-3);

            //cout << "DCF: " << dcf << endl;

            kSample newSample = {kx, ky, 0.0, dcf};
            kSamples.push_back(newSample);
        }
    }

    //------------------------------- Foward Transform  ---------------------------------------------
    kSamples = NUDFT(xSpatial, ySpatial, kSamples, img);

    if(timePercent < 100.0) {
        addMovement(kSamples, timePercent);
    }
    
    if(SNR < 1000.0) {
        addKspaceNoise(kSamples, SNR);
    }

    displayTrajectory(kSamples, 256, "Radial Trajectory");

    //------------------------------ image reconstruction --------------------------------------------------------------

    if(useNUFFT == true){
        return INUFFT(kSamples, newWidth, newHeight, debug);
    } else {
        return INUDFT(kSamples, newWidth, newHeight);
    }
    
}