
#include "../../include/MRI/Trajectories.hpp"
#include "../../include/MRI/NUDFT.hpp"

using namespace std;

vector<vector<double>> mriCartesianInterpolation(
    vector<vector<double>> img, 
    int newWidth, int newHeight, 
    int kxSampleSize, int kySampleSize, 
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
    double dkx = 2.0 * kxMax / kxSampleSize;
    double dky = 2.0 * kyMax / kySampleSize;

    for (int j = 0; j < kySampleSize; j++)
    {
        double ky = -kyMax + j * dky;

        for (int i = 0; i < kxSampleSize; i++)
        {
            double kx = -kxMax + i * dkx;

            kSamples.push_back({kx, ky, 0.0, 1.0});
        }
    }

    displayTrajectory(kSamples, 256, "Cartesian Trajectory");

    //------------------------------- Foward Transform  ---------------------------------------------
    kSamples = NUDFT(xSpatial, ySpatial, kSamples, img);
    
    if(timePercent < 100.0) {
        addMovement(kSamples, timePercent);
    }
    
    if(SNR < 1000.0) {
        addKspaceNoise(kSamples, SNR);
    }


    //------------------------------ image reconstruction --------------------------------------------------------------

    if(useNUFFT == true){
        return INUFFT(kSamples, newWidth, newHeight, debug);
    } else {
        return INUDFT(kSamples, newWidth, newHeight);
    }
}