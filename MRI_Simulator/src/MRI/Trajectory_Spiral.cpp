#include "../../include/MRI/Trajectories.hpp"
#include "../../include/MRI/NUDFT.hpp"

using namespace std;

vector<vector<double>> mriSpiralInterpolation(
    vector<vector<double>> img, 
    int newWidth, 
    int newHeight, 
    int numOfSamples, 
    int numOfTurns,
    int numInterleaves, 
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

    double kMax = max(kxMax, kyMax);  
    double tMax = numOfTurns*2*M_PI;

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
    double ax = kxMax/tMax;
    double ay = kyMax/tMax;

    int samplesPerInterleaf = numOfSamples / numInterleaves;

    for(int interleaf = 0; interleaf < numInterleaves; interleaf++)
    {
        double theta = 2*M_PI*interleaf/numInterleaves;

        for(int i = 0; i < samplesPerInterleaf; i++)
        {
            double t = tMax * i / (samplesPerInterleaf - 1);

            kSample newSample;

            newSample.kx = ax*t*cos(t + theta);
            newSample.ky = ay*t*sin(t + theta);
            newSample.value = 0.0;

            // exact Jacobian DCF for this spiral parametrization
            double dcfWeight = ax*ay*t;   // = (a*t)^2, i.e. r^2
            newSample.dcf = max(dcfWeight, 0.0);    
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

    displayTrajectory(kSamples, 256, "Spiral Trajectory");

    //------------------------------ image reconstruction --------------------------------------------------------------
    if(useNUFFT == true){
        return INUFFT(kSamples, newWidth, newHeight, debug);
    } else {
        return INUDFT(kSamples, newWidth, newHeight);
    }
}
