#include "../../include/MRI/Trajectories.hpp"
#include "../../include/MRI/NUDFT.hpp"

#include <random>

using namespace std;

vector<vector<double>> mriRandomInterpolation(
    vector<vector<double>> img, 
    int newWidth, 
    int newHeight, 
    int numOfSamples, 
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

    vector<double> xSpatial(imgCols);
    vector<double> ySpatial(imgRows);

    double kxMax = imgCols/(2.0*xFOV);
    double kyMax = imgRows/(2.0*yFOV);

    double maxFreq = max(kxMax, kyMax);

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
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> distrib_X(-kxMax, kxMax);
    uniform_real_distribution<double> distrib_Y(-kyMax, kyMax);

    for (int i = 0; i < numOfSamples; i++)
    {
        double kx = distrib_X(gen);
        double ky = distrib_Y(gen);

        kSample newSample = {kx, ky, 0.0, 1.0};
        kSamples.push_back(newSample);

        //cout << "kx: " << kx << "hz" << endl;
        //cout << "ky: " << ky << "hz" << endl;
    }

    //------------------------------- Foward Transform  ---------------------------------------------
    kSamples = NUDFT(xSpatial, ySpatial, kSamples, img);

    displayTrajectory(kSamples, 256, "Random Trajectory");

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