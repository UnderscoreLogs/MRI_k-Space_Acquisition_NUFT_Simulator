#include "../../include/MRI/Trajectories.hpp"
#include "../../include/MRI/NUDFT.hpp"

#include <random>

using namespace std;

vector<vector<double>> mriVariableDensityRandomInterpolation(
    vector<vector<double>> img, 
    int newWidth, 
    int newHeight, 
    double alpha,
    double centerFraction,
    double acceleration,
    string reconType, 
    double SNR,
    double timePercent, 
    bool debug
) 
    {

    int numOfSamples = round((newWidth*newHeight)/acceleration);

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
    uniform_real_distribution<double> distribXFreq(-kxMax, kxMax);
    uniform_real_distribution<double> distribYFreq(-kyMax, kyMax);
    uniform_real_distribution<double> distribProb(0.0, 1.0);

    double kyRadius = kyMax*centerFraction;
    double kxRadius = kxMax*centerFraction;

    while(kSamples.size() < numOfSamples) {

        double kx = distribXFreq(gen);
        double ky = distribYFreq(gen);

        double r = abs(ky) / kyMax;

        double p = exp(-alpha * r * r);

        if(fabs(ky) < kyRadius && fabs(kx) < kxRadius) {
            kSample newSample = {kx, ky, 0.0, 1.0};
            kSamples.push_back(newSample);
        } else if (distribProb(gen) < p) {
            double dcfWeight = 1.0 / max(p, 1e-2);
            
            kSample newSample = {kx, ky, 0.0, dcfWeight};
            kSamples.push_back(newSample);
        }
    }

    //cout << "Number of Samples: " << kSamples.size() << endl;

    //------------------------------- Foward Transform  ---------------------------------------------
    kSamples = NUDFT(xSpatial, ySpatial, kSamples, img);

    displayTrajectory(kSamples, 256, "VDR Trajectory");

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

