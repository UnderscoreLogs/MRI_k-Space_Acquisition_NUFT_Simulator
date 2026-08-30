#include "../../include/MRI/Trajectories.hpp"
#include "../../include/MRI/NUDFT.hpp"

#include <random>

using namespace std;

struct KyCandidate
{
    int index;
    double weight;
};

vector<vector<double>> mriVariableDensityCartesianInterpolation(
    vector<vector<double>> img, 
    int newWidth, 
    int newHeight, 
    int kxSampleSize, 
    int kySampleSize, 
    double alpha,
    double centerFraction,
    double acceleration, 
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

    vector<double> kx;
    vector<double> ky;
    double kxMax = imgCols/(2.0*xFOV);
    double kyMax = imgRows/(2.0*yFOV);
    double kMax = max(kxMax, kyMax);
    double dkx = 2.0 * kxMax / kxSampleSize;
    double dky = 2.0 * kyMax / kySampleSize;

    double centerFreqWidth = kyMax * centerFraction;

    int totalLines = static_cast<int>(kySampleSize / acceleration);

    int centerLines = static_cast<int>(kySampleSize * centerFraction);

    int remainingLines = totalLines - centerLines;
    int linesUsed = 0;

    random_device rd;
    mt19937 gen(12345); // rd() -> 123345
    uniform_real_distribution<double> distrib(0.0, 1.0);

    for (int i = 0; i < kxSampleSize; i++)
    {
        kx.push_back(-kxMax + i * dkx);
    }

    for (int i = 0; i < kySampleSize; i++)
    {
        ky.push_back(-kyMax + i * dky);
    }

    // build candidates for non-center lines, keyed by density weight
    vector<KyCandidate> candidates;
    for (int j = 0; j < kySampleSize; j++)
    {
        if (fabs(ky[j]) < centerFreqWidth) continue; // center handled separately, always kept

        double r = fabs(ky[j]) / kMax;
        double p = exp(-alpha * r * r);
        candidates.push_back({j, p});
    }

    // turn each weight into a priority key u^(1/weight); higher weight -> key tends higher
    for (auto &c : candidates)
    {
        double u = distrib(gen);
        c.weight = pow(u, 1.0 / c.weight);
    }

    int numToSelect = min(remainingLines, static_cast<int>(candidates.size()));

    partial_sort(candidates.begin(), candidates.begin() + numToSelect, candidates.end(),
        [](const KyCandidate &a, const KyCandidate &b) { return a.weight > b.weight; });

    vector<int> selectedRows;
    for (int i = 0; i < numToSelect; i++){
        selectedRows.push_back(candidates[i].index);
    }
    for (int j = 0; j < kySampleSize; j++){
        if (fabs(ky[j]) < centerFreqWidth) {
            selectedRows.push_back(j);
        }
    }
    sort(selectedRows.begin(), selectedRows.end()); // keep k-space rows in order

    for (int j : selectedRows){
        for (int i = 0; i < kxSampleSize; i++) {
            kSamples.push_back({kx[i], ky[j], 0.0, 1.0});
        }
    }
    displayTrajectory(kSamples, 256, "VDCartesian Trajectory");

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

