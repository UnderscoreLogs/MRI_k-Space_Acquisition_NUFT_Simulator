#ifndef kSpaceTraj_HPP
#define kSpaceTraj_HPP

#include <vector>
#include <complex>

using namespace std;

struct kSample {
    double kx;
    double ky;
    complex<double> value;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// NUDFTs
///////////////////////////////////////////////////////////////////////////////////////////////////////////

vector<kSample> NUDFT(vector<double> xSpatial, vector<double> ySpatial, vector<kSample> kSamples, vector<vector<double>> img);

vector<vector<double>> INUFFT(vector<kSample> kSamples, int xRes, int yRes, string trajectoryName, double extraInfo);

vector<vector<double>> INUDFT(vector<kSample> kSamples, int xRes, int yRes, string trajectoryName, double extraInfo);

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// NUDFT Trajectories
///////////////////////////////////////////////////////////////////////////////////////////////////////////

vector<vector<double>> mriRadialInterpolation(vector<vector<double>> img, int newWidth, int newHeight, int numOfSpokes,  int numSamplesPerSpoke);

vector<vector<double>> mriRandomInterpolation(vector<vector<double>> img, int newWidth, int newHeight, int numOfSamples);

vector<vector<double>> mriCartesianInterpolation(vector<vector<double>> img, int newWidth, int newHeight, int kxSampleSize, int kySampleSize);

vector<vector<double>> mriSpiralInterpolation(vector<vector<double>> img, int newWidth, int newHeight, int numOfSamples, int numOfTurns);

vector<vector<double>> mriVariableDensityCartesianInterpolation(vector<vector<double>> img, int newWidth, int newHeight, int kxSampleSize, int kySampleSize, double alpha);

vector<vector<double>> mriVariableDensityRandomInterpolation(vector<vector<double>> img, int newWidth, int newHeight, int numOfSamples, double alpha);

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Utilities
///////////////////////////////////////////////////////////////////////////////////////////////////////////

vector<vector<double>> getImage(string imgName);

void displayKSpace(vector<vector<complex<double>>>& kSpace, string name);

void displayTrajectory(vector<kSample>& samples, int size, string windowName);

double calcPSNR(vector<vector<double>> K, vector<vector<double>> controlImage);

double calcSSIM(const vector<vector<double>>& img2, const vector<vector<double>>& img1);

vector<vector<double>> normalizeImage(const vector<vector<double>>& img, double newMin, double newMax);

void saveImg(vector<vector<double>> image_double, string fileName, string folder);

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Helpers
///////////////////////////////////////////////////////////////////////////////////////////////////////////

cv::Mat doubleToMat(vector<vector<double>> imgD);
cv::Mat complexVecToMat(const vector<vector<complex<double>>> &data);

#endif
