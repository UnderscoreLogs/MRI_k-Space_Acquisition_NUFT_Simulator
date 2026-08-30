#include "../../include/MRI/NUDFT.hpp"

#include <random>
#include <cmath>

using namespace std;

vector<kSample> NUDFT(vector<double> xSpatial, vector<double> ySpatial, vector<kSample> kSamples, vector<vector<double>> img) {

    int imgRows = img.size();                      // number of rows
    int imgCols = img.empty() ? 0 : img[0].size(); // number of columns (from first row)

    int numSamples = kSamples.size();

    //#pragma omp parallel for schedule(dynamic)
    for (int sample_i = 0; sample_i < numSamples; sample_i++) {
        double kx = kSamples[sample_i].kx; 
        double ky = kSamples[sample_i].ky;

        vector<complex<double>> phaseX(imgCols);
        vector<complex<double>> phaseY(imgRows);

        for (int x = 0; x < imgCols; x++) {
            double s, c; 
            sincos(-2*M_PI*kx*xSpatial[x], &s, &c);
            phaseX[x] = {c, s};
        }
        for (int y = 0; y < imgRows; y++) {
            double s, c; 
            sincos(-2*M_PI*ky*ySpatial[y], &s, &c);
            phaseY[y] = {c, s};
        }

        complex<double> value = 0.0;
        for (int y = 0; y < imgRows; y++) {
            complex<double> rowSum = 0.0;
            for (int x = 0; x < imgCols; x++)
                rowSum += img[y][x] * phaseX[x];
            value += rowSum * phaseY[y];
        }
        kSamples[sample_i].value = value;
    }

    return kSamples;
}

void addKspaceNoise(vector<kSample>& kSamples, double SNR) {

    random_device rd;
    mt19937 gen(rd());

    int numSamples = kSamples.size();
    double krms_sum = {};

    for(int sample_i = 0; sample_i < numSamples; sample_i++){        
        krms_sum += norm(kSamples[sample_i].value);
    }

    double k_rms = sqrt(krms_sum / static_cast<double>(numSamples));
    double sigma = k_rms / pow(10.0, (SNR / 20.0)); //sigma is the standard deviation of each of the real and imaginary noise components.
    normal_distribution<double> noise(0.0, sigma);

    for(int sample_i = 0; sample_i < numSamples; sample_i++){
        complex<double> noisySample = kSamples[sample_i].value + complex<double>(noise(gen), noise(gen));
        kSamples[sample_i].value = noisySample;
    }
}

void addMovement(vector<kSample>& kSamples, double timePercent) {
    double xDistance = 0.05;
    double yDistance = 0.0; 

    int numSamples = kSamples.size();

    for(int t = 0; t < numSamples; t++){

        double dx = 0.0;
        double dy = 0.0;

        if(t >= (double)numSamples * timePercent / 100.0) {
            dx = xDistance;
            dy = yDistance;
        }

        double kx = kSamples[t].kx;
        double ky = kSamples[t].ky;
        
        complex<double> motionPhase = exp(complex<double>(0, -2.0 * M_PI *(kx * dx + ky * dy)));

        kSamples[t].value *= motionPhase;
    }
}

