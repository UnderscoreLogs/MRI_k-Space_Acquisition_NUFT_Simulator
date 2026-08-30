#include "../../include/MRI/NUDFT.hpp"

using namespace std;

vector<vector<double>> INUDFT(vector<kSample> kSamples, int xRes, int yRes) {
    double xFOV = 1.0;
    double yFOV = 1.0;
    
    int numSamples = kSamples.size();

    vector<double> xSpatial(xRes);
    vector<double> ySpatial(yRes);

    double dx = xFOV / xRes;
    double dy = yFOV / yRes;

    for (int i = 0; i < xRes; i++)
    {
        //xSpatial[i] = dx * i;
        xSpatial[i] = (i - xRes/2.0) * dx;
    }
    for (int i = 0; i < yRes; i++)
    {
        //ySpatial[i] = dy * i;
        ySpatial[i] = (i - yRes/2.0) * dy;
    }

    vector<vector<double>> image(yRes, vector<double>(xRes));

    double dcfSum = 0.0;
    for (const auto& s : kSamples) {
        dcfSum += s.dcf;
    }
    double dcfMean = dcfSum / kSamples.size();

    if (dcfMean > 0.0) {
        for (auto& s : kSamples) {
            s.dcf /= dcfMean;
        }
    }

    for(int y = 0; y < yRes; y++){
        for(int x = 0; x < xRes; x++) {
            complex<double> pixel = 0.0;

            for(int sample_i = 0; sample_i < numSamples; sample_i++){
                
                double angle = 2*M_PI*(kSamples[sample_i].kx * xSpatial[x] + kSamples[sample_i].ky*ySpatial[y]);
                double s, c;
                sincos(angle, &s, &c);

                pixel += kSamples[sample_i].value * kSamples[sample_i].dcf * complex<double>(c, s);
            }

            pixel /= (double)(xRes*yRes);
            image[y][x] = pixel.real();
        }
    }

    return image;

}
