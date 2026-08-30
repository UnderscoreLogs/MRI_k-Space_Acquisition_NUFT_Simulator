
#include "../../include/Imaging/Metrics.hpp"

using namespace std;

double calcPSNR(vector<vector<double>> K, vector<vector<double>> controlImage) {
    int N = K.size();
    int M = K[0].size();

    double sum = {};

    for(int y = 0; y < N; y++){
        for(int x = 0; x < M; x++){
            sum += (controlImage[y][x] - K[y][x]) * (controlImage[y][x] - K[y][x]);
        }
    }

   double MSE = sum/(M*N);
    
   double PSNR = 10.0 * log10((255.0 * 255.0) / MSE);
   return PSNR;
}

double calcSSIM(const vector<vector<double>>& img2, const vector<vector<double>>& img1)
{// AI made this function, Im too lazy to code this 
    
    int rows = img1.size();
    int cols = img1[0].size();

    if (rows != img2.size() || cols != img2[0].size())
        throw runtime_error("Images must be the same size.");

    const double L = 255.0;          // Pixel range
    const double C1 = pow(0.01 * L, 2);
    const double C2 = pow(0.03 * L, 2);

    double mu1 = 0.0;
    double mu2 = 0.0;
    int N = rows * cols;

    // Mean
    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < cols; x++)
        {
            mu1 += img1[y][x];
            mu2 += img2[y][x];
        }
    }

    mu1 /= N;
    mu2 /= N;

    double sigma1 = 0.0;
    double sigma2 = 0.0;
    double sigma12 = 0.0;

    // Variance and covariance
    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < cols; x++)
        {
            double d1 = img1[y][x] - mu1;
            double d2 = img2[y][x] - mu2;

            sigma1 += d1 * d1;
            sigma2 += d2 * d2;
            sigma12 += d1 * d2;
        }
    }

    sigma1 /= (N - 1);
    sigma2 /= (N - 1);
    sigma12 /= (N - 1);

    double numerator = (2.0 * mu1 * mu2 + C1) * (2.0 * sigma12 + C2);

    double denominator = (mu1 * mu1 + mu2 * mu2 + C1) * (sigma1 + sigma2 + C2);

    return numerator / denominator;
}




