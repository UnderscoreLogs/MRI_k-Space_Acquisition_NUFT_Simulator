#define _USE_MATH_DEFINES
#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>
#include <complex>
#include <iostream>
#include <chrono>
#include <random>
#include <filesystem>

#include "FFT.hpp"

using namespace std;

struct kSample {
    double kx;
    double ky;
    complex<double> value;
};

vector<vector<double>> getImage(string imgName)
{
    cv::Mat image = cv::imread(imgName, cv::IMREAD_GRAYSCALE);

    if (image.empty())
    {
        cout << "Could not load image\n";
        return {};
    }

    vector<vector<double>> imageVector(image.rows, vector<double>(image.cols));

    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            // Read as uchar (the real storage type), then cast to double
            imageVector[i][j] = static_cast<double>(image.at<uchar>(i, j));
        }
    }

    cout << "Vector size: " << imageVector.size() << " x " << imageVector[0].size() << endl;

    return imageVector;
}

cv::Mat doubleToMat(vector<vector<double>> imgD) {
    int rows = imgD.size();
    int cols = imgD[0].size();

    cv::Mat magMat(rows, cols, CV_64F);

    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < cols; x++)
        {
            magMat.at<double>(y, x) = imgD[y][x];
        }
    }

    return magMat;
}

cv::Mat complexVecToMat(const vector<vector<complex<double>>> &data)
{
    int rows = data.size();
    int cols = rows ? data[0].size() : 0;

    cv::Mat mag(rows, cols, CV_64F);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            mag.at<double>(i, j) = std::abs(data[i][j]);
        }
    }

    return mag;
}

double getMax(vector<vector<double>> vec)
{ // Finds the max value within a given 2D vector
    double highVal = vec[0][0];

    int rows = vec.size();                      // number of rows
    int cols = vec.empty() ? 0 : vec[0].size(); // number of columns (from first row)

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (vec[i][j] > highVal)
            {
                highVal = vec[i][j];
            }
        }
    }

    return highVal;
}

void printVec(vector<vector<double>> vec)
{ // for debugging
    for (int i = 0; i < vec.size(); i++)
    {
        for (int j = 0; j < vec[0].size(); j++)
        {
            cout << vec[i][j] << ", ";
        }
        cout << endl;
    }
}

vector<vector<double>> DIFT(vector<kSample> kSamples, int xRes, int yRes, string trajectoryName) {
    double alpha = 0;

    char trajectory = {};
    if(trajectoryName == "CARTESIAN") {
        trajectory = 'c';
    } else if(trajectoryName == "RADIAL") {
        trajectory = 'r';
    } else if(trajectoryName == "SPIRAL") {
        trajectory = 's';
    } else if(trajectoryName == "RANDOM") {
        trajectory = 'd';
    } else if (trajectoryName == "VD_RANDOM") {
        trajectory = 'v';
        alpha = 6;
    }

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
    vector<double> dcf(numSamples);

    switch(trajectory)
    {
    case 'c':
        for (int i = 0; i < numSamples; i++){
            dcf[i] = 1.0;
        }
        break;

    case 'r':
        for (int i = 0; i < numSamples; i++){
            dcf[i] = sqrt(kSamples[i].kx*kSamples[i].kx + kSamples[i].ky*kSamples[i].ky);
        };
        break;

    case 's':
        for (int i = 0; i < numSamples; i++){
            dcf[i] = sqrt(kSamples[i].kx*kSamples[i].kx + kSamples[i].ky*kSamples[i].ky);
        } // first approximation
        break;
    case 'v':
        for(int i = 0; i < numSamples; i++) {
            double r = sqrt(kSamples[i].kx*kSamples[i].kx + kSamples[i].ky*kSamples[i].ky);
            double p = exp(-(r*r)); //not working properly
            dcf[i] = 1.0;
        }
        break;
    default:
        cout << "Invalid trajectory, dcf = 1." << endl;
        for (int i = 0; i < numSamples; i++){
            dcf[i] = 1.0;
        }
        break;
    }

    for(int y = 0; y < yRes; y++){
        for(int x = 0; x < xRes; x++) {
            complex<double> pixel = 0.0;

            for(int sample_i = 0; sample_i < numSamples; sample_i++){
                
                double angle = 2*M_PI*(kSamples[sample_i].kx * xSpatial[x] + kSamples[sample_i].ky*ySpatial[y]);
                double s, c;
                sincos(angle, &s, &c);

                pixel += kSamples[sample_i].value * complex<double>(c, s);
            }

            pixel /= numSamples;
            image[y][x] = pixel.real();
        }
    }

    return image;

}

vector<vector<double>> mriRadialInterpolation(vector<vector<double>> img, int newWidth, int newHeight, int numOfSpokes,  int numSamplesPerSpoke) {
    
    double xFOV = 1.0; //changes nothing
    double yFOV = 1.0;

    int imgRows = img.size();                      // number of rows
    int imgCols = img.empty() ? 0 : img[0].size(); // number of columns (from first row)
    
    double kMax = imgCols/(2.0*xFOV);  
    double R = kMax;

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
    double dTheta = M_PI/(double)numOfSpokes;
    double dr = 2.0 * R / (numSamplesPerSpoke - 1);


    for (int spoke = 0; spoke < numOfSpokes; spoke++)
    {
        for(int i = 0; i < numSamplesPerSpoke; i++) {

                double r = -R + i * dr;
                double theta = dTheta*spoke;

            kSample newSample = {r*cos(theta), r*sin(theta), 0.0};
            kSamples.push_back(newSample);
        }
    }

    //------------------------------- Foward Transform  ---------------------------------------------
    cout << "Foward Transform Started." << endl;
    int numSamples = kSamples.size();

    for(int sample_i = 0; sample_i < numSamples; sample_i++){
        complex<double> value = 0.0;

        for(int y = 0; y < imgRows; y++){

            for(int x = 0; x < imgCols; x++) {

                double angle = -2*M_PI*(kSamples[sample_i].kx * xSpatial[x] + kSamples[sample_i].ky * ySpatial[y]);
                double s, c;
                sincos(angle, &s, &c); //saves time instead of calling cos(angle) and sin(angle)
                value += img[y][x] * complex<double>(c, s);
            }
        }
        kSamples[sample_i].value = value;
    }

    cout << "k-Space Constructed." << endl;

    //------------------------------ image reconstruction --------------------------------------------------------------

    return DIFT(kSamples, newWidth, newHeight, "RADIAL");
}

vector<vector<double>> mriRandomInterpolation(vector<vector<double>> img, int newWidth, int newHeight, int numOfSamples) {
    
    double xFOV = 1.0; //changes nothing
    double yFOV = 1.0;

    int imgRows = img.size();                      // number of rows
    int imgCols = img.empty() ? 0 : img[0].size(); // number of columns (from first row)

    vector<double> xSpatial(imgCols);
    vector<double> ySpatial(imgRows);

    double maxFreq = imgCols/(2.0*xFOV);

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
    uniform_real_distribution<double> distrib(-maxFreq, maxFreq);

    for (int i = 0; i < numOfSamples; i++)
    {
        double kx = distrib(gen);
        double ky = distrib(gen);

        kSample newSample = {kx, ky, 0.0};
        kSamples.push_back(newSample);

        //cout << "kx: " << kx << "hz" << endl;
        //cout << "ky: " << ky << "hz" << endl;
    }

    //------------------------------- Foward Transform  ---------------------------------------------
    cout << "Foward Transform Started." << endl;
    int numSamples = kSamples.size();

    for(int sample_i = 0; sample_i < numSamples; sample_i++){
        complex<double> value = 0.0;

        for(int y = 0; y < imgRows; y++){

            for(int x = 0; x < imgCols; x++) {

                double angle = -2*M_PI*(kSamples[sample_i].kx * xSpatial[x] + kSamples[sample_i].ky * ySpatial[y]);
                double s, c;
                sincos(angle, &s, &c); //saves time instead of calling cos(angle) and sin(angle)
                value += img[y][x] * complex<double>(c, s);
            }
        }
        kSamples[sample_i].value = value;
    }

    cout << "k-Space Constructed." << endl;

    //------------------------------ image reconstruction --------------------------------------------------------------

    return DIFT(kSamples, newWidth, newHeight, "RANDOM");
}

vector<vector<double>> mriCartesianInterpolation(vector<vector<double>> img, int newWidth, int newHeight, int kxSampleSize, int kySampleSize) {
    
    double xFOV = 1.0; //changes nothing
    double yFOV = 1.0;

    int imgRows = img.size();                      // number of rows
    int imgCols = img.empty() ? 0 : img[0].size(); // number of columns (from first row)
    
    double kMax = imgCols/(2.0*xFOV);  
    double R = kMax;

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

    for (int j = 0; j < kySampleSize; j++)
    {
        for(int i = 0; i < kxSampleSize; i++){
            kSample newSample = {((i - (kxSampleSize / 2.0))), ((j - (kySampleSize / 2.0))), 0.0};
            kSamples.push_back(newSample);
        }
    }
    

    //------------------------------- Foward Transform  ---------------------------------------------
    cout << "Foward Transform Started." << endl;
    int numSamples = kSamples.size();

    for(int sample_i = 0; sample_i < numSamples; sample_i++){
        complex<double> value = 0.0;

        for(int y = 0; y < imgRows; y++){

            for(int x = 0; x < imgCols; x++) {

                double angle = -2*M_PI*(kSamples[sample_i].kx * xSpatial[x] + kSamples[sample_i].ky * ySpatial[y]);
                double s, c;
                sincos(angle, &s, &c); //saves time instead of calling cos(angle) and sin(angle)
                value += img[y][x] * complex<double>(c, s);
            }
        }
        kSamples[sample_i].value = value;
    }

    cout << "k-Space Constructed." << endl;

    //------------------------------ image reconstruction --------------------------------------------------------------

    return DIFT(kSamples, newWidth, newHeight, "CARTESIAN");
}

vector<vector<double>> mriSpiralInterpolation(vector<vector<double>> img, int newWidth, int newHeight, int numOfSamples, int numOfTurns) {
    
    double xFOV = 1.0; //changes nothing
    double yFOV = 1.0;

    int imgRows = img.size();                      // number of rows
    int imgCols = img.empty() ? 0 : img[0].size(); // number of columns (from first row)
    
    double kMax = imgCols/(2.0*xFOV);  
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
    double a = kMax/tMax;

    for (int i = 0; i < numOfSamples; i++)
    {
        double t = tMax * i / (numOfSamples - 1);

        kSample newSample = {a*t*cos(t), a*t*sin(t), 0.0};
        kSamples.push_back(newSample);
        
    }

    //------------------------------- Foward Transform  ---------------------------------------------
    cout << "Foward Transform Started." << endl;
    int numSamples = kSamples.size();

    for(int sample_i = 0; sample_i < numSamples; sample_i++){
        complex<double> value = 0.0;

        for(int y = 0; y < imgRows; y++){

            for(int x = 0; x < imgCols; x++) {

                double angle = -2*M_PI*(kSamples[sample_i].kx * xSpatial[x] + kSamples[sample_i].ky * ySpatial[y]);
                double s, c;
                sincos(angle, &s, &c); //saves time instead of calling cos(angle) and sin(angle)
                value += img[y][x] * complex<double>(c, s);
            }
        }
        kSamples[sample_i].value = value;
    }

    cout << "k-Space Constructed." << endl;

    //------------------------------ image reconstruction --------------------------------------------------------------

    return DIFT(kSamples, newWidth, newHeight, "SPIRAL");
}

vector<vector<double>> mriVariableDensityCartesianInterpolation(vector<vector<double>> img, int newWidth, int newHeight, int kxSampleSize, int kySampleSize, double alpha) {
    
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
    double kyMax = imgCols/(2.0*xFOV);

    double centerFreqWidth = kyMax * 0.08;

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> distrib(0.0, 1.0);

    for (int i = 0; i < kxSampleSize; i++)
    {
        kx.push_back(i - kxSampleSize / 2.0 / xFOV);
    }

    for (int i = 0; i < kySampleSize; i++)
    {
        ky.push_back(i - kySampleSize / 2.0 / yFOV);
    }

    for (int j = 0; j < kySampleSize; j++)
    {
        double r = abs(ky[j]) / kyMax;

        double p = exp(-alpha * r * r);

        if(fabs(ky[j]) < centerFreqWidth) {
            for (int i = 0; i < kxSampleSize; i++)
            {
                kSample newSample = {kx[i], ky[j], 0.0};
                kSamples.push_back(newSample);
            }
        } else if (distrib(gen) < p) {
            for (int i = 0; i < kxSampleSize; i++)
            {
                kSample newSample = {kx[i], ky[j], 0.0};
                kSamples.push_back(newSample);
            }
        }
    }

    //------------------------------- Foward Transform  ---------------------------------------------
    cout << "Foward Transform Started." << endl;
    int numSamples = kSamples.size();

    for(int sample_i = 0; sample_i < numSamples; sample_i++){
        complex<double> value = 0.0;

        for(int y = 0; y < imgRows; y++){

            for(int x = 0; x < imgCols; x++) {

                double angle = -2*M_PI*(kSamples[sample_i].kx * xSpatial[x] + kSamples[sample_i].ky * ySpatial[y]);
                double s, c;
                sincos(angle, &s, &c); //saves time instead of calling cos(angle) and sin(angle)
                value += img[y][x] * complex<double>(c, s);
            }
        }
        kSamples[sample_i].value = value;
    }

    cout << "k-Space Constructed." << endl;

    //------------------------------ image reconstruction --------------------------------------------------------------

    return DIFT(kSamples, newWidth, newHeight, "CARTESIAN");
}

vector<vector<double>> mriVariableDensityRandomInterpolation(vector<vector<double>> img, int newWidth, int newHeight, int numOfSamples, double alpha) {
    
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

    double kyRadius = kyMax*0.1;
    double kxRadius = kxMax*0.1;

    while(kSamples.size() < numOfSamples) {

        double kx = distribXFreq(gen);
        double ky = distribYFreq(gen);

        double r = abs(ky) / kyMax;

        double p = exp(-alpha * r * r);

        if(ky < kyRadius && kx < kxRadius) {
            kSample newSample = {kx, ky, 0.0};
            kSamples.push_back(newSample);
        } else if (distribProb(gen) < p) {
            kSample newSample = {kx, ky, 0.0};
            kSamples.push_back(newSample);
        }
    }

    //cout << "Number of Samples: " << kSamples.size() << endl;

    //------------------------------- Foward Transform  ---------------------------------------------
    cout << "Foward Transform Started." << endl;
    int numSamples = kSamples.size();

    for(int sample_i = 0; sample_i < numSamples; sample_i++){
        complex<double> value = 0.0;

        for(int y = 0; y < imgRows; y++){

            for(int x = 0; x < imgCols; x++) {

                double angle = -2*M_PI*(kSamples[sample_i].kx * xSpatial[x] + kSamples[sample_i].ky * ySpatial[y]);
                double s, c;
                sincos(angle, &s, &c); //saves time instead of calling cos(angle) and sin(angle)
                value += img[y][x] * complex<double>(c, s);
            }
        }
        kSamples[sample_i].value = value;
    }

    cout << "k-Space Constructed." << endl;

    //------------------------------ image reconstruction --------------------------------------------------------------

    return DIFT(kSamples, newWidth, newHeight, "VD_RANDOM");
}

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

void saveImg(vector<vector<double>> image_double, string fileName, string folder) {
    string fullFileName =  fileName + ".png";
    string windowName = fileName + " Image";

    cv::Mat image_mat = doubleToMat(image_double);

    cv::Mat displayMat;
    cv::normalize(image_mat, displayMat, 0, 255, cv::NORM_MINMAX);
    displayMat.convertTo(displayMat, CV_8U);

    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::imshow(windowName, displayMat);

    cv::imwrite(folder + "/" + fullFileName, displayMat);

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

int main()
{
    //------------------------Settings-------------------------------------------------------------------------------------------------
    string imgName = "../standard_test_images/lena_gray_256.tif";

    int xRes = 256;
    int yRes = 256;

    int numOfSamples = 40000;

    vector<vector<double>> img = getImage(imgName);
    
    namespace fs = filesystem;

    int run = 1;
    while (fs::exists("Results/Run" + std::to_string(run))) {
        run++;
    }

    string folder = "Results/Run" + to_string(run);
    fs::create_directories(folder);

    /*----------------------------------- VDR ---------------------------------------------------
    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    auto VDR_double = mriVariableDensityRandomInterpolation(img, xRes, yRes, numOfSamples, 12);
    
    // Stop timer
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "VDR finished in " << duration.count() << " ms" << std::endl;

    //----------------------------------- Cartesian ---------------------------------------------------
    // Start timer
    start = std::chrono::high_resolution_clock::now();

    auto Cart_double = mriCartesianInterpolation(img, xRes, yRes, round(sqrt(numOfSamples)), round(sqrt(numOfSamples)));
    
    // Stop timer
    end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Cartesian finished in " << duration.count() << " ms" << std::endl;

    //----------------------------------- Radial ---------------------------------------------------*/
    // Start timer
    start = std::chrono::high_resolution_clock::now();

    auto Rad_double = mriRadialInterpolation(img, xRes, yRes, 128, round(numOfSamples/128));
    
    // Stop timer
    end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Radial finished in " << duration.count() << " ms" << std::endl;

    /*----------------------------------- Random ---------------------------------------------------
    // Start timer
    start = std::chrono::high_resolution_clock::now();

    auto Rand_double = mriRandomInterpolation(img, xRes, yRes, numOfSamples);
    
    // Stop timer
    end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Random finished in " << duration.count() << " ms" << std::endl;

    //----------------------------------- Spiral ---------------------------------------------------
    // Start timer
    start = std::chrono::high_resolution_clock::now();

    auto Spir_double = mriSpiralInterpolation(img, xRes, yRes, numOfSamples, 32);
    
    // Stop timer
    end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Spiral finished in " << duration.count() << " ms" << std::endl;

    //----------------------------------- VDC ---------------------------------------------------
    // Start timer
    start = std::chrono::high_resolution_clock::now();

    auto VDC_double = mriVariableDensityCartesianInterpolation(img, xRes, yRes, round(sqrt(numOfSamples)), round(sqrt(numOfSamples)), 12);
    
    // Stop timer
    end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "VDC finished in " << duration.count() << " ms" << std::endl;
    //------------------------------------ Save & Display Images -----------------------------------------------*/
   // cout << "      VDR PSNR: " <<         calcPSNR(VDR_double, img) << endl;
    //cout << "Cartesian PSNR: " <<   calcPSNR(Cart_double, img) << endl;
    cout << "   Radial PSNR: " <<      calcPSNR(Rad_double, img) << endl;
    //cout << "   Random PSNR: " <<      calcPSNR(Rand_double, img) << endl;
    //cout << "   Spiral PSNR: " <<      calcPSNR(Spir_double, img) << endl;
    //cout << "      VDC PSNR: " <<         calcPSNR(VDC_double, img) << endl;    
    
    //cout << "\n    VDR SSIM: " <<         calcSSIM(VDR_double, img) << endl;
    //cout << "Cartesian SSIM: " <<   calcSSIM(Cart_double, img) << endl;
    cout << "   Radial SSIM: " <<      calcSSIM(Rad_double, img) << endl;
    //cout << "   Random SSIM: " <<      calcSSIM(Rand_double, img) << endl;
    //cout << "   Spiral SSIM: " <<      calcSSIM(Spir_double, img) << endl;
    //cout << "      VDC SSIM: " <<         calcSSIM(VDC_double, img) << endl;   

    //saveImg(VDR_double, "Variable-Density-Random", folder);

    //saveImg(Cart_double, "Cartesian", folder);

    saveImg(Rad_double, "Radial", folder);

    //saveImg(Rand_double, "Random", folder);

    //saveImg(Spir_double, "Spiral", folder);

    //saveImg(VDC_double, "Variable-Density-Cartesian", folder);

    cout << "Program Finished." << endl;

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
