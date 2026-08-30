#define _USE_MATH_DEFINES
#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>
#include <complex>
#include <iostream>
#include <chrono>

#include "FFT.hpp"

using namespace std;

vector<vector<double>> getImage(string imgName){
    cv::Mat image = cv::imread(imgName, cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
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

cv::Mat complexVecToMat(const vector<vector<complex<double>>>& data) {
    int rows = data.size();
    int cols = rows ? data[0].size() : 0;

    cv::Mat mag(rows, cols, CV_64F);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mag.at<double>(i, j) = std::abs(data[i][j]);
        }
    }

    return mag;
}

double getMax(vector<vector<double>> vec) { //Finds the max value within a given 2D vector
    double highVal = vec[0][0];

    int rows = vec.size();          // number of rows
    int cols = vec.empty() ? 0 : vec[0].size();  // number of columns (from first row)

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++){
            if(vec[i][j] > highVal) {
                highVal = vec[i][j];
            }
        }
    }

    return highVal;
}

void printVec(vector<vector<double>> vec) { //for debugging
    for(int i = 0; i < vec.size(); i++){
        for(int j = 0; j < vec[0].size(); j++ ){
            cout << vec[i][j] << ", ";
        }
        cout << endl;
    }
}

int main() {
    //-------------------------Timer-------------------------------------
    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    //------------------------Settings-------------------------------------------------------------------------------------------------
    string imgName = "C:/Users/smegl/OneDrive/Desktop/Code/Matlab/Imaging/standard_test_images/lena_gray_256.tif";

    double gamma = (double)(5.0E7) * (2*M_PI);

    int xRes = 64;
    int yRes = 64;

    double xFOV = 1.0;
    double yFOV = 1.0;

    double B0 = 1.0;
    double xGradStep = (double)(4.0E-7); //need to be double?
    double yGradStep = (double)(2.5E-7);

    double Fs = (xGradStep*xFOV) * (gamma/(2*M_PI));

    double totalTime = xRes/Fs;
    double pulseDuration = (2*M_PI)/(gamma*yGradStep*yFOV);

    vector<double> timeLine(xRes);
    for(int i = 0; i < xRes; i++){ //creates "time line"
        timeLine[i] = (totalTime/xRes)*i - (totalTime/2);
    }
    //------------------------ Quadrature Detection ---------------------------------------------------------
    double targetLamour = gamma * B0;
    
    vector<double> refCos(xRes);
    vector<double> refSin(xRes);

    for(int i = 0; i < xRes; i++){
        refCos[i] = cos(targetLamour*timeLine[i]);
    }
    for(int i = 0; i < xRes; i++){
        refSin[i] = -1*sin(targetLamour*timeLine[i]);
    }
    //---------------------------------- Image Prep -----------------------------------------------------------------------------------------
    vector<vector<double>> img = getImage(imgName);

    double normFactor = getMax(img);
    int imgRows = img.size();          // number of rows
    int imgCols = img.empty() ? 0 : img[0].size();  // number of columns (from first row)

    vector<vector<complex<double>>> kSpace(yRes, vector<complex<double>>(xRes));

    vector<double> xSpatial(imgCols);
    vector<double> ySpatial(imgRows);

    double delta_X = xFOV/imgCols;
    double delta_Y = yFOV/imgRows;

    vector<double> phaseSteps(yRes);

    vector<vector<double>> field(imgRows, vector<double>(imgCols));

    for(int i = 0; i < yRes; i++){
        phaseSteps[i] = i - yRes/2; //need to be int?
    }

    for(int i = 0; i < imgCols; i++){
        xSpatial[i] = delta_X * i;
        //cout << i << ": " << xSpatial[i] << endl;
    }
    for(int i = 0; i < imgRows; i++){
        ySpatial[i] = delta_Y * i;
    }

    for(int i = 0; i < imgRows; i++){ //creates "magnetic field" with x gradient applied
        for(int j = 0; j < imgCols; j++){
            field[i][j] = B0 + xGradStep * xSpatial[j]; 
        }
    }

    for(int i = 0; i < imgRows; i++) { //normalize image
        for(int j = 0; j < imgCols; j++){
            img[i][j] = img[i][j] / normFactor;
        }
    }

    vector<vector<double>> lamourFreqs(imgRows, vector<double>(imgCols)); //reduntant?
    for(int i = 0; i < imgRows; i++){
        for(int j = 0; j < imgCols; j++) {
            lamourFreqs[i][j] = gamma * field[i][j];
        }
    }

    //------------------------------- Main Loop Precompute ---------------------------------------------
    vector<vector<complex<double>>> M(imgRows, vector<complex<double>>(xRes));


    //instead of "acc += img[k][j] * exp(complex<double>(0.0, -(lamourFreqs[k][j] * timeLine[i] + phase_y[k])));" precompute before loop, then split exp using exp properties
    for (int k = 0; k < imgRows; k++) {
        for (int i = 0; i < xRes; i++) {
            complex<double> acc(0.0, 0.0);
            for (int j = 0; j < imgCols; j++) {
                double angle = -lamourFreqs[k][j] * timeLine[i];
                acc += img[k][j] * complex<double>(cos(angle), sin(angle)); 
            }                                                                               
            M[k][i] = acc;
        }
    }
    
    //--------------------------------------- Main Loop ---------------------------------------------------------
    for(int idx = 0; idx < yRes; idx++){

        vector<double> phase_y(imgRows);
        for(int i = 0; i < imgRows; i++) {
            phase_y[i] = (yGradStep*phaseSteps[idx]*ySpatial[i]) * gamma * pulseDuration; //making y field gradient
        }

    vector<complex<double>> signal(xRes);
    for (int i = 0; i < xRes; i++) {
        complex<double> acc(0.0, 0.0);
        for (int k = 0; k < imgRows; k++) {
            double angle = -phase_y[k];
            acc += M[k][i] * complex<double>(cos(angle), sin(angle));
        }
        signal[i] = acc * complex<double>(refCos[i], refSin[i]);
    }

        kSpace[idx] = signal;
        cout << "Line [" << idx << "] finished" << endl;
    }
    cout << "k-Space Constructed." << endl;
    
    //------------------------------ image reconstruction --------------------------------------------------------------
    auto imgRecon = ifft2D(kSpace);

    cv::Mat magMat = complexVecToMat(imgRecon);

    cv::Mat displayMat;
    cv::normalize(magMat, displayMat, 0, 255, cv::NORM_MINMAX);
    displayMat.convertTo(displayMat, CV_8U);

    double minVal = 1e30;
    double maxVal = -1e30;

    for (int i = 0; i < imgRecon.size(); i++)
    {
        for (int j = 0; j < imgRecon[0].size(); j++)
        {
            double m = abs(imgRecon[i][j]);

            minVal = std::min(minVal, m);
            maxVal = std::max(maxVal, m);
        }
    }

    cout << "Recon magnitude range: " << minVal << "  " << maxVal << endl;

    // Stop timer
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Program finished in " << duration.count() << " ms" << std::endl;    

    cv::namedWindow("Reconstructed Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Reconstructed Image", displayMat);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
