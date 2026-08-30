#define _USE_MATH_DEFINES
#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>
#include <complex>
#include <iostream>
#include <chrono>

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

vector<vector<double>> DIFT(vector<vector<complex<double>>> kSpace) {
    
    double xFOV = 1.0;
    double yFOV = 1.0;
    
    int kySize = kSpace.size();
    int kxSize = kSpace[0].size();

    vector<double> xSpatial(kxSize);
    vector<double> ySpatial(kySize);

    double dx = xFOV / kxSize;
    double dy = yFOV / kySize;

    for (int i = 0; i < kxSize; i++)
    {
        xSpatial[i] = dx * i;
        // cout << i << ": " << xSpatial[i] << endl;
    }
    for (int i = 0; i < kySize; i++)
    {
        ySpatial[i] = dy * i;
    }

    vector<double> kx(kxSize);
    for (int i = 0; i < kxSize; i++)
    {
        kx[i] = (i - kxSize / 2.0) / xFOV;
    }

    vector<double> ky(kySize);
    for (int i = 0; i < kxSize; i++)
    {
        ky[i] = (i - kySize / 2.0) / yFOV;
    }

    vector<vector<double>> resizedImg(kySize, vector<double>(kxSize));
    vector<vector<complex<double>>> M(kySize, vector<complex<double>>(kxSize));

    for (int k_xi = 0; k_xi < kxSize; k_xi++)
    {
        for (int y = 0; y < kySize; y++)
        {
            complex<double> acc = 0.0;
            for (int k_yi = 0; k_yi < kySize; k_yi++)
            {
                double angle = 2 * M_PI * (ky[k_yi] * ySpatial[y]);

                acc += kSpace[k_yi][k_xi] * complex<double>(cos(angle), sin(angle));
            }
            M[y][k_xi] = acc;
        }
    }

    for (int y = 0; y < kySize; y++)
    {
        for (int x = 0; x < kxSize; x++)
        {
            complex<double> pixel = 0.0;
            for (int k_xi = 0; k_xi < kxSize; k_xi++)
            {
                double angle = 2 * M_PI * (kx[k_xi] * xSpatial[x]);

                pixel += M[y][k_xi] * complex<double>(cos(angle), sin(angle));
            }
            pixel /= (kxSize * kySize);
            resizedImg[y][x] = pixel.real();
        }
    }

    return resizedImg;

}

vector<vector<double>> mriInterpolation(vector<vector<double>> img, int newWidth, int newHeight) {
    
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
        xSpatial[i] = dx * i;
        // cout << i << ": " << xSpatial[i] << endl;
    }
    for (int i = 0; i < imgRows; i++)
    {
        ySpatial[i] = dy * i;
    }

    vector<vector<kSample>> kSamples(imgRows, vector<kSample>(imgCols, {0.0, 0.0, 0.0}));
    for (int j = 0; j < imgRows; j++)
    {
        for(int i = 0; i < imgCols; i++){
            kSamples[j][i].kx = (i - (imgCols / 2.0)) / xFOV;
            kSamples[j][i].ky = (j - (imgRows / 2.0)) / yFOV;
        }
    }

    //------------------------------- Main Loop Precompute ---------------------------------------------
    vector<vector<complex<double>>> M(imgRows, vector<complex<double>>(imgCols));

    for (int k = 0; k < imgRows; k++) {
        for (int i = 0; i < imgCols; i++) {
            complex<double> acc(0.0, 0.0);
            for (int j = 0; j < imgCols; j++) {
                double angle = -2 * M_PI * kSamples[0][i].kx * xSpatial[j];
                acc += img[k][j] * complex<double>(cos(angle), sin(angle));
            }
            M[k][i] = acc;
        }
    }
    //--------------------------------------- Main Loop ---------------------------------------------------------
    //vector<vector<complex<double>>> kSpaceTrue(imgRows, vector<complex<double>>(imgCols));
    for (int idx = 0; idx < imgRows; idx++) {
        for (int i = 0; i < imgCols; i++)
        {
            complex<double> acc(0.0, 0.0);
            for (int k = 0; k < imgRows; k++)
            {
                double angle = -2 * M_PI * kSamples[idx][0].ky * ySpatial[k];
                acc += M[k][i] * complex<double>(cos(angle), sin(angle));
            }
            kSamples[idx][i].value = acc;
        }
    }

    vector<vector<complex<double>>> kSpace(newHeight, vector<complex<double>>(newWidth, complex<double>(0.0, 0.0)));

    if(newWidth >= imgCols && newHeight >= imgRows) {
        int xOffset = (newWidth - imgCols) / 2;
        int yOffset = (newHeight - imgRows) / 2;
        for (int y = 0; y < imgRows; y++) {
            for (int x = 0; x < imgCols; x++) {
                kSpace[y + yOffset][x + xOffset] = kSamples[y][x].value;
            }
        }
    } else if( newWidth < imgCols && newHeight < imgRows) {
        int xOffset = (imgCols - newWidth) / 2;
        int yOffset = (imgRows - newHeight) / 2;
        for (int y = 0; y < newHeight; y++) {
            for (int x = 0; x < newWidth; x++) {
                kSpace[y][x] = kSamples[y + yOffset][x + xOffset].value;
            }
        }        
    }

    cout << "k-Space Constructed." << endl;

    //------------------------------ image reconstruction --------------------------------------------------------------


    return DIFT(kSpace);
}

int main()
{
    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    //------------------------Settings-------------------------------------------------------------------------------------------------
    string imgName = "C:/Users/smegl/OneDrive/Desktop/Code/Matlab/Imaging/standard_test_images/lena_gray_256.tif";

    int xRes = 512;
    int yRes = 512;

    vector<vector<double>> img = getImage(imgName);

    auto MRI_double = mriInterpolation(img, xRes, yRes);
    // Stop timer
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Program finished in " << duration.count() << " ms" << std::endl;

    cv::Mat MRI_Img = doubleToMat(MRI_double);

    cv::Mat displayMat;
    cv::normalize(MRI_Img, displayMat, 0, 255, cv::NORM_MINMAX);
    displayMat.convertTo(displayMat, CV_8U);

    cv::namedWindow("Reconstructed Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Reconstructed Image", displayMat);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
