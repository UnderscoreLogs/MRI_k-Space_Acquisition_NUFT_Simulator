#define _USE_MATH_DEFINES
#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>
#include <complex>
#include <iostream>
#include <chrono>

#include "FFT.hpp"

using namespace std;

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

int main()
{
    //-------------------------Timer-------------------------------------
    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    //------------------------Settings-------------------------------------------------------------------------------------------------
    string imgName = "C:/Users/smegl/OneDrive/Desktop/Code/Matlab/Imaging/standard_test_images/lena_gray_256.tif";

    int xRes = 128;
    int yRes = 128;

    double xFOV = 1.0; //changes nothing
    double yFOV = 1.0;


    //---------------------------------- Image Prep -----------------------------------------------------------------------------------------
    vector<vector<double>> img = getImage(imgName);

    double normFactor = getMax(img);
    int imgRows = img.size();                      // number of rows
    int imgCols = img.empty() ? 0 : img[0].size(); // number of columns (from first row)

    //vector<vector<complex<double>>> kSpace(yRes, vector<complex<double>>(xRes));

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

    for (int i = 0; i < imgRows; i++) //not needed
    { // normalize image
        for (int j = 0; j < imgCols; j++)
        {
            img[i][j] = img[i][j] / normFactor;
        }
    }

    vector<double> kx(imgCols);
    for (int i = 0; i < imgCols; i++)
    {
        kx[i] = (i - imgCols / 2.0) / xFOV;
    }

    vector<double> ky(imgRows);
    for (int i = 0; i < imgRows; i++)
    {
        ky[i] = (i - imgRows / 2.0) / yFOV;
    }

    //------------------------------- Main Loop Precompute ---------------------------------------------
    vector<vector<complex<double>>> M(imgRows, vector<complex<double>>(imgCols));

    for (int k = 0; k < imgRows; k++) {
        for (int i = 0; i < imgCols; i++) {
            complex<double> acc(0.0, 0.0);
            for (int j = 0; j < imgCols; j++) {
                double angle = -2 * M_PI * kx[i] * xSpatial[j];
                acc += img[k][j] * complex<double>(cos(angle), sin(angle));
            }
            M[k][i] = acc;
        }
    }
    //--------------------------------------- Main Loop ---------------------------------------------------------
    vector<vector<complex<double>>> kSpaceTrue(imgRows, vector<complex<double>>(imgCols));
    for (int idx = 0; idx < imgRows; idx++) {
        for (int i = 0; i < imgCols; i++)
        {
            complex<double> acc(0.0, 0.0);
            for (int k = 0; k < imgRows; k++)
            {
                double angle = -2 * M_PI * ky[idx] * ySpatial[k];
                acc += M[k][i] * complex<double>(cos(angle), sin(angle));
            }
            kSpaceTrue[idx][i] = acc;
        }
    }

    vector<vector<complex<double>>> kSpace(yRes, vector<complex<double>>(xRes, complex<double>(0.0, 0.0)));

    if(xRes >= imgCols && yRes >= imgRows) {
        int xOffset = (xRes - imgCols) / 2;
        int yOffset = (yRes - imgRows) / 2;
        for (int y = 0; y < imgRows; y++) {
            for (int x = 0; x < imgCols; x++) {
                kSpace[y + yOffset][x + xOffset] = kSpaceTrue[y][x];
            }
        }
    } else if( xRes < imgCols && yRes < imgRows) {
        int xOffset = (imgCols - xRes) / 2;
        int yOffset = (imgRows - yRes) / 2;
        for (int y = 0; y < yRes; y++) {
            for (int x = 0; x < xRes; x++) {
                kSpace[y][x] = kSpaceTrue[y + yOffset][x + yOffset];
            }
        }        
    }

    cout << "k-Space Constructed." << endl;

    //------------------------------ image reconstruction --------------------------------------------------------------
    
    
    auto imgRecon = ifft2D(ifft2DShift(kSpace));

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

    cv::Mat magKMat = complexVecToMat(ifft2DShift(kSpace));
    
    cv::Mat displayKspace;

    cv::normalize(magKMat, displayKspace, 0, 255, cv::NORM_MINMAX);

    displayKspace.convertTo(displayKspace,CV_8U);
    imshow("kspace",displayKspace);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
