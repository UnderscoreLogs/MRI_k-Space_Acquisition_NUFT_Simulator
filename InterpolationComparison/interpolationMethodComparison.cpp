#define _USE_MATH_DEFINES
#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>
#include <complex>
#include <iostream>
#include <chrono>

#include "FFT.hpp"

using namespace std;

double sinc(double x) {
    if (x == 0.0) {
        return 1.0;
    }

    return sin(M_PI * x) / (M_PI * x);
}

double lanczosKernel(double x, int a) {
    if(abs(x) < a) {
        return sinc(x) * sinc(x/(double)a);
    } else {
        return 0.0;
    }
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

vector<vector<double>> nearestNeighborInterpolation(vector<vector<double>> img, int newWidth, int newHeight) {

    vector<vector<double>> resizedImg(newHeight, vector<double>(newWidth));
    int oldHeight = img.size();
    int oldWidth = img[0].size();   

    for(int y = 0; y < newHeight; y++) {
        for(int x = 0; x < newWidth; x++) {

            int oldX = round((float)x * oldWidth / newWidth);
            int oldY = round((float)y * oldHeight / newHeight);
            
            // prevent out of bounds
            if(oldX >= oldWidth) {
                oldX = oldWidth - 1;
            }

            if(oldY >= oldHeight) {
                oldY = oldHeight - 1;
            }

            resizedImg[y][x] = img[oldY][oldX];
        }
    }

    return resizedImg;
}

vector<vector<double>> bilinearInterpolation(vector<vector<double>> img, int newWidth, int newHeight) {
    vector<vector<double>> resizedImg(newHeight, vector<double>(newWidth));
    int oldHeight = img.size();
    int oldWidth = img[0].size();  
    
    double xScale = (double)(oldWidth - 1) / (newWidth - 1);
    double yScale = (double)(oldHeight - 1) / (newHeight - 1);

    for(int y = 0; y < newHeight; y++) {
        for(int x = 0; x < newWidth; x++) {

            double corX = x * xScale;
            double corY = y * yScale;

            int x_0 = floor(corX);
            int y_0 = floor(corY);

            int x_1 = min(x_0 + 1, oldWidth - 1);
            int y_1 = min(y_0 + 1, oldHeight - 1);

            double dx = corX - x_0;
            double dy = corY - y_0;

            resizedImg[y][x] =
                (1-dx)*(1-dy)*img[y_0][x_0] +
                dx*(1-dy)*img[y_0][x_1] +
                (1-dx)*dy*img[y_1][x_0] +
                dx*dy*img[y_1][x_1];
        }
    }

    return resizedImg;
}

vector<vector<double>> bicubicInterpolation(vector<vector<double>> img, int newWidth, int newHeight) {
    vector<vector<double>> resizedImg(newHeight, vector<double>(newWidth));
    int oldHeight = img.size();
    int oldWidth = img[0].size();  
    
    double xScale = (double)(oldWidth - 1) / (newWidth - 1);
    double yScale = (double)(oldHeight - 1) / (newHeight - 1);
    for(int y = 0; y < newHeight; y++) {
        for(int x = 0; x < newWidth; x++) {

            vector<double> R(4);

            double corX = x * xScale;
            double corY = y * yScale;

            int x1 = floor(corX);
            int x2 = min(x1 + 1, oldWidth - 1);
            int x3 = min(x1 + 2, oldWidth - 1);
            int x0 = max(x1 - 1, 0);
            
            double t = corX - x1;

            int y1 = floor(corY);
            int y2 = min(y1 + 1, oldHeight - 1);
            int y3 = min(y1 + 2, oldHeight - 1);
            int y0 = max(y1 - 1, 0);

            double u = corY - y1;

            int yi[4] = {y0, y1, y2, y3};

            for (int i = 0; i < 4; i++)
            {
                R[i] =
                    img[yi[i]][x1] +
                    0.5 * t * (img[yi[i]][x2] - img[yi[i]][x0]) +
                    0.5 * t * t * (2 * img[yi[i]][x0] - 5 * img[yi[i]][x1] + 4 * img[yi[i]][x2] - img[yi[i]][x3]) +
                    0.5 * t * t * t * (3 * (img[yi[i]][x1] - img[yi[i]][x2]) + img[yi[i]][x3] - img[yi[i]][x0]);
            }

            resizedImg[y][x] =
                    R[1] +
                    0.5 * u * (R[2] - R[0]) +
                    0.5 * u * u * (2 * R[0] - 5 * R[1] + 4 * R[2] - R[3]) +
                    0.5 * u * u * u * (3 * (R[1] - R[2]) + R[3] - R[0]); 
        }
    }

    return resizedImg;    
}

vector<vector<double>> lanczos3Interpolation(vector<vector<double>> img, int newWidth, int newHeight) {

    vector<vector<double>> resizedImg(newHeight, vector<double>(newWidth));
    int oldHeight = img.size();
    int oldWidth = img[0].size();  
    
    double xScale = (double)(oldWidth - 1) / (newWidth - 1);
    double yScale = (double)(oldHeight - 1) / (newHeight - 1);

    for(int y = 0; y < newHeight; y++){
        for(int x = 0; x < newWidth; x++){
            double corX = x * xScale;
            double corY = y * yScale;
            
            double numSum = {};
            double domSum = {};

            int x3 = floor(corX);
            int y3 = floor(corY);

            int xi[6] = {max(x3 - 2, 0), max(x3 - 1, 0), (x3), min(x3 + 1, oldWidth - 1), min(x3 + 2, oldWidth - 1), min(x3 + 3, oldWidth - 1)};
            int yi[6] = {max(y3 - 2, 0), max(y3 - 1, 0), (y3), min(y3 + 1, oldHeight - 1), min(y3 + 2, oldHeight - 1), min(y3 + 3, oldHeight - 1)};

            double dx_i[6] = {corX - xi[0], corX - xi[1], corX - xi[2], corX - xi[3], corX - xi[4], corX - xi[5]};
            double dy_i[6] = {corY - yi[0], corY - yi[1], corY - yi[2], corY - yi[3], corY - yi[4], corY - yi[5]};


            for(int j = 0; j < 6; j++){
                for(int i = 0; i < 6; i++) {
                    numSum += img[yi[j]][xi[i]] * lanczosKernel(dx_i[i], 3) * lanczosKernel(dy_i[j], 3);
                }
            }

            for(int j = 0; j < 6; j++){
                for(int i = 0; i < 6; i++) {
                    domSum += lanczosKernel(dx_i[i], 3) * lanczosKernel(dy_i[j], 3);
                }
            }

            resizedImg[y][x] = numSum / domSum;
        }
    }

    return resizedImg;
}

vector<vector<double>> fourierZeroPadInterpolation(vector<vector<double>> img, int newWidth, int newHeight)
{
    vector<vector<double>> resizedImg(newHeight, vector<double>(newWidth));
    int oldHeight = img.size();
    int oldWidth = img[0].size();

    vector<vector<complex<double>>> kspace = fft2DShift(fft2D(img));

    vector<vector<complex<double>>> resizedKspace(newHeight, vector<complex<double>>(newWidth, {0.0, 0.0}));

    if (newWidth >= oldWidth && newHeight >= oldHeight)
    {
        int xOffset = (newWidth - oldWidth) / 2;
        int yOffset = (newHeight - oldHeight) / 2;

        for (int y = 0; y < oldHeight; y++)
        {
            for (int x = 0; x < oldWidth; x++)
            {
                resizedKspace[y+yOffset][x+xOffset] = kspace[y][x];
            }
        }
    } else {
        int xOffset = (oldWidth - newWidth) / 2;
        int yOffset = (oldHeight - newHeight) / 2;

        for (int y = 0; y < newHeight; y++)
        {
            for (int x = 0; x < newWidth; x++)
            {
                resizedKspace[y][x] = kspace[y+yOffset][x+xOffset];
            }
        }
    }

    vector<vector<complex<double>>> result = ifft2D(ifft2DShift(resizedKspace));

    for (int y = 0; y < newHeight; y++)
    {
        for (int x = 0; x < newWidth; x++)
        {
            resizedImg[y][x] = result[y][x].real();
        }
    }

    return resizedImg;
}

vector<vector<double>> sincInterpolation(vector<vector<double>> img, int newWidth, int newHeight) {

    vector<vector<double>> resizedImg(newHeight, vector<double>(newWidth));
    int oldHeight = img.size();
    int oldWidth = img[0].size();  
    
    double xScale = (double)(oldWidth - 1) / (newWidth - 1);
    double yScale = (double)(oldHeight - 1) / (newHeight - 1);

    for(int y = 0; y < newHeight; y++){
        for(int x = 0; x < newWidth; x++){
            double corX = x * xScale;
            double corY = y * yScale;

            for(int j = 0; j < oldHeight; j++){
                for(int i = 0; i < oldWidth; i++) {
                    resizedImg[y][x] += img[j][i] * sinc(corX - i) * sinc(corY - j);
                }
            }    
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

    vector<vector<complex<double>>> kSpace(newHeight, vector<complex<double>>(newWidth, complex<double>(0.0, 0.0)));

    if(newWidth >= imgCols && newHeight >= imgRows) {
        int xOffset = (newWidth - imgCols) / 2;
        int yOffset = (newHeight - imgRows) / 2;
        for (int y = 0; y < imgRows; y++) {
            for (int x = 0; x < imgCols; x++) {
                kSpace[y + yOffset][x + xOffset] = kSpaceTrue[y][x];
            }
        }
    } else if( newWidth < imgCols && newHeight < imgRows) {
        int xOffset = (imgCols - newWidth) / 2;
        int yOffset = (imgRows - newHeight) / 2;
        for (int y = 0; y < newHeight; y++) {
            for (int x = 0; x < newWidth; x++) {
                kSpace[y][x] = kSpaceTrue[y + yOffset][x + yOffset];
            }
        }        
    }

    cout << "k-Space Constructed." << endl;

    //------------------------------ image reconstruction --------------------------------------------------------------
    
    
    vector<vector<complex<double>>> imgRecon = ifft2D(ifft2DShift(kSpace));
    vector<vector<double>> resizedImg(imgRecon.size(), vector<double>(imgRecon[0].size()));

    for (int y = 0; y < newHeight; y++)
    {
        for (int x = 0; x < newWidth; x++)
        {
            resizedImg[y][x] = imgRecon[y][x].real();
        }
    }

    return resizedImg;
}

void displayInterpolation(string name, vector<vector<double>> img_double)
{
    cv::Mat img = doubleToMat(img_double);

    cv::Mat display;
    cv::normalize(img, display, 0, 255, cv::NORM_MINMAX);
    display.convertTo(display, CV_8U);

    cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
    cv::imshow(name, display);
}

double calcPSNR(vector<vector<double>> K, vector<vector<double>> I) {
    int N = K.size();
    int M = K[0].size();

    double sum = {};

    for(int y = 0; y < N; y++){
        for(int x = 0; x < M; x++){
            sum += (I[y][x] - K[y][x]) * (I[y][x] - K[y][x]);
        }
    }

   double MSE = sum/(M*N);
    
   double PSNR = 10.0 * log10((255.0 * 255.0) / MSE);
   return PSNR;
}

int main(){
    
    int xRes = 512;
    int yRes = 512;

    string imgName = "C:/Users/smegl/OneDrive/Desktop/Code/Matlab/Imaging/standard_test_images/lena_gray_256.tif";

    string compareImgName = "C:/Users/smegl/OneDrive/Desktop/Code/Matlab/Imaging/standard_test_images/lena_gray_512.tif";
    
    vector<vector<double>> img = getImage(imgName);

    vector<vector<double>> compareImg = getImage(compareImgName);

    //-----------------------------------------MRI--------------------------------------------------
    auto mriStart = std::chrono::high_resolution_clock::now(); 
    //start timer
    auto MRI_double = mriInterpolation(img, xRes, yRes);
    cv::Mat MRI_Img = doubleToMat(MRI_double);
    
    auto mriEnd = std::chrono::high_resolution_clock::now();// stop timer

    auto MRIduration = std::chrono::duration_cast<std::chrono::milliseconds>(mriEnd - mriStart);
    std::cout << "MRI Interpolation finished in " << MRIduration.count() << " ms" << std::endl;

    cout << "MRI PSNR: " << calcPSNR(MRI_double, compareImg) << endl;

    //Image Display
    cv::Mat MRI_Img_Display;
    cv::normalize(MRI_Img, MRI_Img_Display, 0, 255, cv::NORM_MINMAX);
    MRI_Img_Display.convertTo(MRI_Img_Display, CV_8U);

    cv::namedWindow("MRI Interpolation Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("MRI Interpolation Image", MRI_Img_Display);

    //----------------------------------------------- Near Neighbor ----------------------------------------------
    auto nearestNeighborStart = std::chrono::high_resolution_clock::now();  //start timer
    auto NearNeighbor_double =nearestNeighborInterpolation(img, xRes, yRes);
    cv::Mat nearestNeighbor_Img = doubleToMat(NearNeighbor_double);
    
    auto nearestNeighborEnd = std::chrono::high_resolution_clock::now();// stop timer

    auto nearestNeighborduration = std::chrono::duration_cast<std::chrono::milliseconds>(nearestNeighborEnd - nearestNeighborStart);
    std::cout << "Nearest Neighbor Interpolation finished in " << nearestNeighborduration.count() << " ms" << std::endl;
    cout << "Nearest Neighbor PSNR: " << calcPSNR(NearNeighbor_double, compareImg) << endl;

    //Image Display
    cv::Mat nearestNeighbor_Img_Display;
    cv::normalize(nearestNeighbor_Img, nearestNeighbor_Img_Display, 0, 255, cv::NORM_MINMAX);
    nearestNeighbor_Img_Display.convertTo(nearestNeighbor_Img_Display, CV_8U);

    cv::namedWindow("Nearest Neighbor Interpolation Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Nearest Neighbor Interpolation Image", nearestNeighbor_Img_Display);

    //----------------------------------------------- Bilinear ----------------------------------------------    
    auto bilinearStart = std::chrono::high_resolution_clock::now(); //start timer
    auto bilinear_double =bilinearInterpolation(img, xRes, yRes);

    cv::Mat bilinear_Img = doubleToMat(bilinear_double);
    
    auto bilinearEnd = std::chrono::high_resolution_clock::now();// stop timer

    auto bilinearduration = std::chrono::duration_cast<std::chrono::milliseconds>(bilinearEnd - bilinearStart);
    std::cout << "Bilinear Interpolation finished in " << bilinearduration.count() << " ms" << std::endl;
    cout << "Bilinear PSNR: " << calcPSNR(bilinear_double, compareImg) << endl;

    //Image Display
    cv::Mat bilinear_Img_Display;
    cv::normalize(bilinear_Img, bilinear_Img_Display, 0, 255, cv::NORM_MINMAX);
    bilinear_Img_Display.convertTo(bilinear_Img_Display, CV_8U);

    cv::namedWindow("bilinear Interpolation Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("bilinear Interpolation Image", bilinear_Img_Display);

    //----------------------------------------------- Bicubic ----------------------------------------------        
    auto bicubicStart = std::chrono::high_resolution_clock::now(); //start timer
    auto bicubic_double =bicubicInterpolation(img, xRes, yRes);

    cv::Mat bicubic_Img = doubleToMat(bicubic_double);
    
    auto bicubicEnd = std::chrono::high_resolution_clock::now();// stop timer

    auto bicubicduration = std::chrono::duration_cast<std::chrono::milliseconds>(bicubicEnd - bicubicStart);
    std::cout << "bicubic Interpolation finished in " << bicubicduration.count() << " ms" << std::endl;
    cout << "bicubic PSNR: " << calcPSNR(bicubic_double, compareImg) << endl;

    //Image Display
    cv::Mat bicubic_Img_Display;
    cv::normalize(bicubic_Img, bicubic_Img_Display, 0, 255, cv::NORM_MINMAX);
    bicubic_Img_Display.convertTo(bicubic_Img_Display, CV_8U);

    cv::namedWindow("bicubic Interpolation Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("bicubic Interpolation Image", bicubic_Img_Display);

    //----------------------------------------------- Lanczos-3 ----------------------------------------------        
    auto lanczos3Start = std::chrono::high_resolution_clock::now(); //start timer
    auto lanczos3_double =lanczos3Interpolation(img, xRes, yRes);

    cv::Mat lanczos3_Img = doubleToMat(lanczos3_double);
    
    auto lanczos3End = std::chrono::high_resolution_clock::now();// stop timer

    auto lanczos3duration = std::chrono::duration_cast<std::chrono::milliseconds>(lanczos3End - lanczos3Start);
    std::cout << "lanczos3 Interpolation finished in " << lanczos3duration.count() << " ms" << std::endl;
    cout << "lanczos3 PSNR: " << calcPSNR(lanczos3_double, compareImg) << endl;

    //Image Display
    cv::Mat lanczos3_Img_Display;
    cv::normalize(lanczos3_Img, lanczos3_Img_Display, 0, 255, cv::NORM_MINMAX);
    lanczos3_Img_Display.convertTo(lanczos3_Img_Display, CV_8U);

    cv::namedWindow("lanczos3 Interpolation Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("lanczos3 Interpolation Image", lanczos3_Img_Display);

    //----------------------------------------------- fourierZeroPad ----------------------------------------------            
    auto fourierZeroPadStart = std::chrono::high_resolution_clock::now(); //start timer
    auto fourierZeroPad_double =fourierZeroPadInterpolation(img, xRes, yRes);

    cv::Mat fourierZeroPad_Img = doubleToMat(fourierZeroPad_double);
    
    auto fourierZeroPadEnd = std::chrono::high_resolution_clock::now();// stop timer

    auto fourierZeroPadduration = std::chrono::duration_cast<std::chrono::milliseconds>(fourierZeroPadEnd - fourierZeroPadStart);
    std::cout << "fourierZeroPad Interpolation finished in " << fourierZeroPadduration.count() << " ms" << std::endl;
    cout << "Fourier Zero Pad PSNR: " << calcPSNR(fourierZeroPad_double, compareImg) << endl;

    //Image Display
    cv::Mat fourierZeroPad_Img_Display;
    cv::normalize(fourierZeroPad_Img, fourierZeroPad_Img_Display, 0, 255, cv::NORM_MINMAX);
    fourierZeroPad_Img_Display.convertTo(fourierZeroPad_Img_Display, CV_8U);

    cv::namedWindow("fourierZeroPad Interpolation Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("fourierZeroPad Interpolation Image", fourierZeroPad_Img_Display);

    //----------------------------------------------- sinc ----------------------------------------------            
    auto sincStart = std::chrono::high_resolution_clock::now(); //start timer
    auto sinc_double =sincInterpolation(img, xRes, yRes);

    cv::Mat sinc_Img = doubleToMat(sinc_double);
    
    auto sincEnd = std::chrono::high_resolution_clock::now();// stop timer

    auto sincduration = std::chrono::duration_cast<std::chrono::milliseconds>(sincEnd - sincStart);
    std::cout << "sinc Interpolation finished in " << sincduration.count() << " ms" << std::endl;
    cout << "Sinc PSNR: " << calcPSNR(sinc_double, compareImg) << endl;

    //Image Display
    cv::Mat sinc_Img_Display;
    cv::normalize(sinc_Img, sinc_Img_Display, 0, 255, cv::NORM_MINMAX);
    sinc_Img_Display.convertTo(sinc_Img_Display, CV_8U);

    cv::namedWindow("sinc Interpolation Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("sinc Interpolation Image", sinc_Img_Display);

    //-------------------------------------------------------------------------------------------------     
    
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
}