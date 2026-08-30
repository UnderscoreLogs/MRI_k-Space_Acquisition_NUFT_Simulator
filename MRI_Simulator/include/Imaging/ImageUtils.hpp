#pragma once

#include "../MRI/KSpace.hpp"

#define _USE_MATH_DEFINES
#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>
#include <complex>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <filesystem>

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Utilities
///////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::vector<double>> getImage(std::string imgName);

void displayKSpace(std::vector<std::vector<std::complex<double>>>& kSpace, std::string name);

void displayTrajectory(std::vector<kSample>& samples, int size, std::string windowName);

double calcPSNR(std::vector<std::vector<double>> K, std::vector<std::vector<double>> controlImage);

double calcSSIM(const std::vector<std::vector<double>>& img2, const std::vector<std::vector<double>>& img1);

std::vector<std::vector<double>> normalizeImage(const std::vector<std::vector<double>>& img, double newMin, double newMax);

void saveImg(std::vector<std::vector<double>> image_double, std::string fileName, std::string folder);

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Helpers
///////////////////////////////////////////////////////////////////////////////////////////////////////////

cv::Mat doubleToMat(std::vector<std::vector<double>> imgD);
cv::Mat complexVecToMat(const std::vector<std::vector<std::complex<double>>> &data);