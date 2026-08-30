#pragma once

#include "KSpace.hpp"
#include "../Imaging/ImageUtils.hpp"

#include <vector>
#include <string>

std::vector<std::vector<double>> mriRadialInterpolation(
    std::vector<std::vector<double>> img, 
    int newWidth, 
    int newHeight, 
    int numOfSpokes,  
    int numSamplesPerSpoke, 
    std::string reconType,
    double SNR,
    double timePercent,
    bool debug
);

std::vector<std::vector<double>> mriRandomInterpolation(
    std::vector<std::vector<double>> img, 
    int newWidth,
    int newHeight, 
    int numOfSamples, 
    std::string reconType,
    double SNR,
    double timePercent,
    bool debug
);

std::vector<std::vector<double>> mriCartesianInterpolation(
    std::vector<std::vector<double>> img, 
    int newWidth, 
    int newHeight, 
    int kxSampleSize, 
    int kySampleSize, 
    std::string reconType,
    double SNR,
    double timePercent,
    bool debug
);

std::vector<std::vector<double>> mriSpiralInterpolation(
    std::vector<std::vector<double>> img, 
    int newWidth, 
    int newHeight, 
    int numOfSamples, 
    int numOfTurns,
    int numInterleaves, 
    std::string reconType,
    double SNR,
    double timePercent,
    bool debug
);

std::vector<std::vector<double>> mriVariableDensityCartesianInterpolation(
    std::vector<std::vector<double>> img, 
    int newWidth, 
    int newHeight, 
    int kxSampleSize, 
    int kySampleSize, 
    double alpha,
    double centerFraction,
    double acceleration,  
    std::string reconType,
    double SNR,
    double timePercent,
    bool debug
);

std::vector<std::vector<double>> mriVariableDensityRandomInterpolation(
    std::vector<std::vector<double>> img, 
    int newWidth, 
    int newHeight, 
    double alpha,
    double centerFraction,
    double acceleration,
    std::string reconType, 
    double SNR,
    double timePercent,
    bool debug
);
