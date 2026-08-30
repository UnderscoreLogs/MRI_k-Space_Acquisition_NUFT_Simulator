#pragma once

#include "KSpace.hpp"
#include "../Math/FFT.hpp"
#include "../Imaging/ImageUtils.hpp"

#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <complex> 
#include <vector>
#include <numeric>

std::vector<kSample> NUDFT(
    std::vector<double> xSpatial,
    std::vector<double> ySpatial,
    std::vector<kSample> kSamples,
    std::vector<std::vector<double>> img
);

std::vector<std::vector<double>> INUDFT(
    std::vector<kSample> kSamples, 
    int xRes, int yRes
);

std::vector<std::vector<double>> INUFFT(
    std::vector<kSample> kSamples,
    int xRes,
    int yRes,
    bool debug
);

void addKspaceNoise(
    std::vector<kSample>& kSamples, 
    double SNR
);

void addMovement(
    std::vector<kSample>& kSamples, 
    double timePercent
);