#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <stdexcept>

double calcPSNR(std::vector<std::vector<double>> K, std::vector<std::vector<double>> controlImage);

double calcSSIM(const std::vector<std::vector<double>>& img2, const std::vector<std::vector<double>>& img1);