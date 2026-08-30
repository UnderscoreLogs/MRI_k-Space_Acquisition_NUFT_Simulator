#pragma once

#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <complex> 
#include <vector>
#include <sstream>
#include <string>

#include <math.h>

// 1D FFT / IFFT
std::vector<std::complex<double>> fft(const std::vector<double>& realSamples);

std::vector<std::complex<double>> fft(const std::vector<std::complex<double>>& samples);

std::vector<std::complex<double>> ifft(const std::vector<std::complex<double>>& samples);

std::vector<std::complex<double>> bluesteinFFT(const std::vector<std::complex<double>>& input);

std::vector<std::complex<double>> bluesteinFFT(const std::vector<double>& input);

std::vector<std::complex<double>> bluesteinIFFT(const std::vector<std::complex<double>>& input);

// 2D FFT / IFFT (row-column decomposition)
std::vector<std::vector<std::complex<double>>> transpose(
    const std::vector<std::vector<std::complex<double>>>& matrix);

std::vector<std::vector<std::complex<double>>> fft2D(
    const std::vector<std::vector<double>>& input);

std::vector<std::vector<std::complex<double>>> ifft2D(
    const std::vector<std::vector<std::complex<double>>>& input);

std::vector<std::vector<std::complex<double>>> bluesteinFFT2D(
    const std::vector<std::vector<double>>& input);

std::vector<std::vector<std::complex<double>>> bluesteinFFT2D(
    const std::vector<std::vector<std::complex<double>>>& input);

std::vector<std::vector<std::complex<double>>> bluesteinIFFT2D(
    const std::vector<std::vector<std::complex<double>>>& input);    

//Untils
std::vector<std::vector<std::complex<double>>> fft2DShift(
    const std::vector<std::vector<std::complex<double>>>& input);   
    
std::vector<std::vector<std::complex<double>>> ifft2DShift(
    const std::vector<std::vector<std::complex<double>>>& input);

size_t nextPowerOf2(size_t n);

std::vector<double> padSignal(const std::vector<double>& input);

std::vector<double> padSignal(const std::vector<double>& input, size_t targetSize);

std::vector<std::complex<double>> unpadSignal(const std::vector<std::complex<double>>& input, size_t originalSize);

std::vector<std::vector<double>> padMatrix2D(const std::vector<std::vector<double>>& input);

std::vector<std::vector<double>> padMatrix2D(const std::vector<std::vector<double>>& input, size_t targetRows, size_t targetCols);

// --- new: complex-valued overloads, needed for padding gridded k-space before ifft2D ---
std::vector<std::vector<std::complex<double>>> padMatrix2D(
    const std::vector<std::vector<std::complex<double>>>& input);

std::vector<std::vector<std::complex<double>>> padMatrix2D(
    const std::vector<std::vector<std::complex<double>>>& input,
    size_t targetRows,
    size_t targetCols);

std::vector<std::vector<std::complex<double>>> unpadMatrix2D(const std::vector<std::vector<std::complex<double>>>& input, size_t originalRows, size_t originalCols);

//helpers

std::vector<std::complex<double>> fftRecursive(const std::vector<std::complex<double>>& samples);

