#ifndef DFT_HPP
#define DFT_HPP

#include <vector>
#include <complex>

// 1D FFT / IFFT
std::vector<std::complex<double>> fft(std::vector<double> realSamples);

std::vector<std::complex<double>> ifft(std::vector<std::complex<double>> samples);

// 2D FFT / IFFT (row-column decomposition)
std::vector<std::vector<std::complex<double>>> transpose(
    const std::vector<std::vector<std::complex<double>>>& matrix);

std::vector<std::vector<std::complex<double>>> fft2D(
    const std::vector<std::vector<double>>& input);

std::vector<std::vector<std::complex<double>>> ifft2D(
    const std::vector<std::vector<std::complex<double>>>& input);

std::vector<std::vector<std::complex<double>>> fft2DShift(
    const std::vector<std::vector<std::complex<double>>>& input);   
    
std::vector<std::vector<std::complex<double>>> ifft2DShift(
    const std::vector<std::vector<std::complex<double>>>& input);

#endif

