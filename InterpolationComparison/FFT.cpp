#include "FFT.hpp"
#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <complex> 
#include <vector>
#include <sstream>
#include <string>

#include <math.h>

//black magic stuff going on here

std::vector<std::complex<double>> fftRecursive(std::vector<std::complex<double>> samples){
    
    size_t N = samples.size();

    //cout << "FFT called with N = " << N << endl; //debug for checking length of the called vector

    size_t M = N/2; //half the size of samples

    if(N <= 0 || (N & (N-1)) != 0) { //checks dataset if it is a power of 2, if it is not then it gives an errror
        std::cout << "Error: Dataset is not a power of 2." << std::endl;
        return {};
    }

    if(N <= 1) { //preventing something..?
        return samples;
    }

    std::vector<std::complex<double>> even(M);
    std::vector<std::complex<double>> odd(M);

    for(size_t i = 0; i < N/2; i++){
        even[i] = samples[2*i];
        odd[i] = samples[2*i+1];
    }

    even = fftRecursive(even);
    odd = fftRecursive(odd);

    std::vector<std::complex<double>> fftOutput(N);

    for(size_t k = 0; k<M; k++) {

        double angle = (-2*M_PI*k)/N; //calculating the angle of the DFT equation -(2pi/N)*k*n

        std::complex<double> expTerm = exp(std::complex<double>(0,angle));     

        std::complex<double> twiddleOdd = odd[k] * expTerm;

        fftOutput[k] = even[k] + twiddleOdd;
        fftOutput[k+M] = even[k] - twiddleOdd;
    }
    return fftOutput;
}

std::vector<std::complex<double>> fft(std::vector<double> realSamples) {

    //converts the real samples to complex and then returns the actual fft
    size_t N = realSamples.size();

    std::vector<std::complex<double>> complexSamples(N);

    for(size_t i = 0; i<N; i++) {
        complexSamples[i] += realSamples[i];
    }

    return fftRecursive(complexSamples);
}

std::vector<std::complex<double>> ifftRecursive(std::vector<std::complex<double>> samples){
    
    size_t N = samples.size();

    //cout << "FFT called with N = " << N << endl; //debug for checking length of the called vector

    size_t M = N/2; //half the size of samples

    if(N <= 0 || (N & (N-1)) != 0) { //checks dataset if it is a power of 2, if it is not then it gives an errror
        std::cout << "Error: Dataset is not a power of 2." << std::endl;
        return {};
    }

    if(N <= 1) { //preventing something..?
        return samples;
    }

    std::vector<std::complex<double>> even(M);
    std::vector<std::complex<double>> odd(M);

    for(size_t i = 0; i < N/2; i++){
        even[i] = samples[2*i];
        odd[i] = samples[2*i+1];
    }

    even = ifftRecursive(even);
    odd = ifftRecursive(odd);

    std::vector<std::complex<double>> fftOutput(N);

    for(size_t k = 0; k<M; k++) {

        double angle = (2*M_PI*k)/N; //calculating the angle of the DFT equation -(2pi/N)*k*n

        std::complex<double> expTerm = exp(std::complex<double>(0,angle));     

        std::complex<double> twiddleOdd = odd[k] * expTerm;

        fftOutput[k] = (even[k] + twiddleOdd);
        fftOutput[k+M] = (even[k] - twiddleOdd);
    }
    return fftOutput;
}

std::vector<std::complex<double>> ifft(std::vector<std::complex<double>> samples) {

    size_t N = samples.size();

    std::vector<std::complex<double>> x = ifftRecursive(samples);

    for(size_t i = 0; i<N; i++) {
        x[i] /= (double)N;
    }

    return x;
}

// Im too lazy to code all this so this is all Claude

// --- add these to FFT.hpp / FFT.cpp ---

// Transposes a 2D complex matrix (rows x cols -> cols x rows)
std::vector<std::vector<std::complex<double>>> transpose(
    const std::vector<std::vector<std::complex<double>>>& matrix)
{
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    std::vector<std::vector<std::complex<double>>> result(cols, std::vector<std::complex<double>>(rows));

    for(size_t i = 0; i < rows; i++) {
        for(size_t j = 0; j < cols; j++) {
            result[j][i] = matrix[i][j];
        }
    }

    return result;
}

// 2D FFT of a real-valued matrix. Both dimensions must be powers of 2.
std::vector<std::vector<std::complex<double>>> fft2D(
    const std::vector<std::vector<double>>& input)
{
    size_t rows = input.size();

    // Step 1: FFT every row (real -> complex)
    std::vector<std::vector<std::complex<double>>> rowResult(rows);
    for(size_t i = 0; i < rows; i++) {
        rowResult[i] = fft(input[i]);
    }

    // Step 2: transpose so columns become rows
    std::vector<std::vector<std::complex<double>>> transposed = transpose(rowResult);

    // Step 3: FFT every (former) column
    for(size_t i = 0; i < transposed.size(); i++) {
        transposed[i] = fftRecursive(transposed[i]);
    }

    // Step 4: transpose back to original orientation
    return transpose(transposed);
}

// Inverse 2D FFT. Returns a complex matrix; take .real() per element if you
// know the original data was real-valued.
std::vector<std::vector<std::complex<double>>> ifft2D(
    const std::vector<std::vector<std::complex<double>>>& input)
{
    size_t rows = input.size();

    // Step 1: IFFT every row
    std::vector<std::vector<std::complex<double>>> rowResult(rows);
    for(size_t i = 0; i < rows; i++) {
        rowResult[i] = ifft(input[i]);
    }

    // Step 2: transpose
    std::vector<std::vector<std::complex<double>>> transposed = transpose(rowResult);

    // Step 3: IFFT every (former) column
    for(size_t i = 0; i < transposed.size(); i++) {
        transposed[i] = ifft(transposed[i]);
    }

    // Step 4: transpose back
    return transpose(transposed);
}

std::vector<std::vector<std::complex<double>>> fft2DShift(
    const std::vector<std::vector<std::complex<double>>>& input)
{
    int rows = input.size();
    int cols = input[0].size();

    std::vector<std::vector<std::complex<double>>> output(
        rows,
        std::vector<std::complex<double>>(cols)
    );

    int halfRows = rows / 2;
    int halfCols = cols / 2;

    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < cols; x++)
        {
            int newY = (y + halfRows) % rows;
            int newX = (x + halfCols) % cols;

            output[newY][newX] = input[y][x];
        }
    }

    return output;
}

std::vector<std::vector<std::complex<double>>> ifft2DShift(
    const std::vector<std::vector<std::complex<double>>>& input)
{
    return fft2DShift(input);
}