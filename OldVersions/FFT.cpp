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

// --- Zero-padding utilities ---
// fftRecursive/fft/fft2D all require power-of-2 dimensions, otherwise they
// print an error and hand back an empty vector. These helpers let you pad
// an arbitrary-length real signal (or MxN real matrix) up to the next
// power of 2 with zeros before transforming, and crop the result back down
// to the original size afterward.

// Returns the smallest power of 2 that is >= n. nextPowerOf2(0) == 1.
size_t nextPowerOf2(size_t n) {
    if (n <= 1) return 1;

    size_t power = 1;
    while (power < n) {
        power <<= 1;
    }
    return power;
}

// Zero-pads a 1D real signal up to the next power of 2 above its current size.
// e.g. a 300-sample signal becomes 512 samples, with the extra 212 set to 0.
std::vector<double> padSignal(const std::vector<double>& input) {
    return padSignal(input, nextPowerOf2(input.size()));
}

// Zero-pads a 1D real signal up to an explicit targetSize (must be >= input size,
// and should itself be a power of 2 if you intend to run fft() on the result).
std::vector<double> padSignal(const std::vector<double>& input, size_t targetSize) {
    size_t N = input.size();

    if (targetSize < N) {
        std::cout << "Error: target size is smaller than input size, cannot pad." << std::endl;
        return input;
    }

    std::vector<double> padded(targetSize, 0.0); // zero-filled by default
    for (size_t i = 0; i < N; i++) {
        padded[i] = input[i];
    }
    return padded;
}

// Crops a 1D complex result (e.g. the output of ifft()) back down to
// originalSize, discarding the padded tail. Use this after fft()/ifft() on
// a signal you padded with padSignal(), to get back to the original length.
std::vector<std::complex<double>> unpadSignal(
    const std::vector<std::complex<double>>& input, size_t originalSize)
{
    if (originalSize > input.size()) {
        std::cout << "Error: originalSize is larger than input size, cannot unpad." << std::endl;
        return input;
    }
    return std::vector<std::complex<double>>(input.begin(), input.begin() + originalSize);
}

// Zero-pads a 2D real matrix so both dimensions become powers of 2 (each
// dimension pads independently to its own next power of 2). Handy before
// calling fft2D() on an image/matrix whose size isn't already a power of 2.
std::vector<std::vector<double>> padMatrix2D(const std::vector<std::vector<double>>& input) {
    size_t rows = input.size();
    size_t cols = rows > 0 ? input[0].size() : 0;
    return padMatrix2D(input, nextPowerOf2(rows), nextPowerOf2(cols));
}

// Zero-pads a 2D real matrix up to explicit targetRows x targetCols
// (top-left aligned; new cells default-init to 0.0).
std::vector<std::vector<double>> padMatrix2D(
    const std::vector<std::vector<double>>& input,
    size_t targetRows,
    size_t targetCols)
{
    size_t rows = input.size();
    size_t cols = rows > 0 ? input[0].size() : 0;

    if (targetRows < rows || targetCols < cols) {
        std::cout << "Error: target dimensions are smaller than input dimensions, cannot pad." << std::endl;
        return input;
    }

    std::vector<std::vector<double>> padded(targetRows, std::vector<double>(targetCols, 0.0));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            padded[i][j] = input[i][j];
        }
    }
    return padded;
}

std::vector<std::vector<std::complex<double>>> padMatrix2D(
    const std::vector<std::vector<std::complex<double>>>& input,
    size_t targetRows,
    size_t targetCols)
{
    size_t rows = input.size();
    size_t cols = rows > 0 ? input[0].size() : 0;

    if (targetRows < rows || targetCols < cols) {
        std::cout << "Error: target dimensions are smaller than input dimensions, cannot pad." << std::endl;
        return input;
    }

    std::vector<std::vector<std::complex<double>>> padded(
        targetRows, std::vector<std::complex<double>>(targetCols, {0.0, 0.0}));
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            padded[i][j] = input[i][j];
    return padded;
}

std::vector<std::vector<std::complex<double>>> padMatrix2D(
    const std::vector<std::vector<std::complex<double>>>& input)
{
    size_t rows = input.size();
    size_t cols = rows > 0 ? input[0].size() : 0;
    return padMatrix2D(input, nextPowerOf2(rows), nextPowerOf2(cols));
}

// Crops a 2D complex result (e.g. the output of ifft2D()) back down to
// originalRows x originalCols, keeping the top-left block. Use this after
// fft2D()/ifft2D() on a matrix you padded with padMatrix2D().
std::vector<std::vector<std::complex<double>>> unpadMatrix2D(
    const std::vector<std::vector<std::complex<double>>>& input,
    size_t originalRows,
    size_t originalCols)
{
    size_t rows = input.size();
    size_t cols = rows > 0 ? input[0].size() : 0;

    if (originalRows > rows || originalCols > cols) {
        std::cout << "Error: original dimensions are larger than input dimensions, cannot unpad." << std::endl;
        return input;
    }

    std::vector<std::vector<std::complex<double>>> cropped(
        originalRows, std::vector<std::complex<double>>(originalCols));
    for (size_t i = 0; i < originalRows; i++) {
        for (size_t j = 0; j < originalCols; j++) {
            cropped[i][j] = input[i][j];
        }
    }
    return cropped;
}