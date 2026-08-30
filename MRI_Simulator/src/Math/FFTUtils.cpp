#include "../../include/Math/FFT.hpp"


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
std::vector<std::vector<double>> padMatrix2D(const std::vector<std::vector<double>>& input, size_t targetRows, size_t targetCols)
{
    size_t rows = input.size();
    size_t cols = rows > 0 ? input[0].size() : 0;

    int xOffset = (targetCols - cols)/2;
    int yOffset = (targetRows - rows)/2;

    if (targetRows < rows || targetCols < cols) {
        std::cout << "Error: target dimensions are smaller than input dimensions, cannot pad." << std::endl;
        return input;
    }

    std::vector<std::vector<double>> padded(targetRows, std::vector<double>(targetCols, 0.0));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            padded[yOffset+i][xOffset+j] = input[i][j];
        }
    }
    return padded;
}

std::vector<std::vector<std::complex<double>>> padMatrix2D(const std::vector<std::vector<std::complex<double>>>& input,size_t targetRows, size_t targetCols)
{
    size_t rows = input.size();
    size_t cols = rows > 0 ? input[0].size() : 0;

    int xOffset = (targetCols - cols)/2;
    int yOffset = (targetRows - rows)/2;

    if (targetRows < rows || targetCols < cols) {
        std::cout << "Error: target dimensions are smaller than input dimensions, cannot pad." << std::endl;
        return input;
    }

    std::vector<std::vector<std::complex<double>>> padded(
        targetRows, std::vector<std::complex<double>>(targetCols, {0.0, 0.0}));
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            padded[yOffset+i][xOffset+j] = input[i][j];
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

    int xOffset = (cols - originalCols)/2;
    int yOffset = (rows - originalRows)/2;    

    if (originalRows > rows || originalCols > cols) {
        std::cout << "Error: original dimensions are larger than input dimensions, cannot unpad." << std::endl;
        return input;
    }

    std::vector<std::vector<std::complex<double>>> cropped(
        originalRows, std::vector<std::complex<double>>(originalCols));
    for (size_t i = 0; i < originalRows; i++) {
        for (size_t j = 0; j < originalCols; j++) {
            cropped[i][j] = input[i+yOffset][j+xOffset];
        }
    }
    return cropped;
}