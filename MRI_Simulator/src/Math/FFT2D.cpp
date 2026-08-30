#include "../../include/Math/FFT.hpp"


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

//==========================================================
// 2D Bluestein FFT (works for arbitrary matrix dimensions)
//==========================================================
std::vector<std::vector<std::complex<double>>> bluesteinFFT2D(
    const std::vector<std::vector<double>>& input)
{
    size_t rows = input.size();

    // FFT each row
    std::vector<std::vector<std::complex<double>>> rowResult(rows);
    for(size_t i = 0; i < rows; i++)
    {
        rowResult[i] = bluesteinFFT(input[i]);
    }

    // Transpose
    auto transposed = transpose(rowResult);

    // FFT each column
    for(size_t i = 0; i < transposed.size(); i++)
    {
        transposed[i] = bluesteinFFT(transposed[i]);
    }

    // Transpose back
    return transpose(transposed);
}

//==========================================================
// 2D Bluestein FFT (complex input)
//==========================================================
std::vector<std::vector<std::complex<double>>> bluesteinFFT2D(
    const std::vector<std::vector<std::complex<double>>>& input)
{
    std::vector<std::vector<std::complex<double>>> data = input;

    // FFT rows
    for(size_t i = 0; i < data.size(); i++)
    {
        data[i] = bluesteinFFT(data[i]);
    }

    // FFT columns
    data = transpose(data);

    for(size_t i = 0; i < data.size(); i++)
    {
        data[i] = bluesteinFFT(data[i]);
    }

    return transpose(data);
}

//==========================================================
// 2D Bluestein IFFT
//==========================================================
std::vector<std::vector<std::complex<double>>> bluesteinIFFT2D(
    const std::vector<std::vector<std::complex<double>>>& input)
{
    std::vector<std::vector<std::complex<double>>> data = input;

    // IFFT rows
    for(size_t i = 0; i < data.size(); i++)
    {
        data[i] = bluesteinIFFT(data[i]);
    }

    // IFFT columns
    data = transpose(data);

    for(size_t i = 0; i < data.size(); i++)
    {
        data[i] = bluesteinIFFT(data[i]);
    }

    return transpose(data);
}