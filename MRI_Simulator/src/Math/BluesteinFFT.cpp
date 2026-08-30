#include "../../include/Math/FFT.hpp"

//black magic stuff going on here
//Slower than cooleytukey, but does not have power of 2 limitation

std::vector<std::complex<double>> bluesteinFFT(const std::vector<std::complex<double>>& input){
    int N = input.size();

    int M = 1;

    while(M < 2*N-1) {
        M <<= 1;
    }

    std::vector<std::complex<double>> a(M);
    std::vector<std::complex<double>> b(M);

    //---------------------------------------------------------
    // Chirp multiplication
    //---------------------------------------------------------

    for (int n = 0; n < N; n++)
    {
        double angle = -M_PI * (double)(n * n) / N;

        std::complex<double> w(cos(angle), sin(angle));

        a[n] = input[n] * w;
    }

    //---------------------------------------------------------
    // Chirp kernel
    //---------------------------------------------------------

    for (int n = 0; n < N; n++)
    {
        double angle = M_PI * (double)(n * n) / N;

        std::complex<double> w(cos(angle), sin(angle));

        b[n] = w;

        if (n != 0) {
            b[M - n] = w;
        }
    }

    //---------------------------------------------------------
    // Convolution using CooleyTukey
    //---------------------------------------------------------

    std::vector<std::complex<double>> aFFT = fft(a);
    std::vector<std::complex<double>> bFFT = fft(b);

    for (int i = 0; i < M; i++){
        aFFT[i] *= bFFT[i];
    }

    std::vector<std::complex<double>> A = ifft(aFFT);    

    //---------------------------------------------------------
    // Final chirp
    //---------------------------------------------------------

    std::vector<std::complex<double>> result(N);

    for (int n = 0; n < N; n++)
    {
        double angle = -M_PI * (double)(n * n) / N;

        std::complex<double> w(cos(angle), sin(angle));

        result[n] = A[n] * w;

    }

    return result;

}

//for real values
std::vector<std::complex<double>> bluesteinFFT(const std::vector<double>& input) {
    //converts the real samples to complex and then returns the actual fft
    size_t N = input.size();

    std::vector<std::complex<double>> complexSamples(N);

    for(size_t i = 0; i<N; i++) {
        complexSamples[i] = std::complex<double>(input[i], 0.0);
    }

    return bluesteinFFT(complexSamples);
}

std::vector<std::complex<double>> bluesteinIFFT(const std::vector<std::complex<double>>& input){
    int N = input.size();

    int M = 1;

    while(M < 2*N-1) {
        M <<= 1;
    }

    std::vector<std::complex<double>> a(M);
    std::vector<std::complex<double>> b(M);

    //---------------------------------------------------------
    // Chirp multiplication
    //---------------------------------------------------------

    for (int n = 0; n < N; n++)
    {
        double angle = M_PI * (double)(n * n) / N;

        std::complex<double> w(cos(angle), sin(angle));

        a[n] = input[n] * w;
    }

    //---------------------------------------------------------
    // Chirp kernel
    //---------------------------------------------------------

    for (int n = 0; n < N; n++)
    {
        double angle = -M_PI * (double)(n * n) / N;

        std::complex<double> w(cos(angle), sin(angle));

        b[n] = w;

        if (n != 0) {
            b[M - n] = w;
        }
    }

    //---------------------------------------------------------
    // Convolution using CooleyTukey
    //---------------------------------------------------------

    std::vector<std::complex<double>> aFFT = fft(a);
    std::vector<std::complex<double>> bFFT = fft(b);

    for (int i = 0; i < M; i++){
        aFFT[i] *= bFFT[i];
    }

    std::vector<std::complex<double>> A = ifft(aFFT);    

    //---------------------------------------------------------
    // Final chirp
    //---------------------------------------------------------

    std::vector<std::complex<double>> result(N);

    for (int n = 0; n < N; n++)
    {
        double angle = M_PI * (double)(n * n) / N;

        std::complex<double> w(cos(angle), sin(angle));

        result[n] = A[n] * w;

        //result[n] /= N; needed?

    }

    return result;

}