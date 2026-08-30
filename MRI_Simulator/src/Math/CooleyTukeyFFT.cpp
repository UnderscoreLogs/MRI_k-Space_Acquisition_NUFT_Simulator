#include "../../include/Math/FFT.hpp"

//black magic stuff going on here

std::vector<std::complex<double>> fftRecursive(const std::vector<std::complex<double>>& samples){
    
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

std::vector<std::complex<double>> fft(const std::vector<double>& realSamples) {

    //converts the real samples to complex and then returns the actual fft
    size_t N = realSamples.size();

    std::vector<std::complex<double>> complexSamples(N);

    for(size_t i = 0; i<N; i++) {
        complexSamples[i] += realSamples[i];
    }

    return fftRecursive(complexSamples);
}

std::vector<std::complex<double>> fft(const std::vector<std::complex<double>>& samples) {
    return fftRecursive(samples);
}

std::vector<std::complex<double>> ifftRecursive(const std::vector<std::complex<double>>& samples){
    
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

std::vector<std::complex<double>> ifft(const std::vector<std::complex<double>>& samples) {

    size_t N = samples.size();

    std::vector<std::complex<double>> x = ifftRecursive(samples);

    for(size_t i = 0; i<N; i++) {
        x[i] /= (double)N;
    }

    return x;
}