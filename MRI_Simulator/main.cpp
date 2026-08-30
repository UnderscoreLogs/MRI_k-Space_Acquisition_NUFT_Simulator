
#include "include/Math/FFT.hpp"
#include "include/MRI/NUDFT.hpp"
#include "include/MRI/Trajectories.hpp"
#include "include/Imaging/Metrics.hpp"
#include "include/Imaging/ImageUtils.hpp"
#include "include/Imaging/niftiReader.hpp"
#include <fstream>

using namespace std;

vector<vector<double>> getSlice(string fileName, int sliceIndex) {
    
    NiftiHeader header;

    vector<double> image = loadNifti(fileName, header);

    int nx = header.dim[1]; 
    int ny = header.dim[2]; 
    int nz = header.dim[3];

    cout << "\nExtracting slice " << sliceIndex << " of " << nz - 1 << endl;

    return normalizeImage(extractSlice(image, nx, ny, nz, sliceIndex), 0.0, 255.0);
}

void testing(const vector<vector<double>>& img, const vector<vector<double>>& img2, int xRes, int yRes, int numOfSamples, string folder) {
    
    string imgName = "../standard_test_images/lena_gray_256_WithDCF_25_spiral.png";
    double acceleration = ((double)yRes*(double)xRes)/numOfSamples;

    //vector<vector<double>> img = getSlice(imgName, 100); // for *.nii files
    vector<vector<double>> img_compare = getImage(imgName);

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    auto Rad_double = normalizeImage(mriSpiralInterpolation(img, xRes, yRes, numOfSamples, 4, 32, "INUFFT", 30.0, 100, true), 0.0, 255.0);
    
    // Stop timer
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time
    auto RadDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    cout << "Time: " << RadDuration.count() << "ms" << endl;
    cout << "SSIM: " << calcSSIM(Rad_double, img_compare) << endl;
    cout << "PSNR: " << calcPSNR(Rad_double, img_compare) << endl;

    saveImg(Rad_double, "Cartesian", folder);

    cout << "Program Finished." << endl;
}

void loopTesting(const vector<vector<double>>& img, const vector<vector<double>>& img2, int xRes, int yRes, int numOfSamples, string folder) {
//----------------------------------------------- Testing -----------------------------------------------------------
    int maxRadius = max(xRes,yRes)/2;
    cout << "Number of samples = " << numOfSamples << endl;
    bool printFirstLine = true;
    double acceleration = ((double)yRes*(double)xRes)/numOfSamples;

    for(int gamma = 131072; gamma <= 131072; gamma += 131072) {
        for(int beta = 5; beta <= 5; beta++) {
            for(int alpha = 0; alpha <= 4; alpha++) {
                switch (alpha)
                {
                case 0:
                    numOfSamples = 65536*0.1;
                    break;
                case 1:
                    numOfSamples = 65536*0.25;
                    break;
                case 2:
                    numOfSamples = 65536*0.5;
                    break;
                case 3:
                    numOfSamples = 65536*0.75;
                    break;
                case 4:
                    numOfSamples = 65536*1.0;
                    break;
                default:
                    break;
                }
                
                double alphaD = alpha * 0.5;
                double gammaD = gamma * 0.5;
                double betaD = beta * 0.05 + 0.01;


                // Start timer
                auto start = std::chrono::high_resolution_clock::now();

                auto reconImg_double = mriRadialInterpolation(img, xRes, yRes, round(numOfSamples/xRes), xRes, "INUFFT", 10000.0, 100.0, false);
                auto reconImg_norm = normalizeImage(reconImg_double, 0.0, 255.0);

                // Stop timer
                auto end = std::chrono::high_resolution_clock::now();

                // Calculate elapsed time
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);  

                double psnr = calcPSNR(reconImg_norm, img);
                double ssim = calcSSIM(reconImg_norm, img);

                cout << "Alpha = " << alpha << " finished in " << duration.count() << " ms" << endl;
                cout << "Alpha = " << alpha << " PSNR: " << psnr << endl;
                cout << "Alpha = " << alpha << " SSIM: " << ssim << endl << endl;
                saveImg(reconImg_norm, "VDC_Alpha" + to_string(alpha) + 
                                        " -PSNR_" + to_string(psnr)+ 
                                        " -SSIM_" + to_string(ssim) + 
                                        "-Time_" + to_string(duration.count()) + "ms"
                                        , folder);

                ofstream file(folder + "/data.txt", ios::app);

                if (!file.is_open()) {
                    cerr << "Could not open data.txt\n";
                    return;
                }

                if(printFirstLine) {
                    file << "Alpha" << '\t' << "PSNR" << '\t' << "SSIM" << '\t' << "Time" << '\t' << "# Samples" << endl;
                    printFirstLine = false;
                }

                file << alpha << '\t' << psnr << '\t' << ssim << '\t' << duration.count() << '\t' << numOfSamples << endl;

                /*file << "Num of Samples = \t" << numOfSamples << '\n';

                file << "Wraps = \t" << alpha << '\t'
                    << " Interleafs = \t" << beta<< '\t'
                    << " finished in \t" << duration.count() << "\t ms\n";

                file << "Wraps = \t" << alpha << '\t'
                    << " Interleafs = \t" << beta << '\t'                 
                    << " PSNR: \t" << psnr << '\n';

                file << "Wraps = \t" << alpha << '\t'
                    << " Interleafs = \t" << beta << '\t'                  
                    << " SSIM: \t" << ssim << "\n\n";*/

                file.close();                         
            }
        }
    }
}

void spiralLoopTesting(const vector<vector<double>>& img, const vector<vector<double>>& img2, int xRes, int yRes, int numOfSamples, string folder) {
//----------------------------------------------- Testing -----------------------------------------------------------
    int maxRadius = max(xRes,yRes)/2;
    cout << "Number of samples = " << numOfSamples << endl;
    bool printFirstLine = true;

    for(int gamma = 131072; gamma <= 131072; gamma += 131072) {
        for(int beta = 1; beta <= 1024; beta = beta * 2) {
            for(int alpha = 1; alpha <= 1024; alpha = alpha * 2) {
                //numOfSamples = gamma;
                int alphaD = alpha * 0.5;
                int gammaD = gamma * 0.5;
                int betaD = beta * 0.05 + 0.01;


                // Start timer
                auto start = std::chrono::high_resolution_clock::now();

                auto reconImg_double = mriSpiralInterpolation(img, xRes, yRes, numOfSamples, alpha, beta, "INUFFT",  10000.0, 100.0, false);
                auto reconImg_norm = normalizeImage(reconImg_double, 0.0, 255.0);

                // Stop timer
                auto end = std::chrono::high_resolution_clock::now();

                // Calculate elapsed time
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

                double psnr = calcPSNR(reconImg_norm, img);
                double ssim = calcSSIM(reconImg_norm, img);

                cout << "Num Warps = " << alpha << " Interleafs = " << beta << " finished in " << duration.count() << " ms" << endl;
                cout << "Num Warps = " << alpha << " Interleafs = " << beta << " PSNR: " << psnr << endl;
                cout << "Num Warps = " << alpha << " Interleafs = " << beta << " SSIM: " << ssim << endl << endl;
                saveImg(reconImg_norm, "Spiral_Wraps" + to_string(alpha) + 
                                        "_Interleafs" + to_string(beta) + 
                                        " -PSNR_" + to_string(psnr)+ 
                                        " -SSIM_" + to_string(ssim) + 
                                        "-Time_" + to_string(duration.count()) + "ms"
                                        , folder);

                ofstream file(folder + "/data.txt", ios::app);

                if (!file.is_open()) {
                    cerr << "Could not open data.txt\n";
                    return;
                }

                if(printFirstLine) {
                    file << "# Wraps" << '\t' << "# Interleafs" << '\t' << "PSNR" << '\t' << "SSIM" << '\t' << "Time" << '\t' << "# Samples" << endl;
                    printFirstLine = false;
                }

                file << alpha << '\t' << beta << '\t' << psnr << '\t' << ssim << '\t' << duration.count() << '\t' << numOfSamples << endl;

                /*file << "Num of Samples = \t" << numOfSamples << '\n';

                file << "Wraps = \t" << alpha << '\t'
                    << " Interleafs = \t" << beta<< '\t'
                    << " finished in \t" << duration.count() << "\t ms\n";

                file << "Wraps = \t" << alpha << '\t'
                    << " Interleafs = \t" << beta << '\t'                 
                    << " PSNR: \t" << psnr << '\n';

                file << "Wraps = \t" << alpha << '\t'
                    << " Interleafs = \t" << beta << '\t'                  
                    << " SSIM: \t" << ssim << "\n\n";*/

                file.close();                         
            }
        }
    }
}

void runAll(const vector<vector<double>>& img, const vector<vector<double>>& img2, int xRes, int yRes, int numOfSamples, string folder) {
    
    double SNR = 10000.0;
    double timePercent = 100.0;

    int Og_yRes = img.size();
    int Og_xRes = img[0].size();

    double acceleration = ((double)yRes*(double)xRes)/numOfSamples;

    //----------------------------------- VDR ---------------------------------------------------
    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    auto VDR_double = normalizeImage(mriVariableDensityRandomInterpolation(img, xRes, yRes, 1.0, 0.16, acceleration, "INUFFT", SNR, timePercent, false), 0.0, 255.0);
    
    // Stop timer
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time
    auto VDRduration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    //----------------------------------- Cartesian ---------------------------------------------------
    // Start timer
    start = std::chrono::high_resolution_clock::now();

    auto Cart_double = normalizeImage(mriCartesianInterpolation(img, xRes, yRes, round(sqrt(numOfSamples)), round(sqrt(numOfSamples)), "INUFFT", SNR, timePercent, false), 0.0, 255.0);
    
    // Stop timer
    end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time
    auto CarteDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    //----------------------------------- Radial ---------------------------------------------------//
    // Start timer
    start = std::chrono::high_resolution_clock::now();

    int maxDiameter = max(xRes, yRes);

    auto Rad_double = normalizeImage(mriRadialInterpolation(img, xRes, yRes, round(numOfSamples/maxDiameter), maxDiameter, "INUFFT", SNR, timePercent, false), 0.0, 255.0);
    
    // Stop timer
    end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time
    auto RadDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    //----------------------------------- Random ---------------------------------------------------
    // Start timer
    start = std::chrono::high_resolution_clock::now();

    auto Rand_double = normalizeImage(mriRandomInterpolation(img, xRes, yRes, numOfSamples, "INUFFT", SNR, timePercent, false), 0.0, 255.0);
    
    // Stop timer
    end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time
    auto RandDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    //----------------------------------- Spiral ---------------------------------------------------//
    // Start timer
    start = std::chrono::high_resolution_clock::now();

    auto Spir_double = normalizeImage(mriSpiralInterpolation(img, xRes, yRes, numOfSamples, 4, 32, "INUFFT", SNR, timePercent, false), 0.0, 255.0);
    
    // Stop timer
    end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time
    auto SpirDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    //----------------------------------- VDC ---------------------------------------------------
    // Start timer
    start = std::chrono::high_resolution_clock::now();

    auto VDC_double = normalizeImage(mriVariableDensityCartesianInterpolation(img, xRes, yRes, xRes, yRes, 1.0, 0.16, acceleration, "INUFFT", SNR, timePercent, false), 0.0, 255.0);
    
    // Stop timer
    end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time
    auto VDCduration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    //------------------------------------ Save & Display Images -----------------------------------------------//
    cout << "      VDR PSNR: " <<         calcPSNR(VDR_double, img2) << endl;
    cout << "Cartesian PSNR: " <<   calcPSNR(Cart_double, img2) << endl;
    cout << "   Radial PSNR: " <<      calcPSNR(Rad_double, img2) << endl;
    cout << "   Random PSNR: " <<      calcPSNR(Rand_double, img2) << endl;
    cout << "   Spiral PSNR: " <<      calcPSNR(Spir_double, img2) << endl;
    cout << "      VDC PSNR: " <<         calcPSNR(VDC_double, img2) << endl << endl;    
    
    cout << "      VDR SSIM: " <<         calcSSIM(VDR_double, img2) << endl;
    cout << "Cartesian SSIM: " <<   calcSSIM(Cart_double, img2) << endl;
    cout << "   Radial SSIM: " <<      calcSSIM(Rad_double, img2) << endl;
    cout << "   Random SSIM: " <<      calcSSIM(Rand_double, img2) << endl;
    cout << "   Spiral SSIM: " <<      calcSSIM(Spir_double, img2) << endl;
    cout << "      VDC SSIM: " <<         calcSSIM(VDC_double, img2) << endl << endl;   

    cout << "      VDR finished in: " << VDRduration.count() << " ms" << endl;
    cout << "Cartesian finished in: " << CarteDuration.count() << " ms" << endl;
    cout << "   Radial finished in: " << RadDuration.count() << " ms" << endl;
    cout << "   Random finished in: " << RandDuration.count() << " ms" << endl;
    cout << "   Spiral finished in: " << SpirDuration.count() << " ms" << endl;
    cout << "      VDC finished in: " << VDCduration.count() << " ms" << endl;

    saveImg(VDR_double, "Variable-Density-Random", folder);

    saveImg(Cart_double, "Cartesian", folder);

    saveImg(Rad_double, "Radial", folder);

    saveImg(Rand_double, "Random", folder);

    saveImg(Spir_double, "Spiral", folder);

    saveImg(VDC_double, "Variable-Density-Cartesian", folder);

        /*auto printStats = [](const vector<vector<double>> &img, const string &name)
    {
        double mn = img[0][0], mx = img[0][0], sum = 0;
        int N = 0;
        for (auto &row : img)
            for (double v : row)
            {
                mn = min(mn, v);
                mx = max(mx, v);
                sum += v;
                N++;
            }
        cout << name << " -> min: " << mn << " max: " << mx << " mean: " << sum / N << endl;
    };

    printStats(VDR_double, "VDR");
    printStats(Cart_double, "Cartesian");
    printStats(Rad_double, "Radial");
    printStats(Rand_double, "Random");
    printStats(Spir_double, "Spiral");
    printStats(VDC_double, "VDC");
    printStats(img2, "Original");*/

    ofstream file(folder + "/data.txt", ios::app);

    if (!file.is_open())
    {
        cerr << "Could not open data.txt\n";
        return;
    }

    file << "      VDR PSNR: \t" <<         calcPSNR(VDR_double, img2) << endl;
    file << "Cartesian PSNR: \t" <<   calcPSNR(Cart_double, img2) << endl;
    file << "   Radial PSNR: \t" <<      calcPSNR(Rad_double, img2) << endl;
    file << "   Random PSNR: \t" <<      calcPSNR(Rand_double, img2) << endl;
    file << "   Spiral PSNR: \t" <<      calcPSNR(Spir_double, img2) << endl;
    file << "      VDC PSNR: \t" <<         calcPSNR(VDC_double, img2) << endl << endl;    
    
    file << "      VDR SSIM: \t" <<         calcSSIM(VDR_double, img2) << endl;
    file << "Cartesian SSIM: \t" <<   calcSSIM(Cart_double, img2) << endl;
    file << "   Radial SSIM: \t" <<      calcSSIM(Rad_double, img2) << endl;
    file << "   Random SSIM: \t" <<      calcSSIM(Rand_double, img2) << endl;
    file << "   Spiral SSIM: \t" <<      calcSSIM(Spir_double, img2) << endl;
    file << "      VDC SSIM: \t" <<         calcSSIM(VDC_double, img2) << endl << endl;   

    file << "      VDR finished in: \t" << VDRduration.count() << "\t ms" << endl;
    file << "Cartesian finished in: \t" << CarteDuration.count() << "\t ms" << endl;
    file << "   Radial finished in: \t" << RadDuration.count() << "\t ms" << endl;
    file << "   Random finished in: \t" << RandDuration.count() << "\t ms" << endl;
    file << "   Spiral finished in: \t" << SpirDuration.count() << "\t ms" << endl;
    file << "      VDC finished in: \t" << VDCduration.count() << "\t ms" << endl;

    file.close();
}

void averageRun(const string imgName, int xRes, int yRes, int numOfSamples, string folder) {

    string trajectory = "Spiral";

    double ssimAvg = 0.0;
    double psnrAvg = 0.0;
    double timeAvg = 0.0;
    int numRuns = 0;
    int numRunsPSNR = 0;

    for(int i = 0; i < 186; i += 5) {
        vector<vector<double>> img = getSlice(imgName, i);
        int xRes = img[0].size();
        int yRes = img.size();

            // Start timer
        auto start = std::chrono::high_resolution_clock::now();

        int maxRadius = max(xRes/2, yRes/2);

        auto recon_double = normalizeImage(mriVariableDensityCartesianInterpolation(img, xRes, yRes, xRes, yRes, 1, 0.10, ((xRes*yRes)/numOfSamples), "INUFFT", 10000.0, 100.0, false), 0.0, 255.0);
        
        // Stop timer
        auto end = std::chrono::high_resolution_clock::now();

        // Calculate elapsed time
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);


        double psnr = calcPSNR(recon_double, img);
        if(isfinite(psnr)) {
            psnrAvg += psnr;
            numRunsPSNR++;
        }

        ssimAvg += calcSSIM(recon_double, img);
        timeAvg += duration.count();
        numRuns++;

        cout << psnr << endl;

        saveImg(recon_double, "MRI_Brain-Slice" + to_string(i) + "_" + trajectory, folder);
    }

    ssimAvg = ssimAvg / (double) numRuns;
    psnrAvg = psnrAvg / (double) numRunsPSNR;
    timeAvg = timeAvg / (double) numRuns;

    cout << "File Name: " << imgName << endl;
    cout << "Trajectory: " << trajectory << endl;
    cout << "xRes: " << xRes << endl;
    cout << "yRes: " << yRes << endl;
    cout << "#Samples: " << numOfSamples << endl;
    
    cout << "Average SSIM: " << ssimAvg << endl;
    cout << "Average PSNR: " << psnrAvg << endl;
    cout << "Average Time: " << timeAvg << endl;

    ofstream file(folder + "/data.txt", ios::app);

    if (!file.is_open())
    {
        cerr << "Could not open data.txt\n";
        return;
    }

    file << "File Name: " << imgName << endl;
    file << "Trajectory: " << trajectory << endl;
    file << "xRes: " << xRes << endl;
    file << "yRes: " << yRes << endl;
    file << "#Samples:" << numOfSamples << endl;

    file << "Average SSIM: " << ssimAvg << endl;
    file << "Average PSNR: " << psnrAvg << endl;
    file << "Average Time: " << timeAvg << endl;

    file.close();

}
int main()
{
    //------------------------Settings-------------------------------------------------------------------------------------------------
    //string imgName = "../OpenNeuroDatasets/sub-0117_run-2_T1w.nii";
    //string imgName = "../FastMRIDatasets/FastMRIKneeSlice20.png";
    string imgName = "../standard_test_images/lena_gray_256.tif";

    //vector<vector<double>> img = getSlice(imgName, 100); // for *.nii files
    vector<vector<double>> img = getImage(imgName);

    int xRes = img[0].size();
    int yRes = img.size();

    for(int i = 3; i < 4; i++) {
        double underSamplePercent = 0.25;//((double)i * 0.25) + 0.25;

        int numOfSamples = round(xRes*yRes*underSamplePercent);

        //vector<vector<double>> img2 = getImage(imgName2);
        
        namespace fs = filesystem;

        int run = 1;
        while (fs::exists("Results/Run" + std::to_string(run))) {
            run++;
        }

        string folder = "Results/Run" + to_string(run);
        fs::create_directories(folder);

        testing(img, img, xRes, yRes, numOfSamples, folder);

        //averageRun(imgName, xRes, yRes, numOfSamples, folder);
    }

    cout << "Program Finished." << endl;

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
