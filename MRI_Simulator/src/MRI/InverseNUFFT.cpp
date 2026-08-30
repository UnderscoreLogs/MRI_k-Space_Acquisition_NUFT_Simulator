#include "../../include/MRI/NUDFT.hpp"

using namespace std;

double besselI0(double x)
{
    double ax = std::fabs(x);

    if (ax < 3.75)
    {
        // Polynomial fit in t = (x/3.75)^2
        double t = x / 3.75;
        double t2 = t * t;

        return 1.0 + t2 * (3.5156229 + t2 * (3.0899424 + t2 * (1.2067492
               + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813)))));
    } else {
        // Polynomial fit in 1/ax, scaled by e^ax / sqrt(ax)
        double t = 3.75 / ax;

        double poly = 0.39894228 + t * (0.01328592 + t * (0.00225319
                    + t * (-0.00157565 + t * (0.00916281 + t * (-0.02057706
                    + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377)))))));

        return (exp(ax) / sqrt(ax)) * poly;
    }
}

double kaiserBesselBeta(double W, double osf) {
    double beta = M_PI*sqrt( ((W/osf)*(W/osf)) * ((osf-0.5)*(osf-0.5)) - 0.8);
    return beta;
}

double kaiserBesselKernel(double kappa, double W, double beta) {
    double halfW = W/2.0;

    if(fabs(kappa) > halfW) {
        return 0.0;
    }

    double arg = beta * sqrt(1.0 - (2.0*kappa/W) * (2.0*kappa/W));
    return besselI0(arg) / besselI0(beta);
}


double kaiserBesselKernel2D(double dx, double dy, double W, double beta) {
    double r = sqrt(dx*dx + dy*dy);
    double halfW = W / 2.0;
    if (r > halfW) return 0.0;
    double arg = beta * sqrt(1.0 - (2.0*r/W)*(2.0*r/W));
    return besselI0(arg) / besselI0(beta);
}

vector<vector<complex<double>>> griddingTransform(const vector<kSample>& kSamples, int xSize, int ySize, double kxMax, double kyMax, double beta, double W){
    int numOfSamples = kSamples.size();

    vector<vector<complex<double>>> grid(ySize, vector<complex<double>>(xSize, 0.0));

    double gridScaleX = (xSize-1) / (2.0 * kxMax);
    double gridScaleY = (ySize-1) / (2.0 * kyMax);

    for (int s = 0; s < numOfSamples; s++)
    {
        complex<double> value = kSamples[s].value;

        // convert this sample's continuous k-space coords to fractional grid coords
        double gx = kSamples[s].kx * gridScaleX + xSize / 2.0;
        double gy = kSamples[s].ky * gridScaleY + ySize / 2.0;

        // only touch grid points within the kernel's support (W/2), not the whole grid
        int xMin = max(0, (int)ceil(gx - W/2.0));
        int xMax = min(xSize - 1, (int)floor(gx + W/2.0));
        int yMin = max(0, (int)ceil(gy - W/2.0));
        int yMax = min(ySize - 1, (int)floor(gy + W/2.0));

        for (int y = yMin; y <= yMax; y++)
        {
            double wy = kaiserBesselKernel(y - gy, W, beta);
            for (int x = xMin; x <= xMax; x++)
            {
                double wx = kaiserBesselKernel(x - gx, W, beta);
                grid[y][x] += value * (wx * wy);  // separable 2D kernel
            }
        }
    }

    return grid;
}

vector<vector<double>> kaiserBesselDeapodization(int xGrid, int yGrid, double W, double beta)
{
    vector<vector<complex<double>>> kernel(yGrid, vector<complex<double>>(xGrid, 0.0));

    int cx = xGrid / 2;
    int cy = yGrid / 2;

    for (int y = max(0, (int)floor(cy - W/2)); y <= min(yGrid-1, (int)ceil(cy + W/2)); y++)
    {
        double wy = kaiserBesselKernel(y - cy, W, beta);

        for (int x = max(0, (int)floor(cx - W/2)); x <= min(xGrid-1, (int)ceil(cx + W/2)); x++)
        {
            double wx = kaiserBesselKernel(x - cx, W, beta);
            kernel[y][x] = wx * wy;
        }
    }

    kernel = ifft2DShift(kernel);
    auto correction = bluesteinFFT2D(kernel);
    correction = fft2DShift(correction);

    vector<vector<double>> deapod(yGrid, vector<double>(xGrid));

    double center = abs(correction[cy][cx]);

    for (int y = 0; y < yGrid; y++)
    {
        for (int x = 0; x < xGrid; x++)
        {
            deapod[y][x] = abs(correction[y][x]) / center;
        }
    }

    //displayKSpace(correction, "Deapodization");

    return deapod;
}

vector<vector<double>> INUFFT(
    vector<kSample> kSamples, 
    int xRes, 
    int yRes, 
    bool debug
) 
    {
    
    double osf = 1.5;
    double W = 6.0;

    double beta = kaiserBesselBeta(W, osf);

    double xFOV = 1.0;
    double yFOV = 1.0;
    
    int numSamples = kSamples.size();

    vector<double> xSpatial(xRes);
    vector<double> ySpatial(yRes);

    double dx = xFOV / xRes;
    double dy = yFOV / yRes;

    double kxMax = fabs(kSamples[0].kx);
    double kyMax = fabs(kSamples[0].ky);

    for (int i = 0; i < numSamples; i++) {
        kxMax = max(kxMax, fabs(kSamples[i].kx));
    }

    for (int i = 0; i < numSamples; i++) {
        kyMax = max(kyMax, fabs(kSamples[i].ky));
    }

    for (int i = 0; i < xRes; i++)
    {
        //xSpatial[i] = dx * i;
        xSpatial[i] = (i - xRes/2.0) * dx;
    }
    for (int i = 0; i < yRes; i++)
    {
        //ySpatial[i] = dy * i;
        ySpatial[i] = (i - yRes/2.0) * dy;
    }

    double dcfSum = 0.0;
    for (const auto& s : kSamples) {
        dcfSum += s.dcf;
    }
    double dcfMean = dcfSum / kSamples.size();

    if (dcfMean > 0.0) {
        for (auto& s : kSamples) {
            s.dcf /= dcfMean;
        }
    }

    vector<kSample> weightedSamples = kSamples;
    for(int i = 0; i < numSamples; i++) {
        weightedSamples[i].value = kSamples[i].value * kSamples[i].dcf;
    }

    int xGrid = (int)(xRes * osf);
    int yGrid = (int)(yRes * osf);

    vector<vector<complex<double>>> griddedKspace = griddingTransform(weightedSamples, xGrid, yGrid, kxMax, kyMax, beta, W);

    vector<vector<complex<double>>> shiftedKspace = ifft2DShift(griddedKspace);

    vector<vector<complex<double>>> complexImage = bluesteinIFFT2D(shiftedKspace);
    
    vector<vector<complex<double>>> complexImageShift = fft2DShift(complexImage);

    vector<vector<complex<double>>> croppedComplexImage(yRes, vector<complex<double>>(xRes));

    if(debug == true) {
        displayKSpace(griddedKspace, "Gridded K-Space");
        displayKSpace(shiftedKspace, "Shifted K-space");
        displayKSpace(complexImage, "complexImage");
        displayKSpace(complexImageShift, "complexImage");    
    }

    int xOffset = (xGrid - xRes) / 2;
    int yOffset = (yGrid - yRes) / 2;

    for (int y = 0; y < yRes; y++) {
        for (int x = 0; x < xRes; x++) {
            croppedComplexImage[y][x] = complexImageShift[y + yOffset][x + xOffset];
        }
    }

    vector<vector<double>> image(yRes, vector<double>(xRes));

    vector<vector<double>> deapod = kaiserBesselDeapodization(xGrid, yGrid, W, beta);

    for(int y = 0; y < yRes; y++) {
        for(int x = 0; x < xRes; x++) {

            image[y][x] = abs(croppedComplexImage[y][x]) / deapod[y + yOffset][x + xOffset];
        }
    }

    return image;

}




