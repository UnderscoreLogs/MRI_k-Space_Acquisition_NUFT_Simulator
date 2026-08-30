#include "../../include/Imaging/ImageUtils.hpp"

using namespace std;

cv::Mat doubleToMat(vector<vector<double>> imgD) {
    int rows = imgD.size();
    int cols = imgD[0].size();

    cv::Mat magMat(rows, cols, CV_64F);

    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < cols; x++)
        {
            magMat.at<double>(y, x) = imgD[y][x];
        }
    }

    return magMat;
}

cv::Mat complexVecToMat(const vector<vector<complex<double>>> &data)
{
    int rows = data.size();
    int cols = rows ? data[0].size() : 0;

    cv::Mat mag(rows, cols, CV_64F);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            mag.at<double>(i, j) = std::abs(data[i][j]);
        }
    }

    return mag;
}

vector<vector<double>> normalizeImage(const vector<vector<double>>& img, double newMin, double newMax)
{
    int rows = img.size();
    int cols = img[0].size();

    double minVal = img[0][0];
    double maxVal = img[0][0];

    // Find min and max
    for (const auto& row : img)
    {
        for (double v : row)
        {
            minVal = min(minVal, v);
            maxVal = max(maxVal, v);
        }
    }

    vector<vector<double>> out(rows, vector<double>(cols));

    // Avoid divide-by-zero
    if (fabs(maxVal - minVal) < 1e-12)
        return out;

    double scale = (newMax - newMin) / (maxVal - minVal);

    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < cols; x++)
        {
            out[y][x] = (img[y][x] - minVal) * scale + newMin;
        }
    }

    return out;
}

void saveImg(vector<vector<double>> image_double, string fileName, string folder) {
    string fullFileName =  fileName + ".png";
    string windowName = fileName + " Image";

    cv::Mat image_mat = doubleToMat(image_double);

    image_mat.convertTo(image_mat, CV_8U);

    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::imshow(windowName, image_mat);

    cv::imwrite(folder + "/" + fullFileName, image_mat);

}

void displayTrajectory(vector<kSample>& samples, int size, string windowName)
{
    cv::Mat img = cv::Mat::zeros(size,size,CV_8UC1);

    for(auto& s : samples)
    {
        int x = (int)(s.kx + size/2);
        int y = (int)(s.ky + size/2);

        if(x >=0 && x<size && y>=0 && y<size)
            img.at<uchar>(y,x)=255;
    }

    cv::Mat dis_img;
    cv::normalize(img, dis_img, 0, 255, cv::NORM_MINMAX);
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::imshow(windowName, dis_img);
}

void displayKSpace(vector<vector<complex<double>>>& kSpace, string name) {

    int ySize = kSpace.size();
    int xSize = kSpace[0].size();    

    cv::Mat img(ySize, xSize, CV_32F);

    for (int y = 0; y < ySize; y++) {
        for (int x = 0; x < xSize; x++) {
            img.at<float>(y, x) = log(1.0f + static_cast<float>(abs(kSpace[y][x])));
        }
    }

    cv::Mat display;
    cv::normalize(img, display, 0, 255, cv::NORM_MINMAX);
    display.convertTo(display, CV_8U);

    cv::imshow(name, display);
}

vector<vector<double>> getImage(string imgName)
{
    cv::Mat image = cv::imread(imgName, cv::IMREAD_GRAYSCALE);

    if (image.empty())
    {
        cout << "Could not load image\n";
        return {};
    }

    vector<vector<double>> imageVector(image.rows, vector<double>(image.cols));

    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            // Read as uchar (the real storage type), then cast to double
            imageVector[i][j] = static_cast<double>(image.at<uchar>(i, j));
        }
    }

    cout << "Vector size: " << imageVector.size() << " x " << imageVector[0].size() << endl;

    return imageVector;
}





