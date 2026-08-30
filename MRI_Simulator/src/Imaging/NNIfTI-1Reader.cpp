#include "../../include/Imaging/niftiReader.hpp"

using namespace std;

// Ai coded this

// ============================================================
// Read one voxel
// ============================================================

double readVoxel(
    ifstream& file,
    int datatype,
    double slope,
    double intercept
)
{
    double value;

    switch (datatype)
    {
        // uint8
        case 2:
        {
            uint8_t x;
            file.read(reinterpret_cast<char*>(&x), sizeof(x));
            value = x;
            break;
        }

        // int16
        case 4:
        {
            int16_t x;
            file.read(reinterpret_cast<char*>(&x), sizeof(x));
            value = x;
            break;
        }

        // int32
        case 8:
        {
            int32_t x;
            file.read(reinterpret_cast<char*>(&x), sizeof(x));
            value = x;
            break;
        }

        // float32
        case 16:
        {
            float x;
            file.read(reinterpret_cast<char*>(&x), sizeof(x));
            value = x;
            break;
        }

        // float64
        case 64:
        {
            double x;
            file.read(reinterpret_cast<char*>(&x), sizeof(x));
            value = x;
            break;
        }

        default:
            cerr << "Unsupported NIfTI datatype: "
                 << datatype << endl;

            exit(1);
    }

    return value * slope + intercept;
}


// ============================================================
// Load a 3D NIfTI image
// ============================================================

vector<double> loadNifti(
    const string& filename,
    NiftiHeader& header
)
{
    ifstream file(filename, ios::binary);

    if (!file)
    {
        cerr << "Could not open: " << filename << endl;
        exit(1);
    }


    // --------------------------------------------------------
    // Read header
    // --------------------------------------------------------

    file.read(
        reinterpret_cast<char*>(&header),
        sizeof(NiftiHeader)
    );

    if (!file)
    {
        cerr << "Could not read NIfTI header." << endl;
        exit(1);
    }


    // --------------------------------------------------------
    // Verify NIfTI-1
    // --------------------------------------------------------

    if (header.sizeof_hdr != 348)
    {
        cerr << "Invalid NIfTI header.\n";
        cerr << "sizeof_hdr = "
             << header.sizeof_hdr << endl;

        exit(1);
    }


    // --------------------------------------------------------
    // Dimensions
    // --------------------------------------------------------

    int nx = header.dim[1];
    int ny = header.dim[2];
    int nz = header.dim[3];

    int ndim = header.dim[0];

    cout << "\n========== NIfTI Information ==========\n";

    cout << "Dimensions: "
         << nx << " x "
         << ny << " x "
         << nz << endl;

    cout << "Number of dimensions: "
         << ndim << endl;

    cout << "Datatype code: "
         << header.datatype << endl;

    cout << "Bits per voxel: "
         << header.bitpix << endl;

    cout << "Voxel size: "
         << header.pixdim[1] << " x "
         << header.pixdim[2] << " x "
         << header.pixdim[3]
         << endl;

    cout << "Voxel offset: "
         << header.vox_offset
         << endl;

    cout << "Scale slope: "
         << header.scl_slope
         << endl;

    cout << "Scale intercept: "
         << header.scl_inter
         << endl;

    cout << "=======================================\n\n";


    // --------------------------------------------------------
    // Scaling
    // --------------------------------------------------------

    double slope = header.scl_slope;

    if (slope == 0)
        slope = 1.0;

    double intercept = header.scl_inter;


    // --------------------------------------------------------
    // Number of voxels
    // --------------------------------------------------------

    long long totalVoxels =
        static_cast<long long>(nx) *
        static_cast<long long>(ny) *
        static_cast<long long>(nz);


    vector<double> image(totalVoxels);


    // --------------------------------------------------------
    // Move to voxel data
    // --------------------------------------------------------

    file.seekg(
        static_cast<streamoff>(header.vox_offset),
        ios::beg
    );

    if (!file)
    {
        cerr << "Could not seek to voxel data.\n";
        exit(1);
    }


    // --------------------------------------------------------
    // Read voxels
    // --------------------------------------------------------

    cout << "Reading "
         << totalVoxels
         << " voxels...\n";

    for (long long i = 0; i < totalVoxels; i++)
    {
        image[i] =
            readVoxel(
                file,
                header.datatype,
                slope,
                intercept
            );
    }

    cout << "Finished reading image.\n";

    return image;
}


// ============================================================
// Extract axial slice
// ============================================================

vector<vector<double>> extractSlice(
    const vector<double>& image,
    int nx,
    int ny,
    int nz,
    int slice
)
{
    if (slice < 0 || slice >= nz)
    {
        cerr << "Invalid slice index.\n";
        exit(1);
    }


    vector<vector<double>> result(
        ny,
        vector<double>(nx)
    );


    /*
        NIfTI voxel ordering is generally:

            x changes fastest
            then y
            then z

        Therefore:

        index = x
              + y * nx
              + z * nx * ny
    */


    for (int y = 0; y < ny; y++)
    {
        for (int x = 0; x < nx; x++)
        {
            long long index =
                x
                + y * nx
                + slice * nx * ny;

            result[y][x] = image[index];
        }
    }

    return result;
}

