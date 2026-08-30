#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>

// ============================================================
// NIfTI-1 Header
// ============================================================

#pragma pack(push, 1)

struct NiftiHeader
{
    int32_t sizeof_hdr;          // 0
    char data_type[10];          // 4
    char db_name[18];             // 14
    int32_t extents;              // 32
    int16_t session_error;        // 36
    char regular;                 // 38
    char dim_info;                // 39

    int16_t dim[8];               // 40

    float intent_p1;              // 56
    float intent_p2;              // 60
    float intent_p3;              // 64

    int16_t intent_code;          // 68
    int16_t datatype;              // 70
    int16_t bitpix;               // 72
    int16_t slice_start;          // 74

    float pixdim[8];               // 76

    float vox_offset;             // 108
    float scl_slope;              // 112
    float scl_inter;              // 116

    int16_t slice_end;            // 120
    char slice_code;              // 122
    char xyzt_units;              // 123

    float cal_max;                // 124
    float cal_min;                // 128

    float slice_duration;         // 132
    float toffset;                // 136

    int32_t glmax;                // 140
    int32_t glmin;                // 144

    char descrip[80];             // 148
    char aux_file[24];             // 228

    int16_t qform_code;            // 252
    int16_t sform_code;            // 254

    float quatern_b;               // 256
    float quatern_c;               // 260
    float quatern_d;               // 264

    float qoffset_x;               // 268
    float qoffset_y;               // 272
    float qoffset_z;               // 276

    float srow_x[4];               // 280
    float srow_y[4];               // 296
    float srow_z[4];               // 312

    char intent_name[16];           // 328
    char magic[4];                  // 344
};

#pragma pack(pop)

double readVoxel(
    std::ifstream& file,
    int datatype,
    double slope,
    double intercept
);

// ============================================================
// Load a 3D NIfTI image
// ============================================================

std::vector<double> loadNifti(
    const std::string& filename,
    NiftiHeader& header
);

// ============================================================
// Extract axial slice
// ============================================================

std::vector<std::vector<double>> extractSlice(
    const std::vector<double>& image,
    int nx,
    int ny,
    int nz,
    int slice
);