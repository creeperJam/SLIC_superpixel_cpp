#pragma once

#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <omp.h>
#include <opencv2/opencv.hpp>

constexpr int m = 10;
constexpr int NUM_RUNS = 15;
constexpr int WARMUP_RUNS = 2;

using namespace cv;

inline int height, width;
inline constexpr std::array<int, 6> TILE_SIZES = {16, 32, 64, 128, 256, 512};
inline const std::array<std::pair<std::string, int>, 4> IMAGES = {
    std::pair{std::string(PROJECT_SOURCE_DIR)+"/Images/COCO-1.jpg"    , 200},
          {std::string(PROJECT_SOURCE_DIR)+"/Images/COCO-2.jpg"    , 300},
          {std::string(PROJECT_SOURCE_DIR)+"/Images/1080p-test.jpg", 1000},
          {std::string(PROJECT_SOURCE_DIR)+"/Images/4k-test-2.png" , 2000},
};

struct pixel {
    float l, a, b;
    int x, y;
};

struct pixels_SoA {
    std::vector<float> L;
    std::vector<float> a;
    std::vector<float> b;
    std::vector<int> x;
    std::vector<int> y;

    void emplace_back(float L, float a, float b, int x, int y) {
        this->L.emplace_back(L);
        this->a.emplace_back(a);
        this->b.emplace_back(b);
        this->x.emplace_back(x);
        this->y.emplace_back(y);
    }

    void reserve(const int size) {
        L.reserve(size);
        a.reserve(size);
        b.reserve(size);
        x.reserve(size);
        y.reserve(size);
    }

    void clear() {
        L.clear();
        a.clear();
        b.clear();
        x.clear();
        y.clear();
    }

    explicit pixels_SoA(const int& size) {
        reserve(size);
    }

    explicit pixels_SoA() = default;
};

struct image_SoA {
    std::vector<float> L;
    std::vector<float> a;
    std::vector<float> b;

    explicit image_SoA() = default;

    explicit image_SoA(const Mat& img) {
        int size = width * height;
        L.resize(size); a.resize(size); b.resize(size);

        for (int i = 0; i < height; ++i) {
            const auto* row = img.ptr<Vec3f>(i);
            for (int j = 0; j < width; ++j) {
                int idx = i * width + j;
                L[idx] = row[j][0];
                a[idx] = row[j][1];
                b[idx] = row[j][2];
            }
        }
    }

    image_SoA(const image_SoA& img) {
        const int size = width * height;
        L.resize(size); a.resize(size); b.resize(size);

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                int idx = i * width + j;
                L[idx] = img.L[idx];
                a[idx] = img.a[idx];
                b[idx] = img.b[idx];
            }
        }
    }

    Mat to_Mat(int height, int width) const {
        if (height * width != L.size()) return {};

        Mat out = Mat(height, width, CV_32FC3);

        for (int r = 0; r < height; ++r) {
            auto out_row = out.ptr<Vec3f>(r);
            for (int c = 0; c < width; ++c) {
                out_row[c][0] = L[r * width + c];
                out_row[c][1] = a[r * width + c];
                out_row[c][2] = b[r * width + c];
            }
        }

        cvtColor(out, out, COLOR_Lab2BGR);
        return out;
    }
};

struct logs {
    std::array<double, NUM_RUNS> wall_times{};
    std::array<double, NUM_RUNS> cpu_times{};
    int thread_num = 1;
    int tile_size = 0;

    void add(const std::pair<double, double>& times, int pos) {
        wall_times[pos] = times.first;
        cpu_times[pos] = times.second;
    }
};

/**
 * @brief given a starting pixels, it finds all other directly touching pixels and return a vector containing their position
 *
 * @param labels cv::Mat that contains the index of the SP that contains the pixel at coordinates (x, y)
 * @param visited vector that contains true if the pixel at (x, y. Written in x * width + y) has alredy been visited or not
 * @param x, y pixel from which the BFS should start
 * @return the vector (or island) of pairs (x, y) of pixels all touching each other
 */
std::vector<std::pair<int, int>> floodFillBFS(const Mat& labels, std::vector<bool>& visited, int x, int y);

Mat run_sequential(const image_SoA& image_SoA, const int& k);
Mat run_parallel(const image_SoA& image_SoA, const int& k);
Mat run_tile(const image_SoA& image_SoA, const int& k, const int& tile_size);

void applySegmentationColored(Mat& img, const Mat& labels, const pixels_SoA& centers);

/**
 * @brief Given two pixels, calculates the L1 distance between the two. Used for error calculation
 *
 * @param p1 first pixel
 * @param p2 second pixel
 * @return a float value equal to the L1 distance (or Manhattan distance) between the specified pixel
 */
inline float manhattanDist(const pixel& p1, const pixel& p2) {
    return std::abs(p1.l - p2.l) + std::abs(p1.a - p2.a) + std::abs(p1.b - p2.b) + std::abs(p1.x - p2.x) + std::abs(p1.y - p2.y);
}