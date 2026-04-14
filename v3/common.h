#pragma once

#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <opencv2/opencv.hpp>

constexpr int m = 10;
constexpr int NUM_ITERATIONS = 10;

using namespace cv;

inline int height, width;

struct pixel_AoS {
    float l, a, b;
    int x, y;
    pixel_AoS(const float& l, const float& a, const float& b, const int& x, const int& y) : l(l), a(a), b(b), x(x), y(y) {}
};

struct pixels_SoA {
    int width, height;
    std::vector<float> L;
    std::vector<float> a;
    std::vector<float> b;

    pixels_SoA(const cv::Mat& img) {
        height = img.rows;
        width = img.cols;
        int size = width * height;
        L.resize(size); a.resize(size); b.resize(size);

        for (int i = 0; i < height; ++i) {
            const auto* row = img.ptr<cv::Vec3f>(i);
            for (int j = 0; j < width; ++j) {
                int idx = i * width + j;
                L[idx] = row[j][0];
                a[idx] = row[j][1];
                b[idx] = row[j][2];
            }
        }
    }
};

float distance(const pixel_AoS& p1, const pixel_AoS& p2, const float& S);
std::vector<std::pair<int, int>> floodFillBFS(const Mat& labels, std::vector<bool>& visited, int x, int y);

double run_parallel(const std::string& img_path, const int& k);
double run_sequential(const std::string& img_path, const int& k);

inline float manhattanDist(const pixel_AoS& p1, const pixel_AoS& p2) {
    return std::abs(p1.l - p2.l) + std::abs(p1.a - p2.a) + std::abs(p1.b - p2.b) + std::abs(p1.x - p2.x) + std::abs(p1.y - p2.y);
}