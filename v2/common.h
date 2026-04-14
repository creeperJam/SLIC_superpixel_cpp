#pragma once

#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <opencv2/opencv.hpp>

#define m 10
#define NUM_ITERATIONS 10

using namespace cv;

inline int height, width;

struct pixel_AoS {
    float l, a, b; // Colore
    int x, y;      // Posizione: x = riga, y = colonna
    pixel_AoS(const float& l, const float& a, const float& b, const int& x, const int& y) : l(l), a(a), b(b), x(x), y(y) {}
};

float distance(const pixel_AoS& p1, const pixel_AoS& p2, const float& S);
std::vector<std::pair<int, int>> floodFillBFS(const Mat& labels, std::vector<bool>& visited, int x, int y);

double run_parallel(const std::string& img_path, const int& k);
double run_sequential(const std::string& img_path, const int& k);

inline float manhattanDist(const pixel_AoS& p1, const pixel_AoS& p2) {
    return std::abs(p1.l - p2.l) + std::abs(p1.a - p2.a) + std::abs(p1.b - p2.b) + std::abs(p1.x - p2.x) + std::abs(p1.y - p2.y);
}