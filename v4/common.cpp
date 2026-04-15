#include "common.h"

std::vector<std::pair<int, int>> floodFillBFS(const Mat& labels, std::vector<bool>& visited, int x, int y) { // La matrice dei label di ogni pixel_AoS, la riga e colonna iniziale
    std::array<std::pair<int, int>, 4> dir = {std::pair{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    std::vector<std::pair<int, int>> current_segment;
    current_segment.reserve(10);
    std::queue<std::pair<int, int>> q;

    visited[x * width + y] = true;
    q.emplace(x, y);
    int currentSP = labels.at<int>(x, y);

    while (!q.empty()) {
        std::pair<int, int> front = q.front();
        int x = front.first, y = front.second;
        q.pop();

        for (std::pair<int, int>& it : dir) {
            int nx = x + it.first;
            int ny = y + it.second;

            if (nx >= 0 && nx < height &&
                ny >= 0 && ny < width &&
                !visited.at(nx * width + ny) &&
                 labels.at<int>(nx, ny) == currentSP) {
                    visited[nx * width + ny] = true;
                    current_segment.emplace_back(nx, ny);
                    q.emplace(nx, ny);
                 }
        }
    }

    return current_segment;
}

void applySegmentationColored(Mat& img, const Mat& labels, const pixels_SoA& centers) {
    const int K = centers.L.size();
    const int H = img.rows;
    const int W = img.cols;

    Mat palette_lab(1, K, CV_32FC3);
    for (int k = 0; k < K; k++) {
        palette_lab.at<Vec3f>(0, k) = Vec3f(centers.L[k], centers.a[k], centers.b[k]);
    }

    Mat palette_bgr_float;
    cvtColor(palette_lab, palette_bgr_float, COLOR_Lab2BGR);

    Mat palette_bgr_byte;
    palette_bgr_float.convertTo(palette_bgr_byte, CV_8UC3, 255.0);

    std::vector<Vec3b> fast_palette(K);
    for(int k=0; k<K; k++) {
        fast_palette[k] = palette_bgr_byte.at<Vec3b>(0, k);
    }

    // Makes the whole cluster the same color
#pragma omp parallel for schedule(static)
    for (int i = 0; i < H; i++) {
        const int* label_ptr = labels.ptr<int>(i);
        Vec3b* pixel_ptr = img.ptr<Vec3b>(i);

        for (int j = 0; j < W; j++) {
            int cluster_id = label_ptr[j];
            if (cluster_id >= 0 && cluster_id < K) {
                pixel_ptr[j] = fast_palette[cluster_id];
            } else {
                pixel_ptr[j] = Vec3b(0, 0, 0);
            }
        }
    }

    // Colors the border between each cluster
    // const Vec3b color = Vec3b(0, 255, 0);
    // for (int i = 0; i < height - 1; i++) {
    //     for (int j = 0; j < width - 1; j++) {
    //         const auto& etC = labels.at<int>(i, j);
    //         const auto& etR = labels.at<int>(i, j + 1);
    //         const auto& etD = labels.at<int>(i + 1, j);
    //
    //         if (etC != etR || etC != etD) {
    //             img.at<Vec3b>(i, j) = color;
    //         }
    //     }
    // }
}