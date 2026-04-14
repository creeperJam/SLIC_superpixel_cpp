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

float distance(const pixel_AoS& p1, const pixel_AoS& p2, const float& S) {
    auto dis_lab = std::pow((p1.l - p2.l), 2) + std::pow((p1.a - p2.a), 2) + std::pow((p1.b - p2.b), 2);
    auto dis_xy = std::pow((p1.x - p2.x), 2) + std::pow((p1.y - p2.y), 2);

    return dis_lab + std::pow(m/S, 2) * dis_xy;
}