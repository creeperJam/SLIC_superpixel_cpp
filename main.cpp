//
// Created by albi0 on 15/11/2025.
//

#include <opencv2/opencv.hpp>
#include <iostream>
#include <limits>
#include <ctime>
#include <filesystem>
#define m 10

using namespace cv;

struct pixel {
    float l, a, b; // Colore
    int x, y;      // Posizione: x = riga, y = colonna
    pixel(float l, float a, float b, int x, int y) : l(l), a(a), b(b), x(x), y(y) {}
};

int height, width; // Size of the image

inline float manhattanDist(const pixel& p1, const pixel& p2);
float distance(const pixel& p1, const pixel& p2, const float& S);
std::vector<std::pair<int, int>> floodFillBFS(const Mat& labels, std::vector<bool>& visited, int x, int y);

int main(int argc, char *argv[]) {
    clock_t start = clock();

    if (argc <= 2) {
        printf("Errore, argomenti mancanti!! Formato comando: ./SLIC_SP_req path/to/image SP-amount");
        return -2;
    }
    std::string img_path = argv[1];
    const int k = std::stoi(argv[2]);

    Mat img = imread(img_path, IMREAD_COLOR_BGR);
    if (img.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    Mat img_lab = img.clone();
    img.convertTo(img_lab, CV_32F, 1.0/255.0); // convertita a BGR 32-bit float
    cvtColor(img_lab, img_lab, COLOR_BGR2Lab ); // Convertita al colorspace CIE Lab

    height = img_lab.rows;
    width = img_lab.cols;
    const float step = static_cast<float>(sqrt((height * width) / k));

    std::vector<pixel> clusters_centers;
    clusters_centers.reserve(k);

    // Definizione della posizione iniziale dei centri di tutti i cluster
    Vec3f lab;
    for (int i = step; i < height; i += step) {
        for (int j = step; j < width; j += step) {
            lab = img_lab.at<Vec3f>(i, j);

            clusters_centers.emplace_back(lab[0], lab[1], lab[2], i, j);
        }
    }

    const int real_k = clusters_centers.size();

    // Perturbazione dei centri dei cluster nel vicinato 3x3 tramite l'uso del gradiente
    for (auto&[l, a, b, x, y] : clusters_centers) {
        float min_gradient = std::numeric_limits<float>::max();
        int min_x, min_y;

        for (int i = x - 1; i < x + 2; i++) {
            if (i < 1 || i >= height - 1) {
                continue;
            }
            for (int j = y - 1; j < y + 2; j++) {
                if (j < 1 || j >= width - 1) {
                    continue;
                }
                auto lab1 = img_lab.at<Vec3f>(i + 1, j);
                auto lab2 = img_lab.at<Vec3f>(i - 1, j);
                auto lab3 = img_lab.at<Vec3f>(i, j + 1);
                auto lab4 = img_lab.at<Vec3f>(i, j - 1);

                Vec3f norm_l = {lab1[0] - lab2[0], lab1[1] - lab2[1], lab1[2] - lab2[2]};
                Vec3f norm_r = {lab3[0] - lab4[0], lab3[1] - lab4[1], lab3[2] - lab4[2]};

                float gradient = norm_l[0]*norm_l[0] + norm_l[1]*norm_l[1] + norm_l[2]*norm_l[2]
                    + norm_r[0]*norm_r[0] + norm_r[1]*norm_r[1] + norm_r[2]*norm_r[2];

                if (gradient < min_gradient) {
                    min_gradient = gradient;
                    min_x = i;
                    min_y = j;
                }
            }
        }

        x = min_x;
        y = min_y;
        auto lab_min = img_lab.at<Vec3f>(min_x, min_y);
        l = lab_min[0];
        a = lab_min[1];
        b = lab_min[2];
    }

    auto labels = Mat(height, width, CV_32S, -1);
    auto distances = Mat(height, width, CV_32F, std::numeric_limits<float>::max());

    double errore_finale = 0;

    for (int iterazioni = 0; iterazioni < 10; iterazioni++) {
        int sp_counter = 0;
        for (const pixel& pixel_c : clusters_centers) {
            const int iMax = pixel_c.x < height - (int)step ? pixel_c.x + (int)step : height; // Riga finale
            const int jMax = pixel_c.y < width - (int)step ? pixel_c.y + (int)step : width;   // Colonna finale
            int i = pixel_c.x > (int)step ? pixel_c.x - (int)step : 0;
            int j = 0;
            Vec3f lab;
            for (; i < iMax; i++) {
                j = pixel_c.y > (int)step ? pixel_c.y - (int)step : 0;

                for (; j < jMax; j++) {
                    lab = img_lab.at<Vec3f>(i, j);
                    pixel pixel_k{lab[0], lab[1], lab[2], i, j};
                    float distance_kc = distance(pixel_c, pixel_k, step);

                    if (distance_kc < distances.at<float>(i, j)) {
                        distances.at<float>(i, j) = distance_kc;
                        labels.at<int>(i, j) = sp_counter;
                    }
                }
            }
            sp_counter++;
        }

        std::vector<float> l_avg(real_k, 0), a_avg(real_k, 0), b_avg(real_k, 0);
        std::vector<float> x_avg(real_k, 0), y_avg(real_k, 0);
        std::vector<int> pixel_counter(real_k, 0);

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int cluster_id = labels.at<int>(i, j);
                if (cluster_id == -1) continue;

                auto lab = img_lab.at<Vec3f>(i, j);
                l_avg[cluster_id] += lab[0];
                a_avg[cluster_id] += lab[1];
                b_avg[cluster_id] += lab[2];
                x_avg[cluster_id] += i; // riga
                y_avg[cluster_id] += j; // colonna
                pixel_counter[cluster_id]++;
            }
        }

        for (int k_id = 0; k_id < real_k; k_id++) {
            if (pixel_counter[k_id] == 0) continue; // Evita divisione per zero

            pixel newC {
                (l_avg[k_id] / pixel_counter[k_id]),
                (a_avg[k_id] / pixel_counter[k_id]),
                (b_avg[k_id] / pixel_counter[k_id]),
                static_cast<int>(std::round(x_avg[k_id] / pixel_counter[k_id])),
                static_cast<int>(std::round(y_avg[k_id] / pixel_counter[k_id]))
            };
            errore_finale += manhattanDist(newC, clusters_centers[k_id]);
            clusters_centers[k_id] = newC;
        }
    }
    // Passo 9
    auto visited = std::vector<bool>(width * height, false);
    std::vector<std::vector<std::vector<std::pair<int, int>>>> segments_per_label(real_k);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (visited[i * width + j]) continue;
            segments_per_label[labels.at<int>(i, j)].push_back(floodFillBFS(labels, visited, i, j));
        }
    }

    for (int i = 0; i < real_k; i++) {
        if (segments_per_label[i].size() == 1) continue;

        auto& islandsVec = segments_per_label[i];

        // Trova isola più grande e eliminala dal vettore islandsVec
        int pixel_count = islandsVec[0].size();
        int biggest_index = 0;
        for (int island = 1; island < islandsVec.size(); island++) {
            if (islandsVec[island].size() > pixel_count) {
                pixel_count = islandsVec[island].size();
                biggest_index = island;
            }
        }

        for (int island = 0; island < islandsVec.size(); island++) {
            if (island == biggest_index) continue;

            std::vector<int> neighbor_counts(real_k, 0);
            for (const auto&[x, y] : islandsVec[island]) {
                std::array<std::pair<int, int>, 4> dir = {std::pair{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
                for (std::pair<int, int>& it : dir) {
                    int nx = x + it.first;
                    int ny = y + it.second;

                    // Check bordi
                    if (nx >= 0 && nx < height && ny >= 0 && ny < width) {
                        int cluster = labels.at<int>(nx, ny);
                        if (cluster != i) {
                            neighbor_counts[cluster]++;
                        }
                    }
                }
            }

            auto it_max = std::max_element(neighbor_counts.begin(), neighbor_counts.end());
            if (*it_max > 0) {
                for (const auto&[x, y] : islandsVec[island]) {
                    labels.at<int>(x, y) = std::distance(neighbor_counts.begin(), it_max);
                }
            }
        }
    }

    clock_t end = clock();
    printf( "Tempo finale: %.10f\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    Vec3b color = Vec3b(0, 255, 255);
    for (int i = 0; i < height - 1; i++) {     // y
        for (int j = 0; j < width - 1; j++) {  // x
            auto etC = labels.at<int>(i, j);
            auto etR = labels.at<int>(i, j + 1);
            auto etD = labels.at<int>(i + 1, j);

            if (etC != etR || etC != etD) {
                img.at<Vec3b>(i, j) = color;
            }
        }
    }

    std::filesystem::path output_path = std::filesystem::path(PROJECT_SOURCE_DIR) / "output" ;
    if (!std::filesystem::exists(output_path)) {
        std::filesystem::create_directory(output_path);
    }
    output_path = output_path / "result.png";

    imwrite(output_path.string(), img);

    return 0;
}


// Definizione della posizione iniziale dei centri di tutti i cluster
void clustersInizialization(const Mat& img, std::vector<pixel>& clusters_centers, const float& step) {
    Vec3f lab;
    for (int i = step; i < height; i += step) {
        for (int j = step; j < width; j += step) {
            lab = img.at<Vec3f>(i, j);

            clusters_centers.emplace_back(lab[0], lab[1], lab[2], i, j);
        }
    }
}

std::vector<std::pair<int, int>> floodFillBFS(const Mat& labels, std::vector<bool>& visited, int x, int y) { // La matrice dei label di ogni pixel, la riga e colonna iniziale
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

inline float manhattanDist(const pixel& p1, const pixel& p2) {
    return std::abs(p1.l - p2.l) + std::abs(p1.a - p2.a) + std::abs(p1.b - p2.b) + std::abs(p1.x - p2.x) + std::abs(p1.y - p2.y);
}

float distance(const pixel& p1, const pixel& p2, const float& S) {
    auto dis_lab = (p1.l - p2.l)*(p1.l - p2.l) + (p1.a - p2.a)*(p1.a - p2.a) + (p1.b - p2.b)*(p1.b -p2.b);
    auto dis_xy = (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);

    return dis_lab + m/S*m/S * dis_xy;
}


// float distance(const pixel& p1, const pixel& p2, const float& S) {
//     auto dis_lab = sqrt((p1.l - p2.l)*(p1.l - p2.l) + (p1.a - p2.a)*(p1.a - p2.a) + (p1.b - p2.b)*(p1.b -p2.b));
//     auto dis_xy = static_cast<float>(sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y)));
//
//     return dis_lab + m/S * dis_xy;
// }
