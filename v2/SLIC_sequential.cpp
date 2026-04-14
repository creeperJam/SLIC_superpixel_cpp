#include "SLIC_sequential.h"

double run_sequential(const std::string &img_path, const int &k) {
    std::string time_results = "";
    double time_results_avg = 0;

    for (int it_totali = 1; it_totali < NUM_ITERATIONS + 1; it_totali++) {
        Mat img = imread(img_path, IMREAD_COLOR_BGR);
        if (img.empty()) {
            std::cerr << "Error: Could not load image!" << std::endl;
            return -1;
        }

        Mat img_lab;
        GaussianBlur(img, img_lab, Size(5,5) ,0);
        img_lab.convertTo(img_lab, CV_32F, 1.0/255.0); // convertita a BGR 32-bit float
        cvtColor(img_lab, img_lab, COLOR_BGR2Lab ); // Convertita al colorspace CIE Lab

        clock_t start = clock();

        height = img_lab.rows;
        width = img_lab.cols;
        const float step = static_cast<float>(sqrt((height * width) / k));

        std::vector<pixel_AoS> clusters_centers;
        clusters_centers.reserve(k);

        clustersInizialization(img_lab, clusters_centers, step);
        const int real_k = clusters_centers.size();

        clustersPerturbation(img_lab, clusters_centers);

        auto labels = Mat(height, width, CV_32S, -1);
        auto distances = Mat(height, width, CV_32F, std::numeric_limits<float>::max());

        double errore_finale = 0;

        for (int iterazioni = 0; iterazioni < 10; iterazioni++) {
            int sp_counter = 0;
            bestMatchPixelNeighborhoood(img_lab, clusters_centers, distances, labels, step, sp_counter);
            newClustersCenters(img_lab, clusters_centers, labels, errore_finale, real_k);
        }
        enforceConnectivity(labels, real_k);

        clock_t end = clock();
        double time_diff = (double)(end - start) / CLOCKS_PER_SEC;
        // printf( "Tempo finale %i° iterazione: %.10f\n", it_totali, time_diff);
        time_results.append(std::to_string(time_diff) + " s\n");
        time_results_avg += time_diff;

        const Vec3b color = Vec3b(0, 255, 255);
        for (int i = 0; i < height - 1; i++) {     // y
            for (int j = 0; j < width - 1; j++) {  // x
                const auto& etC = labels.at<int>(i, j);
                const auto& etR = labels.at<int>(i, j + 1);
                const auto& etD = labels.at<int>(i + 1, j);

                if (etC != etR || etC != etD) {
                    img.at<Vec3b>(i, j) = color;
                }
            }
        }

        std::filesystem::path output_path = std::filesystem::path(PROJECT_SOURCE_DIR) / "output" / "sequenziale" ;
        output_path.append("result_" + std::to_string(it_totali % 11) + ".png");
        // std::cout << output_path.string() << "\n";
        imwrite(output_path.string(), img);
    }

    std::filesystem::path output_path = std::filesystem::path(PROJECT_SOURCE_DIR) / "output" / "sequenziale";
    output_path.append("timeResults.txt");
    time_results_avg /= NUM_ITERATIONS;
    std::ofstream ofs(output_path);
    ofs << time_results << "------------\n";
    ofs << time_results_avg << " s";
    ofs.close();

    return time_results_avg;
}


// Definizione della posizione iniziale dei centri di tutti i cluster
void clustersInizialization(const Mat& img, std::vector<pixel_AoS>& clusters_centers, const float& step) {
    int step_int = static_cast<int> (step);
    Vec3f lab;
    for (int i = step_int; i < height; i += step_int) {
        for (int j = step_int; j < width; j += step_int) {
            lab = img.at<Vec3f>(i, j);

            clusters_centers.emplace_back(lab[0], lab[1], lab[2], i, j);
        }
    }
}

// Perturbazione dei cluster in un vicinato 3x3
void clustersPerturbation(const Mat& img, std::vector<pixel_AoS>& clusters_centers) {
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
                const auto& lab1 = img.at<Vec3f>(i + 1, j);
                const auto& lab2 = img.at<Vec3f>(i - 1, j);
                const auto& lab3 = img.at<Vec3f>(i, j + 1);
                const auto& lab4 = img.at<Vec3f>(i, j - 1);

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
        const auto& lab_min = img.at<Vec3f>(min_x, min_y);
        l = lab_min[0];
        a = lab_min[1];
        b = lab_min[2];
    }
}

void bestMatchPixelNeighborhoood(Mat& img, const std::vector<pixel_AoS>& clusters_centers, Mat& distances, Mat& labels, const float& step, int& sp_counter) {
    const int step_int = static_cast<int> (step);
    for (const pixel_AoS& pixel_c : clusters_centers) {
        const int iMax = std::min(pixel_c.x + step_int, height);
        const int jMax = std::min(pixel_c.y + step_int, width);
        int j, i;

        for (i = std::max(pixel_c.x - step_int, 0); i < iMax; i++) {
            for (j = std::max(pixel_c.y - step_int, 0); j < jMax; j++) {
                auto& lab = img.at<Vec3f>(i, j);
                pixel_AoS pixel_k{lab[0], lab[1], lab[2], i, j};
                float distance_kc = distance(pixel_c, pixel_k, step);

                if (distance_kc < distances.at<float>(i, j)) {
                    distances.at<float>(i, j) = distance_kc;
                    labels.at<int>(i, j) = sp_counter;
                }
            }
        }
        sp_counter++;
    }
}

void newClustersCenters(const Mat& img, std::vector<pixel_AoS>& clusters_centers, const Mat& labels, double& errore_finale, const int& real_k) {
    std::vector<float> l_avg(real_k, 0), a_avg(real_k, 0), b_avg(real_k, 0);
    std::vector<float> x_avg(real_k, 0), y_avg(real_k, 0);
    std::vector<int> pixel_counter(real_k, 0);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int cluster_id = labels.at<int>(i, j);
            if (cluster_id == -1) continue;

            const auto& lab = img.at<Vec3f>(i, j);
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

        pixel_AoS newC {
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

void enforceConnectivity(Mat& labels, const int& real_k) {
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
                for (const auto&[vertical, horizontal] : dir) {
                    const int nx = x + vertical;
                    const int ny = y + horizontal;

                    // Check bordi
                    if (nx >= 0 && nx < height && ny >= 0 && ny < width) {
                        const int& cluster = labels.at<int>(nx, ny);
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
}