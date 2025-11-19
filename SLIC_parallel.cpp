//
// Created by albi0 on 15/11/2025.
//

#include "SLIC_parallel.h"

double run_parallel(const std::string& img_path, const int& k) {
    std::string time_results = "";
    double time_results_avg = 0;

    for (int it_totali = 1; it_totali < NUM_ITERATIONS + 1; it_totali++) {
        Mat img = imread(img_path, IMREAD_COLOR_BGR);
        if (img.empty()) {
            std::cerr << "Error: Could not load image!" << std::endl;
            return -1;
        }

        clock_t start = clock();

        Mat img_lab;
        GaussianBlur(img, img_lab, Size(5,5) ,0);
        // img.convertTo(img_lab, CV_32F, 1.0/255.0); // convertita a BGR 32-bit float
        img_lab.convertTo(img_lab, CV_32F, 1.0/255.0); // convertita a BGR 32-bit float
        cvtColor(img_lab, img_lab, COLOR_BGR2Lab ); // Convertita al colorspace CIE Lab

        height = img_lab.rows;
        width = img_lab.cols;
        const float step = static_cast<float>(sqrt((height * width) / k));

        std::vector<pixel> clusters_centers;
        clusters_centers.reserve(k);

        // Definizione della posizione iniziale dei centri di tutti i cluster
        parClustersInizialization(img_lab, clusters_centers, step, k);
        const int real_k = clusters_centers.size();

        // Perturbazione dei centri dei cluster nel vicinato 3x3 tramite l'uso del gradiente
        parClustersPerturbation(img_lab, clusters_centers);

        auto labels = Mat(height, width, CV_32S, -1);
        auto distances = Mat(height, width, CV_32F, std::numeric_limits<float>::max());

        double errore_finale = 0;

        for (int iterazioni = 0; iterazioni < 10; iterazioni++) {
            parBestMatchPixelNeighborhood(img_lab, clusters_centers, distances, labels, step);
            parNewClustersCenters(img_lab, clusters_centers, labels, errore_finale, real_k);
        }
        // Passo 9
        parEnforceConnectivity(labels, real_k);

        clock_t end = clock();
        double time_diff = (double)(end - start) / CLOCKS_PER_SEC;
        // printf( "Tempo finale %i° iterazione: %.10f\n", it_totali, time_diff);
        time_results.append(std::to_string(time_diff) + " s\n");
        time_results_avg += time_diff;

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

        std::filesystem::path output_path = std::filesystem::path(PROJECT_SOURCE_DIR) / "output" / "parallelo" ;
        if (!std::filesystem::exists(output_path)) {
            std::filesystem::create_directory(output_path);
        }

        output_path.append("result_par_" + std::to_string(it_totali % 11) + ".png");
        // std::cout << output_path.string() << "\n";
        imwrite(output_path.string(), img);
    }

    std::filesystem::path output_path = std::filesystem::path(PROJECT_SOURCE_DIR) / "output" / "parallelo";
    output_path.append("timeResults.txt");
    time_results_avg /= NUM_ITERATIONS;
    std::ofstream ofs(output_path);
    ofs << time_results << "------------\n";
    ofs << time_results_avg << " s";
    ofs.close();

    return time_results_avg;
}


// Definizione della posizione iniziale dei centri di tutti i cluster
void parClustersInizialization(const Mat& img, std::vector<pixel>& clusters_centers, const float& step, const int& k) {
    int step_i = static_cast<int>(step);
#pragma omp parallel
    {
        std::vector<pixel> private_centers;
        // Riserviamo un po' di memoria per evitare riallocazioni frequenti nel thread
        private_centers.reserve(k / omp_get_num_threads());
        Vec3f lab;

        #pragma omp for nowait
        for (int i = step_i; i < height; i += step_i) {
            for (int j = step_i; j < width; j += step_i) {
                lab = img.at<Vec3f>(i, j);

                private_centers.emplace_back(lab[0], lab[1], lab[2], i, j);
            }
        }

        #pragma omp critical
        {
            clusters_centers.insert(
                clusters_centers.end(),
                private_centers.begin(),
                private_centers.end()
            );
        }
    }
}

// Perturbazione dei cluster in un vicinato 3x3
void parClustersPerturbation(const Mat& img, std::vector<pixel>& clusters_centers) {
    int num_centers = clusters_centers.size();
#pragma omp parallel for schedule(static)
    for (int k = 0; k < num_centers; k++) {
        // for (auto&[l, a, b, x, y] : clusters_centers) {
        pixel& p = clusters_centers[k];

        int x = p.x, y = p.y;

        float min_gradient = std::numeric_limits<float>::max();
        int min_x = x, min_y = y;

        for (int i = x - 1; i < x + 2; i++) {
            if (i < 1 || i >= height - 1) {
                continue;
            }
            for (int j = y - 1; j < y + 2; j++) {
                if (j < 1 || j >= width - 1) {
                    continue;
                }
                auto lab1 = img.at<Vec3f>(i + 1, j);
                auto lab2 = img.at<Vec3f>(i - 1, j);
                auto lab3 = img.at<Vec3f>(i, j + 1);
                auto lab4 = img.at<Vec3f>(i, j - 1);

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
        auto lab_min = img.at<Vec3f>(min_x, min_y);
        p.l = lab_min[0];
        p.a = lab_min[1];
        p.b = lab_min[2];
    }
}

void parBestMatchPixelNeighborhood(Mat& img, const std::vector<pixel>& clusters_centers, Mat& distances, Mat& labels, const float& step) {
    std::vector<std::mutex> row_mutexes(height);

    int num_centers = static_cast<int>(clusters_centers.size());
    int step_i = static_cast<int>(step);

#pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < num_centers; ++k) {
        const pixel& pixel_c = clusters_centers[k];
        int current_cluster_id = k;

        int r_min = std::max(0, pixel_c.x - step_i);
        int r_max = std::min(height, pixel_c.x + step_i);
        int c_min = std::max(0, pixel_c.y - step_i);
        int c_max = std::min(width, pixel_c.y + step_i);

        for (int i = r_min; i < r_max; ++i) {
            std::lock_guard<std::mutex> lock(row_mutexes[i]);

            for (int j = c_min; j < c_max; ++j) {
                Vec3f lab = img.at<Vec3f>(i, j);
                pixel pixel_k{lab[0], lab[1], lab[2], i, j};

                float distance_kc = distance(pixel_c, pixel_k, step);

                if (distance_kc < distances.at<float>(i, j)) {
                    distances.at<float>(i, j) = distance_kc;
                    labels.at<int>(i, j) = current_cluster_id;
                }
            }
        }
    }
}

void parNewClustersCenters(const Mat& img, std::vector<pixel>& clusters_centers, const Mat& labels, double& errore_finale, const int& real_k) {
    std::vector<float> l_avg(real_k, 0), a_avg(real_k, 0), b_avg(real_k, 0);
    std::vector<float> x_avg(real_k, 0), y_avg(real_k, 0);
    std::vector<int> pixel_counter(real_k, 0);

#pragma omp parallel
    {
        std::vector<float> l_loc(real_k, 0), a_loc(real_k, 0), b_loc(real_k, 0);
        std::vector<float> x_loc(real_k, 0), y_loc(real_k, 0);
        std::vector<int> pixel_counter_loc(real_k, 0);
#pragma omp for collapse(2) nowait
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int cluster_id = labels.at<int>(i, j);
                if (cluster_id == -1) continue;

                auto lab = img.at<Vec3f>(i, j);
                l_loc[cluster_id] += lab[0];
                a_loc[cluster_id] += lab[1];
                b_loc[cluster_id] += lab[2];
                x_loc[cluster_id] += i; // riga
                y_loc[cluster_id] += j; // colonna
                pixel_counter_loc[cluster_id]++;
            }
        }
#pragma omp critical
        {
            for (int k = 0; k < real_k; k++) {
                l_avg[k] += l_loc[k];
                a_avg[k] += a_loc[k];
                b_avg[k] += b_loc[k];
                x_avg[k] += x_loc[k];
                y_avg[k] += y_loc[k];
                pixel_counter[k] += pixel_counter_loc[k];
            }
        }
    }

#pragma omp parallel for schedule(static) reduction(+:errore_finale)
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

void parEnforceConnectivity(Mat& labels, const int& real_k) {
    std::vector<std::vector<std::pair<int, int>>> pixels_by_cluster(real_k);
    std::vector<bool> visited(width * height, false);
    std::vector<std::vector<std::vector<std::pair<int, int>>>> segments_per_label(real_k);

#pragma omp parallel
    {
        // Vettori privati per evitare locking continuo
        std::vector<std::vector<std::pair<int, int>>> local_pixels(real_k);

#pragma omp for nowait
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int label = labels.at<int>(i, j);
                if (label >= 0 && label < real_k) {
                    local_pixels[label].emplace_back(i, j);
                }
            }
        }

        // Merge critico (veloce perché fatto una volta per thread)
#pragma omp critical
        {
            for (int k = 0; k < real_k; ++k) {
                if (!local_pixels[k].empty()) {
                    pixels_by_cluster[k].insert(
                        pixels_by_cluster[k].end(),
                        local_pixels[k].begin(),
                        local_pixels[k].end()
                    );
                }
            }
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < real_k; k++) {
        const auto& cluster_pixels = pixels_by_cluster[k];
        if (cluster_pixels.empty()) continue;

        for (const auto& pixel : cluster_pixels) {
            int px = pixel.first;
            int py = pixel.second;
            if (visited[px * width + py]) continue;
            segments_per_label[k].push_back(floodFillBFS(labels, visited, px, py));
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < real_k; i++) {
        if (segments_per_label[i].size() == 1) continue;

        auto& islandsVec = segments_per_label[i];

        // Trova isola più grande ed eliminala dal vettore islandsVec
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
}


// float distance(const pixel& p1, const pixel& p2, const float& S) {
//     auto dis_lab = sqrt((p1.l - p2.l)*(p1.l - p2.l) + (p1.a - p2.a)*(p1.a - p2.a) + (p1.b - p2.b)*(p1.b -p2.b));
//     auto dis_xy = static_cast<float>(sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y)));
//
//     return dis_lab + m/S * dis_xy;
// }
