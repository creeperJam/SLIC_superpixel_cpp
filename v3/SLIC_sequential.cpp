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
        img_lab.convertTo(img_lab, CV_32F, 1.0/255.0);
        cvtColor(img_lab, img_lab, COLOR_BGR2Lab );

        pixels_SoA img_SoA(img_lab);

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
            bestMatchPixelNeighborhoood(img_SoA, clusters_centers, distances, labels, step, sp_counter);
            newClustersCenters(img_SoA, clusters_centers, labels, errore_finale, real_k);
        }
        enforceConnectivity(labels, real_k);

        clock_t end = clock();
        double time_diff = (double)(end - start) / CLOCKS_PER_SEC;
        time_results.append(std::to_string(time_diff) + " s\n");
        time_results_avg += time_diff;

        apply_segmentation_colored(img, labels, clusters_centers);

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

void apply_segmentation_colored(Mat& img, const Mat& labels, const std::vector<pixel_AoS>& centers) {
    const int K = centers.size();
    const int H = img.rows;
    const int W = img.cols;

    Mat palette_lab(1, K, CV_32FC3);

    for (int k = 0; k < K; k++) {
        palette_lab.at<Vec3f>(0, k) = Vec3f(centers[k].l, centers[k].a, centers[k].b);
    }

    Mat palette_bgr_float;
    cvtColor(palette_lab, palette_bgr_float, COLOR_Lab2BGR);

    Mat palette_bgr_byte;
    palette_bgr_float.convertTo(palette_bgr_byte, CV_8UC3, 255.0);

    std::vector<Vec3b> fast_palette(K);
    for(int k=0; k<K; k++) {
        fast_palette[k] = palette_bgr_byte.at<Vec3b>(0, k);
    }

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
}


void clustersInizialization(const pixels_SoA& img, std::vector<pixel_AoS>& clusters_centers, const float& step) {
    int step_i = static_cast<int>(step);

    for (int i = step_i; i < height; i += step_i) {
        for (int j = step_i; j < width; j += step_i) {
            int idx = i * img.width + j;
            clusters_centers.emplace_back(img.L[idx], img.a[idx], img.b[idx], i, j);
        }
    }
}

// Perturbazione dei cluster in un vicinato 3x3
void clustersPerturbation(const pixels_SoA& img, std::vector<pixel_AoS>& clusters_centers) {
    int num_centers = clusters_centers.size();
    for (int k = 0; k < num_centers; k++) {
        pixel_AoS& p = clusters_centers[k];
        int x = p.x, y = p.y;

        float min_gradient = std::numeric_limits<float>::max();
        int min_x = x, min_y = y;

        for (int i = x - 1; i < x + 2; i++) {
            if (i < 1 || i >= height - 1) continue;

            int row_prev = (i - 1) * width;
            int row_curr = i * width;
            int row_next = (i + 1) * width;

            for (int j = y - 1; j < y + 2; j++) {
                if (j < 1 || j >= width - 1) continue;

                int idx_up = row_curr + (j - 1);
                int idx_down = row_curr + (j + 1);
                int idx_left = row_prev + j;
                int idx_right = row_next + j;

                float l1 = img.L[idx_right];
                float a1 = img.a[idx_right];
                float b1 = img.b[idx_right];

                float l2 = img.L[idx_left];
                float a2 = img.a[idx_left];
                float b2 = img.b[idx_left];

                float l3 = img.L[idx_down];
                float a3 = img.a[idx_down];
                float b3 = img.b[idx_down];

                float l4 = img.L[idx_up];
                float a4 = img.a[idx_up];
                float b4 = img.b[idx_up];

                float grad_sq =
                    (l1-l2)*(l1-l2) + (a1-a2)*(a1-a2) + (b1-b2)*(b1-b2) +
                    (l3-l4)*(l3-l4) + (a3-a4)*(a3-a4) + (b3-b4)*(b3-b4);

                if (grad_sq < min_gradient) {
                    min_gradient = grad_sq;
                    min_x = i;
                    min_y = j;
                }
            }
        }

        p.x = min_x;
        p.y = min_y;
        int idx_min = min_x * width + min_y;
        p.l = img.L[idx_min];
        p.a = img.a[idx_min];
        p.b = img.b[idx_min];
    }
}

void bestMatchPixelNeighborhoood(const pixels_SoA& img, const std::vector<pixel_AoS>& clusters_centers, Mat& distances, Mat& labels, const float& step, int& sp_counter) {
    const int step_int = static_cast<int>(step);

    float m_div_s = m / step;
    float W = m_div_s * m_div_s;

    for (const pixel_AoS& pixel_c : clusters_centers) {
        const int r_min = std::max(pixel_c.x - step_int, 0);
        const int r_max = std::min(pixel_c.x + step_int, height);
        const int c_min = std::max(pixel_c.y - step_int, 0);
        const int c_max = std::min(pixel_c.y + step_int, width);

        for (int i = r_min; i < r_max; i++) {

            int row_offset = i * img.width;

            float dx = static_cast<float>(i) - pixel_c.x;
            float dx2 = dx * dx;

            for (int j = c_min; j < c_max; j++) {

                int idx = row_offset + j;

                float L = img.L[idx];
                float a = img.a[idx];
                float b = img.b[idx];

                float dL = L - pixel_c.l;
                float da = a - pixel_c.a;
                float db = b - pixel_c.b;
                float dist_color = (dL*dL) + (da*da) + (db*db);

                float dy = static_cast<float>(j) - pixel_c.y;
                float dy2 = dy * dy;

                float dist_space = (dx2 + dy2);

                float distance_kc = dist_color + (dist_space * W);

                if (distance_kc < distances.at<float>(i, j)) {
                    distances.at<float>(i, j) = distance_kc;
                    labels.at<int>(i, j) = sp_counter;
                }
            }
        }
        sp_counter++;
    }
}

void newClustersCenters(const pixels_SoA& img, std::vector<pixel_AoS>& clusters_centers, const Mat& labels, double& errore_finale, const int& real_k) {
    std::vector<float> l_avg(real_k, 0.0), a_avg(real_k, 0.0), b_avg(real_k, 0.0);
    std::vector<float> x_avg(real_k, 0.0), y_avg(real_k, 0.0);
    std::vector<int> pixel_counter(real_k, 0);

    for (int r = 0; r < height; r++) {
        int row_offset = r * img.width;

        for (int c = 0; c < width; c++) {
            int idx = row_offset + c;

            int label = labels.at<int>(r, c);
            if (label == -1) continue;

            l_avg[label] += img.L[idx];
            a_avg[label] += img.a[idx];
            b_avg[label] += img.b[idx];

            x_avg[label] += r;
            y_avg[label] += c;

            pixel_counter[label]++;
        }
    }

    for (int k_id = 0; k_id < real_k; k_id++) {
        if (pixel_counter[k_id] == 0) continue;

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