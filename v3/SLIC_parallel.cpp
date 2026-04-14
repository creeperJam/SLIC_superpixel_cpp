#include "SLIC_parallel.h"

void parApplySegmentationColored(Mat& img, const Mat& labels, const std::vector<pixel_AoS>& centers) {
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
}

double run_parallel(const std::string& img_path, const int& k) {
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

        pixels_SoA img_soa(img_lab);

        clock_t start = clock();

        height = img_lab.rows;
        width = img_lab.cols;
        const float step = static_cast<float>(sqrt((height * width) / k));

        std::vector<pixel_AoS> clusters_centers;
        clusters_centers.reserve(k);

        parClustersInizialization(img_soa, clusters_centers, step, k);
        const int real_k = clusters_centers.size();

        parClustersPerturbation(img_soa, clusters_centers);

        auto labels = Mat(height, width, CV_32S, -1);
        auto distances = Mat(height, width, CV_32F, std::numeric_limits<float>::max());

        double errore_finale = 0;

        for (int iterazioni = 0; iterazioni < 10; iterazioni++) {
            parBestMatchPixelNeighborhood(img_soa, clusters_centers, distances, labels, step);
            parNewClustersCenters(img_soa, clusters_centers, labels, errore_finale, real_k);
        }

        parEnforceConnectivity(labels, real_k);

        clock_t end = clock();
        double time_diff = (double)(end - start) / CLOCKS_PER_SEC;

        time_results.append(std::to_string(time_diff) + " s\n");
        time_results_avg += time_diff;

        parApplySegmentationColored(img, labels, clusters_centers);

        std::filesystem::path output_path = std::filesystem::path(PROJECT_SOURCE_DIR) / "output" / "parallelo" ;
        output_path.append("result_par_" + std::to_string(it_totali % 11) + ".png");
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


void parClustersInizialization(const pixels_SoA& img, std::vector<pixel_AoS>& clusters_centers, const float& step, const int& k) {
    int step_i = static_cast<int>(step);
#pragma omp parallel
    {
        std::vector<pixel_AoS> private_centers;
        int threads = omp_get_num_threads();
        if(threads > 0) private_centers.reserve((k / threads) + 10);

        #pragma omp for nowait
        for (int i = step_i; i < height; i += step_i) {

            #pragma omp simd
            for (int j = step_i; j < width; j += step_i) {
                int idx = i * img.width + j;
                private_centers.emplace_back(img.L[idx], img.a[idx], img.b[idx], i, j);
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
void parClustersPerturbation(const pixels_SoA& img, std::vector<pixel_AoS>& clusters_centers) {
    int num_centers = clusters_centers.size();
#pragma omp parallel for schedule(static)
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

void parBestMatchPixelNeighborhood(const pixels_SoA& img, const std::vector<pixel_AoS>& clusters_centers, Mat& distances, Mat& labels, const float& step) {
    std::vector<std::mutex> row_mutexes(height);
    if(row_mutexes.size() != height) row_mutexes = std::vector<std::mutex>(height);

    int num_centers = static_cast<int>(clusters_centers.size());
    int step_i = static_cast<int>(step);

    float m_div_s = m / step;
    float W = m_div_s * m_div_s;

#pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < num_centers; ++k) {
        const pixel_AoS& center = clusters_centers[k];
        int current_cluster_id = k;

        int r_min = std::max(0, center.x - step_i);
        int r_max = std::min(height, center.x + step_i);
        int c_min = std::max(0, center.y - step_i);
        int c_max = std::min(width, center.y + step_i);

        for (int i = r_min; i < r_max; ++i) {
            int row_offset = i * img.width;

            float* dist_ptr = distances.ptr<float>(i);
            int* label_ptr = labels.ptr<int>(i);

            float dx = static_cast<float>(i) - center.x;
            float dx2_W = (dx * dx) * W;

            for (int j = c_min; j < c_max; ++j) {
                int idx = row_offset + j;

                float dL = img.L[idx] - center.l;
                float da = img.a[idx] - center.a;
                float db = img.b[idx] - center.b;
                float col_dist = (dL*dL) + (da*da) + (db*db);

                float dy = static_cast<float>(j) - center.y;
                float space_dist = dx2_W + (dy * dy) * W;

                float dist_tot = col_dist + space_dist;

                if (dist_tot < dist_ptr[j]) {
                    std::lock_guard<std::mutex> lock(row_mutexes[i]);

                    if (dist_tot < dist_ptr[j]) {
                        dist_ptr[j] = dist_tot;
                        label_ptr[j] = current_cluster_id;
                    }
                }
            }
        }
    }
}

void parNewClustersCenters(const pixels_SoA& img, std::vector<pixel_AoS>& clusters_centers, const Mat& labels, double& errore_finale, const int& real_k) {
    std::vector<float> l_avg(real_k, 0), a_avg(real_k, 0), b_avg(real_k, 0);
    std::vector<float> x_avg(real_k, 0), y_avg(real_k, 0);
    std::vector<int> pixel_counter(real_k, 0);

#pragma omp parallel
    {
        std::vector<float> l_loc(real_k, 0), a_loc(real_k, 0), b_loc(real_k, 0);
        std::vector<float> x_loc(real_k, 0), y_loc(real_k, 0);
        std::vector<int> pixel_counter_loc(real_k, 0);

        #pragma omp for nowait
        for (int i = 0; i < height; i++) {
            const int* label_ptr = labels.ptr<int>(i);
            int row_offset = i * width;

            #pragma omp simd
            for (int j = 0; j < width; j++) {
                int cluster_id = label_ptr[j];
                if (cluster_id == -1) continue;

                int idx = row_offset + j;

                l_loc[cluster_id] += img.L[idx];
                a_loc[cluster_id] += img.a[idx];
                b_loc[cluster_id] += img.b[idx];

                x_loc[cluster_id] += i;
                y_loc[cluster_id] += j;

                pixel_counter_loc[cluster_id]++;
            }
        }

        #pragma omp critical
        {
            #pragma omp simd
            for (int k = 0; k < real_k; k++) {
                if (pixel_counter_loc[k] > 0) {
                    l_avg[k] += l_loc[k];
                    a_avg[k] += a_loc[k];
                    b_avg[k] += b_loc[k];
                    x_avg[k] += x_loc[k];
                    y_avg[k] += y_loc[k];
                    pixel_counter[k] += pixel_counter_loc[k];
                }
            }
        }
    }

#pragma omp parallel for schedule(static) reduction(+:errore_finale)
    for (int k_id = 0; k_id < real_k; k_id++) {
        if (pixel_counter[k_id] <= 0) continue;

        float inv_count = 1.0f / pixel_counter[k_id];

        pixel_AoS newC {
            l_avg[k_id] * inv_count,
            a_avg[k_id] * inv_count,
            b_avg[k_id] * inv_count,
            static_cast<int>(std::round(x_avg[k_id] * inv_count)),
            static_cast<int>(std::round(y_avg[k_id] * inv_count))
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
        if (segments_per_label[i].size() <= 1) continue;

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
                int dx[] = {1, -1, 0, 0};
                int dy[] = {0, 0, 1, -1};

                for(int d=0; d<4; ++d) {
                    int nx = x + dx[d];
                    int ny = y + dy[d];

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
                int new_label = std::distance(neighbor_counts.begin(), it_max);
                for (const auto&[x, y] : islandsVec[island]) {
                    labels.at<int>(x, y) = new_label;
                }
            }
        }
    }
}