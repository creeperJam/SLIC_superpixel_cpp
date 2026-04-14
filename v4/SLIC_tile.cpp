#include "SLIC_tile.h"

image_SoA run_tile(const image_SoA& image_SoA, const int& k, const int& tile_size) {
    const float step = static_cast<float>(sqrt((height * width) / k));

    pixels_SoA clusters_centers;
    clusters_centers.reserve(k);

    tileClustersInitialization(image_SoA, clusters_centers, step, k);
    const int real_k = static_cast<int>(clusters_centers.L.size());

    tileClustersPerturbation(image_SoA, clusters_centers);

    auto labels = Mat(height, width, CV_32S, -1);
    auto distances = Mat(height, width, CV_32F, std::numeric_limits<float>::max());

    double error = 0;

    for (int iterazioni = 0; iterazioni < 10; iterazioni++) {
        tileBestMatchPixelNeighborhood(image_SoA, clusters_centers, distances, labels, step, tile_size);
        tileNewClustersCenters(image_SoA, clusters_centers, labels, error, real_k);
    }
    tileEnforceConnectivity(labels, real_k);

    Mat output = image_SoA.to_Mat(height, width);
    cvtColor(output, output, COLOR_Lab2BGR);
    output.convertTo(output, CV_8UC3, 255.0);
    applySegmentationColored(output, labels, clusters_centers);
    imwrite(std::string(PROJECT_SOURCE_DIR) + "/output/tile/result_" + std::to_string(tile_size) + ".png", output);

    return image_SoA;
}

void tileClustersInitialization(const image_SoA& img, pixels_SoA& clusters_centers, const float& step, const int& k) {
    int step_i = static_cast<int>(step);
#pragma omp parallel
    {
        pixels_SoA private_centers;
        int threads = omp_get_num_threads();
        if (threads > 0) {
            private_centers.reserve((k / threads) + 10);
        }

        #pragma omp for nowait
        for (int i = step_i >> 1; i < height; i += step_i) {
            for (int j = step_i >> 1; j < width; j += step_i) {
                int pos = i * width + j;

                private_centers.L.emplace_back(img.L[pos]);
                private_centers.a.emplace_back(img.a[pos]);
                private_centers.b.emplace_back(img.b[pos]);
                private_centers.x.emplace_back(i);
                private_centers.y.emplace_back(j);
            }
        }

        #pragma omp critical
        {
            clusters_centers.L.insert(
                clusters_centers.L.end(),
                private_centers.L.begin(),
                private_centers.L.end()
                );
            clusters_centers.a.insert(
                clusters_centers.a.end(),
                private_centers.a.begin(),
                private_centers.a.end()
                );
            clusters_centers.b.insert(
                clusters_centers.b.end(),
                private_centers.b.begin(),
                private_centers.b.end()
                );
            clusters_centers.x.insert(
                clusters_centers.x.end(),
                private_centers.x.begin(),
                private_centers.x.end()
                );
            clusters_centers.y.insert(
                clusters_centers.y.end(),
                private_centers.y.begin(),
                private_centers.y.end()
            );
        }
    }
}

// Perturbazione dei cluster in un vicinato 3x3
void tileClustersPerturbation(const image_SoA& img, pixels_SoA& clusters_centers) {
    int num_centers = clusters_centers.L.size();
#pragma omp parallel for schedule(static)
    for (int cc_index = 0; cc_index < num_centers; cc_index++) {
        int x = clusters_centers.x[cc_index];
        int y = clusters_centers.y[cc_index];

        float min_gradient = std::numeric_limits<float>::max();
        int min_x = x, min_y = y;

        for (int i = x - 1; i < x + 2; i++) {
            if (i < 1 || i >= height - 1) continue;

            int row_prev = (i - 1) * width;
            int row_curr = i * width;
            int row_next = (i + 1) * width;

            for (int j = y - 1; j < y + 2; j++) {
                if (j < 1 || j >= width - 1) continue;

                // Calcolo indici per il gradiente usando SoA
                int idx_up = row_curr + (j - 1); // i, j-1
                int idx_down = row_curr + (j + 1); // i, j+1
                int idx_left = row_prev + j; // i-1, j
                int idx_right = row_next + j; // i+1, j

                float l1 = img.L[idx_right]; // i+1
                float a1 = img.a[idx_right];
                float b1 = img.b[idx_right];

                float l2 = img.L[idx_left]; // i-1
                float a2 = img.a[idx_left];
                float b2 = img.b[idx_left];

                float l3 = img.L[idx_down]; // j+1
                float a3 = img.a[idx_down];
                float b3 = img.b[idx_down];

                float l4 = img.L[idx_up]; // j-1
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

        clusters_centers.x[cc_index] = min_x;
        clusters_centers.y[cc_index] = min_y;
        int idx_min = min_x * width + min_y;
        clusters_centers.L[cc_index] = img.L[idx_min];
        clusters_centers.a[cc_index] = img.a[idx_min];
        clusters_centers.b[cc_index] = img.b[idx_min];
    }
}

void tileBestMatchPixelNeighborhood(const image_SoA& img, const pixels_SoA& clusters_centers, Mat& distances, Mat& labels, const float& step, const int tile_size) {
    float m_div_s = m / step;
    float W = m_div_s * m_div_s;
    int reserve_size = ((tile_size / step) + 2) * ((tile_size / step) + 2);

#pragma omp parallel
    {
        pixels_SoA local_centers(reserve_size);
        std::vector<int> id_local_centers;
        id_local_centers.reserve(reserve_size);
#pragma omp for schedule(dynamic)
        for (int r = 0; r < height; r+=tile_size) {
            for (int c = 0; c < width; c+=tile_size) {
                local_centers.clear();
                id_local_centers.clear();

                for (int i = 0; i < clusters_centers.L.size(); i++) {
                    int x = clusters_centers.x[i];
                    int y = clusters_centers.y[i];

                    if (r <= x + step && r + tile_size >= x - step && c <= y + step && c + tile_size >= y - step) {
                        local_centers.emplace_back(clusters_centers.L[i], clusters_centers.a[i], clusters_centers.b[i], x, y);
                        id_local_centers.emplace_back(i);
                    }
                }
                int height_upper_bound = std::min(r + tile_size, height);
                int width_upper_bound = std::min(c + tile_size, width);
                for (int i = r; i < height_upper_bound; i++) {
                    int row_offset = i * width;

                    float* dist_ptr = distances.ptr<float>(i);
                    int* label_ptr = labels.ptr<int>(i);

                    for (int j = c; j < width_upper_bound; j++) {
                        int idx = row_offset + j;
                        for (int cc_index = 0; cc_index < local_centers.L.size(); cc_index++) {
                            float dL = img.L[idx] - local_centers.L[cc_index];
                            float da = img.a[idx] - local_centers.a[cc_index];
                            float db = img.b[idx] - local_centers.b[cc_index];
                            float col_dist = (dL*dL) + (da*da) + (db*db);

                            float dx = static_cast<float>(i) - local_centers.x[cc_index];
                            float dy = static_cast<float>(j) - local_centers.y[cc_index];
                            float space_dist = W * ((dx * dx) +  (dy * dy));

                            float dist_tot = col_dist + space_dist;

                            if (dist_tot < dist_ptr[j]) {
                                dist_ptr[j] = dist_tot;
                                label_ptr[j] = id_local_centers[cc_index];
                            }
                        }
                    }
                }
            }
        }
    }
}

void tileNewClustersCenters(const image_SoA& img, pixels_SoA& clusters_centers, const Mat& labels, double& error, const int& real_k) {
    std::vector<float> l_avg(real_k, 0.0f), a_avg(real_k, 0.0f), b_avg(real_k, 0.0f);
    std::vector<float> x_avg(real_k, 0.0f), y_avg(real_k, 0.0f);
    std::vector<int> pixel_counter(real_k, 0);

    float* ptr_l = l_avg.data();
    float* ptr_a = a_avg.data();
    float* ptr_b = b_avg.data();
    float* ptr_x = x_avg.data();
    float* ptr_y = y_avg.data();
    int* ptr_counter = pixel_counter.data();

    #pragma omp parallel for schedule(static) \
        reduction(+: ptr_l[:real_k], ptr_a[:real_k], ptr_b[:real_k], ptr_x[:real_k], ptr_y[:real_k], ptr_counter[:real_k])
    for (int i = 0; i < height; i++) {
        const int* label_ptr = labels.ptr<int>(i);
        int row_offset = i * width;

        for (int j = 0; j < width; j++) {
            int cluster_id = label_ptr[j];
            if (cluster_id == -1) continue;

            int idx = row_offset + j;

            ptr_l[cluster_id] += img.L[idx];
            ptr_a[cluster_id] += img.a[idx];
            ptr_b[cluster_id] += img.b[idx];

            ptr_x[cluster_id] += i;
            ptr_y[cluster_id] += j;

            ptr_counter[cluster_id]++;
        }
    }

    #pragma omp parallel for schedule(static) reduction(+:error)
    for (int k_id = 0; k_id < real_k; k_id++) {
        if (pixel_counter[k_id] <= 0) continue;

        float inv_count = 1.0f / pixel_counter[k_id];

        pixel newC {
            (l_avg[k_id] * inv_count),
            (a_avg[k_id] * inv_count),
            (b_avg[k_id] * inv_count),
            static_cast<int>((x_avg[k_id] * inv_count) + 0.5f),
            static_cast<int>((y_avg[k_id] * inv_count) + 0.5f)
        };

        error += manhattanDist(newC, {
            clusters_centers.L[k_id],
            clusters_centers.a[k_id],
            clusters_centers.b[k_id],
            clusters_centers.x[k_id],
            clusters_centers.y[k_id]}
        );

        clusters_centers.L[k_id] = newC.l;
        clusters_centers.a[k_id] = newC.a;
        clusters_centers.b[k_id] = newC.b;
        clusters_centers.x[k_id] = newC.x;
        clusters_centers.y[k_id] = newC.y;
    }
}

void tileEnforceConnectivity(Mat& labels, const int& real_k) {
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
                int new_label = std::distance(neighbor_counts.begin(), it_max);
                for (const auto&[x, y] : islandsVec[island]) {
                    labels.at<int>(x, y) = new_label;
                }
            }
        }
    }
}