#include "SLIC_sequential.h"

image_SoA run_sequential(const image_SoA& image_SoA, const int &k) {
    const float step = static_cast<float>(sqrt((height * width) / k));

    pixels_SoA clusters_centers;
    clusters_centers.reserve(k);

    clustersInitialization(image_SoA, clusters_centers, step);
    const int real_k = static_cast<int>(clusters_centers.L.size());

    clustersPerturbation(image_SoA, clusters_centers);

    auto labels = Mat(height, width, CV_32S, -1);
    auto distances = Mat(height, width, CV_32F, std::numeric_limits<float>::max());

    double error = 0;

    for (int iterazioni = 0; iterazioni < 10; iterazioni++) {
        bestMatchPixelNeighborhood(image_SoA, clusters_centers, distances, labels, step);
        newClustersCenters(image_SoA, clusters_centers, labels, error, real_k);
    }
    enforceConnectivity(labels, real_k);

    Mat output = image_SoA.to_Mat(height, width);
    cvtColor(output, output, COLOR_Lab2BGR);
    output.convertTo(output, CV_8UC3, 255.0);
    applySegmentationColored(output, labels, clusters_centers);
    imwrite(std::string(PROJECT_SOURCE_DIR) + "/output/sequenziale/result.png", output);
    
    return image_SoA;
}

void clustersInitialization(const image_SoA& img, pixels_SoA& clusters_centers, const float& step) {
    int step_i = static_cast<int>(step);
    for (int i = step_i >> 1; i < height; i += step_i) {
        for (int j = step_i >> 1; j < width; j += step_i) {
            int pos = i * width + j;

            clusters_centers.L.emplace_back(img.L[pos]);
            clusters_centers.a.emplace_back(img.a[pos]);
            clusters_centers.b.emplace_back(img.b[pos]);
            clusters_centers.x.emplace_back(i);
            clusters_centers.y.emplace_back(j);
        }
    }

    clusters_centers.L.shrink_to_fit();
    clusters_centers.a.shrink_to_fit();
    clusters_centers.b.shrink_to_fit();
    clusters_centers.x.shrink_to_fit();
    clusters_centers.y.shrink_to_fit();
}

void clustersPerturbation(const image_SoA& img, pixels_SoA& clusters_centers) {
    for (int cc_index = 0; cc_index < clusters_centers.L.size(); cc_index++) {
        int x   = clusters_centers.x[cc_index];
        int y   = clusters_centers.y[cc_index];

        float min_gradient = std::numeric_limits<float>::max();
        int min_x = x, min_y = y;

        for (int i = x - 1; i < x + 2; i++) {
            if (i < 1 || i >= height - 1) continue;

            int row_prev = (i - 1) * width;
            int row_curr = i * width;
            int row_next = (i + 1) * width;

            for (int j = y - 1; j < y + 2; j++) {
                if (j < 1 || j >= width - 1) continue;

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

void bestMatchPixelNeighborhood(const image_SoA& img, const pixels_SoA& clusters_centers, Mat& distances, Mat& labels, const float& step) {
    const float W = (m * m) / (step * step);

    // Formula distanza: D^2 = dc^2 + (ds^2 * (m/S)^2)
    for (int cc_index = 0; cc_index < clusters_centers.L.size(); cc_index++) {
        float pixelc_l = clusters_centers.L[cc_index];
        float pixelc_a = clusters_centers.a[cc_index];
        float pixelc_b = clusters_centers.b[cc_index];
        int pixelc_x = clusters_centers.x[cc_index];
        int pixelc_y = clusters_centers.y[cc_index];

        int r_min = std::max(0, static_cast<int>(pixelc_x - step));
        int r_max = std::min(height, static_cast<int>(pixelc_x + step));
        int c_min = std::max(0, static_cast<int>(pixelc_y - step));
        int c_max = std::min(width, static_cast<int>(pixelc_y + step));

        for (int i = r_min; i < r_max; i++) {
            int row_offset = i * width;

            float dx = static_cast<float>(i) - pixelc_x;
            float dx2 = dx * dx;

            float* dist_row = distances.ptr<float>(i);
            int* labels_row = labels.ptr<int>(i);

            for (int j = c_min; j < c_max; j++) {
                int idx = row_offset + j;

                float L = img.L[idx];
                float a = img.a[idx];
                float b = img.b[idx];

                float dL = L - pixelc_l;
                float da = a - pixelc_a;
                float db = b - pixelc_b;
                float dist_color = (dL*dL) + (da*da) + (db*db);

                float dy = static_cast<float>(j) - pixelc_y;
                float dy2 = dy * dy;

                float dist_space = (dx2 + dy2);
                float distance_kc = dist_color + (dist_space * W);

                if (distance_kc < dist_row[j]) {
                    dist_row[j] = distance_kc;
                    labels_row[j] = cc_index;
                }
            }
        }
    }
}

void newClustersCenters(const image_SoA& img, pixels_SoA& clusters_centers, const Mat& labels, double& error, const int& real_k) {
    std::vector<float> l_avg(real_k, 0.0), a_avg(real_k, 0.0), b_avg(real_k, 0.0);
    std::vector<float> x_avg(real_k, 0.0), y_avg(real_k, 0.0);
    std::vector<int> pixel_counter(real_k, 0);

    for (int i = 0; i < height; i++) {
        int row_offset = i * width;

        for (int j = 0; j < width; j++) {
            int idx = row_offset + j;

            int label = labels.at<int>(i, j); // labels è ancora Mat
            if (label == -1) continue;

            l_avg[label] += img.L[idx];
            a_avg[label] += img.a[idx];
            b_avg[label] += img.b[idx];

            x_avg[label] += i;
            y_avg[label] += j;

            pixel_counter[label]++;
        }
    }

    for (int k_id = 0; k_id < real_k; k_id++) {
        if (pixel_counter[k_id] == 0) continue;
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

void enforceConnectivity(Mat& labels, const int& real_k) {
    auto visited = std::vector<bool>(width * height, false);
    std::vector<std::vector<std::vector<std::pair<int, int>>>> segments_per_label(real_k);

    for (int i = 0; i < height; i++) {
        int* label_row = labels.ptr<int>(i);

        for (int j = 0; j < width; j++) {
            if (visited[i * width + j]) continue;

            segments_per_label[label_row[j]].push_back(floodFillBFS(labels, visited, i, j));
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