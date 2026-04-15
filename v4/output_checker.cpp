#include "common.h"

// NOTE: When using this function, make sure the last part in each implementation is NOT commented, otherwise the result will be wrong
//  or give error
double compare_results(const Mat& image_seq, const Mat& image_par);

int main() {
    omp_set_num_threads(omp_get_num_procs());
    std::cout << std::fixed << std::setprecision(4);
    for (auto& image : IMAGES) {
        // if (image.first != std::string(PROJECT_SOURCE_DIR)+"/Images/COCO-2.jpg") continue;
        auto& image_path = image.first;
        auto& sp_count = image.second;
        const Mat img = imread(image_path, IMREAD_COLOR_BGR);
        if (img.empty()) {
            std::cerr << "Error during image loading, check that the path is correct. If on Linux check for upper/lower case letters as well.\n";
            return -1;
        }

        std::cout << "===== RUNNING SLIC ALGORITHM ON IMAGE " << image_path.substr(image_path.find_last_of('/')+1) << " =====\n";

        Mat img_lab;
        GaussianBlur(img, img_lab, Size(5,5) ,0);
        img_lab.convertTo(img_lab, CV_32F, 1.0/255.0);
        cvtColor(img_lab, img_lab, COLOR_BGR2Lab );
        height = img_lab.rows;
        width = img_lab.cols;

        image_SoA base_img_SoA(img_lab);
        image_SoA img_SoA{};
        Mat img_to_save;

        img_SoA = {base_img_SoA};
        Mat output_seq = run_sequential(img_SoA, sp_count);

        img_SoA = {base_img_SoA};
        Mat output_par = run_parallel(img_SoA, sp_count);
        std::cout << "Parallel accuracy: " << compare_results(output_seq, output_par) << "%\n";

        Mat output_tile;
        for (int tile_size : TILE_SIZES) {
            img_SoA = {base_img_SoA};
            output_tile = run_tile(img_SoA, sp_count, tile_size);

            std::cout << "Tile size "<< tile_size << " accuracy: " << compare_results(output_seq, output_tile)<< "%\n";
        }
    }
}

double compare_results(const Mat& image_seq, const Mat& image_par) {
    int count_diff = 0, i = 0, j = 0;
    double image_size = image_seq.rows * image_seq.cols;

    Vec3b color_seq;
    Vec3b color_par;
    for (i = 0; i < image_seq.rows; ++i) {
        auto row_seq = image_seq.row(i);
        auto row_par = image_par.row(i);
        for (j = 0; j < image_seq.cols; ++j) {
            color_seq = row_seq.at<Vec3b>(j);
            color_par = row_par.at<Vec3b>(j);
            if (std::abs(color_seq[0] - color_par[0]) > 1 || // Tolerance to avoid flagging as error micro differences
                std::abs(color_seq[1] - color_par[1]) > 1 ||
                std::abs(color_seq[2] - color_par[2]) > 1) {
                count_diff++;
                }
        }
    }

    return (image_size - count_diff) / (image_size + 0.0f) * 100.0f;
}