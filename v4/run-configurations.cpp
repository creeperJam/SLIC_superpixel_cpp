#include "common.h"

void logs_saver(const logs& logs, const std::string& image, const std::string& algorithm, const std::filesystem::path& logs_path);

int main(const int argc, char *argv[]) {
    std::time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time), "%Y-%m-%d_%H-%M-%S");
    const std::filesystem::path logs_path = std::filesystem::path(PROJECT_SOURCE_DIR) / "output" / "times" / (oss.str() + "_time_logs.csv");
    std::cout << logs_path.string() << std::endl;
    std::ofstream logs_file(logs_path.string());
    logs_file << "Image,Algorithm,Thread_count,Tile_size,"
                 "Wall_mean_ms,Wall_max_ms,Wall_min_ms,Wall_std_ms,"
                 "CPU_mean_ms,CPU_max_ms,CPU_min_ms,CPU_std_ms\n";
    logs_file.close();


    const int THREADS_COUNT = omp_get_num_procs();

    std::filesystem::path output_path = std::filesystem::path(PROJECT_SOURCE_DIR) / "output"  ;
    if (!std::filesystem::exists(output_path)) {
        std::filesystem::create_directory(output_path);
    }
    output_path = output_path / "sequenziale";
    if (!std::filesystem::exists(output_path)) {
        std::filesystem::create_directory(output_path);
    }
    output_path = std::filesystem::path(PROJECT_SOURCE_DIR) / "output" / "parallelo"  ;
    if (!std::filesystem::exists(output_path)) {
        std::filesystem::create_directory(output_path);
    }
    output_path = std::filesystem::path(PROJECT_SOURCE_DIR) / "output" / "tile"  ;
    if (!std::filesystem::exists(output_path)) {
        std::filesystem::create_directory(output_path);
    }
    output_path = std::filesystem::path(PROJECT_SOURCE_DIR) / "output" / "times"  ;
    if (!std::filesystem::exists(output_path)) {
        std::filesystem::create_directory(output_path);
    }

    for (const auto& image : IMAGES) {
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
        image_SoA image_SoA{};

        // ESECUZIONE SEQUENZIALE
        std::cout << "Running sequential warmup runs --- ";
        for (int i = 0; i < WARMUP_RUNS; i++) {
            image_SoA = {base_img_SoA};
            run_sequential(image_SoA, sp_count);
        }
        std::cout << "FINISHED\n";

        std::cout << "Running sequential runs        --- ";
        logs sequential_logs;
        for (int i = 0; i < NUM_RUNS; i++) {
            image_SoA = {base_img_SoA};
            const clock_t cpu_start = clock();
            const auto wall_start = std::chrono::high_resolution_clock::now();

            run_sequential(image_SoA, sp_count);

            const clock_t cpu_end = std::clock();
            const auto wall_end = std::chrono::high_resolution_clock::now();
            double wall_time = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
            double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

            sequential_logs.add({wall_time, cpu_time}, i);
        }
        std::cout << "FINISHED\n";
        logs_saver(sequential_logs, image_path, "SEQUENTIAL", logs_path);

        for (int num_threads = 1; num_threads <= THREADS_COUNT; num_threads <<= 1) {
            std::cout << " ---- Running threads count: " << num_threads << " ----\n";
            omp_set_num_threads(num_threads);

            std::cout << "Running parallel version warmup --- ";
            for (int i = 0; i < WARMUP_RUNS; i++) {
                image_SoA = {base_img_SoA};
                run_parallel(image_SoA, sp_count);
            }
            std::cout << "FINISHED\n";

            std::cout << "Running parallel runs           --- ";
            logs parallel_logs;
            parallel_logs.thread_num = num_threads;
            for (int i = 0; i < NUM_RUNS; i++) {
                image_SoA = {base_img_SoA};
                const clock_t cpu_start = clock();
                const auto wall_start = std::chrono::high_resolution_clock::now();

                run_parallel(image_SoA, sp_count);

                const clock_t cpu_end = std::clock();
                const auto wall_end = std::chrono::high_resolution_clock::now();
                double wall_time = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
                double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

                parallel_logs.add({wall_time, cpu_time}, i);
            }
            std::cout << "FINISHED\n";
            logs_saver(parallel_logs, image_path, "PARALLEL", logs_path);

            std::cout << "---- Running tile version runs ----\n";
            for (auto tile_size : TILE_SIZES) {
                if (img.rows * img.cols < tile_size * tile_size * num_threads) {
                    std::cout << " ----- SKIPPING TILE SIZE " << tile_size << " -----\n";
                    continue;
                }
                std::cout << "Running tile size " << tile_size << " warmups\n";
                for (int i = 0; i < WARMUP_RUNS; i++) {
                    image_SoA = {base_img_SoA};
                    run_tile(image_SoA, sp_count, tile_size);
                }
                std::cout << "Running tile size " << tile_size << " runs\n";
                logs tile_logs;
                tile_logs.tile_size = tile_size;
                tile_logs.thread_num = num_threads;
                for (int i = 0; i < NUM_RUNS; i++) {
                    image_SoA = {base_img_SoA};
                    const clock_t cpu_start = clock();
                    const auto wall_start = std::chrono::high_resolution_clock::now();

                    run_tile(image_SoA, sp_count, tile_size);

                    const clock_t cpu_end = std::clock();
                    const auto wall_end = std::chrono::high_resolution_clock::now();
                    double wall_time = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
                    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

                    tile_logs.add({wall_time, cpu_time}, i);
                }
                std::cout << "Finished tile size " << tile_size << " runs\n";
                logs_saver(tile_logs, image_path, "TILE", logs_path);
            }
            std::cout << "Finished tile version runs\n";
        }
    }
}

void logs_saver(const logs& logs, const std::string& image, const std::string& algorithm, const std::filesystem::path& logs_path) {
    std::ofstream logs_file(logs_path, std::ios::app);
    std::string image_name = image.substr(image.find_last_of('/')+1);

    double wall_mean_ms = 0.0;
    for (auto& wall_time : logs.wall_times) {
        wall_mean_ms += wall_time;
    }
    wall_mean_ms /= NUM_RUNS;
    double wall_std_ms = 0.0;
    for (auto& wall_time : logs.wall_times) {
        wall_std_ms += (wall_time - wall_mean_ms) * (wall_time - wall_mean_ms);
    }
    wall_std_ms /= NUM_RUNS - 1;
    wall_std_ms = std::sqrt(wall_std_ms);
    const double wall_max_ms = *std::max_element(logs.wall_times.begin(), logs.wall_times.end());
    const double wall_min_ms = *std::min_element(logs.wall_times.begin(), logs.wall_times.end());

    double cpu_mean_ms = 0.0;
    for (auto& cpu_time : logs.cpu_times) {
        cpu_mean_ms += cpu_time;
    }
    cpu_mean_ms /= NUM_RUNS;
    double cpu_std_ms = 0.0;
    for (auto& cpu_time : logs.cpu_times) {
        cpu_std_ms += (cpu_time - wall_mean_ms) * (cpu_time - wall_mean_ms);
    }
    cpu_std_ms /= NUM_RUNS - 1;
    cpu_std_ms = std::sqrt(cpu_std_ms);

    const double cpu_max_ms = *std::max_element(logs.cpu_times.begin(), logs.cpu_times.end());
    const double cpu_min_ms = *std::min_element(logs.cpu_times.begin(), logs.cpu_times.end());

    logs_file << image_name << "," << algorithm << "," << logs.thread_num << "," << logs.tile_size << "," <<
        wall_mean_ms << "," << wall_max_ms << "," << wall_min_ms << "," << wall_std_ms << "," <<
        cpu_mean_ms << "," << cpu_max_ms << "," << cpu_min_ms << "," << cpu_std_ms << "\n";

    logs_file.close();
}