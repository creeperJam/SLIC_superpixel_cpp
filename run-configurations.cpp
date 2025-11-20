//
// Created by albi0 on 19/11/2025.
//

#include "common.h"

int main(const int argc, char *argv[]) {
    if (argc <= 2) {
        printf("Errore, argomenti mancanti!! Formato comando: ./SLIC_SP_req path/to/image SP-amount");
        return -2;
    }
    const std::string img_path = argv[1];
    const int k = std::stoi(argv[2]);

    const double avg_seq_time = run_sequential(img_path, k);
    const double avg_par_time = run_parallel(img_path, k);

    printf("Sequential time: %.4f\n", avg_seq_time);
    printf("Parallel time: %.4f\n", avg_par_time);
    printf("Speedup parallel vs sequential: %.4f", avg_seq_time / avg_par_time);
}