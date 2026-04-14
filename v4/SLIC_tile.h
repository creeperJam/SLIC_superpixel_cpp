#ifndef SLIC_SUPERPIXEL_SLIC_TILE_H
#define SLIC_SUPERPIXEL_SLIC_TILE_H

#pragma once

#include <omp.h>
#include <mutex>
#include "common.h"

void tileClustersInitialization(const image_SoA& img, pixels_SoA& clusters_centers, const float& step, const int& k);
void tileClustersPerturbation(const image_SoA& img, pixels_SoA& clusters_centers);
void tileBestMatchPixelNeighborhood(const image_SoA& img, const pixels_SoA& clusters_centers, Mat& distances, Mat& labels, const float& step, const int tile_size);
void tileNewClustersCenters(const image_SoA& img, pixels_SoA& clusters_centers, const Mat& labels, double& errore_finale, const int& real_k);
void tileEnforceConnectivity(Mat& labels, const int& real_k);

#endif //SLIC_SUPERPIXEL_SLIC_TILE_H