#pragma once

#include <omp.h>
#include <mutex>
#include "common.h"

void parClustersInizialization(const pixels_SoA& img, std::vector<pixel_AoS>& clusters_centers, const float& step, const int& k);
void parClustersPerturbation(const pixels_SoA& img, std::vector<pixel_AoS>& clusters_centers);
void parBestMatchPixelNeighborhood(const pixels_SoA& img, const std::vector<pixel_AoS>& clusters_centers, Mat& distances, Mat& labels, const float& step);
void parNewClustersCenters(const pixels_SoA& img, std::vector<pixel_AoS>& clusters_centers, const Mat& labels, double& errore_finale, const int& real_k);
void parEnforceConnectivity(Mat& labels, const int& real_k);