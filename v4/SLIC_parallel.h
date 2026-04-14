#pragma once

#include <mutex>
#include "common.h"

void parClustersInitialization(const image_SoA& img, pixels_SoA& clusters_centers, const float& step, const int& k);
void parClustersPerturbation(const image_SoA& img, pixels_SoA& clusters_centers);
void parBestMatchPixelNeighborhood(const image_SoA& img, const pixels_SoA& clusters_centers, Mat& distances, Mat& labels, const float& step, std::vector<std::mutex>& row_mutexes);
void parNewClustersCenters(const image_SoA& img, pixels_SoA& clusters_centers, const Mat& labels, double& errore_finale, const int& real_k);
void parEnforceConnectivity(Mat& labels, const int& real_k);