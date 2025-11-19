//
// Created by albi0 on 19/11/2025.
//
#pragma once

#include <omp.h>
#include "common.h"

void parClustersInizialization(const Mat& img, std::vector<pixel>& clusters_centers, const float& step, const int& k);
void parClustersPerturbation(const Mat& img, std::vector<pixel>& clusters_centers);
void parBestMatchPixelNeighborhood(Mat& img, const std::vector<pixel>& clusters_centers, Mat& distances, Mat& labels, const float& step);
void parNewClustersCenters(const Mat& img, std::vector<pixel>& clusters_centers, const Mat& labels, double& errore_finale, const int& real_k);
void parEnforceConnectivity(Mat& labels, const int& real_k);