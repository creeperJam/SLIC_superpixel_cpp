#pragma once

#include "common.h"

void clustersInizialization(const Mat& img, std::vector<pixel_AoS>& clusters_centers, const float& step);
void clustersPerturbation(const Mat& img, std::vector<pixel_AoS>& clusters_centers);
void bestMatchPixelNeighborhoood(Mat& img, const std::vector<pixel_AoS>& clusters_centers, Mat& distances, Mat& labels, const float& step, int& sp_counter);
void newClustersCenters(const Mat& img, std::vector<pixel_AoS>& clusters_centers, const Mat& labels, double& errore_finale, const int& real_k);
void enforceConnectivity(Mat& labels, const int& real_k);