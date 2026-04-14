#pragma once

#include "common.h"

void clustersInizialization(const pixels_SoA& img, std::vector<pixel_AoS>& clusters_centers, const float& step);
void clustersPerturbation(const pixels_SoA& img, std::vector<pixel_AoS>& clusters_centers);
void bestMatchPixelNeighborhoood(const pixels_SoA& img, const std::vector<pixel_AoS>& clusters_centers, Mat& distances, Mat& labels, const float& step, int& sp_counter);
void newClustersCenters(const pixels_SoA& img, std::vector<pixel_AoS>& clusters_centers, const Mat& labels, double& errore_finale, const int& real_k);
void enforceConnectivity(Mat& labels, const int& real_k);

void apply_segmentation_colored(Mat& original_img, const Mat& labels, const std::vector<pixel_AoS>& centers);