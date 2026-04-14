#pragma once

#include "common.h"

/**
 * @brief Given the image, a vector that will contain the cluster center and the step, this will save all the values of
 * cluster center into the vector.
 *
 * @param img contains all image pixels in the CIE-Lab color-space
 * @param clusters_centers contains the center pixel of all clusters
 * @param step distance between each cluster centers
 */
void clustersInitialization(const image_SoA& img, pixels_SoA& clusters_centers, const float& step);

/**
 * @brief Perturbs each cluster center in a 3x3 neighborhood to the lowest gradient position
 *
 * @param img contains all image pixels in the CIE-Lab color-space
 * @param clusters_centers contains the center pixel of all clusters
 */
void clustersPerturbation(const image_SoA& img, pixels_SoA& clusters_centers);

/**
 * @brief Each cluster centers checks all pixels in a 2Sx2S area, calculates the distance from it and if smaller than the
 *  one already present, assigns the pixels to its own cluster.
 *
 * @param img contains all image pixels in the CIE-Lab color-space
 * @param clusters_centers contains the center pixel of all clusters
 * @param distances contains the current lowest distance of each pixel to its SP center (if max float value, not a part of any SP)
 * @param labels contains the ID of the SP the pixel is part of (if -1, not a part of any SP)
 * @param step distance between each cluster centers
 */
void bestMatchPixelNeighborhood(const image_SoA& img, const pixels_SoA& clusters_centers, Mat& distances, Mat& labels, const float& step);

/**
 * @brief Calculates the average color and position of all the pixels in every SP and makes it the new cluster center
 *
 * @param img contains all image pixels in the CIE-Lab color-space
 * @param clusters_centers contains the center pixel of all clusters
 * @param labels contains the ID of the SP the pixel is part of (if -1, not a part of any SP)
 * @param error current error of the algorithm, can be used in the main to run the algorithm until it's lower than a threshold
 * @param real_k contains the actual amount of superpixels (not the input value)
 */
void newClustersCenters(const image_SoA& img, pixels_SoA& clusters_centers, const Mat& labels, double& error, const int& real_k);

/**
 * @brief Checks for any orphaned islands of pixels disconnected from the main SP and assigns them to the best matching cluster
 *  nearby.
 *
 * @param labels contains the ID of the SP the pixel is part of (if -1, not a part of any SP)
 * @param real_k contains the actual amount of superpixels (not the input value)
 */
void enforceConnectivity(Mat& labels, const int& real_k);