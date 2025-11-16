/*
 * SP = Superpixel
 * K = number of approx. equally sized SP
 * N = Number of pixels in the image
 * So the approx. size of each SP comes out to be: N/K
 *
 * S = sqrt(SP size) interval for superpixel centers
 *
 * For the normalization in a 5D space we'll use the following formulas:
 * d_(lab) = sqrt( (l_k - l_i)^2 + (a_k - a_i)^2 + (b_k - b_i)^2 )
 * d_(xy)  = sqrt( (x_k - x_i)^2 + (y_k - y_i)^2 )
 * D_s     = d_(lab) + ( m/s * d_(xy) )
 *
 * for m we'll use the same value as the research paper used, m = 10, but it can be customized with any value in the range [1,20]
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <limits>
#include <ctime>

#define m 10
#define K 400 // TODO: Change into command line input instead of constant declaration

using namespace cv;

// SoA per gestire i SP
struct pixel {
    float l, a, b; // Color
    int x, y; // Position
};
int height;
int width;

float gradient(const Vec3f& lab1, const Vec3f& lab2, const Vec3f& lab3, const Vec3f& lab4);
float distance(const pixel& p1, const pixel& p2, const float& S);
inline float manhattanDist(const pixel& p1, const pixel& p2);
std::vector<std::pair<int, int>> floodFillBFS(const Mat& labels, std::vector<bool>& visited, int x, int y);

int main(int argc, char *argv[]) {

    if (argc <= 2) {
        printf("Errore, argomenti mancanti!! Formato comando: ./SLIC_SP_req path/to/image SP-amount");
        return -2;
    }
    std::string img_path = argv[1];
    const int k = std::stoi(argv[2]);

    Mat img = imread(img_path, IMREAD_COLOR_BGR);
    if (img.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    clock_t start = clock();

    std::vector<pixel> centers;
    centers.reserve(K);

    Mat labImg;
    Mat img_fin_test = img.clone();
    img.convertTo(img, CV_32F, 1.0/255.0); // convertita a BGR 32-bit float
    cvtColor(img, labImg, COLOR_BGR2Lab ); // Convertita al colorspace CIE Lab
    // cvtColor(img, labImg, COLOR_BGR2Lab ); // Convertita al colorspace CIE Lab

    height = labImg.rows;
    width = labImg.cols;
    int size = height * width;
    const auto S = static_cast<float>(sqrt(size / K));

    for (int i = static_cast<int>(S/2); i < height; i += static_cast<int>(S)) {
        for (int j = static_cast<int>(S/2); j < width; j += static_cast<int>(S)) {
            auto lab = labImg.at<Vec3f>(i, j);

            centers.push_back({lab[0], lab[1], lab[2], j, i});
        }
    }

    int real_k = centers.size();

    // Passo 2 dell'algoritmo, perturbazione in 3x3
    for (auto&[l, a, b, x, y] : centers) {
        float gradientMin = std::numeric_limits<float>::max();
        int xMin = x, yMin = y;

        for (int i = y - 1; i <= y + 1; i++) {
            if (i == 0 || i == height - 1) continue;
            for (int j = x - 1; j <= x + 1; j++) {
                if (j == 0 || j == width - 1) continue;

                auto lab1 = labImg.at<Vec3f>(i + 1, j);
                auto lab2 = labImg.at<Vec3f>(i - 1, j);
                auto lab3 = labImg.at<Vec3f>(i, j + 1);
                auto lab4 = labImg.at<Vec3f>(i, j - 1);

                float g = gradient(lab1, lab2, lab3, lab4);
                if (g < gradientMin) {
                    gradientMin = g;
                    xMin = j;
                    yMin = i;
                }
            }
        }

        x = xMin; // colonna
        y = yMin; // riga
        auto labMin = labImg.at<Vec3f>(yMin, xMin);
        l = labMin[0];
        a = labMin[1];
        b = labMin[2];
    }

    // Passo 3-6
    auto labels = Mat(height, width, CV_32S, -1);
    auto distances = Mat(height, width, CV_32F, std::numeric_limits<float>::max());

    double erroreFinale = 0;
    for (int iterazioni = 0; iterazioni < 10; iterazioni++) {
        int spCounter = 0;
        double erroreTotale = 0;
        for (const pixel& pC : centers) {
            const int iMax = pC.y < height - (int)S ? pC.y + (int)S : height; // Riga finale
            const int jMax = pC.x < width - (int)S ? pC.x + (int)S : width;   // Colonna finale

            for (int i = pC.y > (int)S ? pC.y - (int)S : 0; i < iMax; i++) {
                for (int j = pC.x > (int)S ? pC.x - (int)S : 0; j < jMax; j++) {
                    Vec3f labMin = labImg.at<Vec3f>(i, j);
                    pixel p = { labMin[0],labMin[1] ,labMin[2], j, i};
                    float distanceKI = distance(pC, p, S);

                    if (distanceKI < distances.at<float>(i, j)) {
                        distances.at<float>(i, j) = distanceKI;
                        labels.at<int>(i, j) = spCounter;
                    }
                }
            }
            spCounter++;
        }
        // Passo 7
        for (int k = 0; k < spCounter; k++) {
            float l_avg = 0, a_avg = 0, b_avg = 0;
            float x_avg = 0, y_avg = 0;
            int pixelCounter = 0;

            for ( int i = 0; i < height; i++) {
                for ( int j = 0; j < width; j++) {
                    if (labels.at<int>(i, j) == k) {
                        auto lab = labImg.at<Vec3f>(i, j);
                        l_avg += lab[0];
                        a_avg += lab[1];
                        b_avg += lab[2];
                        x_avg += j;
                        y_avg += i;
                        pixelCounter++;
                    }
                }
            }

            pixel newC = {(l_avg / pixelCounter), (a_avg / pixelCounter), (b_avg / pixelCounter), static_cast<int>(std::round(x_avg / pixelCounter)), static_cast<int>(std::round(y_avg / pixelCounter))};
            erroreTotale += manhattanDist(newC, centers[k]);
            centers[k] = newC;
        }
        erroreFinale = erroreTotale;
    }
    // Passo 9
    auto visited = std::vector<bool>(width * height, false);
    std::vector<std::vector<std::vector<std::pair<int, int>>>> segments_per_label(real_k);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (visited[i * width + j]) continue;
            segments_per_label[labels.at<int>(i, j)].push_back(floodFillBFS(labels, visited, i, j));
        }
    }

    for (int i = 0; i < real_k; i++) {
        if (segments_per_label[i].size() == 1) continue;

        auto& islandsVec = segments_per_label[i];

        // Trova isola più grande e eliminala dal vettore islandsVec
        int pixel_count = islandsVec[0].size();
        int biggest_index = 0;
        for (int island = 1; island < islandsVec.size(); island++) {
            if (islandsVec[island].size() > pixel_count) {
                pixel_count = islandsVec[island].size();
                biggest_index = island;
            }
        }

        for (int island = 0; island < islandsVec.size(); island++) {
            if (island == biggest_index) continue;

            std::vector<int> neighbor_counts(real_k, 0);
            for (const auto&[x, y] : islandsVec[island]) {
                std::array<std::pair<int, int>, 4> dir = {std::pair{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
                for (std::pair<int, int>& it : dir) {
                    int nx = x + it.first;
                    int ny = y + it.second;

                    // Check bordi
                    if (nx >= 0 && nx < height && ny >= 0 && ny < width) {
                        int cluster = labels.at<int>(nx, ny);
                        if (cluster != i) {
                            neighbor_counts[cluster]++;
                        }
                    }
                }
            }

            auto it_max = std::max_element(neighbor_counts.begin(), neighbor_counts.end());
            if (*it_max > 0) {
                for (const auto&[x, y] : islandsVec[island]) {
                    labels.at<int>(x, y) = std::distance(neighbor_counts.begin(), it_max);
                }
            }
        }
    }

    clock_t end = clock();
    printf( "Tempo finale: %.10f", ((double)(end - start)) / CLOCKS_PER_SEC);

    // namedWindow("Immagine", WINDOW_NORMAL);
    // imshow("Immagine", img_fin_test);
    // waitKey(0);
    Vec3b color = Vec3b(0, 255, 255);
    for (int i = 0; i < height - 1; i++) {     // y
        for (int j = 0; j < width - 1; j++) {  // x
            auto etC = labels.at<int>(i, j);
            auto etR = labels.at<int>(i, j + 1);
            auto etD = labels.at<int>(i + 1, j);

            if (etC != etR || etC != etD) {
                img_fin_test.at<Vec3b>(i, j) = color;
            }
        }
    }
    imwrite(R"(C:\\Users\\albi0\\ProgettiPersonali\\C\\SLIC_SP_seq\\Images\\OUTPUT\\img-v1.png)", img_fin_test);
    // imshow("Immagine", img_fin_test);
    // waitKey(0);

    // destroyAllWindows();

    return 0;
}

std::vector<std::pair<int, int>> floodFillBFS(const Mat& labels, std::vector<bool>& visited, int x, int y) { // La matrice dei label di ogni pixel, la riga e colonna iniziale
    std::array<std::pair<int, int>, 4> dir = {std::pair{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    std::vector<std::pair<int, int>> current_segment;
    current_segment.reserve(10);
    std::queue<std::pair<int, int>> q;

    visited[x * width + y] = true;
    q.emplace(x, y);
    int currentSP = labels.at<int>(x, y);

    while (!q.empty()) {
        std::pair<int, int> front = q.front();
        int x = front.first, y = front.second;
        q.pop();

        for (std::pair<int, int>& it : dir) {
            int nx = x + it.first;
            int ny = y + it.second;

            if (nx >= 0 && nx < height &&
                ny >= 0 && ny < width &&
                !visited.at(nx * width + ny) &&
                 labels.at<int>(nx, ny) == currentSP) {
                    visited[nx * width + ny] = true;
                    current_segment.emplace_back(nx, ny);
                    q.emplace(nx, ny);
                 }
        }
    }

    return current_segment;
}

inline float manhattanDist(const pixel& p1, const pixel& p2) {
    return std::abs(p1.l - p2.l) + std::abs(p1.a - p2.a) + std::abs(p1.b - p2.b) + std::abs(p1.x - p2.x) + std::abs(p1.y - p2.y);
}

float distance(const pixel& p1, const pixel& p2, const float& S) {
    auto dis_lab = sqrt((p1.l - p2.l)*(p1.l - p2.l) + (p1.a - p2.a)*(p1.a - p2.a) + (p1.b - p2.b)*(p1.b -p2.b));
    auto dis_xy = static_cast<float>(sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y)));

    return dis_lab + m/S * dis_xy;
}

float gradient(const Vec3f& lab1, const Vec3f& lab2, const Vec3f& lab3, const Vec3f& lab4) {
    Vec3f normLeft = {lab1[0] - lab2[0], lab1[1] - lab2[1], lab1[2] - lab2[2]};
    Vec3f normRight = {lab3[0] - lab4[0], lab3[1] - lab4[1], lab3[2] - lab4[2]};

    return normLeft[0]*normLeft[0] + normLeft[1]*normLeft[1] + normLeft[2]*normLeft[2] +
            normRight[0]*normRight[0] + normRight[1]*normRight[1] + normRight[2]*normRight[2];
}