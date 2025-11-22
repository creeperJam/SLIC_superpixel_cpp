# SLIC_superpixel_cpp

This project is an implementation of the SLIC (Simple Linear Iterative Clustering) superpixel algorithm in the C++ language. \
For the realization the C++17 standard was used, as for the image reading, processing e saving the OpenCV library with its types and functions were used.

This was made for a project at UniFi (Università degli studi di Firenze).

## Algorithm

For the source of the algorithm used you can read the full paper at [1](#References) and the implementation is made by following the algorithm as described by them, with a couple of small differences. \
The algorithm is the following:

  ```math
    \begin{aligned}
      &\color{white}\rule{25cm}{0.75pt}\\ \\
      &\space\space\textbf{Algorithm} - \text{Efficient superpixel segmentation} \\
      &\color{white}\rule{25cm}{0.4pt} \\ 
    \end{aligned}
    \begin{aligned}
      &1: &&\text{Initialize cluster centers } C_k[l_k, a_k, b_k, x_k, y_k]^T` \text{ by sampling pixels at regular grid steps S.} \\
      &2: &&\text{Perturb cluster centers in an } n \times n \text{ neighborhood, to the lowest gradient position. }\\
      &3: &&\textbf{repeat} \\
      &4: && \quad \textbf{for } \text{each cluster center } C_k \textbf{ do} \\
      &5: && \quad \quad\text{Assign the best matching pixels from a } 2S \times 2S \text{ square neighborhood around the cluster center according to the distance measure} \\
      &6: && \quad \textbf{end for} \\
      &7: && \quad \text{Compute new cluster centers and residual error E \{L1 distance between previous centers and recomputed centers\}} \\
      &8: &&\textbf{until } E ≤ \text{ threshold} \\
      &9: &&\text{Enforce connectivity} \\
    \end{aligned}
      \color{white}\rule{25cm}{0.4pt}
  ```

In both the parallel and sequential versions there are functions for every step. The only inputs that the  algorithm takes are:

1. Image to process
2. Number of wanted superpixels, which will be called K for the rest of the explanations

Having this information we can calculate everything else, starting with S, the interval on the superpixel grid, which roughly indicates were each superpixel center will be located on the image. For example, if we have S=5, the first superpixel center will be located in (5, 5), the second in (5, 10) and so on until we reach the end and go to the next line.

To calculate this we use the following formula: $`S=\sqrt{N/K}`$, where N is the number of pixels and K the number of desired superpixels.

Another variable is $`m`$, this is not inserted by the user as an input but changed directly in the code. What it does is indicate if we want to the algorithm to care more about how close each pixel is to the other rather than their color distance, or the opposite. For all the test the value $`m=10`$ was used.
As for the formulas for all the calculations, while visible in the cited paper, they are the following:
- **Gradient**: $`G(x,y)=\| \textbf{I}(x+1,y)-\textbf{I}(x-1,y) \|^2 + \| \textbf{I}(x,y+1)-\textbf{I}(x,y-1) \|^2`$
- **Distance**: It's a combination of three parts
  - $`d_{lab}=\sqrt{(l_k-l_i)^2 + (a_k-a_i)^2 + (b_k-b_i)^2}`$
  - $`d_{xy}=\sqrt{(x_k-x_i)^2 + (y_k-y_i)^2}`$
  - $`D_s=d_{lab}+\frac{m}{S}d_{xy}`$
- **L1 distance** - or [taxicab/Manhattan distance](https://en.wikipedia.org/wiki/Taxicab_geometry) with $`p=(l_p, a_p, b_p, x_p, y_p)`$ and $`q=(l_q, a_q, b_q, x_q, y_q)`$ being two points:
  $`d_T(\textbf{p},\textbf{q})=\|\textbf{p}-\textbf{q}\|_T=\sum_{i=1}^{n}{|p_i-q_i|}`$

The difference in the code is on the Distance formula where, instead of doing all the costly square roots, I choose to square everything, removing the square roots and only have to square the fraction. This works since the value is mainly used to check if this new distance is lower then the previous one and since $`A < B \Longrightarrow A^2 < B^2`$ we can do this.

The other thing to note is that, while the L2 norm required in the calculation of the gradient does normally need to be square rooted, the paper also square the L2 norm, negating each other and the only remaining this is:
```math
\begin{aligned}
  &\text{Suppose that }x = \textbf{I}(x+1,y)-\textbf{I}(x-1,y) \text{ and } y = \textbf{I}(x,y+1)-\textbf{I}(x,y-1) \\
  &G(x,y) = (x_l^2+x_a^2+x_b^2+x_x^2+x_y^2) + (y_l^2+y_a^2+y_b^2+y_x^2+y_y^2)
\end{aligned}
```

### Extras

While the algorith alone might be good enough, on most pictures I noticed that, when a certain area, while being the same thing with roughly the same color, kept coming out as a lot of tiny clusters. To fix this i choose to add a very small Gaussian blur that would allow the extremely small details and general noise of the image to get blurred out without impacting the actual borders of the various objects.

If not wanted the only change necessary is to comment out the line where the Gaussian blur is applied.

Here's a before and after:
<div align="center">
  <img src="/examples/gaussian-blur/before_blur.png" alt="Before blur" width="400">
  <img src="/examples/gaussian-blur/after_blur.png" alt="After blur" width="400">
</div>

## Building the project
### Requirements

The two requirements are:
- OpenCV - This has multiple options based on the OS:
  - Windows: precombiled binaries from OpenCV for MSVC (although, without certain flags, it supports only up to OpenMP 2.0) or compile it from source for any compiler following their guide.
  - Linux/MacOS: compiling it from source following their guide.
- OpenMP: I used the g++ compiler from MSYS2. Clang and MSVC can work but i haven't tested them.

For Windows there's also the option of using package managers like VCPKG and MSYS2 (the one I used), just make sure you install the correct ones and not anything malicious (just in case).

In case it still doesn't find libraries, check if the environment variables are set correctly.

### Building

After you have all the requirements installed and working, the steps to build the program are the following:
1. Go into the project folder and run `mkdir build` and `cd build`
2. `cmake ..`
3. `make`
   And, supposing everything is installed correctly and i what i wrote is in itself correct (i hope, but it should) you get the file you have to run in the build folder.

If, instead, you wanted to see image while running with a window, supposing you installed OpenCV with GUI support, you'll have to include in the `common.h` file `#include <opencv2/highgui.hpp>`, comment the part where the program runs `imwrite(args)` and write instead `imshow(window_name, image_to_show)`. After you've done this just compile again following the previous steps.

### Usage

The command to run it is:
- Linux: `./SLIC_superpixel_cpp /abs/path/to/image num_of_superpixel`
- Windows `SLIC_superpixel_cpp.exe /abs/path/to/image num_of_superpixel`

All the outputs, both 20 image results (10 for sequential and 10 for parallel) and time average results, are available in the output folder created in the project, not the build, folder if not already present.

## Time differences

As already stated, this project has both a sequential version and a parallel one and here I'll show the difference in time and the improvements. The following times were gained from the first version and last version of the sequential and the parallel.

For the tests, the variable $`m`$ was set to the value 10, just like in the cited paper to keep consistency. \
As for the device used for the tests, they were run on my own laptop ([Lenovo LOQ 3 15APH8](https://www.lenovo.com/it/it/p/laptops/loq-laptops/lenovo-loq-15aph8/len101q0004)) which has a Ryzen 7 7840HS + RTX 4060 with 16 GB of DDR5 RAM and an SSD. \
All the tests were run a total of 10 times each while plugged to avoid battery problems.

All images samples were obtained from websites like COCO or just random images found around. All sources are cited in the [References](#Referencecs).



| Image                          | Superpixel count | Average time sequential (old) | Average time sequential (new) | Average time parallel |
|--------------------------------|:----------------:|:-----------------------------:|:-----------------------------:|:---------------------:|
| War Thunder 4k                 |       5000       |           8.9681 s            |           1.6691 s            |       0.3254 s        |
| Princess Mononoke 8k           |      10000       |           36.1382 s           |           6.0243 s            |       1.8047 s        |
| Coco train2017 samples 0..0009 |       400        |           0.3699 s            |           0.0539 s            |       0.0159 s        |
| Coco train2017 samples 0..0127 |       400        |           0.3707 s            |           0.0559 s            |       0.0152 s        |
| Coco train2017 samples 0..4749 |       400        |           0.2979 s            |           0.0468 s            |       0.0148 s        |
| Coco train2017 samples 0..8019 |       400        |           0.1995 s            |           0.0273 s            |       0.0103 s        |

Based on this times we can see that the speedup from using OpenMP depends on image size and the superpixel count. \
In the testing done above, the speedups come from switching to OpenMP instead of sequential comes out to:

| Image                          | Superpixel count | Average time sequential (new) | Average time parallel  | Speedup par/seq |
|--------------------------------|:----------------:|:-----------------------------:|:----------------------:|:---------------:|
| War Thunder 4k                 |       5000       |           1.6691 s            |        0.3254 s        |     5.1293      |
| Princess Mononoke 8k           |      10000       |           6.0243 s            |        1.8047 s        |     3.3381      |
| Coco train2017 samples 0..0009 |       400        |           0.0539 s            |        0.0159 s        |     3.3899      |
| Coco train2017 samples 0..0127 |       400        |           0.0559 s            |        0.0152 s        |     3.6776      |
| Coco train2017 samples 0..4749 |       400        |           0.0468 s            |        0.0148 s        |     0.3162      |
| Coco train2017 samples 0..8019 |       400        |           0.0273 s            |        0.0103 s        |     2.6504      |

## Examples results

Here are the before and after picture applying the algorithm:

<div align="center">
  <img src="/examples/results/before_1.jpg" alt="Before blur" width="200">
  <img src="/examples/results/after_1.png" alt="After blur" width="200">
  <img src="/examples/results/before_2.jpg" alt="Before blur" width="200">
  <img src="/examples/results/after_2.png" alt="After blur" width="200"> 
</div>

<div align="center">
  <img src="/examples/results/before_3.png" alt="Before blur" width="200">
  <img src="/examples/results/after_3.png" alt="After blur" width="200">
  <img src="/examples/results/before_4.jpg" alt="Before blur" width="200">
  <img src="/examples/results/after_4.png" alt="After blur" width="200">
</div>

This images, as you can guess, are not mine and only served the purpose to test my program, any and all credit goes to the original creators.

## References

1. [SLIC superpixel paper](https://infoscience.epfl.ch/entities/publication/2dd26d47-3d00-43eb-9e31-4610db94a26e) - EPFL - Achanta Radhakrishna,  Shaji Appu, Smith Kevin, Lucchi Aurélien, Fua Pascal, Süsstrunk Sabine
2. [COCO train2017 samples](https://cocodataset.org/#download) - Only a tiny portion was used, too many photos for me
3. [War Thunder 4k images](https://warthunder.com/en/media/wallpapers)  - Only one of the pictures was tested
4. [Princess Mononoke 8K image](https://www.artstation.com/artwork/XBJJOw) - The original was 1080p, but found an upscaled version on [Reddit](https://www.reddit.com/r/wallpapers/comments/ncxor3/3840x2160_princess_mononoke_8k_link_in_comments/)
