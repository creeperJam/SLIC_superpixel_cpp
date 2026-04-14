# SLIC_superpixel_cpp

This project is an implementation of the SLIC (Simple Linear Iterative Clustering) superpixel algorithm in the C++ language. \
For the realization the C++17 standard was used, as for the image reading, processing e saving the OpenCV library with its types and functions were used.

This was made for a project at UniFi (Università degli studi di Firenze).

This project provides 4 implementation:
- v1: first made as a quick test, but avoid using since it's slow
- v2: pure AoS implementation of the algorithm, much faster than v1
- v3: hybrid implementation with image stored as an SoA and the cluster centers as AoS
- v4: pure SoA implementation, generally the fastest of the 4.

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

The first three version take the input as command line arguments. They take the following inputs in this order:

1. Image to process
2. Number of wanted superpixels, which will be called K for the rest of the explanations

If the version is the fourth, than the input has been moved inside the `common.h` file, to simplify the testing of different input. This is easily changeable.
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
   And, supposing everything is installed correctly and what i wrote is in itself correct (i hope, but it should) you get the file you have to run in the build folder.

If, instead, you wanted to see image while running with a window, supposing you installed OpenCV with GUI support, you'll have to include in the `common.h` file `#include <opencv2/highgui.hpp>`, comment the part where the program runs `imwrite(args)` and write instead `imshow(window_name, image_to_show)`. After you've done this just compile again following the previous steps.

### Usage

Since there are 4 version of the program the usage varies depending on which is used:
If the version in use is one of the first three than it's:
- Linux: `./v# /abs/path/to/image num_of_superpixel`
- Windows `v#.exe /abs/path/to/image num_of_superpixel`

Where '#' is the version in use

If the version in use is the fourth one, than it's:
- Linux: `./v#`
- Windows `v#.exe`

All outputs should be found in the output folder after finished.

## References

1. [SLIC superpixel paper](https://infoscience.epfl.ch/entities/publication/2dd26d47-3d00-43eb-9e31-4610db94a26e) - EPFL - Achanta Radhakrishna,  Shaji Appu, Smith Kevin, Lucchi Aurélien, Fua Pascal, Süsstrunk Sabine
2. [COCO train2017 samples](https://cocodataset.org/#download) - Only a tiny portion was used, too many photos for me
3. [Princess Mononoke 1080p image](https://www.artstation.com/artwork/XBJJOw)
