CUDA Path Tracer - using 2 late days
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Caroline Fernandes
  * [LinkedIn](https://www.linkedin.com/in/caroline-fernandes-0-/), [personal website](https://0cfernandes00.wixsite.com/visualfx)
* Tested on: Windows 11, i9-14900HX @ 2.20GHz, Nvidia GeForce RTX 4070

I modified CMakeLists.txt to integrate OIDN Libraries for Denoising and the tinyOBJ library for Mesh loading.

<img src="img/teaScene.png" width="600"> 
teaSceneFull.json 40k Triangles
<img src="img/chair_closer.png" width="600"> 
teaScene.json 80k Triangles
<img src="img/cornell.2025-10-04_23-55-03z.1088samp.png" width="600"> 
customMesh.json 8k Triangles
<img src="img/teaScene_markup.png" width="500">

### Features and Sections

The goal of this project was to implement a CUDA-based path tracer. The base code provided to me had implemented the openGL interop, previewing and saving images, as well as loading a scene description file.

Core Features:
- [Shading](https://github.com/0cfernandes00/Project3-CUDA-Path-Tracer/blob/main/README.md#shading)
- [Stream Compaction](https://github.com/0cfernandes00/Project3-CUDA-Path-Tracer/blob/main/README.md#stream-compaction)
- [Material Sorting](https://github.com/0cfernandes00/Project3-CUDA-Path-Tracer/blob/main/README.md#material-sorting)
- [Anti-Aliasing](https://github.com/0cfernandes00/Project3-CUDA-Path-Tracer/blob/main/README.md#anti-aliasing)

Part 2 Features:
- [Russian Roulette Termination](https://github.com/0cfernandes00/Project3-CUDA-Path-Tracer/blob/main/README.md#russian-roulette-termination)
- [Depth of Field](https://github.com/0cfernandes00/Project3-CUDA-Path-Trace/blob/main/README.md#depth-of-field)
- [Refraction](https://github.com/0cfernandes00/Project3-CUDA-Path-Tracer/blob/main/README.md#refraction)
- [Texture Mapping](https://github.com/0cfernandes00/Project3-CUDA-Path-Tracer/blob/main/README.md#texture-mapping)
- [Mesh Loading](https://github.com/0cfernandes00/Project3-CUDA-Path-Tracer/blob/main/README.md#mesh-loading)
- [Bounding Volume Hierarchies](https://github.com/0cfernandes00/Project3-CUDA-Path-Tracer/blob/main/README.md#bounding-volume-hierarchies)
- [Intel Image Denoise](https://github.com/0cfernandes00/Project3-CUDA-Path-Tracer/blob/main/README.md#intel-image-denoise)

I will be testing the performance of different features under the baseline that denoising, BVH, and anti-aliasing will be enabled. I will be using the customMesh.json scene (an open cornell box) swapping out the meshes starting with potionA.obj, wahoo.obj, and dodecahedron.obj

### Shading
I implemented ideal diffuse and specular shading. Diffuse shading distributes light uniformly and specular materials distribute light in one direction.

<img width="555" height="300" alt="image" src="https://github.com/user-attachments/assets/4b0928eb-f746-4c0d-8c66-b64b82889957" />

### Stream Compaction
Using thrust's library I was able to stream compact away paths that had terminated organizing the paths better in memory. I was surprised to see that stream compaction was more performant for me in open scenes rather than closed scenes. I was expecting to find the opposite. I assume since rays are able to escape to the environment map they can have extreme t values and deemed unuseful rays. Diffuse materials tend to create deep light paths because they allow incoming light to scatter in all directions, explaining why closed scenes don't see much of a benefit with stream compaction.

<img src="img/StreamCompactionChart.png" width="500">
<img src="img/ActiveRaysChart.png" width="500">

### Material Sorting
I had originally expected material sorting to improve performance but it doesn't seem to have a huge speedup. This could potentially be because the way it's sorting the materials is too expensive of an operation to outweigh the benefits of putting similar materials close together in memory. 

<img src="img/MatSortChart.png" width="500">

### Anti-Aliasing
I implemented Stochastic Sampled Antialiasing by jittering the ray that was generated from the camera with a small offset in both the x & y directions.

<img src="img/aliased_close.png" width="400"> <img src="img/antialiased_closeup.png" width="400">

### Russian Roulette Termination
Russian Roulette Termination is a way of randomly terminating paths early that seem to have low throughput contributions. This feature is meant to optimize out unhelpful or unuseful paths. For open scenes, there was not much of a benefit it seemed to slow performance. It seemed more beneficial in closed scenes. The second chart was tested with a closed scene to fully test this feature.

<img src="img/RussianRouletteChart.png" width="500">
<img src="img/RR_closedScene_chart.png" width="500">

### Depth of Field
I implemented Physically Based Depth of Field based on PBRT's method inside the of ray generation kernel on the GPU. This was implemented as a set of physics rules that affect the rays from camera an such much sense to parallelize on the GPU, as such DOF didn't seem to have much of an impact on performance.

<img src="img/cornell.2025-10-04_23-41-19z.900samp.png" width="400"> <img src="img/cornell.2025-10-04_23-55-03z.1088samp.png" width="400">
<img src="img/DOFChart.png" width="400">

### Refraction
I implemented refraction based on PBRT. The fresnel formula with my implementation is using pow() which I know is slow. I imagine there are opportunities for optimization here similar to how we learned in class with the modulo operator. I did a performance analysis of the different materials that I implemented. Surprisingly it seemed texture mapping was the most performant compared to diffuse textures. As expected refractive materials seemed to perform worse or on par with diffuse materials. Each ray's refraction is calculated independently which allows us to classify refraction as embarrassingly parallel. The performance speedup for the GPU implementation is extreme compared to other features implemented for part 2.

<img src="img/MaterialsChart.png" width="600">

### Texture Mapping
This was one of the last features I implemented and spent a good amount of time debugging uvs and textures. At the bottom I mention a blooper that helped me discover a texture flipping issue that I noticed after submitting my code. I have not pushed this small change so as to not utilize another late day.

<img src="img/cornell.2025-10-03_17-17-22z.81samp.png" width="400">

Environment Mapping
My project also supports HDR environment map loading. Previously, when a ray bounced outside of the open cornell box the ray terminated and returned a color value of zero. Now it returns the color sampled in the hdr texture. An open scene with an HDR map renders quicker than the same obj inside of the cornell box, this make sense because it's less surfaces for the ray to bounce off of. This is reflected in the performance analysis as scenes with the map enabled has a higher FPS than without. The loading of the image was done on the CPU but the sampling was done on the GPU. I believe sampling the map on the GPU doesn't provide much of a speedup as compared to the GPU. It's a slight but not hugely significant improvement. With its current implementation, my pathtracer produces fireflies from HDR maps with hotspots. A future optimization would be implementing MIS or mipmapping to combat this problem. 

<img src="img/cornell.2025-10-04_03-36-46z.1025samp.png" width="400">
<img src="img/EnvtMapChart.png" width="400">

### Mesh Loading
I implemented OBJ loading using the tinyOBJ library. As the number of vertices increases in a scene, the FPS gets slower. I loaded the parsed and loaded the mesh data on the CPU and checked for intersection on the GPU. I assume checking for intersection is slower on the CPU since the GPU can compute the triangle intersection test faster. This is an expected result which lead to implementing a spatial data structure.

### Bounding Volume Hierarchies
Bounding Volume Hierarchies provide a way of breaking the scene into smaller subcomponents to check against for ray intersection tests. I implemented a Naive BVH and used AABB for the bounds test. This was a difficult part of the project for me. I had expected BVH to improve performance but this was most evident for even larger meshes. In one sample scene, (with the potion mesh(8k triangles) and environment map) I started off with a speed of 6 FPS. I was able to optimize my BVH to get up to 16 FPS for the same scene. I found a few places to optimize my BVH structure, but the largest improvement was checking the bounds for each child first and then traversing to the nearest child first. Previously, I was only checking the parent then automatically traversing to the left and right children. One future optimization would be using the Surface Area Heuristic as a splitting method. BVH Traversal on the CPU would be significantly slower, there are several computations required throughout the process from the triangle intersection tests to AABB testing making the GPU superior. Tree building may have advantages on the GPU if the scene was dynamic but things like SAH are easier to implement without the overhead of parallelization.

<img src="img/BVHChart.png" width="500">

### Intel Image Denoise
I integrated [Intel's Denoiser](https://github.com/RenderKit/oidn) which utilizes Deep Learning methods to converge the results faster. The result was nicer but it effectively blurred hard edges in geometry. Intel offers a CUDA GPU version for denoising, but I had an easier time working with the CPU denoising. The largest pitfall of the CPU implementation is the amount of time the GPU is idle. Additionally, I implemented a blending function for the two passes. I believe having denoising on the GPU would increase performance but only minimally since the filter is only running every 50(user-defined) iterations. I was pleased with the improvement to my renders but am interested to test out other denoisers, as expected denoising did not have a huge impact on performance since it was more of a post-process.

<img src="img/noise_closeup.png" width="500"> <img src="img/denoise_closeup.png" width="500">
<img src="img/DenoiseChart.png" width="400">

### Bloopers
The image below was a result of not flipping the v coordinate when reading in a texture.
<img src="img/cornell.2025-10-05_20-32-15z.87samp.png" width="400">
<img src="img/cornell.2025-10-04_22-02-03z.6samp.png" width="400">

### Resources
Potion Model - given permission from artist [Caitlin Cheek](https://caitlincheek.com/potions)
[Tea Set Model](https://sketchfab.com/3d-models/tea-set-cc8666654d7f4da8a0398cab82292f13)
[Queen of Hearts Chair Model](https://sketchfab.com/3d-models/alice-in-wonderland-red-queens-throne-cbe155e9f492404d964124ce284b5f1e)
