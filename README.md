CUDA Path Tracer - using 2 late days
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Caroline Fernandes
  * [LinkedIn](https://www.linkedin.com/in/caroline-fernandes-0-/), [personal website](https://0cfernandes00.wixsite.com/visualfx)
* Tested on: Windows 11, i9-14900HX @ 2.20GHz, Nvidia GeForce RTX 4070

I modified CMakeLists.txt to integrate OIDN Libraries for Denoising and the tinyOBJ library for Mesh loading.

<img src="img/teaScene.png" width="600">
<img src="img/chair_closer.png" width="600">
<img src="img/cornell.2025-10-04_23-55-03z.1088samp.png" width="600">
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

### Shading
I implemented ideal diffuse and specular shading. Diffuse shading distributes light uniformly and specular materials distribute light in one direction.

<img width="555" height="300" alt="image" src="https://github.com/user-attachments/assets/4b0928eb-f746-4c0d-8c66-b64b82889957" />

### Stream Compaction
Using thrust's library I was able to stream compact away paths that had terminated organizing the paths better in memory.

### Material Sorting
I had originally expected material sorting to improve performance but it doesn't seem to have a huge speedup. This could potentially be because the way it's sorting the materials is too expensive of an operation to outweigh the benefits of putting similar materials close together in memory. 

### Anti-Aliasing
I implemented Stochastic Sampled Antialiasing by jittering the ray that was generated from the camera with a small offset in both the x & y directions.

<img src="img/aliased_close.png" width="400"> <img src="img/antialiased_closeup.png" width="400">

### Russian Roulette Termination
Russian Roulette Termination is a way of randomly terminating paths early that seem to have low throughput contributions. This feature is meant to optimize out unhelpful or unuseful paths. For open scenes, there was not much of a benefit it seemed to slow performance. It seemed more beneficial in closed scenes.

### Depth of Field
I implemented Physically Based Depth of Field based on the PBRT's method.

<img src="img/cornell.2025-10-04_23-41-19z.900samp.png" width="400"> <img src="img/cornell.2025-10-04_23-55-03z.1088samp.png" width="400">

### Refraction
I implemented refraction to recreate glass materials.

### Texture Mapping
This was one of the last features I implemented and spent a good amount of time debugging uvs and textures. At the bottom I mention a blooper that helped me discover a texture flipping issue that I noticed after submitting. I have not pushed this small change so as to not utilize another late day but I have made the fix.
<img src="img/cornell.2025-10-03_17-17-22z.81samp.png" width="400">

Environment Mapping
My project supports HDR environment map loading. Previously, when a ray bounced outside of the open cornell box, the ray terminated and returned a color value of zero. Now it returns the color sampled in the hdr texture. (TODO) An open scene with an HDR map renders quicker than the same obj inside of the cornell box, this make sense because it's less surfaces for the ray to bounce off of. (TODO) The loading of the image was done on the CPU but the sampling was done on the GPU. With its current implementation, my pathtracer produces fireflies from HDR maps with hotspots. A future optimization would be implementing MIS or use mipmapping to combat this problem.

<img src="img/cornell.2025-10-04_03-36-46z.1025samp.png" width="400">


### Mesh Loading
I implemented OBJ loading using the tinyOBJ library. As the number of vertices increases in a scene, the FPS gets slower. This is an expected result that lead to implementing a spatial data structure.

### Bounding Volume Hierarchies
I implemented a Naive BVH and used AABB for the bounds test. This was a diffiuclt part of the project for me, there is a lot of oppurtunity for improvement and optimization for this particular feature. In one sample scene with a the potion mesh(8k triangles) and environment map I started off with a speed of 6 FPS. I was able to optimize my BVH to get up to 16 FPS for the same scene. 

### Intel Image Denoise
I integrated [Intel's Denoiser](https://github.com/RenderKit/oidn) which utilizes Deep Learning methods to converge the results faster. This feature provided a much nicer image but it effectively blurred hard edges in geometry. (TODO Performance Impact) Intel offers a CUDA GPU version for denoising, but I only had an easier time getting the CPU denoising working. In addition to running the denoising filter it also runs a blending function. I believe having these on the GPU would increase performance but only minimally since the filter is only running every 50(user-defined) iterations. I was pleased with the improvement to my renders but I am interested to test out other denoisers to see if they produce nicer results.

<img src="img/noise_closeup.png" width="500"> <img src="img/denoise_closeup.png" width="500">

### Bloopers
The image below was a result of not flipping the v coordinate when reading in a texture.
<img src="img/cornell.2025-10-05_20-32-15z.87samp.png" width="400">
<img src="img/cornell.2025-10-04_22-02-03z.6samp.png" width="400">

### Resources
Potion Model - given permission from artist [Caitlin Cheek](https://caitlincheek.com/potions)
[Tea Set Model](https://sketchfab.com/3d-models/tea-set-cc8666654d7f4da8a0398cab82292f13)
[Queen of Hearts Chair Model](https://sketchfab.com/3d-models/alice-in-wonderland-red-queens-throne-cbe155e9f492404d964124ce284b5f1e)
