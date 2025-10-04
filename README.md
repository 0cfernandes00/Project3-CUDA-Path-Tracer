CUDA Path Tracer - using 1 late day
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Caroline Fernandes
  * [LinkedIn](https://www.linkedin.com/in/caroline-fernandes-0-/), [personal website](https://0cfernandes00.wixsite.com/visualfx)
* Tested on: Windows 11, i9-14900HX @ 2.20GHz, Nvidia GeForce RTX 4070
![](img/cornell.2025-10-04_03-36-46z.1025samp.png)

### Features and Sections

The goal of this project was to implement a CUDA-based path tracer. The base code was provided which implemented the openGL interop, previewing and saving images, as well as loading a scene description file.

Core Features:
- [Shading](https://github.com/0cfernandes00/Project3-CUDA-Path-Trace/blob/main/README.md#shading)
- [Stream Compaction](https://github.com/0cfernandes00/Project3-CUDA-Path-Trace/blob/main/README.md#stream-compaction)
- [Material Sorting](https://github.com/0cfernandes00/Project3-CUDA-Path-Trace/blob/main/README.md#material-sorting)
- [Anti-Aliasing](https://github.com/0cfernandes00/Project3-CUDA-Path-Trace/blob/main/README.md#anti-aliasing)

Part 2 Features:
- [Russian Roulette Termination](https://github.com/0cfernandes00/Project3-CUDA-Path-Trace/blob/main/README.md#russian-roulette-termination)
- [Texture Mapping](https://github.com/0cfernandes00/Project3-CUDA-Path-Trace/blob/main/README.md#texture-mapping)
- [Mesh Loading](https://github.com/0cfernandes00/Project3-CUDA-Path-Trace/blob/main/README.md#mesh-loading)
- [Bounding Volume Hierarchies](https://github.com/0cfernandes00/Project3-CUDA-Path-Trace/blob/main/README.md#bouding-volume-hierarchies)
- [Intel Image Denoise](https://github.com/0cfernandes00/Project3-CUDA-Path-Trace/blob/main/README.md#intel-image-denoise)

### Shading
I implemented ideal diffuse and specular shading. Dffuse shading distributes light uniformly and specular materials distribute light in one direction.
<img width="1731" height="933" alt="image" src="https://github.com/user-attachments/assets/4b0928eb-f746-4c0d-8c66-b64b82889957" />


### Stream Compaction
Using thrust's library I was able to stream compact away paths that had terminated organizing the paths better in memory.

### Material Sorting
I had originally expected material sorting to improve performance but it doesn't seem to have a huge speedup. This could potentially be because the way it's sorting the materials is too expensive of an operation to outweigh the benefits of putting similar materials close together in memory. 

### Anti-Aliasing

### Russian Roulette Termination

### Texture Mapping

**Environment Mapping **

### Mesh Loading

### Bounding Volume Hierarchies

### Intel Image Denoise
