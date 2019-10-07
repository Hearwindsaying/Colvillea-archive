# Colvillea
![Living room scene by benedikt-bitterli](https://github.com/Hearwindsaying/Colvillea/raw/master/examples/Gallery/7200spp.jpg)

## Overview
**Colvillea** is a physically based global illumination renderer running on GPU. It relies on [Nvidia's OptiX](https://developer.nvidia.com/optix) to achieve parallelism by leveraging GPU resources, resulting in high performance ray tracing rendering.

## Features
### Light Transport
 - Direct Lighting
 - Unidirectional Path Tracing

### Reflection Models
 - Lambertian BRDF
 - Specular BRDF (Perfect Mirror)
 - Specular BSDF (Perfect Glass)
 - Ashikhmin-Shirley BRDF (Rough Plastic)
 - GGX Microfacet BRDF (Rough Conductor)
 - GGX Microfacet BSDF (Rough Dielectric)
 - Dielectric-Couductor Two Layered BSDF

### Sampler
 - Random Sampler
 - Halton QMC Sampler (Fast Random Permutation)    
 - Sobol QMC Sampler

### Filter (Progressive)
 - Box filter
 - Gaussian filter

### Rendering Mode
 - Progressive Rendering

### Light Source Models
 - Point Light
 - Quad Light (Spherical Rectangular Sampling)
 - Image-Based Lighting (HDRI Probe)

### Camera 
 - Pinhole Camera

### Geometry
 - Triangle Mesh (Wavefront OBJ)

### Miscellaneous
 - LDR/HDR Image I/O with Gamma Correction

## Work In Progress
 - Interactive rendering supporting editing scene

## Build
Building **Colvillea** requires OptiX 5.1.0 and CUDA 9.0 installed. For graphics driver on Windows platform, driver version 396.65 or later is required. All NVIDIA GPUs of Compute Capability 3.0 (Kepler) or higher are supported. 

**Colvillea** currently builds on Windows only using [CMake](http://www.cmake.org/download/) and could be built using MSVC successfully. It's recommeded that create a separte directory in the same level folder as src folder. Note that you are required to use VS2015 as CUDA_HOST_COMPILER in configuration step. However, generator could use either VS2015 or VS2017 Win64.


