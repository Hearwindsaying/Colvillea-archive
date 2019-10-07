#
# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

include_directories(${SAMPLES_INCLUDE_DIR})

set(SOURCE_LIST
    "${CMAKE_CURRENT_SOURCE_DIR}/Application/Application.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Application/Application.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Application/Colvillea.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Application/GlobalDefs.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Application/SceneGraph.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Application/TWAssert.h"
	
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Camera/PinholeCamera.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Camera/SphericalSkybox.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Geometry/Quad.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Geometry/Quad.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Geometry/TriangleMesh.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Integrator/DirectLighting.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Integrator/PathTracing.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Integrator/Integrator.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Light/HDRILight.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Light/HitProgram.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Light/LightUtil.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Light/PointLight.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Light/QuadLight.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Sampler/HaltonSampler.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Sampler/HaltonSampler_Common.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Sampler/Sampler.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Sampler/SobolSampler.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Shader/Emissive.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Shader/Fresnel.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Shader/FrostedMetal.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Shader/Lambert.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Shader/LambertBRDF.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Shader/MicrofacetBRDF.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Shader/Microsurface.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Shader/Plastic.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Shader/RoughDielectric.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Shader/RoughMetal.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Shader/SmoothGlass.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Shader/SmoothMirror.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Toolkit/CommonStructs.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Toolkit/MCSampling.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Toolkit/NvRandom.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Toolkit/Serialize.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Device/Toolkit/Utility.h"
	
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Camera/Camera.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Camera/Camera.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Camera/CameraController.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Camera/CameraController.h"
	
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Geometry/Quad.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Geometry/Shape.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Geometry/Shape.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Geometry/TriangleMesh.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Geometry/TriangleMesh.h"
	
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Image/ImageLoader.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Image/ImageLoader.h"
	
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Integrator/DirectLighting.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Integrator/Integrator.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Integrator/Integrator.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Integrator/PathTracing.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Integrator/PathTracing.h"
	
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Light/HDRILight.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Light/HDRILight.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Light/Light.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Light/LightPool.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Light/PointLight.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Light/QuadLight.h"
	
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Material/MaterialPool.h"
	
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Sampler/HaltonSampler.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Sampler/HaltonSampler.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Sampler/Sampler.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Sampler/SobolMatrices.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Sampler/SobolMatrices.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Sampler/SobolSampler.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Module/Sampler/SobolSampler.h"
)

set(SOURCE_LIST_LIB
    "${CMAKE_CURRENT_SOURCE_DIR}/../gl/glew.c"
	"${CMAKE_CURRENT_SOURCE_DIR}/../gl/glew.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/../gl/wglew.h"
	
	"${CMAKE_CURRENT_SOURCE_DIR}/../freeImage/FreeImage.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/../freeImage/FreeImagePlus.h"
	
	"${CMAKE_CURRENT_SOURCE_DIR}/../tinyobjloader/tiny_obj_loader.cc"
	"${CMAKE_CURRENT_SOURCE_DIR}/../tinyobjloader/tiny_obj_loader.h"
)

if(OPENGL_FOUND)
  if( WIN32 )
    add_definitions(-DGLEW_BUILD)
  endif()

  if(USING_GNU_CXX OR USING_CLANG_CXX)
    # glew.c #includes glew.h which contains "lax" function prototypes
    set_source_files_properties("glew.c" PROPERTIES COMPILE_FLAGS  "-Wno-strict-prototypes -Wno-missing-prototypes")

  endif()
  
endif()

# Prevent from VS2017 15.8 error C2338 due to the fix for alignment 
if (MSVC_VERSION GREATER_EQUAL "1915")
	add_compile_definitions("_ENABLE_EXTENDED_ALIGNED_STORAGE")
endif()

#SOURCE_GROUP(Application REGULAR_EXPRESSION Application/.*)
SOURCE_GROUP(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_LIST})
SOURCE_GROUP(TREE "${CMAKE_CURRENT_SOURCE_DIR}/../" FILES ${SOURCE_LIST_LIB})

# See top level CMakeLists.txt file for documentation of OPTIX_add_sample_executable.
OPTIX_add_sample_executable( colvillea
    ${CMAKE_CURRENT_SOURCE_DIR}/Application/Application.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Application/Application.h
	${CMAKE_CURRENT_SOURCE_DIR}/Application/Colvillea.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Application/GlobalDefs.h
	${CMAKE_CURRENT_SOURCE_DIR}/Application/SceneGraph.h
	${CMAKE_CURRENT_SOURCE_DIR}/Application/TWAssert.h
	
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Camera/PinholeCamera.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Camera/SphericalSkybox.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Geometry/Quad.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Geometry/Quad.h
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Geometry/TriangleMesh.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Integrator/DirectLighting.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Integrator/PathTracing.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Integrator/Integrator.h
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Light/HDRILight.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Light/HitProgram.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Light/LightUtil.h
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Light/PointLight.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Light/QuadLight.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Sampler/HaltonSampler.h
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Sampler/HaltonSampler_Common.h
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Sampler/Sampler.h
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Sampler/SobolSampler.h
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Shader/Emissive.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Shader/Fresnel.h
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Shader/FrostedMetal.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Shader/Lambert.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Shader/LambertBRDF.h
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Shader/MicrofacetBRDF.h
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Shader/Microsurface.h
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Shader/Plastic.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Shader/RoughDielectric.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Shader/RoughMetal.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Shader/SmoothGlass.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Shader/SmoothMirror.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Toolkit/CommonStructs.h
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Toolkit/MCSampling.h
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Toolkit/NvRandom.h
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Toolkit/Serialize.h
	${CMAKE_CURRENT_SOURCE_DIR}/Device/Toolkit/Utility.h
	
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Camera/Camera.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Camera/Camera.h
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Camera/CameraController.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Camera/CameraController.h
	
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Geometry/Quad.h
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Geometry/Shape.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Geometry/Shape.h
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Geometry/TriangleMesh.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Geometry/TriangleMesh.h
	
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Image/ImageLoader.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Image/ImageLoader.h
	
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Integrator/DirectLighting.h
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Integrator/Integrator.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Integrator/Integrator.h
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Integrator/PathTracing.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Integrator/PathTracing.h
	
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Light/HDRILight.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Light/HDRILight.h
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Light/Light.h
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Light/LightPool.h
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Light/PointLight.h
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Light/QuadLight.h
	
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Material/MaterialPool.h
	
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Sampler/HaltonSampler.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Sampler/HaltonSampler.h
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Sampler/Sampler.h
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Sampler/SobolMatrices.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Sampler/SobolMatrices.h
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Sampler/SobolSampler.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Module/Sampler/SobolSampler.h

	${CMAKE_CURRENT_SOURCE_DIR}/../gl/glew.c
	${CMAKE_CURRENT_SOURCE_DIR}/../gl/glew.h
	${CMAKE_CURRENT_SOURCE_DIR}/../gl/wglew.h
	
	${CMAKE_CURRENT_SOURCE_DIR}/../freeImage/FreeImage.h
	${CMAKE_CURRENT_SOURCE_DIR}/../freeImage/FreeImagePlus.h
	
	${CMAKE_CURRENT_SOURCE_DIR}/../tinyobjloader/tiny_obj_loader.cc
	${CMAKE_CURRENT_SOURCE_DIR}/../tinyobjloader/tiny_obj_loader.h
    )

