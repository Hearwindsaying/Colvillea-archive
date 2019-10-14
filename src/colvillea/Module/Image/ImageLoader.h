#pragma once

#include <src/sampleConfig.h>

#define LIBPATH(param,libname)   param##libname
#pragma comment(lib,LIBPATH(SAMPLES_DIR, "\\dependencies\\FreeImage.lib"))
#pragma comment(lib,LIBPATH(SAMPLES_DIR, "\\dependencies\\FreeImagePlus.lib"))

#include "freeImage/FreeImagePlus.h"

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

//Tiny ImageLoader utility class for loading textures into memory using FreeImage library and create TextureSampler for OptiX
class ImageLoader
{
public:
	static optix::TextureSampler LoadImageTexture(optix::Context & context, const std::string filename, const optix::float4 defaultColor, bool isOpacityMap = false);
	//static optix::TextureSampler LoadHalfConstantTexture(optix::Context & context, const std::string filename);//only for testing cuda uv fetching coordinates
	//static void SaveBufferToImage(optix::Buffer buffer, const std::string filename);
	static void SaveOutputBufferToImage(optix::Buffer buffer, const std::string filename);

    static void saveHDRBufferToImage(optix::Buffer buffer, const std::string &filename);

	/**
	 * @brief save hdriLuminanceBuffer to image,only for internal debugging test
	 * @param buffer hdriLuminanceBuffer which is of float type
	 */
	static void SaveLuminanceBufferToImage(optix::Buffer buffer, const std::string filename);

private:
	static optix::TextureSampler LoadImageTextureLinear(optix::Context & context, const std::string filename, const optix::float4 defaultColor);
	static optix::TextureSampler LoadImageTextureGamma(optix::Context & context, const std::string filename, const optix::float4 defaultColor);
	static optix::TextureSampler LoadImageTextureAlpha(optix::Context & context, const std::string filename, const optix::float4 defaultColor);
};