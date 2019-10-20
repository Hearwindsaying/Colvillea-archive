#include <optix_world.h>
#include <optix_device.h>

#include "../Toolkit/NvRandom.h"
#include "../Toolkit/Utility.h"
#include "../Toolkit/CommonStructs.h"
#include "../Sampler/Sampler.h"


using namespace optix;

//////////////////////////////////////////////////////////////////////////
//Forward declarations:
//system variables related:->Context
rtBuffer<float4, 2>	        sysOutputBuffer;         /*final result buffer. Necessary to keep Alpha=1*/
rtBuffer<float4, 2>         sysHDRBuffer;            /* Final result buffer for exporting to OpenEXR. */
rtBuffer<float, 2>          sysSampleWeightedSum;
rtBuffer<float4, 2>         sysCurrResultBuffer;     /*the sum of weighted radiance(f(dx,dy)*Li) with 
												       respect to the current iteration;
													   A is useless, note that for RenderView display,
													   it's necessary to keep as 1.f to prevent from 
													   alpha cutout by Dear Imgui*/
rtBuffer<float, 2>          sysCurrWeightedSumBuffer;/*weightedSum buffer with respect to the current
												       iteration*/

rtDeclareVariable(float,    sysFilterGaussianAlpha, ,) = 0.25f;//Gaussian filter alpha paramter
rtDeclareVariable(float,    sysFilterWidth, ,) = 1.f;//Gaussian filter width

rtDeclareVariable(float,    sysSceneEpsilon, , );
rtDeclareVariable(rtObject, sysTopObject, , );


#ifndef TWRT_DECLARE_SYSLAUNCH
#define TWRT_DECLARE_SYSLAUNCH
rtDeclareVariable(uint2,    sysLaunch_Dim,       rtLaunchDim, );
rtDeclareVariable(uint2,    sysLaunch_index,     rtLaunchIndex, );
#endif

//camera variables related:->Program
rtDeclareVariable(Matrix4x4, RasterToCamera, , );
rtDeclareVariable(Matrix4x4, CameraToWorld, , );



namespace TwUtil
{
	/**
	 * @brief Evaluate GaussianFilter for given deltaX and deltaY.This version computes filter weight precisely and doesn't employ approximation techniques to optimize computation.
	 * @param dx relative position deltaX to the filter kernel
	 * @param dy relative position deltaY to the filter kernel
	 * @return (e^(-alpha*dx*dx)-e^(-alpha*width*width)) * (e^(-alpha*dy*dy)-e^(-alpha*width*width))
	 */
	static __device__ __inline__ float evaluateFilter(const float dx, const float dy)
	{
		/*todo:prevent from float precision error;
		 *     optimize calculation*/
		float gaussian_exp = expf(-sysFilterGaussianAlpha * sysFilterWidth * sysFilterWidth);
		return fmaxf(0.f, expf(-sysFilterGaussianAlpha * dx*dx) - gaussian_exp) *
			fmaxf(0.f, expf(-sysFilterGaussianAlpha * dy*dy) - gaussian_exp);
		//return 1.f;
	}

	/*
	 * @brief ceilf function for float2
	 * @param val input value
	 * @return the ceiling float2 number for the given value
	 **/
	static __device__ __inline__ float2 ceilf2(const float2 &val)
	{
		return make_float2(ceilf(val.x), ceilf(val.y));
	}

	/*
	 * @brief floorf function for float2
	 * @param val input value
	 * @return the floor float2 number for the given value
	 **/
	static __device__ __inline__ float2 floorf2(const float2 &val)
	{
		return make_float2(floorf(val.x), floorf(val.y));
	}

	/*
	 * @brief ceil function for float2 and cast to int2
	 * @param val input value
	 * @return the ceiling float2 number for the given value, after casting to int2
	 * @see ceilf2()
	 **/
	static __device__ __inline__ int2 ceilf2i(const float2 &val)
	{
		return make_int2(static_cast<int>(ceilf(val.x)), static_cast<int>(ceilf(val.y)));
	}

	/*
	 * @brief floor function for float2 and cast to int2
	 * @param val input value
	 * @return the floor float2 number for the given value, after casting to int2
	 * @see floorf2()
	 **/
	static __device__ __inline__ int2 floorf2i(const float2 &val)
	{
		return make_int2(static_cast<int>(floorf(val.x)), static_cast<int>(floorf(val.y)));
	}

	/*
	 * @brief degamma function converts linear color to sRGB color
	 * for display
	 * @param src input value, alpha channel is not affected
	 * @return corresponding sRGB encoded color in float4. Alpha channel
	 * is left unchanged.
	 * @ref https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_texture_sRGB_decode.txt
	 * @see convertsRGBToLinear
	 **/
	static __device__ __inline__ float4 convertFromLinearTosRGB(const float4 &src)
	{
		float4 dst = src;
		dst.x = (dst.x < 0.0031308f) ? dst.x*12.92f : (1.055f * powf(dst.x, 0.41666f) - 0.055f);
		dst.y = (dst.y < 0.0031308f) ? dst.y*12.92f : (1.055f * powf(dst.y, 0.41666f) - 0.055f);
		dst.z = (dst.z < 0.0031308f) ? dst.z*12.92f : (1.055f * powf(dst.z, 0.41666f) - 0.055f);
		return dst;
	}

	/*
	 * @brief converts one of the sRGB color channel to linear
	 * @param src input value
	 * @return corresponding linear space color channel
	 * @ref https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_texture_sRGB_decode.txt
	 * @see convertFromLinearTosRGB
	 **/
	static __device__ __inline__ float convertsRGBToLinear(const float &src)
	{
		if (src <= 0.f)
			return 0;
		if (src >= 1.f)
			return 1.f;
		if (src <= 0.04045f)
			return src / 12.92f;
		return pow((src + 0.055f) / 1.055f, 2.4f);
	};

	/*
	 * @brief converts sRGB color to linear color
	 * @param src input value, alpha channel is not affected
	 * @return corresponding linear space color in float4. Alpha channel
	 * is left unchanged.
	 * @ref https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_texture_sRGB_decode.txt
	 * @see convertFromLinearTosRGB
	 **/
	static __device__ __inline__ float4 convertsRGBToLinear(const float4 &src)
	{
		return make_float4(convertsRGBToLinear(src.x), convertsRGBToLinear(src.y), convertsRGBToLinear(src.z), 1.f);
	}
};

//////////////////////////////////////////////////////////////////////////
//Program definitions:
RT_PROGRAM void RayGeneration_PinholeCamera()
{
    /* Make sampler and preprocess. */
    GPUSampler localSampler;  /* GPUSampler is a union type, use out parameter instead of return value to avoid copying construct on union, which could lead to problems. */
    makeSampler(RayTracingPipelinePhase::RayGeneration, localSampler);

    /* Fetch camera samples lying on [0,1]^2. */
    float2 qmcSamples = Get2D(&localSampler);

	/* Calculate filmSamples in continuous coordinates. */
	float2 pFilm = qmcSamples + make_float2(static_cast<float>(sysLaunch_index.x), static_cast<float>(sysLaunch_index.y));

	/* Generate ray from camera. */
	float3 rayOrg = make_float3(0.f);
	float3 rayDir = rayOrg;
	TwUtil::GenerateRay(pFilm, rayOrg, rayDir, RasterToCamera, CameraToWorld);

//     rtPrintf("%d %d\n", sysLaunch_index.x, sysLaunch_index.y);

	/* Make ray and trace, goint to next raytracing pipeline phase. */
	Ray ray = make_Ray(rayOrg, rayDir, toUnderlyingValue(RayType::Radiance), sysSceneEpsilon, RT_DEFAULT_MAX);

    CommonStructs::PerRayData_radiance prdRadiance;
	prdRadiance.radiance = make_float4(0.f);

	rtTrace<CommonStructs::PerRayData_radiance>(sysTopObject, ray, prdRadiance, RT_VISIBILITY_ALL, RT_RAY_FLAG_DISABLE_ANYHIT);

	/*--------------------------------------------------------------------------------*/
	/*----Perform filtering and reconstruction so as to write to the output buffer----*/
	/*--------------------------------------------------------------------------------*/
	// todo: use faster filtering technique without atomic ops when width<.5f

	/*compute pFilm's raster extent
	 *--1.get film sample's discrete coordinates*/
	float2 dCoordsSample = pFilm - .5f;

	/*--2.search around the filterWidth for raster pixel boundary*/
	int2 pMin = TwUtil::ceilf2i(dCoordsSample - sysFilterWidth);
	int2 pMax = TwUtil::floorf2i(dCoordsSample + sysFilterWidth);

	/*--3.check for film extent*/
	pMin.x = max(pMin.x, 0);                   pMin.y = max(pMin.y, 0);
	pMax.x = min(pMax.x, sysLaunch_Dim.x - 1); pMax.y = min(pMax.y, sysLaunch_Dim.y - 1);
	
	/*note that assert() only supports in NVRTC*/
	if (pMax.x - pMin.x < 0 || pMax.y - pMin.y < 0)
		rtPrintf("[ERROR]Assert failed at \"filtering\" \n");

	/*loop over raster pixel and add sample with filtering operation*/
	for (int y = pMin.y; y <= pMax.y; ++y)
	{
		for (int x = pMin.x; x <= pMax.x; ++x)
		{
			/*not necessary to distinguish first iteration, because one sample
			 *could possibly contribute to multiple pixels so the 0th iteration doesn't
			 *have a specialized meaning. Instead, we use the modified version of progressive
			 *weighted average formula.*/

			/*Pass 1:accumulate sysCurrResultBuffer with f(dx,dy)*Li and sysCurrWeightedSumBuffer with f(dx,dy)*/
			uint2 pixelIndex = make_uint2(x, y);
			float currentWeight = TwUtil::evaluateFilter(x - dCoordsSample.x, y - dCoordsSample.y);

			float4 &currLi = prdRadiance.radiance;
			/*ignore alpha channel*/
			atomicAdd(&sysCurrResultBuffer[pixelIndex].x, currLi.x * currentWeight);
			atomicAdd(&sysCurrResultBuffer[pixelIndex].y, currLi.y * currentWeight);
			atomicAdd(&sysCurrResultBuffer[pixelIndex].z, currLi.z * currentWeight);
			atomicAdd(&sysCurrWeightedSumBuffer[pixelIndex], currentWeight);
		}
	}
}

//////////////////////////////////////////////////////////////////////////
//Initialize outputbuffer and sampleWeightedSum buffer, much more efficient than using serilized "for" on host 
RT_PROGRAM void RayGeneration_InitializeFilter()
{
	sysCurrResultBuffer[sysLaunch_index] = sysOutputBuffer[sysLaunch_index] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
	sysCurrWeightedSumBuffer[sysLaunch_index] = sysSampleWeightedSum[sysLaunch_index] = 0.0f;
}

//////////////////////////////////////////////////////////////////////////
//Perform filtering
RT_PROGRAM void RayGeneration_Filter()
{
	/*perform gamma correction to resolve correct linear color for computation.*/
	sysOutputBuffer[sysLaunch_index] = convertsRGBToLinear(sysOutputBuffer[sysLaunch_index]);

	sysOutputBuffer[sysLaunch_index] = (sysOutputBuffer[sysLaunch_index] * sysSampleWeightedSum[sysLaunch_index] + sysCurrResultBuffer[sysLaunch_index]) / (sysSampleWeightedSum[sysLaunch_index] + sysCurrWeightedSumBuffer[sysLaunch_index]);
    sysHDRBuffer[sysLaunch_index]    = (sysHDRBuffer[sysLaunch_index] * sysSampleWeightedSum[sysLaunch_index] + sysCurrResultBuffer[sysLaunch_index]) / (sysSampleWeightedSum[sysLaunch_index] + sysCurrWeightedSumBuffer[sysLaunch_index]);
	sysSampleWeightedSum[sysLaunch_index] += sysCurrWeightedSumBuffer[sysLaunch_index];

	/*clear current buffer for next iteration*/
	sysCurrResultBuffer[sysLaunch_index] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
	sysCurrWeightedSumBuffer[sysLaunch_index] = 0.0f;

	/*perform "degamma" operation converting linear color to sRGB for diplaying in RenderView*/
	sysOutputBuffer[sysLaunch_index] = convertFromLinearTosRGB(sysOutputBuffer[sysLaunch_index]);
}


RT_PROGRAM void Exception_Default()
{
	const unsigned int code = rtGetExceptionCode();
	rtPrintf("[Exception]Caught exception 0x%X at launch index (%d,%d)\n", code, sysLaunch_index.x, sysLaunch_index.y);
	rtPrintExceptionDetails();
	sysOutputBuffer[sysLaunch_index] = make_float4(1000.0f, 0.0f, 0.0f, 1.0f);
}


