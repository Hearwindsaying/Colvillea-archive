#include <optix_world.h>
#include <optix_device.h>

#include "colvillea/Device/Toolkit/NvRandom.h"
#include "colvillea/Device/Toolkit/Utility.h"
#include "colvillea/Device/Toolkit/CommonStructs.h"
#include "colvillea/Device/Sampler/Sampler.h"
#include "colvillea/Device/Filter/Filter.h"


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

//rtDeclareVariable(float,    sysFilterGaussianAlpha, ,) = 0.25f;//Gaussian filter alpha paramter
//rtDeclareVariable(float,    sysFilterWidth, ,) = 1.f;//Gaussian filter width>=1.f

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
rtDeclareVariable(float,     focalDistance, ,)=0.0f;
rtDeclareVariable(float,     lensRadius, ,)=0.0f;



namespace TwUtil
{
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
    if (lensRadius > 0.f)
    {
        float2 lensSamples = Get2D(&localSampler);
        TwUtil::GenerateRay(pFilm, rayOrg, rayDir, RasterToCamera, CameraToWorld, lensRadius, focalDistance, &lensSamples);
    }
    else
    {
        TwUtil::GenerateRay(pFilm, rayOrg, rayDir, RasterToCamera, CameraToWorld, lensRadius, focalDistance, nullptr);
    }

	/* Make ray and trace, goint to next raytracing pipeline phase. */
	Ray ray = make_Ray(rayOrg, rayDir, toUnderlyingValue(RayType::Radiance), sysSceneEpsilon, RT_DEFAULT_MAX);

    CommonStructs::PerRayData_radiance prdRadiance;
	prdRadiance.radiance = make_float4(0.f);

	rtTrace<CommonStructs::PerRayData_radiance>(sysTopObject, ray, prdRadiance, RT_VISIBILITY_ALL, RT_RAY_FLAG_DISABLE_ANYHIT);

	/*--------------------------------------------------------------------------------*/
	/*----Perform filtering and reconstruction so as to write to the output buffer----*/
	/*--------------------------------------------------------------------------------*/

    /* If filter width <= 0.5f, one sample could only contribute to one pixel
       -- and there is no chance that two samples not in the same pixel will
       -- contribute to the same pixel. So atomic operation could be saved for
       -- efficenciy. */
    float filterWidth = GetFilterWidth();
    if(filterWidth <= 0.5f)
    {
        float currentWeight = EvaluateFilter(qmcSamples.x - 0.5f, qmcSamples.y - 0.5f);

        float4 &currLi = prdRadiance.radiance;
        /*ignore alpha channel*/
        sysCurrResultBuffer[sysLaunch_index].x    += currLi.x * currentWeight;
        sysCurrResultBuffer[sysLaunch_index].y    += currLi.y * currentWeight;
        sysCurrResultBuffer[sysLaunch_index].z    += currLi.z * currentWeight;
        sysCurrWeightedSumBuffer[sysLaunch_index] += currentWeight;
    }
    else
    {
        /* Compute pFilm's raster extent
         * --1.get film sample's discrete coordinates. */
        float2 dCoordsSample = pFilm - 0.5f;

        /*--2.search around the filterWidth for raster pixel boundary*/
        int2 pMin = TwUtil::ceilf2i(dCoordsSample - filterWidth);
        int2 pMax = TwUtil::floorf2i(dCoordsSample + filterWidth);

        /*--3.check for film extent*/
        pMin.x = max(pMin.x, 0);                       pMin.y = max(pMin.y, 0);
        pMax.x = min(pMax.x, sysLaunch_Dim.x - 1);     pMax.y = min(pMax.y, sysLaunch_Dim.y - 1);

        if ((pMax.x - pMin.x) < 0 || (pMax.y - pMin.y) < 0)
        {
            rtPrintf("invalid samples:%f %f\n", dCoordsSample.x, dCoordsSample.y);
        }

        /* Loop over raster pixel and add sample with filtering operation. */
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
                float currentWeight = EvaluateFilter(x - dCoordsSample.x, y - dCoordsSample.y);

                float4 &currLi = prdRadiance.radiance;
                /*ignore alpha channel*/
                atomicAdd(&sysCurrResultBuffer[pixelIndex].x, currLi.x * currentWeight);
                atomicAdd(&sysCurrResultBuffer[pixelIndex].y, currLi.y * currentWeight);
                atomicAdd(&sysCurrResultBuffer[pixelIndex].z, currLi.z * currentWeight);
                atomicAdd(&sysCurrWeightedSumBuffer[pixelIndex], currentWeight);
            }
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
	/* Perform gamma correction to resolve correct linear color for computation. */
	sysOutputBuffer[sysLaunch_index] = convertsRGBToLinear(sysOutputBuffer[sysLaunch_index]);
	sysOutputBuffer[sysLaunch_index] = (sysOutputBuffer[sysLaunch_index] * sysSampleWeightedSum[sysLaunch_index] + sysCurrResultBuffer[sysLaunch_index]) / 
                                       (sysSampleWeightedSum[sysLaunch_index] + sysCurrWeightedSumBuffer[sysLaunch_index]);
    sysHDRBuffer[sysLaunch_index]    = (sysHDRBuffer[sysLaunch_index] * sysSampleWeightedSum[sysLaunch_index] + sysCurrResultBuffer[sysLaunch_index]) / 
                                       (sysSampleWeightedSum[sysLaunch_index] + sysCurrWeightedSumBuffer[sysLaunch_index]);

	sysSampleWeightedSum[sysLaunch_index] += sysCurrWeightedSumBuffer[sysLaunch_index];

    /* Ensure w component of output buffer is 1.0f in case of being transparent
     * -- in RenderView. */
    sysOutputBuffer[sysLaunch_index].w = 1.f;
    sysHDRBuffer[sysLaunch_index].w    = 1.f;

    /* Prevent from precision error. Note that some filters (such as Gaussian)
     * -- could have zero sample weight to a pixel, which may cause that for some
     * -- pixels, they couldn't have any sample weight contribution at all at first
     * -- launch (almost only one sample distributes sparsely per pixel in the situation).
     * But after a few iterations, all pixels are able to have sample weight contribution.
     * For this reason, we skip the NaN values in output buffer to give a chance for
     * later accumulation. */
    if (isnan(sysOutputBuffer[sysLaunch_index].x) || isnan(sysOutputBuffer[sysLaunch_index].y) || isnan(sysOutputBuffer[sysLaunch_index].z) || 
        isnan(sysHDRBuffer[sysLaunch_index].x)    || isnan(sysHDRBuffer[sysLaunch_index].y)    || isnan(sysHDRBuffer[sysLaunch_index].z))
    {
        sysOutputBuffer[sysLaunch_index] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        sysHDRBuffer[sysLaunch_index]    = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    #if 0
    if (isnan(sysOutputBuffer[sysLaunch_index].x) || isnan(sysOutputBuffer[sysLaunch_index].y) || isnan(sysOutputBuffer[sysLaunch_index].z) || (sysOutputBuffer[sysLaunch_index].x < 0.f || sysOutputBuffer[sysLaunch_index].y < 0.f || sysOutputBuffer[sysLaunch_index].z < 0.f))
    {
        rtPrintf("%d %d out=%f * %f + %f / (%f + %f)\n", sysLaunch_index.x,sysLaunch_index.y,
            sysOutputBuffer[sysLaunch_index].x,
            sysSampleWeightedSum[sysLaunch_index], 
            sysCurrResultBuffer[sysLaunch_index].x, 
            sysSampleWeightedSum[sysLaunch_index], 
            sysCurrWeightedSumBuffer[sysLaunch_index]);
    }
#endif // 
    

	/*clear current buffer for next iteration*/
	sysCurrResultBuffer[sysLaunch_index] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
	sysCurrWeightedSumBuffer[sysLaunch_index] = 0.0f;

	/* Perform "degamma" operation converting linear color to sRGB for diplaying in RenderView. */
	sysOutputBuffer[sysLaunch_index] = convertFromLinearTosRGB(sysOutputBuffer[sysLaunch_index]);
}


RT_PROGRAM void Exception_Default()
{
	const unsigned int code = rtGetExceptionCode();
	rtPrintf("[Exception]Caught exception 0x%X at launch index (%d,%d)\n", code, sysLaunch_index.x, sysLaunch_index.y);
	rtPrintExceptionDetails();
	sysOutputBuffer[sysLaunch_index] = make_float4(1000.0f, 0.0f, 0.0f, 1.0f);
}


