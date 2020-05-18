#include <optix_world.h>
#include <optix_device.h>

#include "colvillea/Device/Toolkit/NvRandom.h"
#include "colvillea/Device/Toolkit/Utility.h"
#include "colvillea/Device/Toolkit/SH.h"
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

#ifndef TW_RT_DECLARE_AREALIGHTCOEFF
#define TW_RT_DECLARE_AREALIGHTCOEFF
/* Flm Diffuse Matrix. */
rtBuffer<float> areaLightFlmVector;

/* Basis Directions. */
rtBuffer<optix::float3, 1> areaLightBasisVector;

rtBuffer<float, 2> areaLightAlphaCoeff;
#endif

static __device__ __inline__ void Test_AP_Flm_MatrixOrder3()
{
    /* Test if AP Matrix is loaded correctly. */
    /*rtPrintf("%f == 0.282095f, %f == 0.0232673, %f == 1.31866 \n", areaLightAPMatrix[make_uint2(0, 0)], areaLightAPMatrix[make_uint2(1, 1)], areaLightAPMatrix[make_uint2(14, 8)]);
    rtPrintf("%f == -1.05266f \n", areaLightAPMatrix[make_uint2(11, 4)]);

    rtPrintf("\n");

    auto sizexy = areaLightAPMatrix.size();
    for (int j = 0; j < sizexy.y; ++j)
    {
        for (int i=0; i<sizexy.x;++i)
            rtPrintf("%f ", areaLightAPMatrix[make_uint2(i, j)]);
        rtPrintf("\n");
    }*/

    /* Test Basis Vector. */
    for (int i = 0; i < areaLightBasisVector.size(); ++i)
        rtPrintf("%f %f %f\n", areaLightBasisVector[i].x, areaLightBasisVector[i].y, areaLightBasisVector[i].z);

    /* Test Flm Vector. */
    rtPrintf("Flm Vector:\n");
    for (int i = 0; i < areaLightFlmVector.size(); ++i)
    {
        rtPrintf("%f ", areaLightFlmVector[i]);
    }
    rtPrintf("\n------------Done printing------------\n");

    /* Test A coeff. */
    rtPrintf("\n");

    //auto sizexy = areaLightAlphaCoeff.size();
    //// sizexy.y == num of col
    //// sizexy.x == num of rows
    //rtPrintf("a[0][0]==%f -> 1.f\n", areaLightAlphaCoeff[make_uint2(0, 0)]);
    //rtPrintf("a[1][0]==%f -> 0.04762f\n", areaLightAlphaCoeff[make_uint2(0, 1)]);
    //rtPrintf("a[0][1]==%f -> 0.0f\n", areaLightAlphaCoeff[make_uint2(1, 0)]);
    //for (int j = 0; j < sizexy.y; ++j)
    //{
    //    for (int i = 0; i < sizexy.x; ++i)
    //        rtPrintf("%f ", areaLightAlphaCoeff[make_uint2(i, j)]);
    //    rtPrintf("\n");
    //}
}



static __device__ __inline__ void TestSolidAngleGPU()
{
    auto sphToCartesian = [](const float theta, const float phi)->float3
    {
        return make_float3(sinf(theta)*cosf(phi), sinf(theta)*sinf(phi), cosf(theta));
    };

    {
        auto A1 = make_float3(0.0f, 1.0f, 0.0f);
        auto B1 = make_float3(-1.0f, 0.0f, 0.0f);
        auto C1 = make_float3(0.0f, -1.0f, 0.0f);
        auto D1 = make_float3(1.0f, 0.0f, 0.0f);
        float3 v2[]{ make_float3(0.f), A1,sphToCartesian(0.f,0.f),D1 };

        float t2 = Cl::computeSolidAngle<3>(v2);
        if (!(t2 == 0.5f*M_PIf))
        {
            rtPrintf("Test failed at %d: t2==%f != %f\n", __LINE__, t2, 0.5f*M_PIf);
        }
    }

    {
        auto A1 = make_float3(0.0f, 1.0f, 0.0f);
        auto C1 = make_float3(0.0f, -1.0f, 0.0f);
        auto D1 = make_float3(1.0f, 0.0f, 0.0f);
        float3 v2[]{ make_float3(0.f), A1,C1,D1 };

        float t2 = Cl::computeSolidAngle<3>(v2);
        {
            rtPrintf("Test failed at %d: t2==%f \n", __LINE__, t2);
        }
    }

    {
        auto A1 = make_float3(0.0f, 1.0f, 0.0f);
        auto B1 = make_float3(-1.0f, 0.0f, 0.0f);
        auto C1 = make_float3(0.0f, -1.0f, 0.0f);
        auto D1 = make_float3(1.0f, 0.0f, 0.0f);
        float3 v2[]{ make_float3(0.f), A1,B1,C1,D1 };

        float t2 = Cl::computeSolidAngle<4>(v2);
        {
            rtPrintf("Test failed at %d: t2==%f -> 6.28 \n", __LINE__, t2);
        }
    }

    {
        auto A1 = (sphToCartesian(M_PI / 2.f, M_PI / 2.f));
        auto B1 = (sphToCartesian(M_PI / 4.f, M_PI / 2.f));
        auto C1 = (sphToCartesian(M_PI / 4.f, 0.f));
        auto D1 = (sphToCartesian(M_PI / 2.f, 0));
        float3 v2[]{ make_float3(0.f), A1,B1,C1,D1 };

        float t2 = Cl::computeSolidAngle<4>(v2);
        {
            rtPrintf("Test failed at %d: t2==%f -> 1.2309 \n", __LINE__, t2);
        }
    }
}


static __device__ __inline__ void TestZHRecurrenceGPU()
{
    auto sphToCartesian = [](const float theta, const float phi)->float3
    {
        return make_float3(sinf(theta)*cosf(phi), sinf(theta)*sinf(phi), cosf(theta));
    };

   /* {
        auto A1 = (sphToCartesian(M_PIf / 2.f, M_PIf / 2.f));
        auto B1 = (sphToCartesian(M_PIf / 4.f, M_PIf / 2.f));
        auto C1 = (sphToCartesian(M_PIf / 4.f, 0.f));
        auto D1 = (sphToCartesian(M_PIf / 2.f, 0.f));

        float3 v[]{ make_float3(0.f),A1,D1,C1 };

        float t2[]{ 0.221557, -0.191874,0.112397,-0.271351,0.257516,-0.106667,-0.157696,-0.182091,0.0910456 };
        float ylmCoeff[9]{ 0.0f,0.0f,0.0f,0.0f,0.0f,0.0f, 0.0f,0.0f,0.0f, };
        computeCoeff<3, 9>(make_float3(0.f), v, ylmCoeff);

        for (int i = 0; i < 9; ++i)
        {
            rtPrintf("Info Line:%d t2[%d]:%f ylmCoeff(GPU)[%d]:%f\n", __LINE__, i, t2[i], i, ylmCoeff[i]);
        }
    }*/
    /*{
        auto A1 = (sphToCartesian(M_PIf / 2.f, M_PIf / 2.f));
        auto B1 = (sphToCartesian(M_PIf / 4.f, M_PIf / 2.f));
        auto C1 = (sphToCartesian(M_PIf / 4.f, 0.f));
        auto D1 = (sphToCartesian(M_PIf / 2.f, 0.f));

        float3 v[]{ make_float3(0.f),A1,C1,D1 };

        float t2[]{ 0.221557, -0.191874,0.112397,-0.271351,0.257516,-0.106667,-0.157696,-0.182091,0.0910456 };
        float ylmCoeff[9]{ 0.0f,0.0f,0.0f,0.0f,0.0f,0.0f, 0.0f,0.0f,0.0f, };
        computeCoeff<3, 9>(make_float3(0.f), v, ylmCoeff);

        for (int i = 0; i < 9; ++i)
        {
            rtPrintf("Info Line:%d t2[%d]:%f ylmCoeff(GPU)[%d]:%f\n", __LINE__, i, t2[i], i, ylmCoeff[i]);
        }
    }*/
    {
        auto A1 = make_float3(0.0f, 1.0f, 0.0f);
        auto B1 = make_float3(-1.0f, 0.0f, 0.0f);
        auto C1 = make_float3(0.0f, -1.0f, 0.0f);
        auto D1 = make_float3(1.0f, 0.0f, 0.0f);

        float3 v[]{ make_float3(0.f),A1,B1,C1,D1 };
        float t2[]{ 0,-2.38419e-07,1.53499,-1.78814e-07,8.0361e-09,1.29553e-08,-1.49964e-09,2.37276e-08,-5.70635e-08,1.90735e-06,0,
1.19209e-07,-0.586183,-1.19209e-07,-1.19209e-07,4.91738e-07,2.90852e-07,-1.73211e-08,3.61446e-07,-5.52696e-07,2.85673e-07,
2.19718e-07,2.16946e-07,4.27483e-07,2.03351e-07,0,3.21865e-06,-3.93391e-06,-9.53674e-07,1.07288e-06,0.367402,
-9.53674e-07,3.8147e-06,-1.66893e-06,-7.45058e-07,4.05312e-06,1.59473e-07,8.44052e-07,-1.00783e-07,4.63194e-07,6.57873e-07,
-1.27605e-07,6.28974e-07,-9.65823e-07,-9.55999e-07,1.80002e-06,-1.09245e-06,-9.892e-07,-3.4858e-07,1.62125e-05,-1.14441e-05,
2.38419e-06,-2.86102e-06,-4.76837e-06,1.90735e-06,1.81198e-05,-0.26816,3.8147e-06,-3.33786e-06,6.67572e-06,7.62939e-06,
3.8147e-06,8.58307e-06,-7.62939e-06,-1.8975e-06,-5.77771e-06,-7.41833e-06,-2.07832e-06,-7.66758e-06,-6.26134e-07,3.82385e-06,
-1.88402e-06,-3.5203e-06,1.18708e-06,8.25938e-06,1.41067e-05,-4.0676e-06,-5.4201e-06,6.67927e-06,-4.89425e-06,-4.6691e-06,
1.5,0.00292969,0.265625,1.625,0.21875,1.375,0.484375,1.875,-0.625,-0.375,
-0.4375,1.375,0.65625,-0.00683594,1.25,-0.8125,0.0859375,0.75,0.1875, };
        
        float ylmCoeff[100];
        Cl::computeCoeff<4, 9>(v, ylmCoeff);
        for (int i = 0; i < 100; ++i)
        {
            rtPrintf("Test failed at Line: t2[%d]:%f ylmCoeff(GPU)[%d]:%f\n", i, t2[i], i, ylmCoeff[i]);
        }
    }
}

#define RUN_TEST_SHI 0

//////////////////////////////////////////////////////////////////////////
//Program definitions:
RT_PROGRAM void RayGeneration_PinholeCamera()
{
#if RUN_TEST_SHI
    if (sysLaunch_index == make_uint2(0, 0))
    {
        //Test_AP_Flm_MatrixOrder3();
        //TestSolidAngleGPU();
        TestZHRecurrenceGPU();
        //testing();
        return;
    }
    else { return; }
#endif

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


