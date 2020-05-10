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

rtBuffer<float> areaLightFlmVector;
rtBuffer<float3, 1> areaLightBasisVector;
rtBuffer<float, 2> areaLightAlphaCoeff;

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

    auto sizexy = areaLightAlphaCoeff.size();
    // sizexy.y == num of col
    // sizexy.x == num of rows
    rtPrintf("a[0][0]==%f -> 1.f\n", areaLightAlphaCoeff[make_uint2(0, 0)]);
    rtPrintf("a[1][0]==%f -> 0.04762f\n", areaLightAlphaCoeff[make_uint2(0, 1)]);
    rtPrintf("a[0][1]==%f -> 0.0f\n", areaLightAlphaCoeff[make_uint2(1, 0)]);
    for (int j = 0; j < sizexy.y; ++j)
    {
        for (int i = 0; i < sizexy.x; ++i)
            rtPrintf("%f ", areaLightAlphaCoeff[make_uint2(i, j)]);
        rtPrintf("\n");
    }
}

/**
 * @brief GPU version compute Solid Angle.
 * @param we spherical projection of polygon, index starting from 1
 */
template<int M>
static __device__ __inline__ float computeSolidAngle(const float3 we[])
{
    float S0 = 0.0f;
    for (int e = 1; e <= M; ++e)
    {
        const optix::float3& we_minus_1 = (e == 1 ? we[M] : we[e - 1]);
        const optix::float3& we_plus_1 = (e == M ? we[1] : we[e + 1]);

        float3 tmpa = cross(we[e], we_minus_1);
        float3 tmpb = cross(we[e], we_plus_1);
        S0 += acosf(dot(tmpa, tmpb) / (length(tmpa)*length(tmpb))); // Typo in Wang's paper, length is inside acos evaluation!
    }
    S0 -= (M - 2)*M_PIf;
    return S0;
}

/**
 * @brief GPU Version computeCoeff
 */
template<int M, int wSize>
static __device__ __inline__ void computeCoeff(float3 x, float3 v[]/*, const float3 w[]*//*, const float a[][5]*/, float ylmCoeff[9])
{
    constexpr int lmax = 2;
    //int wSize = areaLightBasisVector.size();

    auto P1 = [](float z)->float {return z; };
#ifdef __CUDACC__
#undef TW_ASSERT
#define TW_ASSERT(expr) TW_ASSERT_INFO(expr, ##expr)
#define TW_ASSERT_INFO(expr, str)    if (!(expr)) {rtPrintf(str); rtPrintf("Above at Line%d:\n",__LINE__);}
#endif
    //TW_ASSERT(v.size() == M + 1);
    //TW_ASSERT(n == 2);
    // for all edges:
    float3 we[M + 1];

    for (int e = 1; e <= M; ++e)
    {
        v[e] = v[e] - x;
        we[e] = TwUtil::safe_normalize(v[e]);
    }

    float3 lambdae[M + 1];
    float3 ue[M + 1];
    float gammae[M + 1];
    for (int e = 1; e <= M; ++e)
    {
        // Incorrect modular arthmetic: we[(e + 1) % (M+1)] or we[(e + 1) % (M)]
        const optix::float3& we_plus_1 = (e == M ? we[1] : we[e + 1]);
        lambdae[e] = cross(TwUtil::safe_normalize(cross(we[e], we_plus_1)), we[e]);
        ue[e] = cross(we[e], lambdae[e]);
        gammae[e] = acosf(dot(we[e], we_plus_1));
    }
    // Solid angle computation
    float S0 = computeSolidAngle<M>(we);

    float Lw[lmax + 1][wSize];

    for (int i = 0; i < wSize; ++i)
    {
        float ae[M + 1];
        float be[M + 1];
        float ce[M + 1];
        float B0e[M + 1];
        float B1e[M + 1];
        float D0e[M + 1];
        float D1e[M + 1];
        float D2e[M + 1];


        const float3 &wi = areaLightBasisVector[i];
        float S1 = 0;
        for (int e = 1; e <= M; ++e)
        {
            ae[e] = dot(wi, we[e]); be[e] = dot(wi, lambdae[e]); ce[e] = dot(wi, ue[e]);
            S1 = S1 + 0.5*ce[e] * gammae[e];

            B0e[e] = gammae[e];
            B1e[e] = ae[e] * sinf(gammae[e]) - be[e] * cosf(gammae[e]) + be[e];
            D0e[e] = 0; D1e[e] = gammae[e]; D2e[e] = 3 * B1e[e];
        }

        //for l=2 to n
        float C1e[M + 1];
        float B2e[M + 1];

        float B2 = 0;
        for (int e = 1; e <= M; ++e)
        {
            C1e[e] = 1.f / 2.f * ((ae[e] * sin(gammae[e]) - be[e] * cosf(gammae[e]))*P1
            (ae[e] * cosf(gammae[e]) + be[e] * sinf(gammae[e])) +
                be[e] * P1(ae[e]) + (ae[e] * ae[e] + be[e] * be[e] - 1.f)*D1e[e] +
                (1.f)*B0e[e]);
            B2e[e] = 1.5f*(C1e[e]) - 1.f*B0e[e];
            B2 = B2 + ce[e] * B2e[e];
            D2e[e] = 3.f * B1e[e] + D0e[e];
        }

        // my code for B1
        float B1 = 0.f;
        for (int e = 1; e <= M; ++e)
        {
            B1 += ce[e] * B1e[e];
        }
        // B1
        float S2 = 0.5f*B1;

        Lw[0][i] = sqrtf(1.f / (4.f*M_PIf))*S0;
        Lw[1][i] = sqrtf(3.f / (4.f*M_PIf))*S1;
        Lw[2][i] = sqrtf(5.f / (4.f*M_PIf))*S2;
    }

    //TW_ASSERT(9 == a.size());
    for (int j = 0; j <= 2; ++j)
    {
        //TW_ASSERT(2 * j + 1 == 2*lmax+1); // redundant storage
        for (int i = 0; i < 2 * j + 1; ++i)
        {
            float coeff = 0.0f;
            for (int k = 0; k < 2 * j + 1; ++k)
            {
                /* Differ from CPU version! access buffer like a coordinates (need a transpose) */
                coeff += areaLightAlphaCoeff[make_uint2(k, j*j + i)] * Lw[j][k];
            }
            ylmCoeff[j*j + i] = coeff;
        }
    }
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

        float t2 = computeSolidAngle<3>(v2);
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

        float t2 = computeSolidAngle<3>(v2);
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

        float t2 = computeSolidAngle<4>(v2);
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

        float t2 = computeSolidAngle<4>(v2);
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

    {
        auto A1 = (sphToCartesian(M_PIf / 2.f, M_PIf / 2.f));
        auto B1 = (sphToCartesian(M_PIf / 4.f, M_PIf / 2.f));
        auto C1 = (sphToCartesian(M_PIf / 4.f, 0.f));
        auto D1 = (sphToCartesian(M_PIf / 2.f, 0.f));

        float3 v[]{ make_float3(0.f),A1,D1,C1 };

        float t2[]{ 0.221557, -0.191874,0.112397,-0.271351,0.257516,-0.106667,-0.157696,-0.182091,0.0910456 };
        float ylmCoeff[9]{ 0.0f,0.0f,0.0f,0.0f,0.0f,0.0f, 0.0f,0.0f,0.0f, };
        computeCoeff<3, 5>(make_float3(0.f), v, ylmCoeff);

        for (int i = 0; i < 9; ++i)
        {
            rtPrintf("Info Line:%d t2[%d]:%f ylmCoeff(GPU)[%d]:%f\n", __LINE__, i, t2[i], i, ylmCoeff[i]);
        }
    }
    {
        auto A1 = (sphToCartesian(M_PIf / 2.f, M_PIf / 2.f));
        auto B1 = (sphToCartesian(M_PIf / 4.f, M_PIf / 2.f));
        auto C1 = (sphToCartesian(M_PIf / 4.f, 0.f));
        auto D1 = (sphToCartesian(M_PIf / 2.f, 0.f));

        float3 v[]{ make_float3(0.f),A1,C1,D1 };

        float t2[]{ 0.221557, -0.191874,0.112397,-0.271351,0.257516,-0.106667,-0.157696,-0.182091,0.0910456 };
        float ylmCoeff[9]{ 0.0f,0.0f,0.0f,0.0f,0.0f,0.0f, 0.0f,0.0f,0.0f, };
        computeCoeff<3, 5>(make_float3(0.f), v, ylmCoeff);

        for (int i = 0; i < 9; ++i)
        {
            rtPrintf("Info Line:%d t2[%d]:%f ylmCoeff(GPU)[%d]:%f\n", __LINE__, i, t2[i], i, ylmCoeff[i]);
        }
    }
    {
        auto A1 = make_float3(0.0f, 1.0f, 0.0f);
        auto B1 = make_float3(-1.0f, 0.0f, 0.0f);
        auto C1 = make_float3(0.0f, -1.0f, 0.0f);
        auto D1 = make_float3(1.0f, 0.0f, 0.0f);

        float3 v[]{ make_float3(0.f),A1,B1,C1,D1 };
        float t2[]{ 0,-3.72529e-09,1.53499,0,-7.25683e-08,8.81259e-08,-1.43062e-07,-2.99029e-08,7.10429e-08 };

        float ylmCoeff[9]{ 0.0f,0.0f,0.0f,0.0f,0.0f,0.0f, 0.0f,0.0f,0.0f, };
        computeCoeff<4, 5>(make_float3(0.f), v, ylmCoeff);
        for (int i = 0; i < 9; ++i)
        {
            rtPrintf("Test failed at Line: t2[%d]:%f ylmCoeff(GPU)[%d]:%f\n", i, t2[i], i, ylmCoeff[i]);
        }
    }
}

//#define RUN_TEST_SHI

//////////////////////////////////////////////////////////////////////////
//Program definitions:
RT_PROGRAM void RayGeneration_PinholeCamera()
{
#ifdef RUN_TEST_SHI
    if (sysLaunch_index == make_uint2(0, 0))
    {
        //Test_AP_Flm_MatrixOrder3();
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


