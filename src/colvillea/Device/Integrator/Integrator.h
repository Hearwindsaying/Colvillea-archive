#pragma once
#ifndef COLVILLEA_DEVICE_INTEGRATOR_INTEGRATOR_H_
#define COLVILLEA_DEVICE_INTEGRATOR_INTEGRATOR_H_
/* This file is device only. */
#include <optix_world.h>
#include <optix_device.h>
#include <optixu_math_namespace.h>

#include "../Toolkit/CommonStructs.h"
#include "../Toolkit/Utility.h"
#include "../Toolkit/NvRandom.h"
#include "../Light/LightUtil.h"

#include "../Sampler/Sampler.h"

//////////////////////////////////////////////////////////////////////////
//Forward declarations:

//system variables:->Context
rtDeclareVariable(rtObject,				sysTopShadower, , );
rtDeclareVariable(rtObject,				sysTopObject, , );
rtDeclareVariable(float,                sysSceneEpsilon, , );

#ifndef TWRT_DELCARE_SAMPLERTYPE
#define TWRT_DELCARE_SAMPLERTYPE
rtDeclareVariable(int, sysSamplerType, , );         /* Sampler type chosen in GPU program.
                                                       --need to fetch underlying value from
                                                         CommonStructs::SamplerType. */
#endif

//lights:
#ifndef TWRT_DELCARE_LIGHTBUFFER
#define TWRT_DELCARE_LIGHTBUFFER
rtDeclareVariable(CommonStructs::LightBuffers, sysLightBuffers, , );
#endif

rtDeclareVariable(CommonStructs::PerRayData_radiance,  prdRadiance,     rtPayload, );
rtDeclareVariable(CommonStructs::PerRayData_shadow,	   prdShadow,	    rtPayload, );
rtDeclareVariable(Ray,					ray,		     rtCurrentRay, );
rtDeclareVariable(float,				tHit,		     rtIntersectionDistance, );

#ifndef TWRT_DECLARE_SYS_ITERATION_INDEX
#define TWRT_DECLARE_SYS_ITERATION_INDEX
rtDeclareVariable(uint, sysIterationIndex, , ) = 0;
#endif

#ifndef TWRT_DECLARE_SYSLAUNCH
#define TWRT_DECLARE_SYSLAUNCH
rtDeclareVariable(uint2, sysLaunch_Dim, rtLaunchDim, );
rtDeclareVariable(uint2, sysLaunch_index, rtLaunchIndex, );
#endif


//Bindless BSDF callable programs:->Context
rtBuffer< rtCallableProgramId<
    float4(const float3 & wo_World, const float3 & wi_World, const CommonStructs::ShaderParams & shaderParams)> > Eval_f;
rtBuffer< rtCallableProgramId<
    float4(const float3 &wo_World, float3 & outwi_World, float2 & urand, float & outPdf, float bsdfChoiceRand, const CommonStructs::ShaderParams & shaderParams)> >
    Sample_f;
rtBuffer< rtCallableProgramId<
    float(const float3 & wo_World, const float3 & wi_World, const CommonStructs::ShaderParams & shaderParams)> >
    Pdf;

//Bindless Light callable programs:->Context
rtBuffer< rtCallableProgramId<
    float4(const float3 &point, const float & rayEpsilon, float3 & outwi, float & outpdf, float2 lightSample, uint lightBufferIndex, Ray & outShadowRay)> > 
    Sample_Ld;
rtBuffer< rtCallableProgramId<
    float(const float3 & p, const float3 & wi, const int lightId, Ray &shadowRay)> >
    LightPdf;


//////////////////////////////////////////////////////////////////////////
//Common integrator utility functions:

/**
 * @brief Estimate direct lighting contribution to the surface
 * interaction. For performance and implementation consideration,
 * supported light types in Colvillea should provide partial 
 * template specification in accordance with individual 
 * CommonStructs::LightType. It's also recommended that divide
 * light type into HDRILight, AreaLight and Dirac (Delta) Light
 * for more generic programming. 
 * 
 * @param[in] lightId index to light buffer, only avaliable when
 * lightType is not HDRILight
 * @param[in] shaderParams material and differential geometry information
 * @param[in] isectP       intersection point in world space
 * @param[in] isectDir     incoming direction in world space, pointing out of the surface
 * @param[in] localSampler local sampler binded to current launch
 * 
 * @return Return direct lighting contribution due to one specific
 * light.
 */
template<CommonStructs::LightType lightType>
static __device__ __inline__ float4 EstimateDirectLighting(
    int lightId,
    const CommonStructs::ShaderParams & shaderParams,
    const float3 & isectP, const float3 & isectDir,
    GPUSampler &localSampler);

template<>
static __device__ __inline__ float4 EstimateDirectLighting<CommonStructs::LightType::HDRILight>(
    int lightId,
    const CommonStructs::ShaderParams & shaderParams,
    const float3 & isectP, const float3 & isectDir,
    GPUSampler &localSampler)
{
    float4 Ld = make_float4(0.f);

    float3 p = isectP;
    float3 wo_world = isectDir;
    float sceneEpsilon = sysSceneEpsilon;

    float3 outWi = make_float3(0.f);
    float lightPdf = 0.f, bsdfPdf = 0.f;
    Ray shadowRay;

    /* Sample light source with Mulitple Importance Sampling. */
    float2 randSamples = Get2D(&localSampler);

    float4 Li = Sample_Ld[toUnderlyingValue(CommonStructs::LightType::HDRILight)](p, sceneEpsilon, outWi, lightPdf, randSamples, lightId, shadowRay);
    if (lightPdf > 0.f && !isBlack(Li))
    {
        /* Compute BSDF value using sampled outWi from sampling light source. */
        float4 f = Eval_f[toUnderlyingValue(shaderParams.bsdfType)](wo_world, outWi, shaderParams);

        if (!isBlack(f))
        {
            /* Trace shadow ray and find out its visibility. */
            CommonStructs::PerRayData_shadow shadow_prd;
            shadow_prd.blocked = 0;

            const RTrayflags shadowRayFlags = static_cast<RTrayflags>(RTrayflags::RT_RAY_FLAG_DISABLE_ANYHIT | RTrayflags::RT_RAY_FLAG_TERMINATE_ON_FIRST_HIT);
            rtTrace<CommonStructs::PerRayData_shadow>(sysTopShadower, shadowRay, shadow_prd, RT_VISIBILITY_ALL, shadowRayFlags);

            if (!shadow_prd.blocked)
            {
                /* Compute Ld using MIS weight. */

                bsdfPdf = Pdf[toUnderlyingValue(shaderParams.bsdfType)](wo_world, outWi, shaderParams);
                float weight = TwUtil::MonteCarlo::PowerHeuristic(1, lightPdf, 1, bsdfPdf);
                Ld += f * Li * (fabs(dot(outWi, shaderParams.dgShading.nn)) * weight / lightPdf);

                /* Check exceptional value. */
                if (isnan(Ld.x) || isnan(Ld.y) || isnan(Ld.z))
                    rtPrintf("l107f:%f %f %f Li:%f %f %f cosTheta:%f weight:%f lightpdf:%f\n", f.x, f.y, f.z,
                        Li.x, Li.y, Li.z, (fabs(dot(outWi, shaderParams.dgShading.nn)), weight, lightPdf));
                if (weight == 0.f && lightPdf == 0.f)
                    rtPrintf("weirdBSDFPDF %f lightPdf %f weight %f!!!", bsdfPdf, lightPdf, weight);

                if (Ld.x >= 300.f || Ld.y >= 300.f || Ld.z >= 300.f)
                {
                    rtPrintf("LightSampling:f)%f %f %f Li:%f %f %f cosTheta:%f weight:%.7f lightpdf:%.7f bsdfPdf:%.7f Ld+:%f\n", f.x, f.y, f.z,
                        Li.x, Li.y, Li.z, fabs(dot(outWi, shaderParams.dgShading.nn)),
                        weight, lightPdf, bsdfPdf,
                        (f * Li * (fabs(dot(outWi, shaderParams.dgShading.nn)) * weight / lightPdf)).x);
                }
            }
        }
    }


    /* Sample BSDF with Multiple Importance Sampling. */
    randSamples = Get2D(&localSampler);

    float4 f = Sample_f[toUnderlyingValue(shaderParams.bsdfType)](wo_world, outWi, randSamples, bsdfPdf, Get1D(&localSampler), shaderParams);
    if (!isBlack(f) && bsdfPdf > 0.)
    {
        float weight = 1.f;

        //TODO:should use sampledType while involving with compound BSDF type
        if (shaderParams.bsdfType != CommonStructs::BSDFType::SmoothGlass && shaderParams.bsdfType != CommonStructs::BSDFType::SmoothMirror)
        {
            lightPdf = LightPdf[toUnderlyingValue(CommonStructs::LightType::HDRILight)](p, outWi, lightId, shadowRay);
            if (lightPdf == 0.f)
                return Ld;
            weight = TwUtil::MonteCarlo::PowerHeuristic(1, bsdfPdf, 1, lightPdf);
        }


        /* Trace shadow ray to find out whether it's blocked down by object. */
        CommonStructs::PerRayData_shadow shadow_prd;
        shadow_prd.blocked = 0;
        shadowRay = TwUtil::MakeShadowRay(p, sceneEpsilon, outWi);
        
        const RTrayflags shadowRayFlags = static_cast<RTrayflags>(RTrayflags::RT_RAY_FLAG_DISABLE_ANYHIT | RTrayflags::RT_RAY_FLAG_TERMINATE_ON_FIRST_HIT);
        rtTrace<CommonStructs::PerRayData_shadow>(sysTopShadower, shadowRay, shadow_prd, RT_VISIBILITY_ALL,
            shadowRayFlags);

        if (!shadow_prd.blocked)
        {
            Li = TwUtil::Le_HDRILight(outWi, sysLightBuffers.hdriLight.hdriEnvmap, sysLightBuffers.hdriLight.worldToLight);
            if(!isBlack(Li))
                Ld += f * Li * fabsf(dot(outWi, shaderParams.dgShading.nn)) * weight / bsdfPdf;

            /* Check exceptional values. */
            if (isnan(Ld.x) || isnan(Ld.y) || isnan(Ld.z))
            {
                rtPrintf("costheta:%f\n", fabsf(dot(outWi, shaderParams.dgShading.nn)));
                rtPrintf("%f %f %f Li:%f %f %f weight:%f bsdfPdf:%f,outwi:%f %f %f",
                    f.x, f.y, f.z,
                    Li.x, Li.y, Li.z,
                    weight,
                    bsdfPdf,
                    outWi.x, outWi.y, outWi.z);
                rtPrintf("PdfBack:%f\n", Pdf[toUnderlyingValue(shaderParams.bsdfType)](wo_world, outWi, shaderParams));
            }
        }
    }

    return Ld;
}

template<>
static __device__ __inline__ float4 EstimateDirectLighting<CommonStructs::LightType::PointLight>(
    int lightId,
    const CommonStructs::ShaderParams & shaderParams,
    const float3 & isectP, const float3 & isectDir,
    GPUSampler &localSampler)
{
	float4 Ld = make_float4(0.f);

	float3 p = isectP;
	float3 wo_world = isectDir;
    float sceneEpsilon = sysSceneEpsilon;
	
	float3 outWi = make_float3(0.f);
	float lightPdf = 0.f;
	Ray shadowRay;

	//Sample point light, no MIS performed here
	float4 Li = Sample_Ld[toUnderlyingValue(CommonStructs::LightType::PointLight)](p, sceneEpsilon, outWi, lightPdf, make_float2(0.f), lightId, shadowRay);

	if (lightPdf > 0.f && !isBlack(Li))
	{
		float4 f = Eval_f[toUnderlyingValue(shaderParams.bsdfType)](wo_world, outWi, shaderParams);
		if (!isBlack(f))
		{
			//Trace ShadowRay
            CommonStructs::PerRayData_shadow shadow_prd;
			shadow_prd.blocked = 0;

            const RTrayflags shadowRayFlags = static_cast<RTrayflags>(RTrayflags::RT_RAY_FLAG_DISABLE_ANYHIT | RTrayflags::RT_RAY_FLAG_TERMINATE_ON_FIRST_HIT);
			rtTrace<CommonStructs::PerRayData_shadow>(sysTopShadower, shadowRay, shadow_prd, RT_VISIBILITY_ALL, shadowRayFlags);

			if (!shadow_prd.blocked)
				Ld += f * Li * fabs(dot(outWi, shaderParams.dgShading.nn)) / lightPdf;
		}
	}

	return Ld;
}

/************************************************************************/
/*         Integrating Clipped Spherical Harmonics Expansions           */
/************************************************************************/

/* Flm Diffuse Matrix. */
rtBuffer<float> areaLightFlmVector;

/* Basis Directions. */
rtBuffer<float3, 1> areaLightBasisVector;

rtBuffer<float, 2> areaLightAlphaCoeff;
/************************************************************************/
/*         Integrating Clipped Spherical Harmonics Expansions           */
/************************************************************************/
/* BSDF World To Local conversion. */
static __device__ __inline__ optix::float3 BSDFWorldToLocal(const optix::float3 & v, const optix::float3 & sn, const optix::float3 & tn, const optix::float4 & nn, const optix::float3 &worldPoint)
{
    float3 pt;
    pt.x = dot(v, sn) - dot(worldPoint, sn);
    pt.y = dot(v, tn) - dot(worldPoint, tn);
    pt.z = dot(v, nn) - dot(worldPoint, nn);
    return pt;
}

/* Clipping Algorithm. */
static __device__ __inline__ void ClipQuadToHorizon(optix::float3 L[5], int &n)
{
    /* Make a copy of L[]. */
    optix::float3 Lorg[4];
    
    //memcpy(&Lorg[0], &L[0], sizeof(optix::float3) * 4);
    for (int i = 0; i <= 3; ++i)
        Lorg[i] = L[i];

    auto IntersectRayZ0 = [](const optix::float3 &A, const optix::float3 &B)->optix::float3
    {
        float3 o = A;
        float3 d = TwUtil::safe_normalize(B - A);
        float t = -A.z * (length(B - A) / (B - A).z);
        if (!(t >= 0.f))rtPrintf("error in IntersectRayZ0.\n");
        return o + t * d;
    };

    n = 0;
    for (int i = 1; i <= 4; ++i)
    {
        const float3& A = Lorg[i - 1];
        const float3& B = i == 4 ? Lorg[0] : Lorg[i]; // Loop back to zero index
        if (A.z <= 0 && B.z <= 0)
            continue;
        else if (A.z >= 0 && B.z >= 0)
        {
            L[n++] = A;
        }
        else if (A.z >= 0 && B.z <= 0)
        {
            L[n++] = A;
            L[n++] = IntersectRayZ0(A, B);
        }
        else if (A.z <= 0 && B.z >= 0)
        {
            L[n++] = IntersectRayZ0(A, B);
        }
        else
        {
            rtPrintf("ClipQuadToHorizon A B.z.\n");
        }
    }
    if(!(n == 0 || n == 3 || n == 4 || n == 5))
        rtPrintf("ClipQuadToHorizon n.\n");
}

/************************************************************************/
/*   Analytic Spherical Harmonic Coefficients for Polygonal Area Light  */
/************************************************************************/ 
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
                coeff += areaLightAlphaCoeff[make_uint2(k,j*j + i)] * Lw[j][k];
            }
            ylmCoeff[j*j + i] = coeff;
        }
    }
}

template<>
static __device__ __inline__ float4 EstimateDirectLighting<CommonStructs::LightType::QuadLight>(
    int lightId,
    const CommonStructs::ShaderParams & shaderParams,
    const float3 & isectP, const float3 & isectDir,
    GPUSampler &localSampler)
{
    if (shaderParams.bsdfType != BSDFType::Lambert)
    {
        rtPrintf("SH Integration supports lambert only.\n");
    }
#if 1 /* SH Integration */
    /* 1. Get 4 vertices of QuadLight. */
    /*if (sysLaunch_index.y >= 0 && sysLaunch_index.y <= 360)*/
    {
        float3 quadShape[5];
        const CommonStructs::QuadLight &quadLight = sysLightBuffers.quadLightBuffer[lightId];

        float4 L = make_float4(0.f);

        quadShape[0] = TwUtil::xfmPoint(make_float3(-1.f, -1.f, 0.f), quadLight.lightToWorld);
        quadShape[1] = TwUtil::xfmPoint(make_float3(1.f, -1.f, 0.f), quadLight.lightToWorld);
        quadShape[2] = TwUtil::xfmPoint(make_float3(1.f, 1.f, 0.f), quadLight.lightToWorld);
        quadShape[3] = TwUtil::xfmPoint(make_float3(-1.f, 1.f, 0.f), quadLight.lightToWorld);
        quadShape[4] = make_float3(0.f);

        /* 2. Convert QuadLight from World To BSDFLocal (or just use directional information and call WorldToLocal only. */
        quadShape[0] = BSDFWorldToLocal(quadShape[0], shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn, isectP);
        quadShape[1] = BSDFWorldToLocal(quadShape[1], shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn, isectP);
        quadShape[2] = BSDFWorldToLocal(quadShape[2], shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn, isectP);
        quadShape[3] = BSDFWorldToLocal(quadShape[3], shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn, isectP);

//         if (shaderParams.Reflectance.x >= 0.6f && shaderParams.Reflectance.x <= 0.7f)
//         {
//             rtPrintf("%f %f %f, %f %f %f, %f %f %f, %f %f %f\n", quadShape[0].x, quadShape[0].y, quadShape[0].z,
//                 quadShape[1].x, quadShape[1].y, quadShape[1].z,
//                 quadShape[2].x, quadShape[2].y, quadShape[2].z,
//                 quadShape[3].x, quadShape[3].y, quadShape[3].z);
//         }

        /* 3. Clipping. */
        int clippedQuadVertices = 0;
        ClipQuadToHorizon(quadShape, clippedQuadVertices);

        /* 4. Compute Ylm projection coefficient. */
        float ylmCoeff[9];
        if (clippedQuadVertices == 3)
        {
            float3 ABC[4]{ make_float3(0.f),quadShape[0],quadShape[1],quadShape[2] };
            computeCoeff<3, 5>(make_float3(0.f), ABC, ylmCoeff);
        }
        else if (clippedQuadVertices == 4)
        {
            float3 ABCD[5]{ make_float3(0.f),quadShape[0],quadShape[1],quadShape[2],quadShape[3] };
            computeCoeff<4, 5>(make_float3(0.f), ABCD, ylmCoeff);
        }
        else
        {
            float3 ABCDE[6]{ make_float3(0.f),quadShape[0],quadShape[1],quadShape[2],quadShape[3],quadShape[4] };
            computeCoeff<5, 5>(make_float3(0.f), ABCDE, ylmCoeff);
        }

        /* 5. Dot Product of Flm and Ylm. */
        for (int i = 0; i < 9; ++i)
        {
            L += make_float4(areaLightFlmVector[i] * ylmCoeff[i]);
        }

        if (areaLightBasisVector.size() != 5)rtPrintf("assert failed at areaLightBasisVector.size()!=5\n");
        L *= quadLight.intensity * shaderParams.Reflectance / M_PIf;
        return L;
    }
//#else
    //else {
    //    float4 Ld = make_float4(0.f);

    //    float3 p = isectP;
    //    float3 wo_world = isectDir;
    //    float sceneEpsilon = sysSceneEpsilon;

    //    float3 outWi = make_float3(0.f);
    //    float lightPdf = 0.f, bsdfPdf = 0.f;
    //    Ray shadowRay;



    //    /* Sample light source with Mulitple Importance Sampling. */
    //    float2 randSamples = Get2D(&localSampler);
    //    float4 Li = Sample_Ld[toUnderlyingValue(CommonStructs::LightType::QuadLight)](p, sceneEpsilon, outWi, lightPdf, randSamples, lightId, shadowRay);
    //    if (lightPdf > 0.f && !isBlack(Li))
    //    {
    //        /* Compute BSDF value using sampled outWi from sampling light source. */
    //        float4 f = Eval_f[toUnderlyingValue(shaderParams.bsdfType)](wo_world, outWi, shaderParams);
    //        if (!isBlack(f))
    //        {
    //            /* Trace shadow ray and find out its visibility. */
    //            CommonStructs::PerRayData_shadow shadow_prd;
    //            shadow_prd.blocked = 0;

    //            const RTrayflags shadowRayFlags = static_cast<RTrayflags>(RTrayflags::RT_RAY_FLAG_DISABLE_ANYHIT | RTrayflags::RT_RAY_FLAG_TERMINATE_ON_FIRST_HIT);
    //            rtTrace<CommonStructs::PerRayData_shadow>(sysTopShadower, shadowRay, shadow_prd, RT_VISIBILITY_ALL, shadowRayFlags);
    //            if (!shadow_prd.blocked)
    //            {
    //                /* Compute Ld using MIS weight. */

    //                bsdfPdf = Pdf[toUnderlyingValue(shaderParams.bsdfType)](wo_world, outWi, shaderParams);
    //                float weight = TwUtil::MonteCarlo::PowerHeuristic(1, lightPdf, 1, bsdfPdf);
    //                Ld += f * Li * (fabs(dot(outWi, shaderParams.dgShading.nn)) * weight / lightPdf);
    //                /*rtPrintf("%f %f %f Li:%f %f %f weight:%f bsdfPdf:%f,outwi:%f %f %f\n",
    //                    f.x, f.y, f.z,
    //                    Li.x, Li.y, Li.z,
    //                    weight,
    //                    bsdfPdf,
    //                    outWi.x, outWi.y, outWi.z);
    //                rtPrintf("Ld %f %f %f\n", Ld.x, Ld.y, Ld.z);*/
    //            }
    //        }
    //    }


    //    /* Sample BSDF with Multiple Importance Sampling. */
    //    randSamples = Get2D(&localSampler);

    //    float4 f = Sample_f[toUnderlyingValue(shaderParams.bsdfType)](wo_world, outWi, randSamples, bsdfPdf, Get1D(&localSampler), shaderParams);
    //    if (!isBlack(f) && bsdfPdf > 0.)
    //    {
    //        float weight = 1.f;

    //        //todo:should use sampledType while we use overall bsdf type:
    //        if (shaderParams.bsdfType != BSDFType::SmoothGlass && shaderParams.bsdfType != BSDFType::SmoothMirror)
    //        {
    //            lightPdf = LightPdf[toUnderlyingValue(CommonStructs::LightType::QuadLight)](p, outWi, lightId, shadowRay);

    //            if (lightPdf == 0.f)
    //                return Ld;

    //            weight = TwUtil::MonteCarlo::PowerHeuristic(1, bsdfPdf, 1, lightPdf);
    //        }


    //        /* Trace shadow ray to find out whether it's blocked down by object. */
    //        CommonStructs::PerRayData_shadow shadow_prd;
    //        shadow_prd.blocked = 0;

    //        const RTrayflags shadowRayFlags = static_cast<RTrayflags>(RTrayflags::RT_RAY_FLAG_DISABLE_ANYHIT | RTrayflags::RT_RAY_FLAG_TERMINATE_ON_FIRST_HIT);
    //        rtTrace<CommonStructs::PerRayData_shadow>(sysTopShadower, shadowRay, shadow_prd, RT_VISIBILITY_ALL, shadowRayFlags);

    //        /* In LightPdf we have confirmed that ray is able to intersect the light
    //         * -- surface from |p| towards |outWi| if there are no objects between
    //         * them. So now we need to make sure that hypothesis using shadow ray.
    //         * Given |shadowRay| from LightPdf, ray's |tmax| is narrowed down and
    //         * could lead to further optimization when tracing shadow ray. */
    //        if (!shadow_prd.blocked)
    //        {
    //            Li = TwUtil::Le_QuadLight(sysLightBuffers.quadLightBuffer[lightId], -outWi);
    //            if (!isBlack(Li))
    //                Ld += f * Li * fabsf(dot(outWi, shaderParams.dgShading.nn)) * weight / bsdfPdf;
    //        }
    //    }

    //    return Ld;
    //}
#endif
	
}

/**
 * @brief Uniformly sample all lights available in the scene and
 * compute direct lighting contribution to the surface interaction.
 *
 * @param[in] shaderParams material and differential geometry information
 * @param[in] isectP       intersection point in world space
 * @param[in] isectDir     incoming direction in world space, pointing out of the surface
 * @param[in] localSampler local sampler binded to current launch
 *
 * @see EstimateDirectLighting()
 *
 * @return Return direct lighting contribution due to one specific
 * light.
 */
static __device__ __inline__ float4 SampleLightsAggregate(const CommonStructs::ShaderParams & shaderParams, const float3 & isectP, const float3 & isectDir, GPUSampler &localSampler)
{
	float4 L = make_float4(0.f);

    if (sysLightBuffers.hdriLight.hdriEnvmap != RT_TEXTURE_ID_NULL)
    {
        L += EstimateDirectLighting<CommonStructs::LightType::HDRILight>(-1, shaderParams, isectP, isectDir, localSampler);
    }
    
    if (sysLightBuffers.pointLightBuffer != RT_BUFFER_ID_NULL)
    {
        for (int i = 0; i < sysLightBuffers.pointLightBuffer.size(); ++i)
        {
            L += EstimateDirectLighting<CommonStructs::LightType::PointLight>(i, shaderParams, isectP, isectDir, localSampler);
        }
    }
    
    if (sysLightBuffers.quadLightBuffer != RT_BUFFER_ID_NULL)
    {
        for (int i = 0; i < sysLightBuffers.quadLightBuffer.size(); ++i)
        {
            L += EstimateDirectLighting<CommonStructs::LightType::QuadLight>(i, shaderParams, isectP, isectDir, localSampler);
        }
    }
    
    

	return L;
}

#endif // COLVILLEA_DEVICE_INTEGRATOR_INTEGRATOR_H_