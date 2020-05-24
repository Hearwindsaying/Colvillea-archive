#pragma once
#ifndef COLVILLEA_DEVICE_INTEGRATOR_INTEGRATOR_H_
#define COLVILLEA_DEVICE_INTEGRATOR_INTEGRATOR_H_
/* This file is device only. */
#include <optix_world.h>
#include <optix_device.h>
#include <optixu_math_namespace.h>
#include <time.h>

#include "../Toolkit/CommonStructs.h"
#include "../Toolkit/Utility.h"
//#include "../Toolkit/SH.h"
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

#ifndef TW_RT_DECLARE_AREALIGHTCOEFF
#define TW_RT_DECLARE_AREALIGHTCOEFF
/* Flm Diffuse Matrix. */
rtBuffer<float> areaLightFlmVector;

/* BSDF Matrix -- Plastic BSDF. */
rtBuffer<float, 2> BSDFMatrix;

/* Basis Directions. */
rtBuffer<float3, 1> areaLightBasisVector;

//rtBuffer<float, 2> areaLightAlphaCoeff;
#endif

namespace Cl
{
    static __device__ __host__ __inline__ void swap(optix::float3 & lhs, optix::float3 & rhs)
    {
        optix::float3 tmp = lhs; lhs = rhs; rhs = tmp;
    }

    /************************************************************************/
    /*                              Legendre Polynomial                     */
    /************************************************************************/
    template<int N>
    static __device__ __host__ float LegendreP(float x);

    template<>
    static __device__ __host__ float LegendreP<0>(float x)
    {
        return 1.0f;
    }
    template<>
    static __device__ __host__ float LegendreP<1>(float x)
    {
        return x;
    }
    template<>
    static __device__ __host__ float LegendreP<2>(float x)
    {
        return 0.5f*(3.f*x*x - 1);
    }
    template<>
    static __device__ __host__ float LegendreP<3>(float x)
    {
        return 0.5f*(5.f*x*x*x - 3.f*x);
    }
    template<>
    static __device__ __host__ float LegendreP<4>(float x)
    {
        return 0.125f*(35.f*x*x*x*x - 30.f*x*x + 3);
    }
    template<>
    static __device__ __host__ float LegendreP<5>(float x)
    {
        return 0.125f*(63.f*x*x*x*x*x - 70.f*x*x*x + 15.f*x);
    }
    template<>
    static __device__ __host__ float LegendreP<6>(float x)
    {
        return (231.f*x*x*x*x*x*x - 315.f*x*x*x*x + 105.f*x*x - 5.f) / 16.f;
    }
    template<>
    static __device__ __host__ float LegendreP<7>(float x)
    {
        return (429.f*x*x*x*x*x*x*x - 693.f*x*x*x*x*x + 315.f*x*x*x - 35.f*x) / 16.f;
    }
    template<>
    static __device__ __host__ float LegendreP<8>(float x)
    {
        return (6435.f*pow(x, 8) - 12012.f*pow(x, 6) + 6930.f*x*x*x*x - 1260.f*x*x + 35.f) / 128.f;
    }
    template<>
    static __device__ __host__ float LegendreP<9>(float x)
    {
        return (12155.f*pow(x, 9) - 25740.f*pow(x, 7) + 18018.f*pow(x, 5) - 4620.f*x*x*x + 315.f*x) / 128.f;
    }
    template<>
    static __device__ __host__ float LegendreP<10>(float x)
    {
        return (46189.f*pow(x, 10) - 109395.f*pow(x, 8) + 90090.f*pow(x, 6) - 30030.f*x*x*x*x + 3465.f*x*x - 63.f) / 256.f;
    }

    template<int M>
    static __device__ __inline__ bool CheckOrientation(const optix::float3 P[]);

    template<>
    static __device__ __inline__ bool CheckOrientation<3>(const optix::float3 P[])
    {
        const auto D = (P[1] + P[2] + P[3]) / 3.0f;
        const auto N = optix::cross(P[2] - P[1], P[3] - P[1]);
        return optix::dot(D, N) <= 0.0f;
    }

    /* Clipping Algorithm. */
    static __device__ __inline__ void ClipQuadToHorizon(optix::float3 L[5], int &n)
    {
        /* Make a copy of L[]. */
        optix::float3 Lorg[4]{ L[0],L[1],L[2],L[3] };

        auto IntersectRayZ0 = [](const optix::float3 &A, const optix::float3 &B)->optix::float3
        {
            float t = -A.z * (optix::length(B - A) / (B - A).z);
            return A + t * TwUtil::safe_normalize(B - A);
        };

        n = 0;
        for (int i = 1; i <= 4; ++i)
        {
            const optix::float3& A = Lorg[i - 1];
            const optix::float3& B = i == 4 ? Lorg[0] : Lorg[i]; // Loop back to zero index
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
        }
    }

    /**
     * @brief GPU version compute Solid Angle.
     * @param we spherical projection of polygon, index starting from 1
     */
    template<int M>
    static __device__ __inline__ float computeSolidAngle(const optix::float3 we[])
    {
        float S0 = 0.0f;
        for (int e = 1; e <= M; ++e)
        {
            const optix::float3& we_minus_1 = (e == 1 ? we[M] : we[e - 1]);
            const optix::float3& we_plus_1 = (e == M ? we[1] : we[e + 1]);

            float3 tmpa = optix::cross(we[e], we_minus_1);
            float3 tmpb = optix::cross(we[e], we_plus_1);
            S0 += acosf(optix::clamp(optix::dot(tmpa, tmpb) / (optix::length(tmpa)*optix::length(tmpb)), -1.f, 1.f)); // Typo in Wang's paper, length is inside acos evaluation!
            //float cmp = optix::clamp(optix::dot(tmpa, tmpb) / (optix::length(tmpa)*optix::length(tmpb)), -1.f, 1.f);
            //if (isnan(S0))
            //    rtPrintf("acos(%.7f)=%.7f\n", cmp, acosf(cmp));
        }
        /*if (isnan(S0))
        {
            rtPrintf("%d,%d] M:%d ", sysLaunch_index.x, sysLaunch_index.y, M);
            rtPrintf("we:%f,%f,%f %f,%f,%f %f,%f,%f %f,%f,%f\n", we[1].x, we[1].y, we[1].z,
                we[2].x, we[2].y, we[2].z, we[3].x, we[3].y, we[3].z, we[4].x, we[4].y, we[4].z);
        }*/
        S0 -= (M - 2)*M_PIf;
        return S0;
    }

    /**
     * @brief Convert point in world space into TBN local coordinates without
     * hemisphere projection.
     */
    static __device__ __inline__ optix::float3 BSDFWorldToLocal(const optix::float3 & v, const optix::float3 & sn, const optix::float3 & tn, const optix::float4 & nn, const optix::float3 &worldPoint)
    {
        optix::float3 pt;
        pt.x = optix::dot(v, sn) - optix::dot(worldPoint, sn);
        pt.y = optix::dot(v, tn) - optix::dot(worldPoint, tn);
        pt.z = TwUtil::dot(v, nn) - TwUtil::dot(worldPoint, nn);
        return pt;
    }

    /**
     * @brief Convert point in world space into TBN local coordinates, and
     * project into unit hemisphere.
     */
    static __device__ __inline__ optix::float3 BSDFWorldToLocal_Project(const optix::float3 & v, const optix::float3 & sn, const optix::float3 & tn, const optix::float4 & nn, const optix::float3 &worldPoint)
    {
        optix::float3 pt;
        pt.x = optix::dot(v, sn) - optix::dot(worldPoint, sn);
        pt.y = optix::dot(v, tn) - optix::dot(worldPoint, tn);
        pt.z = TwUtil::dot(v, nn) - TwUtil::dot(worldPoint, nn);
        return TwUtil::safe_normalize(pt);
    }

    /************************************************************************/
    /*   Analytic Spherical Harmonic Coefficients for Polygonal Area Light  */
    /************************************************************************/

    // Unrolling loops by recursive template:
    template<int l, int M>
    static __device__ __inline__ void computeLw_unroll(float Lw[][19], float ae[], float gammae[], float be[], float ce[], float D1e[], float B0e[], float D2e[], float B1e[], float D0e[], int i, float Bl_1, float S0, float S1)
    {
        float Bl = 0;
        float Cl_1e[M + 1];
        float B2_e[M + 1];

        for (int e = 1; e <= M; ++e)
        {
            Cl_1e[e] = 1.f / l * ((ae[e] * sinf(gammae[e]) - be[e] * cosf(gammae[e]))*
                LegendreP<l - 1>(ae[e] * cosf(gammae[e]) + be[e] * sinf(gammae[e])) +
                be[e] * LegendreP<l - 1>(ae[e]) + (ae[e] * ae[e] + be[e] * be[e] - 1.f)*D1e[e] +
                (l - 1.f)*B0e[e]);
            B2_e[e] = ((2.f*l - 1.f) / l)*(Cl_1e[e]) - (l - 1.f) / l * B0e[e];
            Bl = Bl + ce[e] * B2_e[e];
            D2e[e] = (2.f*l - 1.f) * B1e[e] + D0e[e];

            D0e[e] = D1e[e];
            D1e[e] = D2e[e];
            B0e[e] = B1e[e];
            B1e[e] = B2_e[e];
        }

        // Optimal storage for S (recurrence relation so that only three terms are kept).
        // S2 is not represented as S2 really (Sl).
        if (l % 2 == 0)
        {
            float S2 = ((2.f*l - 1) / (l*(l + 1))*Bl_1) + ((l - 2.f)*(l - 1.f) / ((l)*(l + 1.f)))*S0;
            S0 = S2;
            Lw[l][i] = sqrtf((2.f * l + 1) / (4.f*M_PIf))*S2;
        }
        else
        {
            float S2 = ((2.f*l - 1) / (l*(l + 1))*Bl_1) + ((l - 2.f)*(l - 1.f) / ((l)*(l + 1.f)))*S1;
            S1 = S2;
            Lw[l][i] = sqrtf((2.f * l + 1) / (4.f*M_PIf))*S2;
        }

        Bl_1 = Bl;

        computeLw_unroll<l + 1, M>(Lw, ae, gammae, be, ce, D1e, B0e, D2e, B1e, D0e, i, Bl_1, S0, S1);
    }

    // Partial specialization is not supported in function templates:
    template<>
    static __device__ __inline__ void computeLw_unroll<9, 5>(float Lw[][19], float ae[], float gammae[], float be[], float ce[], float D1e[], float B0e[], float D2e[], float B1e[], float D0e[], int i, float Bl_1, float S0, float S1)
    {
        float Bl = 0;
        float Cl_1e[5 + 1];
        float B2_e[5 + 1];

        for (int e = 1; e <= 5; ++e)
        {
            Cl_1e[e] = 1.f / 9 * ((ae[e] * sinf(gammae[e]) - be[e] * cosf(gammae[e]))*
                LegendreP<9 - 1>(ae[e] * cosf(gammae[e]) + be[e] * sinf(gammae[e])) +
                be[e] * LegendreP<9 - 1>(ae[e]) + (ae[e] * ae[e] + be[e] * be[e] - 1.f)*D1e[e] +
                (9 - 1.f)*B0e[e]);
            B2_e[e] = ((2.f * 9 - 1.f) / 9)*(Cl_1e[e]) - (9 - 1.f) / 9 * B0e[e];
            Bl = Bl + ce[e] * B2_e[e];
            D2e[e] = (2.f * 9 - 1.f) * B1e[e] + D0e[e];

            D0e[e] = D1e[e];
            D1e[e] = D2e[e];
            B0e[e] = B1e[e];
            B1e[e] = B2_e[e];
        }

        // Optimal storage for S (recurrence relation so that only three terms are kept).
        // S2 is not represented as S2 really (Sl).
        float S2 = ((2.f * 9 - 1) / (9 * (9 + 1))*Bl_1) + ((9 - 2.f)*(9 - 1.f) / ((9)*(9 + 1.f)))*S1;
        S1 = S2;
        Lw[9][i] = sqrtf((2.f * 9 + 1) / (4.f*M_PIf))*S2;

        Bl_1 = Bl;
    }

    template<>
    static __device__ __inline__ void computeLw_unroll<9, 4>(float Lw[][19], float ae[], float gammae[], float be[], float ce[], float D1e[], float B0e[], float D2e[], float B1e[], float D0e[], int i, float Bl_1, float S0, float S1)
    {
        float Bl = 0;
        float Cl_1e[4 + 1];
        float B2_e[4 + 1];

        for (int e = 1; e <= 4; ++e)
        {
            Cl_1e[e] = 1.f / 9 * ((ae[e] * sinf(gammae[e]) - be[e] * cosf(gammae[e]))*
                LegendreP<9 - 1>(ae[e] * cosf(gammae[e]) + be[e] * sinf(gammae[e])) +
                be[e] * LegendreP<9 - 1>(ae[e]) + (ae[e] * ae[e] + be[e] * be[e] - 1.f)*D1e[e] +
                (9 - 1.f)*B0e[e]);
            B2_e[e] = ((2.f * 9 - 1.f) / 9)*(Cl_1e[e]) - (9 - 1.f) / 9 * B0e[e];
            Bl = Bl + ce[e] * B2_e[e];
            D2e[e] = (2.f * 9 - 1.f) * B1e[e] + D0e[e];

            D0e[e] = D1e[e];
            D1e[e] = D2e[e];
            B0e[e] = B1e[e];
            B1e[e] = B2_e[e];
        }

        // Optimal storage for S (recurrence relation so that only three terms are kept).
        // S2 is not represented as S2 really (Sl).
        float S2 = ((2.f * 9 - 1) / (9 * (9 + 1))*Bl_1) + ((9 - 2.f)*(9 - 1.f) / ((9)*(9 + 1.f)))*S1;
        S1 = S2;
        Lw[9][i] = sqrtf((2.f * 9 + 1) / (4.f*M_PIf))*S2;

        Bl_1 = Bl;
    }

    template<>
    static __device__ __inline__ void computeLw_unroll<9, 3>(float Lw[][19], float ae[], float gammae[], float be[], float ce[], float D1e[], float B0e[], float D2e[], float B1e[], float D0e[], int i, float Bl_1, float S0, float S1)
    {
        float Bl = 0;
        float Cl_1e[3 + 1];
        float B2_e[3 + 1];

        for (int e = 1; e <= 3; ++e)
        {
            Cl_1e[e] = 1.f / 9 * ((ae[e] * sinf(gammae[e]) - be[e] * cosf(gammae[e]))*
                LegendreP<9 - 1>(ae[e] * cosf(gammae[e]) + be[e] * sinf(gammae[e])) +
                be[e] * LegendreP<9 - 1>(ae[e]) + (ae[e] * ae[e] + be[e] * be[e] - 1.f)*D1e[e] +
                (9 - 1.f)*B0e[e]);
            B2_e[e] = ((2.f * 9 - 1.f) / 9)*(Cl_1e[e]) - (9 - 1.f) / 9 * B0e[e];
            Bl = Bl + ce[e] * B2_e[e];
            D2e[e] = (2.f * 9 - 1.f) * B1e[e] + D0e[e];

            D0e[e] = D1e[e];
            D1e[e] = D2e[e];
            B0e[e] = B1e[e];
            B1e[e] = B2_e[e];
        }

        // Optimal storage for S (recurrence relation so that only three terms are kept).
        // S2 is not represented as S2 really (Sl).
        float S2 = ((2.f * 9 - 1) / (9 * (9 + 1))*Bl_1) + ((9 - 2.f)*(9 - 1.f) / ((9)*(9 + 1.f)))*S1;
        S1 = S2;
        Lw[9][i] = sqrtf((2.f * 9 + 1) / (4.f*M_PIf))*S2;

        Bl_1 = Bl;
    }



    /**
     * @brief GPU Version computeCoeff
     * @param[in] we Area Light vertices in local shading space.
     *  Note that this parameter will be modified and should
     *  not be used after calling this function. The vertices
     *  indices starts from 1, to M.
     *  They must be normalized (i.e. projected on unit hemisphere)
     * @note Since it's assumed that |we| are in local shading space.
     *  The original |x| parameter is set by default to (0,0,0)
     */
    template<int M, int lmax>
    static __device__ __inline__ void computeCoeff(optix::float3 we[], float ylmCoeff[(lmax + 1)*(lmax + 1)])
    {
#ifdef __CUDACC__
#undef TW_ASSERT
#define TW_ASSERT(expr) TW_ASSERT_INFO(expr, ##expr)
#define TW_ASSERT_INFO(expr, str)    if (!(expr)) {rtPrintf(str); rtPrintf("Above at Line%d:\n",__LINE__);}
#endif
        //TW_ASSERT(v.size() == M + 1);
        //TW_ASSERT(n == 2);
        float3 lambdae[M + 1];
        float3 ue[M + 1];
        float gammae[M + 1];
        for (int e = 1; e <= M; ++e)
        {
            // Incorrect modular arthmetic: we[(e + 1) % (M+1)] or we[(e + 1) % (M)]
            const optix::float3& we_plus_1 = (e == M ? we[1] : we[e + 1]);
            lambdae[e] = optix::cross(TwUtil::safe_normalize(cross(we[e], we_plus_1)), we[e]);
            ue[e] = optix::cross(we[e], lambdae[e]);
            gammae[e] = acosf(optix::dot(we[e], we_plus_1));
        }
        // Solid angle computation
        float solidAngle = computeSolidAngle<M>(we);

        float Lw[lmax + 1][2 * lmax + 1];

        for (int i = 0; i < 2 * lmax + 1; ++i)
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
            float S0 = solidAngle;
            float S1 = 0;
            for (int e = 1; e <= M; ++e)
            {
                ae[e] = optix::dot(wi, we[e]); be[e] = optix::dot(wi, lambdae[e]); ce[e] = optix::dot(wi, ue[e]);
                S1 += 0.5f*ce[e] * gammae[e];

                B0e[e] = gammae[e];
                B1e[e] = ae[e] * sinf(gammae[e]) - be[e] * cosf(gammae[e]) + be[e];
                D0e[e] = 0; D1e[e] = gammae[e]; D2e[e] = 3 * B1e[e];
            }

            // my code for B1
            float Bl_1 = 0.f;
            for (int e = 1; e <= M; ++e)
            {
                Bl_1 += ce[e] * B1e[e];
            }

            // Initial Bands l=0, l=1:
            Lw[0][i] = sqrtf(1.f / (4.f*M_PIf))*S0;
            Lw[1][i] = sqrtf(3.f / (4.f*M_PIf))*S1;

            if (isnan(Lw[0][i]) || isnan(Lw[1][i]))
                rtPrintf("%d,%d]%f %f Lw[0/1][%d]\n", sysLaunch_index.x, sysLaunch_index.y, S0, S1, i);

            computeLw_unroll<2, M>(Lw, ae, gammae, be, ce, D1e, B0e, D2e, B1e, D0e, i, Bl_1, S0, S1);
        }

#pragma region REGION_YLMCoeff_MUL
        ylmCoeff[0] = 1.000000f*Lw[0][0];
        ylmCoeff[1] = 0.684451f*Lw[1][0] + 0.359206f*Lw[1][1] + -1.290530f*Lw[1][2];
        ylmCoeff[2] = 0.491764f*Lw[1][0] + 0.430504f*Lw[1][1] + 0.232871f*Lw[1][2];
        ylmCoeff[3] = -1.668360f*Lw[1][0] + 1.273940f*Lw[1][1] + 0.689107f*Lw[1][2];
        ylmCoeff[4] = 0.330376f*Lw[2][0] + 0.163767f*Lw[2][1] + -0.973178f*Lw[2][2] + -1.585730f*Lw[2][3] + -1.065320f*Lw[2][4];
        ylmCoeff[5] = -0.154277f*Lw[2][0] + 0.466127f*Lw[2][1] + -0.597988f*Lw[2][2] + 0.289095f*Lw[2][3] + -0.324048f*Lw[2][4];
        ylmCoeff[6] = 0.651874f*Lw[2][0] + 0.531830f*Lw[2][1] + 0.467255f*Lw[2][2] + 0.225622f*Lw[2][3] + 0.243216f*Lw[2][4];
        ylmCoeff[7] = -0.675183f*Lw[2][0] + 0.736365f*Lw[2][1] + 0.558372f*Lw[2][2] + 0.231532f*Lw[2][3] + 0.497581f*Lw[2][4];
        ylmCoeff[8] = 1.055130f*Lw[2][0] + -0.712438f*Lw[2][1] + -1.150210f*Lw[2][2] + -0.780808f*Lw[2][3] + 0.626025f*Lw[2][4];
        ylmCoeff[9] = -3.596720f*Lw[3][0] + -4.001470f*Lw[3][1] + -1.962170f*Lw[3][2] + -1.387070f*Lw[3][3] + -4.629640f*Lw[3][4] + 1.838530f*Lw[3][5] + -4.196860f*Lw[3][6];
        ylmCoeff[10] = -1.026310f*Lw[3][0] + -1.141480f*Lw[3][1] + -0.328964f*Lw[3][2] + -0.581013f*Lw[3][3] + -2.094670f*Lw[3][4] + 1.848280f*Lw[3][5] + -1.768730f*Lw[3][6];
        ylmCoeff[11] = -0.276614f*Lw[3][0] + 0.125182f*Lw[3][1] + -0.744750f*Lw[3][2] + 0.647966f*Lw[3][3] + -0.943961f*Lw[3][4] + 0.578956f*Lw[3][5] + -0.379926f*Lw[3][6];
        ylmCoeff[12] = 2.815200f*Lw[3][0] + 2.999510f*Lw[3][1] + 1.559050f*Lw[3][2] + 0.530353f*Lw[3][3] + 2.945050f*Lw[3][4] + -1.805750f*Lw[3][5] + 2.259590f*Lw[3][6];
        ylmCoeff[13] = 0.933848f*Lw[3][0] + 2.366930f*Lw[3][1] + 1.065640f*Lw[3][2] + 0.138454f*Lw[3][3] + 2.523470f*Lw[3][4] + -1.634230f*Lw[3][5] + 1.773880f*Lw[3][6];
        ylmCoeff[14] = -1.531620f*Lw[3][0] + -2.266990f*Lw[3][1] + -1.639440f*Lw[3][2] + -1.279980f*Lw[3][3] + -1.499230f*Lw[3][4] + 0.802996f*Lw[3][5] + -1.877340f*Lw[3][6];
        ylmCoeff[15] = 2.203690f*Lw[3][0] + 2.657180f*Lw[3][1] + 2.377300f*Lw[3][2] + 1.478660f*Lw[3][3] + 2.617180f*Lw[3][4] + -0.154934f*Lw[3][5] + 1.014160f*Lw[3][6];
        ylmCoeff[16] = 0.386892f*Lw[4][0] + -0.234194f*Lw[4][1] + 0.425573f*Lw[4][2] + -2.075420f*Lw[4][3] + -0.998780f*Lw[4][4] + 0.087789f*Lw[4][5] + -1.490880f*Lw[4][6] + -2.101130f*Lw[4][7] + -2.217670f*Lw[4][8];
        ylmCoeff[17] = 0.670532f*Lw[4][0] + 1.115250f*Lw[4][1] + 0.021381f*Lw[4][2] + 0.213706f*Lw[4][3] + -0.775345f*Lw[4][4] + -0.870597f*Lw[4][5] + -0.243734f*Lw[4][6] + -1.073310f*Lw[4][7] + -0.921999f*Lw[4][8];
        ylmCoeff[18] = 3.343630f*Lw[4][0] + 2.883330f*Lw[4][1] + 1.552800f*Lw[4][2] + -0.690988f*Lw[4][3] + -1.211960f*Lw[4][4] + 1.384430f*Lw[4][5] + 0.248891f*Lw[4][6] + -3.335990f*Lw[4][7] + -2.187420f*Lw[4][8];
        ylmCoeff[19] = -4.435760f*Lw[4][0] + -3.106790f*Lw[4][1] + -1.507320f*Lw[4][2] + 1.143550f*Lw[4][3] + 3.059760f*Lw[4][4] + -1.321760f*Lw[4][5] + -0.279068f*Lw[4][6] + 6.365810f*Lw[4][7] + 4.377290f*Lw[4][8];
        ylmCoeff[20] = 0.801925f*Lw[4][0] + -0.589618f*Lw[4][1] + -0.498002f*Lw[4][2] + -1.990030f*Lw[4][3] + -1.171520f*Lw[4][4] + 0.022533f*Lw[4][5] + -1.832880f*Lw[4][6] + -2.482010f*Lw[4][7] + -0.800297f*Lw[4][8];
        ylmCoeff[21] = -0.074162f*Lw[4][0] + 0.463091f*Lw[4][1] + -0.490407f*Lw[4][2] + -1.337940f*Lw[4][3] + -1.193020f*Lw[4][4] + -0.067106f*Lw[4][5] + -0.991982f*Lw[4][6] + -2.579000f*Lw[4][7] + -1.403030f*Lw[4][8];
        ylmCoeff[22] = 2.308980f*Lw[4][0] + 2.595050f*Lw[4][1] + -0.245139f*Lw[4][2] + -0.143466f*Lw[4][3] + -1.280470f*Lw[4][4] + 0.022586f*Lw[4][5] + 0.402924f*Lw[4][6] + -3.233010f*Lw[4][7] + -2.686130f*Lw[4][8];
        ylmCoeff[23] = 2.452170f*Lw[4][0] + 1.307120f*Lw[4][1] + 0.624695f*Lw[4][2] + -1.204530f*Lw[4][3] + -0.992719f*Lw[4][4] + 0.880800f*Lw[4][5] + -1.429300f*Lw[4][6] + -4.161490f*Lw[4][7] + -2.098860f*Lw[4][8];
        ylmCoeff[24] = -3.295530f*Lw[4][0] + -2.754510f*Lw[4][1] + -0.707257f*Lw[4][2] + -3.383790f*Lw[4][3] + 1.159480f*Lw[4][4] + -1.854320f*Lw[4][5] + -2.294330f*Lw[4][6] + 0.282948f*Lw[4][7] + -0.104368f*Lw[4][8];
        ylmCoeff[25] = 0.901645f*Lw[5][0] + 0.385552f*Lw[5][1] + 0.881322f*Lw[5][2] + 0.762582f*Lw[5][3] + 0.062735f*Lw[5][4] + 0.500188f*Lw[5][5] + 0.815467f*Lw[5][6] + 0.016950f*Lw[5][7] + 0.291480f*Lw[5][8] + 1.154980f*Lw[5][9] + 0.629604f*Lw[5][10];
        ylmCoeff[26] = 1.487350f*Lw[5][0] + 0.327641f*Lw[5][1] + 1.036510f*Lw[5][2] + 0.605442f*Lw[5][3] + 1.299000f*Lw[5][4] + 1.289780f*Lw[5][5] + 0.011824f*Lw[5][6] + 0.774944f*Lw[5][7] + -1.535470f*Lw[5][8] + 1.585780f*Lw[5][9] + 0.857565f*Lw[5][10];
        ylmCoeff[27] = -0.139832f*Lw[5][0] + -0.767387f*Lw[5][1] + 0.690406f*Lw[5][2] + -0.647648f*Lw[5][3] + -1.737580f*Lw[5][4] + -0.953175f*Lw[5][5] + -0.415786f*Lw[5][6] + 0.357295f*Lw[5][7] + 0.342909f*Lw[5][8] + -0.860505f*Lw[5][9] + -1.003170f*Lw[5][10];
        ylmCoeff[28] = -1.302540f*Lw[5][0] + -0.029951f*Lw[5][1] + -0.923929f*Lw[5][2] + -1.091530f*Lw[5][3] + -0.484701f*Lw[5][4] + -1.140900f*Lw[5][5] + -1.012180f*Lw[5][6] + 0.732852f*Lw[5][7] + 0.567873f*Lw[5][8] + -1.397640f*Lw[5][9] + -0.588140f*Lw[5][10];
        ylmCoeff[29] = 1.341500f*Lw[5][0] + 0.479091f*Lw[5][1] + 0.816822f*Lw[5][2] + 0.087571f*Lw[5][3] + 0.305267f*Lw[5][4] + 0.492711f*Lw[5][5] + -0.382670f*Lw[5][6] + -0.252676f*Lw[5][7] + -0.294921f*Lw[5][8] + 1.502570f*Lw[5][9] + -0.944971f*Lw[5][10];
        ylmCoeff[30] = -3.185450f*Lw[5][0] + -3.582450f*Lw[5][1] + -1.570540f*Lw[5][2] + -3.915110f*Lw[5][3] + -4.135760f*Lw[5][4] + -3.408710f*Lw[5][5] + -3.036850f*Lw[5][6] + 1.475860f*Lw[5][7] + 1.643680f*Lw[5][8] + -3.679640f*Lw[5][9] + -3.480770f*Lw[5][10];
        ylmCoeff[31] = -1.497580f*Lw[5][0] + -0.795641f*Lw[5][1] + -0.477492f*Lw[5][2] + -0.912100f*Lw[5][3] + -0.961176f*Lw[5][4] + -0.978628f*Lw[5][5] + -0.587473f*Lw[5][6] + 0.514521f*Lw[5][7] + -0.103120f*Lw[5][8] + -0.437121f*Lw[5][9] + -0.999840f*Lw[5][10];
        ylmCoeff[32] = 2.041990f*Lw[5][0] + 1.202230f*Lw[5][1] + 0.339812f*Lw[5][2] + 1.550510f*Lw[5][3] + 1.458890f*Lw[5][4] + 0.618371f*Lw[5][5] + 1.082610f*Lw[5][6] + -0.076546f*Lw[5][7] + -1.467500f*Lw[5][8] + 2.220750f*Lw[5][9] + 0.536480f*Lw[5][10];
        ylmCoeff[33] = 0.045375f*Lw[5][0] + -0.721773f*Lw[5][1] + 0.127358f*Lw[5][2] + 0.344248f*Lw[5][3] + 0.022802f*Lw[5][4] + -0.923741f*Lw[5][5] + -0.898898f*Lw[5][6] + 0.594424f*Lw[5][7] + 0.021107f*Lw[5][8] + 0.407756f*Lw[5][9] + -1.210180f*Lw[5][10];
        ylmCoeff[34] = 0.369802f*Lw[5][0] + -0.429225f*Lw[5][1] + 0.962211f*Lw[5][2] + -0.428983f*Lw[5][3] + 1.227360f*Lw[5][4] + 0.047371f*Lw[5][5] + 0.177308f*Lw[5][6] + 0.884197f*Lw[5][7] + -1.569720f*Lw[5][8] + 1.217300f*Lw[5][9] + 0.321895f*Lw[5][10];
        ylmCoeff[35] = 0.838529f*Lw[5][0] + 1.885890f*Lw[5][1] + 0.571345f*Lw[5][2] + 1.309700f*Lw[5][3] + 1.892460f*Lw[5][4] + 2.360370f*Lw[5][5] + 2.186340f*Lw[5][6] + -0.369590f*Lw[5][7] + -0.305823f*Lw[5][8] + 0.624519f*Lw[5][9] + 1.948200f*Lw[5][10];
        ylmCoeff[36] = 0.130954f*Lw[6][0] + -0.516826f*Lw[6][1] + -0.039086f*Lw[6][2] + -0.055591f*Lw[6][3] + -0.483086f*Lw[6][4] + 0.549499f*Lw[6][5] + -0.425825f*Lw[6][6] + 0.274285f*Lw[6][7] + -0.624874f*Lw[6][8] + 0.704007f*Lw[6][9] + 0.687130f*Lw[6][10] + -0.507504f*Lw[6][11] + 0.163940f*Lw[6][12];
        ylmCoeff[37] = -0.319237f*Lw[6][0] + 0.457155f*Lw[6][1] + 0.050386f*Lw[6][2] + 0.037450f*Lw[6][3] + -0.090040f*Lw[6][4] + -0.060044f*Lw[6][5] + -0.062161f*Lw[6][6] + 1.739800f*Lw[6][7] + 0.379183f*Lw[6][8] + -0.333790f*Lw[6][9] + -0.700205f*Lw[6][10] + -1.530560f*Lw[6][11] + 0.827214f*Lw[6][12];
        ylmCoeff[38] = -1.425090f*Lw[6][0] + 0.341737f*Lw[6][1] + -2.293560f*Lw[6][2] + 0.486814f*Lw[6][3] + 3.322270f*Lw[6][4] + 0.521771f*Lw[6][5] + 0.226620f*Lw[6][6] + 2.093830f*Lw[6][7] + 3.627480f*Lw[6][8] + 1.297470f*Lw[6][9] + 0.113476f*Lw[6][10] + -4.241230f*Lw[6][11] + -1.573040f*Lw[6][12];
        ylmCoeff[39] = 0.110626f*Lw[6][0] + 0.570583f*Lw[6][1] + 0.681116f*Lw[6][2] + 0.393754f*Lw[6][3] + 0.076449f*Lw[6][4] + 0.477050f*Lw[6][5] + 0.031733f*Lw[6][6] + 1.011070f*Lw[6][7] + 0.132282f*Lw[6][8] + -0.207397f*Lw[6][9] + -0.607639f*Lw[6][10] + -0.912909f*Lw[6][11] + -0.027689f*Lw[6][12];
        ylmCoeff[40] = -0.871339f*Lw[6][0] + 0.953107f*Lw[6][1] + -1.235880f*Lw[6][2] + 0.951312f*Lw[6][3] + 2.710710f*Lw[6][4] + -0.676999f*Lw[6][5] + 0.417402f*Lw[6][6] + 1.642490f*Lw[6][7] + 2.111420f*Lw[6][8] + 0.667482f*Lw[6][9] + -0.644610f*Lw[6][10] + -2.838090f*Lw[6][11] + -0.224166f*Lw[6][12];
        ylmCoeff[41] = 1.028620f*Lw[6][0] + -0.207223f*Lw[6][1] + 1.932750f*Lw[6][2] + -0.537461f*Lw[6][3] + -2.939690f*Lw[6][4] + -1.082590f*Lw[6][5] + 0.746330f*Lw[6][6] + -3.075930f*Lw[6][7] + -2.643970f*Lw[6][8] + -2.025530f*Lw[6][9] + 0.324457f*Lw[6][10] + 3.922950f*Lw[6][11] + 1.436580f*Lw[6][12];
        ylmCoeff[42] = -0.596174f*Lw[6][0] + 0.139976f*Lw[6][1] + -1.070570f*Lw[6][2] + 0.925160f*Lw[6][3] + 2.082830f*Lw[6][4] + -0.639657f*Lw[6][5] + 1.001710f*Lw[6][6] + 0.956261f*Lw[6][7] + 1.184230f*Lw[6][8] + -0.420206f*Lw[6][9] + -1.385420f*Lw[6][10] + -2.300770f*Lw[6][11] + -0.244787f*Lw[6][12];
        ylmCoeff[43] = -2.213810f*Lw[6][0] + 0.563199f*Lw[6][1] + -2.755590f*Lw[6][2] + -0.108763f*Lw[6][3] + 3.378680f*Lw[6][4] + -0.351411f*Lw[6][5] + 1.516450f*Lw[6][6] + 0.932440f*Lw[6][7] + 3.012570f*Lw[6][8] + 1.269950f*Lw[6][9] + -0.682257f*Lw[6][10] + -2.697550f*Lw[6][11] + -2.341320f*Lw[6][12];
        ylmCoeff[44] = -0.803653f*Lw[6][0] + 0.486332f*Lw[6][1] + -2.715710f*Lw[6][2] + 0.004861f*Lw[6][3] + 3.268600f*Lw[6][4] + -0.556502f*Lw[6][5] + 1.782420f*Lw[6][6] + 0.779195f*Lw[6][7] + 3.274040f*Lw[6][8] + 1.490380f*Lw[6][9] + -0.610542f*Lw[6][10] + -2.357780f*Lw[6][11] + -2.574510f*Lw[6][12];
        ylmCoeff[45] = 0.847329f*Lw[6][0] + -0.834072f*Lw[6][1] + 2.379160f*Lw[6][2] + 1.241460f*Lw[6][3] + -1.984730f*Lw[6][4] + -0.520790f*Lw[6][5] + -0.488590f*Lw[6][6] + -0.115073f*Lw[6][7] + -1.321640f*Lw[6][8] + -0.886867f*Lw[6][9] + -0.080431f*Lw[6][10] + 1.377650f*Lw[6][11] + 1.797650f*Lw[6][12];
        ylmCoeff[46] = -2.447870f*Lw[6][0] + 0.397090f*Lw[6][1] + -3.776100f*Lw[6][2] + 0.015054f*Lw[6][3] + 6.599580f*Lw[6][4] + 0.329229f*Lw[6][5] + 0.798877f*Lw[6][6] + 3.760580f*Lw[6][7] + 5.891720f*Lw[6][8] + 3.025550f*Lw[6][9] + -0.532234f*Lw[6][10] + -6.770260f*Lw[6][11] + -3.276290f*Lw[6][12];
        ylmCoeff[47] = -0.658281f*Lw[6][0] + 0.529111f*Lw[6][1] + -1.759750f*Lw[6][2] + -0.618552f*Lw[6][3] + 1.902190f*Lw[6][4] + 0.644707f*Lw[6][5] + 0.912778f*Lw[6][6] + 0.531902f*Lw[6][7] + 1.535140f*Lw[6][8] + 1.077950f*Lw[6][9] + 0.382837f*Lw[6][10] + -0.831314f*Lw[6][11] + -1.177790f*Lw[6][12];
        ylmCoeff[48] = 0.902540f*Lw[6][0] + -0.001958f*Lw[6][1] + 0.416233f*Lw[6][2] + -0.534067f*Lw[6][3] + -1.338260f*Lw[6][4] + -0.406687f*Lw[6][5] + 0.157748f*Lw[6][6] + -0.934546f*Lw[6][7] + -1.606370f*Lw[6][8] + -1.041870f*Lw[6][9] + 0.646452f*Lw[6][10] + 1.979130f*Lw[6][11] + 0.095826f*Lw[6][12];
        ylmCoeff[49] = 1.411380f*Lw[7][0] + -1.493180f*Lw[7][1] + -0.267813f*Lw[7][2] + 0.118793f*Lw[7][3] + 0.066779f*Lw[7][4] + 1.111400f*Lw[7][5] + 1.218320f*Lw[7][6] + -0.364957f*Lw[7][7] + 0.996974f*Lw[7][8] + -1.378570f*Lw[7][9] + -0.590952f*Lw[7][10] + -0.990955f*Lw[7][11] + 1.445450f*Lw[7][12] + 0.300644f*Lw[7][13] + 0.521915f*Lw[7][14];
        ylmCoeff[50] = -0.980688f*Lw[7][0] + 0.364658f*Lw[7][1] + 0.617238f*Lw[7][2] + 0.970392f*Lw[7][3] + -0.845702f*Lw[7][4] + 0.328110f*Lw[7][5] + -0.286463f*Lw[7][6] + 0.866263f*Lw[7][7] + -0.592107f*Lw[7][8] + 0.645209f*Lw[7][9] + -0.224906f*Lw[7][10] + 0.547207f*Lw[7][11] + -0.936674f*Lw[7][12] + -0.788088f*Lw[7][13] + -0.536917f*Lw[7][14];
        ylmCoeff[51] = -0.168275f*Lw[7][0] + 0.536730f*Lw[7][1] + -0.365787f*Lw[7][2] + -1.259150f*Lw[7][3] + 0.010743f*Lw[7][4] + -0.413228f*Lw[7][5] + -0.032075f*Lw[7][6] + -0.366879f*Lw[7][7] + -0.353566f*Lw[7][8] + 0.036637f*Lw[7][9] + 0.302125f*Lw[7][10] + 0.738732f*Lw[7][11] + 0.353260f*Lw[7][12] + 0.052342f*Lw[7][13] + -0.221827f*Lw[7][14];
        ylmCoeff[52] = -0.584207f*Lw[7][0] + 0.430717f*Lw[7][1] + -0.130176f*Lw[7][2] + -0.274328f*Lw[7][3] + 0.382646f*Lw[7][4] + -0.992711f*Lw[7][5] + 0.096174f*Lw[7][6] + -0.261535f*Lw[7][7] + 0.094654f*Lw[7][8] + 0.772936f*Lw[7][9] + -0.148429f*Lw[7][10] + 0.808034f*Lw[7][11] + -0.989550f*Lw[7][12] + 0.367983f*Lw[7][13] + -0.497198f*Lw[7][14];
        ylmCoeff[53] = -0.655831f*Lw[7][0] + 0.734315f*Lw[7][1] + 0.474604f*Lw[7][2] + -0.242935f*Lw[7][3] + -0.174109f*Lw[7][4] + 0.226868f*Lw[7][5] + 0.216102f*Lw[7][6] + 0.234184f*Lw[7][7] + 0.075835f*Lw[7][8] + 0.312709f*Lw[7][9] + 0.304648f*Lw[7][10] + 0.691801f*Lw[7][11] + 0.132165f*Lw[7][12] + 0.248373f*Lw[7][13] + -0.771094f*Lw[7][14];
        ylmCoeff[54] = 0.079589f*Lw[7][0] + 1.073380f*Lw[7][1] + -0.292855f*Lw[7][2] + -1.023800f*Lw[7][3] + 0.581984f*Lw[7][4] + -0.873444f*Lw[7][5] + -0.632578f*Lw[7][6] + -0.599404f*Lw[7][7] + -0.774384f*Lw[7][8] + 0.293745f*Lw[7][9] + 0.164963f*Lw[7][10] + 0.878368f*Lw[7][11] + 0.574305f*Lw[7][12] + 0.093858f*Lw[7][13] + 1.008160f*Lw[7][14];
        ylmCoeff[55] = 2.662020f*Lw[7][0] + -2.021760f*Lw[7][1] + 0.195195f*Lw[7][2] + 0.551417f*Lw[7][3] + 0.618997f*Lw[7][4] + 1.443040f*Lw[7][5] + 2.920240f*Lw[7][6] + -0.450233f*Lw[7][7] + 1.023990f*Lw[7][8] + -3.372950f*Lw[7][9] + -0.106694f*Lw[7][10] + -1.960110f*Lw[7][11] + 1.643950f*Lw[7][12] + 0.940143f*Lw[7][13] + 0.462851f*Lw[7][14];
        ylmCoeff[56] = -0.105968f*Lw[7][0] + -1.252840f*Lw[7][1] + 0.864732f*Lw[7][2] + 2.029850f*Lw[7][3] + -0.311623f*Lw[7][4] + 1.517140f*Lw[7][5] + 0.530248f*Lw[7][6] + -0.186852f*Lw[7][7] + -0.190595f*Lw[7][8] + -1.465310f*Lw[7][9] + -0.509711f*Lw[7][10] + -0.848307f*Lw[7][11] + -0.040913f*Lw[7][12] + 0.517662f*Lw[7][13] + -1.192580f*Lw[7][14];
        ylmCoeff[57] = -1.871560f*Lw[7][0] + 1.575850f*Lw[7][1] + 0.171384f*Lw[7][2] + -1.072350f*Lw[7][3] + 0.079502f*Lw[7][4] + -1.881090f*Lw[7][5] + -1.779110f*Lw[7][6] + -0.466125f*Lw[7][7] + -0.225306f*Lw[7][8] + 2.061200f*Lw[7][9] + 0.746487f*Lw[7][10] + 1.152750f*Lw[7][11] + -0.836341f*Lw[7][12] + -1.038500f*Lw[7][13] + -0.058806f*Lw[7][14];
        ylmCoeff[58] = -0.231926f*Lw[7][0] + 1.377850f*Lw[7][1] + -0.192581f*Lw[7][2] + -1.369780f*Lw[7][3] + -0.125444f*Lw[7][4] + -1.938950f*Lw[7][5] + -1.586260f*Lw[7][6] + -0.522810f*Lw[7][7] + -0.007738f*Lw[7][8] + 1.946190f*Lw[7][9] + 1.140060f*Lw[7][10] + 1.364070f*Lw[7][11] + -0.205571f*Lw[7][12] + -0.710586f*Lw[7][13] + 0.220972f*Lw[7][14];
        ylmCoeff[59] = 0.336550f*Lw[7][0] + -0.574124f*Lw[7][1] + -0.732785f*Lw[7][2] + -0.764633f*Lw[7][3] + -0.384849f*Lw[7][4] + -0.013514f*Lw[7][5] + 0.504584f*Lw[7][6] + 0.096723f*Lw[7][7] + 0.278052f*Lw[7][8] + -0.246882f*Lw[7][9] + 0.535610f*Lw[7][10] + 0.588689f*Lw[7][11] + 1.367470f*Lw[7][12] + 0.946260f*Lw[7][13] + 0.718744f*Lw[7][14];
        ylmCoeff[60] = -1.477140f*Lw[7][0] + 1.416470f*Lw[7][1] + 0.480085f*Lw[7][2] + -1.613080f*Lw[7][3] + 0.495642f*Lw[7][4] + -1.874180f*Lw[7][5] + -1.985030f*Lw[7][6] + 0.025550f*Lw[7][7] + -1.036770f*Lw[7][8] + 2.632400f*Lw[7][9] + -0.074327f*Lw[7][10] + 1.930400f*Lw[7][11] + -1.196710f*Lw[7][12] + -0.655958f*Lw[7][13] + 0.104490f*Lw[7][14];
        ylmCoeff[61] = 0.804917f*Lw[7][0] + 0.149715f*Lw[7][1] + -1.295800f*Lw[7][2] + -1.761300f*Lw[7][3] + 1.150100f*Lw[7][4] + -0.565730f*Lw[7][5] + 0.344090f*Lw[7][6] + -0.149350f*Lw[7][7] + 0.177333f*Lw[7][8] + 0.810151f*Lw[7][9] + 0.991728f*Lw[7][10] + 0.996871f*Lw[7][11] + 0.634889f*Lw[7][12] + -0.423213f*Lw[7][13] + 0.898464f*Lw[7][14];
        ylmCoeff[62] = 1.266660f*Lw[7][0] + -0.647383f*Lw[7][1] + -0.706160f*Lw[7][2] + -0.628073f*Lw[7][3] + 0.550705f*Lw[7][4] + -0.287921f*Lw[7][5] + 1.012860f*Lw[7][6] + 0.604584f*Lw[7][7] + 0.565855f*Lw[7][8] + -0.582630f*Lw[7][9] + 0.007751f*Lw[7][10] + 0.532163f*Lw[7][11] + 1.492010f*Lw[7][12] + 0.565321f*Lw[7][13] + 0.325189f*Lw[7][14];
        ylmCoeff[63] = 1.117940f*Lw[7][0] + -1.131550f*Lw[7][1] + -0.282903f*Lw[7][2] + 0.617965f*Lw[7][3] + -0.071718f*Lw[7][4] + 1.578040f*Lw[7][5] + 1.296050f*Lw[7][6] + 0.671933f*Lw[7][7] + 0.738547f*Lw[7][8] + -2.336390f*Lw[7][9] + -0.274473f*Lw[7][10] + -0.262591f*Lw[7][11] + 1.112940f*Lw[7][12] + 0.807418f*Lw[7][13] + 0.257607f*Lw[7][14];
        ylmCoeff[64] = -0.377914f*Lw[8][0] + -0.225818f*Lw[8][1] + -0.429096f*Lw[8][2] + 0.987763f*Lw[8][3] + 0.193171f*Lw[8][4] + 0.714889f*Lw[8][5] + -0.666905f*Lw[8][6] + -0.929931f*Lw[8][7] + -0.588023f*Lw[8][8] + -0.043521f*Lw[8][9] + 0.465649f*Lw[8][10] + 1.111360f*Lw[8][11] + 0.810951f*Lw[8][12] + -0.312313f*Lw[8][13] + -0.070449f*Lw[8][14] + -0.539556f*Lw[8][15] + 0.159737f*Lw[8][16];
        ylmCoeff[65] = -2.327670f*Lw[8][0] + 1.341970f*Lw[8][1] + 0.364760f*Lw[8][2] + 0.485280f*Lw[8][3] + -2.297350f*Lw[8][4] + -0.107763f*Lw[8][5] + -0.157041f*Lw[8][6] + 2.380770f*Lw[8][7] + -1.101030f*Lw[8][8] + 0.969400f*Lw[8][9] + -0.118429f*Lw[8][10] + -1.977090f*Lw[8][11] + -0.678479f*Lw[8][12] + 2.716400f*Lw[8][13] + 1.270780f*Lw[8][14] + 1.776150f*Lw[8][15] + -1.760460f*Lw[8][16];
        ylmCoeff[66] = -0.506219f*Lw[8][0] + 0.361120f*Lw[8][1] + -0.772923f*Lw[8][2] + 0.637045f*Lw[8][3] + -1.983200f*Lw[8][4] + -1.249950f*Lw[8][5] + 0.187018f*Lw[8][6] + 1.296970f*Lw[8][7] + 0.753882f*Lw[8][8] + -0.780084f*Lw[8][9] + -0.108084f*Lw[8][10] + -1.162300f*Lw[8][11] + 0.228745f*Lw[8][12] + 0.782582f*Lw[8][13] + 0.190188f*Lw[8][14] + 1.462190f*Lw[8][15] + -1.241040f*Lw[8][16];
        ylmCoeff[67] = -0.261945f*Lw[8][0] + -0.134153f*Lw[8][1] + -0.861752f*Lw[8][2] + -0.325690f*Lw[8][3] + -1.080220f*Lw[8][4] + -0.635845f*Lw[8][5] + 0.108112f*Lw[8][6] + 0.980172f*Lw[8][7] + 0.272034f*Lw[8][8] + -0.176725f*Lw[8][9] + -0.170833f*Lw[8][10] + -0.771681f*Lw[8][11] + -0.310430f*Lw[8][12] + 0.872530f*Lw[8][13] + 0.529705f*Lw[8][14] + 1.488790f*Lw[8][15] + -0.608076f*Lw[8][16];
        ylmCoeff[68] = -0.652701f*Lw[8][0] + 0.343429f*Lw[8][1] + -0.860292f*Lw[8][2] + 1.396690f*Lw[8][3] + -1.216080f*Lw[8][4] + -0.217333f*Lw[8][5] + 0.624246f*Lw[8][6] + 0.513427f*Lw[8][7] + -0.448237f*Lw[8][8] + 0.419166f*Lw[8][9] + -0.201683f*Lw[8][10] + -0.834232f*Lw[8][11] + 0.630710f*Lw[8][12] + 0.541281f*Lw[8][13] + -0.198191f*Lw[8][14] + 1.732570f*Lw[8][15] + -1.338260f*Lw[8][16];
        ylmCoeff[69] = -0.143953f*Lw[8][0] + 1.265140f*Lw[8][1] + 0.252472f*Lw[8][2] + -0.406242f*Lw[8][3] + -0.671232f*Lw[8][4] + -0.463832f*Lw[8][5] + -0.187793f*Lw[8][6] + -0.053660f*Lw[8][7] + 0.755577f*Lw[8][8] + 0.041813f*Lw[8][9] + -0.613325f*Lw[8][10] + 0.185727f*Lw[8][11] + -0.582403f*Lw[8][12] + 0.168035f*Lw[8][13] + -0.114024f*Lw[8][14] + 0.891265f*Lw[8][15] + -0.929824f*Lw[8][16];
        ylmCoeff[70] = 2.012310f*Lw[8][0] + -1.576260f*Lw[8][1] + -0.800351f*Lw[8][2] + 0.856102f*Lw[8][3] + 2.556560f*Lw[8][4] + 1.950360f*Lw[8][5] + 0.395023f*Lw[8][6] + -3.570100f*Lw[8][7] + 0.742491f*Lw[8][8] + -0.329472f*Lw[8][9] + -0.074153f*Lw[8][10] + 2.637080f*Lw[8][11] + 0.831740f*Lw[8][12] + -2.533290f*Lw[8][13] + -1.547820f*Lw[8][14] + -1.527730f*Lw[8][15] + 1.889530f*Lw[8][16];
        ylmCoeff[71] = -1.013440f*Lw[8][0] + 0.222599f*Lw[8][1] + 0.014803f*Lw[8][2] + 0.204784f*Lw[8][3] + -0.807036f*Lw[8][4] + 0.182928f*Lw[8][5] + -0.523892f*Lw[8][6] + 1.601030f*Lw[8][7] + -0.937233f*Lw[8][8] + 0.743981f*Lw[8][9] + -0.674546f*Lw[8][10] + -0.054782f*Lw[8][11] + -0.667966f*Lw[8][12] + 1.434270f*Lw[8][13] + 0.187707f*Lw[8][14] + 0.861661f*Lw[8][15] + -0.698571f*Lw[8][16];
        ylmCoeff[72] = -0.496894f*Lw[8][0] + 0.258762f*Lw[8][1] + 0.294853f*Lw[8][2] + 0.568549f*Lw[8][3] + -0.587026f*Lw[8][4] + -0.761855f*Lw[8][5] + -0.250601f*Lw[8][6] + 0.208739f*Lw[8][7] + 0.283704f*Lw[8][8] + 0.026877f*Lw[8][9] + 0.470202f*Lw[8][10] + -0.815505f*Lw[8][11] + -0.244517f*Lw[8][12] + -0.188146f*Lw[8][13] + 0.190420f*Lw[8][14] + 0.823236f*Lw[8][15] + -0.070274f*Lw[8][16];
        ylmCoeff[73] = -0.400609f*Lw[8][0] + -0.530642f*Lw[8][1] + -0.030195f*Lw[8][2] + -0.015360f*Lw[8][3] + 0.655302f*Lw[8][4] + -0.239775f*Lw[8][5] + 0.572657f*Lw[8][6] + -0.241502f*Lw[8][7] + 0.260030f*Lw[8][8] + -0.401339f*Lw[8][9] + 0.125290f*Lw[8][10] + -0.017789f*Lw[8][11] + 0.198477f*Lw[8][12] + 0.419563f*Lw[8][13] + -0.149376f*Lw[8][14] + 0.522912f*Lw[8][15] + -0.248691f*Lw[8][16];
        ylmCoeff[74] = 3.022250f*Lw[8][0] + -1.048110f*Lw[8][1] + 0.382163f*Lw[8][2] + -0.814561f*Lw[8][3] + 2.242720f*Lw[8][4] + -0.140416f*Lw[8][5] + 0.693969f*Lw[8][6] + -2.790550f*Lw[8][7] + 1.043390f*Lw[8][8] + -0.215989f*Lw[8][9] + -0.029870f*Lw[8][10] + 1.390150f*Lw[8][11] + 0.197856f*Lw[8][12] + -1.480150f*Lw[8][13] + -1.534680f*Lw[8][14] + -1.017820f*Lw[8][15] + 1.397670f*Lw[8][16];
        ylmCoeff[75] = 3.507190f*Lw[8][0] + -1.328310f*Lw[8][1] + 0.829690f*Lw[8][2] + -1.767170f*Lw[8][3] + 3.129070f*Lw[8][4] + 0.644180f*Lw[8][5] + 0.485468f*Lw[8][6] + -4.616590f*Lw[8][7] + 1.326730f*Lw[8][8] + 0.264079f*Lw[8][9] + -0.585126f*Lw[8][10] + 2.837220f*Lw[8][11] + 0.276637f*Lw[8][12] + -3.210290f*Lw[8][13] + -2.219370f*Lw[8][14] + -3.052650f*Lw[8][15] + 2.956310f*Lw[8][16];
        ylmCoeff[76] = -0.687074f*Lw[8][0] + -0.364741f*Lw[8][1] + 0.182821f*Lw[8][2] + 0.365120f*Lw[8][3] + -0.775456f*Lw[8][4] + 0.474574f*Lw[8][5] + -0.040808f*Lw[8][6] + 0.633208f*Lw[8][7] + -0.087569f*Lw[8][8] + -0.076654f*Lw[8][9] + -0.149420f*Lw[8][10] + -0.318291f*Lw[8][11] + 0.280064f*Lw[8][12] + 0.234616f*Lw[8][13] + 0.977562f*Lw[8][14] + 0.441624f*Lw[8][15] + -0.662151f*Lw[8][16];
        ylmCoeff[77] = 0.089884f*Lw[8][0] + 0.063335f*Lw[8][1] + -1.496280f*Lw[8][2] + 1.369270f*Lw[8][3] + -0.473625f*Lw[8][4] + 0.208693f*Lw[8][5] + -0.458777f*Lw[8][6] + -0.252940f*Lw[8][7] + 0.156376f*Lw[8][8] + -0.349746f*Lw[8][9] + 0.342975f*Lw[8][10] + 0.425743f*Lw[8][11] + -0.288190f*Lw[8][12] + -0.386056f*Lw[8][13] + -1.102830f*Lw[8][14] + 0.639174f*Lw[8][15] + -1.611870f*Lw[8][16];
        ylmCoeff[78] = 0.683439f*Lw[8][0] + -0.256975f*Lw[8][1] + 0.853269f*Lw[8][2] + -1.253060f*Lw[8][3] + 0.689052f*Lw[8][4] + -0.205386f*Lw[8][5] + -0.250166f*Lw[8][6] + -0.095097f*Lw[8][7] + 0.375352f*Lw[8][8] + 0.789996f*Lw[8][9] + -0.948669f*Lw[8][10] + -0.123040f*Lw[8][11] + -0.222474f*Lw[8][12] + 0.474984f*Lw[8][13] + 1.021510f*Lw[8][14] + -1.029300f*Lw[8][15] + 1.257930f*Lw[8][16];
        ylmCoeff[79] = -1.329260f*Lw[8][0] + 0.386258f*Lw[8][1] + -0.413633f*Lw[8][2] + 0.452075f*Lw[8][3] + -1.292370f*Lw[8][4] + 0.123832f*Lw[8][5] + -0.775261f*Lw[8][6] + 2.053530f*Lw[8][7] + -0.438136f*Lw[8][8] + 0.371959f*Lw[8][9] + -0.196067f*Lw[8][10] + -1.724130f*Lw[8][11] + 0.537271f*Lw[8][12] + 1.336480f*Lw[8][13] + 0.961259f*Lw[8][14] + 0.902856f*Lw[8][15] + -0.412672f*Lw[8][16];
        ylmCoeff[80] = -2.266390f*Lw[8][0] + 1.176120f*Lw[8][1] + 0.583651f*Lw[8][2] + 0.185289f*Lw[8][3] + -1.793470f*Lw[8][4] + -0.720326f*Lw[8][5] + -0.414004f*Lw[8][6] + 2.511460f*Lw[8][7] + -1.166780f*Lw[8][8] + -0.257522f*Lw[8][9] + -0.307256f*Lw[8][10] + -2.132790f*Lw[8][11] + -0.371880f*Lw[8][12] + 1.882160f*Lw[8][13] + 1.744210f*Lw[8][14] + 1.330160f*Lw[8][15] + -1.073280f*Lw[8][16];
        ylmCoeff[81] = 18392.000000f*Lw[9][0] + -38257.398438f*Lw[9][1] + 37806.699219f*Lw[9][2] + -21284.699219f*Lw[9][3] + 32237.000000f*Lw[9][4] + -62523.101563f*Lw[9][5] + -41790.898438f*Lw[9][6] + 110133.000000f*Lw[9][7] + 92665.101563f*Lw[9][8] + 4731.580078f*Lw[9][9] + 92667.000000f*Lw[9][10] + 110134.000000f*Lw[9][11] + -41792.601563f*Lw[9][12] + -62522.699219f*Lw[9][13] + 32237.000000f*Lw[9][14] + -21286.500000f*Lw[9][15] + 37806.101563f*Lw[9][16] + -38255.300781f*Lw[9][17] + 18392.300781f*Lw[9][18];
        ylmCoeff[82] = -75.860703f*Lw[9][0] + 157.287994f*Lw[9][1] + -155.574997f*Lw[9][2] + 87.432800f*Lw[9][3] + -132.742996f*Lw[9][4] + 256.299011f*Lw[9][5] + 172.322006f*Lw[9][6] + -453.639008f*Lw[9][7] + -381.201996f*Lw[9][8] + -18.607901f*Lw[9][9] + -381.246002f*Lw[9][10] + -453.673004f*Lw[9][11] + 172.328995f*Lw[9][12] + 256.294006f*Lw[9][13] + -132.735001f*Lw[9][14] + 87.450699f*Lw[9][15] + -155.557007f*Lw[9][16] + 157.266998f*Lw[9][17] + -75.860397f*Lw[9][18];
        ylmCoeff[83] = -2503.149902f*Lw[9][0] + 5205.169922f*Lw[9][1] + -5145.709961f*Lw[9][2] + 2897.199951f*Lw[9][3] + -4387.520020f*Lw[9][4] + 8508.290039f*Lw[9][5] + 5688.740234f*Lw[9][6] + -14988.200195f*Lw[9][7] + -12612.200195f*Lw[9][8] + -644.041992f*Lw[9][9] + -12610.299805f*Lw[9][10] + -14988.500000f*Lw[9][11] + 5687.310059f*Lw[9][12] + 8508.339844f*Lw[9][13] + -4386.490234f*Lw[9][14] + 2896.590088f*Lw[9][15] + -5145.410156f*Lw[9][16] + 5206.819824f*Lw[9][17] + -2502.989990f*Lw[9][18];
        ylmCoeff[84] = -12750.299805f*Lw[9][0] + 26524.900391f*Lw[9][1] + -26210.699219f*Lw[9][2] + 14755.700195f*Lw[9][3] + -22349.599609f*Lw[9][4] + 43346.699219f*Lw[9][5] + 28971.099609f*Lw[9][6] + -76353.703125f*Lw[9][7] + -64242.000000f*Lw[9][8] + -3280.419922f*Lw[9][9] + -64245.398438f*Lw[9][10] + -76354.101563f*Lw[9][11] + 28975.099609f*Lw[9][12] + 43346.000000f*Lw[9][13] + -22348.900391f*Lw[9][14] + 14758.900391f*Lw[9][15] + -26210.400391f*Lw[9][16] + 26520.300781f*Lw[9][17] + -12751.299805f*Lw[9][18];
        ylmCoeff[85] = 3672.850098f*Lw[9][0] + -7639.390137f*Lw[9][1] + 7547.339844f*Lw[9][2] + -4249.919922f*Lw[9][3] + 6436.450195f*Lw[9][4] + -12483.900391f*Lw[9][5] + -8343.339844f*Lw[9][6] + 21989.000000f*Lw[9][7] + 18501.099609f*Lw[9][8] + 944.875000f*Lw[9][9] + 18501.900391f*Lw[9][10] + 21989.199219f*Lw[9][11] + -8344.080078f*Lw[9][12] + -12483.200195f*Lw[9][13] + 6436.799805f*Lw[9][14] + -4250.160156f*Lw[9][15] + 7547.370117f*Lw[9][16] + -7638.160156f*Lw[9][17] + 3672.500000f*Lw[9][18];
        ylmCoeff[86] = -14546.000000f*Lw[9][0] + 30256.099609f*Lw[9][1] + -29899.699219f*Lw[9][2] + 16833.199219f*Lw[9][3] + -25494.300781f*Lw[9][4] + 49446.800781f*Lw[9][5] + 33050.398438f*Lw[9][6] + -87099.000000f*Lw[9][7] + -73285.000000f*Lw[9][8] + -3742.060059f*Lw[9][9] + -73285.703125f*Lw[9][10] + -87100.703125f*Lw[9][11] + 33052.300781f*Lw[9][12] + 49446.300781f*Lw[9][13] + -25495.199219f*Lw[9][14] + 16834.400391f*Lw[9][15] + -29898.900391f*Lw[9][16] + 30254.300781f*Lw[9][17] + -14545.200195f*Lw[9][18];
        ylmCoeff[87] = 2599.250000f*Lw[9][0] + -5405.279785f*Lw[9][1] + 5340.930176f*Lw[9][2] + -3007.280029f*Lw[9][3] + 4554.290039f*Lw[9][4] + -8833.019531f*Lw[9][5] + -5905.069824f*Lw[9][6] + 15560.500000f*Lw[9][7] + 13091.599609f*Lw[9][8] + 666.984009f*Lw[9][9] + 13092.099609f*Lw[9][10] + 15560.599609f*Lw[9][11] + -5905.660156f*Lw[9][12] + -8832.450195f*Lw[9][13] + 4554.750000f*Lw[9][14] + -3007.629883f*Lw[9][15] + 5340.950195f*Lw[9][16] + -5405.049805f*Lw[9][17] + 2598.800049f*Lw[9][18];
        ylmCoeff[88] = 9713.830078f*Lw[9][0] + -20206.300781f*Lw[9][1] + 19968.300781f*Lw[9][2] + -11241.400391f*Lw[9][3] + 17025.699219f*Lw[9][4] + -33021.300781f*Lw[9][5] + -22071.699219f*Lw[9][6] + 58167.300781f*Lw[9][7] + 48941.000000f*Lw[9][8] + 2499.179932f*Lw[9][9] + 48942.898438f*Lw[9][10] + 58167.699219f*Lw[9][11] + -22073.900391f*Lw[9][12] + -33020.601563f*Lw[9][13] + 17025.099609f*Lw[9][14] + -11242.700195f*Lw[9][15] + 19967.199219f*Lw[9][16] + -20203.099609f*Lw[9][17] + 9713.360352f*Lw[9][18];
        ylmCoeff[89] = -15217.599609f*Lw[9][0] + 31652.300781f*Lw[9][1] + -31280.000000f*Lw[9][2] + 17611.000000f*Lw[9][3] + -26671.000000f*Lw[9][4] + 51729.199219f*Lw[9][5] + 34576.000000f*Lw[9][6] + -91119.500000f*Lw[9][7] + -76667.000000f*Lw[9][8] + -3914.729980f*Lw[9][9] + -76668.898438f*Lw[9][10] + -91120.601563f*Lw[9][11] + 34578.101563f*Lw[9][12] + 51727.601563f*Lw[9][13] + -26671.300781f*Lw[9][14] + 17610.599609f*Lw[9][15] + -31278.900391f*Lw[9][16] + 31650.400391f*Lw[9][17] + -15216.799805f*Lw[9][18];
        ylmCoeff[90] = 22110.099609f*Lw[9][0] + -45994.398438f*Lw[9][1] + 45450.800781f*Lw[9][2] + -25586.699219f*Lw[9][3] + 38754.601563f*Lw[9][4] + -75164.703125f*Lw[9][5] + -50238.300781f*Lw[9][6] + 132400.000000f*Lw[9][7] + 111398.000000f*Lw[9][8] + 5688.299805f*Lw[9][9] + 111403.000000f*Lw[9][10] + 132400.000000f*Lw[9][11] + -50244.199219f*Lw[9][12] + -75162.203125f*Lw[9][13] + 38754.300781f*Lw[9][14] + -25591.199219f*Lw[9][15] + 45448.800781f*Lw[9][16] + -45987.300781f*Lw[9][17] + 22111.300781f*Lw[9][18];
        ylmCoeff[91] = 6281.000000f*Lw[9][0] + -13066.700195f*Lw[9][1] + 12911.900391f*Lw[9][2] + -7269.600098f*Lw[9][3] + 11010.400391f*Lw[9][4] + -21354.400391f*Lw[9][5] + -14272.000000f*Lw[9][6] + 37613.101563f*Lw[9][7] + 31647.300781f*Lw[9][8] + 1616.109985f*Lw[9][9] + 31648.099609f*Lw[9][10] + 37613.601563f*Lw[9][11] + -14272.900391f*Lw[9][12] + -21353.699219f*Lw[9][13] + 11010.500000f*Lw[9][14] + -7269.430176f*Lw[9][15] + 12911.400391f*Lw[9][16] + -13066.000000f*Lw[9][17] + 6280.700195f*Lw[9][18];
        ylmCoeff[92] = 9762.080078f*Lw[9][0] + -20306.400391f*Lw[9][1] + 20065.699219f*Lw[9][2] + -11296.400391f*Lw[9][3] + 17110.400391f*Lw[9][4] + -33184.500000f*Lw[9][5] + -22179.300781f*Lw[9][6] + 58452.398438f*Lw[9][7] + 49180.699219f*Lw[9][8] + 2511.159912f*Lw[9][9] + 49182.500000f*Lw[9][10] + 58452.699219f*Lw[9][11] + -22181.599609f*Lw[9][12] + -33183.800781f*Lw[9][13] + 17109.800781f*Lw[9][14] + -11297.700195f*Lw[9][15] + 20064.599609f*Lw[9][16] + -20303.099609f*Lw[9][17] + 9761.610352f*Lw[9][18];
        ylmCoeff[93] = -6209.930176f*Lw[9][0] + 12916.299805f*Lw[9][1] + -12764.400391f*Lw[9][2] + 7185.950195f*Lw[9][3] + -10883.299805f*Lw[9][4] + 21110.000000f*Lw[9][5] + 14108.799805f*Lw[9][6] + -37183.101563f*Lw[9][7] + -31285.400391f*Lw[9][8] + -1598.150024f*Lw[9][9] + -31286.699219f*Lw[9][10] + -37183.300781f*Lw[9][11] + 14110.200195f*Lw[9][12] + 21108.599609f*Lw[9][13] + -10884.400391f*Lw[9][14] + 7186.790039f*Lw[9][15] + -12764.400391f*Lw[9][16] + 12915.799805f*Lw[9][17] + -6208.839844f*Lw[9][18];
        ylmCoeff[94] = -70.574303f*Lw[9][0] + 147.912994f*Lw[9][1] + -146.149994f*Lw[9][2] + 82.140297f*Lw[9][3] + -125.439003f*Lw[9][4] + 243.072006f*Lw[9][5] + 161.074005f*Lw[9][6] + -425.993011f*Lw[9][7] + -358.776001f*Lw[9][8] + -19.075800f*Lw[9][9] + -358.764008f*Lw[9][10] + -426.002991f*Lw[9][11] + 161.080002f*Lw[9][12] + 243.065994f*Lw[9][13] + -125.445999f*Lw[9][14] + 82.145203f*Lw[9][15] + -146.143005f*Lw[9][16] + 147.899994f*Lw[9][17] + -70.570503f*Lw[9][18];
        ylmCoeff[95] = 9020.580078f*Lw[9][0] + -18763.900391f*Lw[9][1] + 18542.300781f*Lw[9][2] + -10439.000000f*Lw[9][3] + 15809.900391f*Lw[9][4] + -30664.699219f*Lw[9][5] + -20495.900391f*Lw[9][6] + 54014.199219f*Lw[9][7] + 45446.500000f*Lw[9][8] + 2320.560059f*Lw[9][9] + 45448.500000f*Lw[9][10] + 54014.699219f*Lw[9][11] + -20497.699219f*Lw[9][12] + -30663.000000f*Lw[9][13] + 15810.700195f*Lw[9][14] + -10439.599609f*Lw[9][15] + 18542.300781f*Lw[9][16] + -18760.900391f*Lw[9][17] + 9019.740234f*Lw[9][18];
        ylmCoeff[96] = 12565.299805f*Lw[9][0] + -26138.900391f*Lw[9][1] + 25829.300781f*Lw[9][2] + -14540.299805f*Lw[9][3] + 22024.699219f*Lw[9][4] + -42715.601563f*Lw[9][5] + -28550.800781f*Lw[9][6] + 75243.101563f*Lw[9][7] + 63307.800781f*Lw[9][8] + 3232.639893f*Lw[9][9] + 63311.199219f*Lw[9][10] + 75243.500000f*Lw[9][11] + -28554.800781f*Lw[9][12] + -42715.000000f*Lw[9][13] + 22024.000000f*Lw[9][14] + -14543.400391f*Lw[9][15] + 25829.000000f*Lw[9][16] + -26134.300781f*Lw[9][17] + 12566.299805f*Lw[9][18];
        ylmCoeff[97] = -1062.069946f*Lw[9][0] + 2209.780029f*Lw[9][1] + -2182.310059f*Lw[9][2] + 1229.069946f*Lw[9][3] + -1862.270020f*Lw[9][4] + 3611.850098f*Lw[9][5] + 2412.590088f*Lw[9][6] + -6359.959961f*Lw[9][7] + -5351.250000f*Lw[9][8] + -273.013000f*Lw[9][9] + -5350.439941f*Lw[9][10] + -6360.089844f*Lw[9][11] + 2411.989990f*Lw[9][12] + 3611.870117f*Lw[9][13] + -1861.829956f*Lw[9][14] + 1228.810059f*Lw[9][15] + -2182.179932f*Lw[9][16] + 2210.479980f*Lw[9][17] + -1062.010010f*Lw[9][18];
        ylmCoeff[98] = 7764.910156f*Lw[9][0] + -16152.299805f*Lw[9][1] + 15962.099609f*Lw[9][2] + -8985.759766f*Lw[9][3] + 13610.299805f*Lw[9][4] + -26396.800781f*Lw[9][5] + -17643.599609f*Lw[9][6] + 46496.101563f*Lw[9][7] + 39121.101563f*Lw[9][8] + 1997.650024f*Lw[9][9] + 39123.300781f*Lw[9][10] + 46497.601563f*Lw[9][11] + -17644.300781f*Lw[9][12] + -26395.699219f*Lw[9][13] + 13609.599609f*Lw[9][14] + -8987.129883f*Lw[9][15] + 15960.500000f*Lw[9][16] + -16150.200195f*Lw[9][17] + 7764.959961f*Lw[9][18];
        ylmCoeff[99] = -7382.979980f*Lw[9][0] + 15356.700195f*Lw[9][1] + -15175.599609f*Lw[9][2] + 8543.610352f*Lw[9][3] + -12939.799805f*Lw[9][4] + 25096.900391f*Lw[9][5] + 16775.500000f*Lw[9][6] + -44208.699219f*Lw[9][7] + -37195.898438f*Lw[9][8] + -1899.550049f*Lw[9][9] + -37196.699219f*Lw[9][10] + -44209.000000f*Lw[9][11] + 16776.199219f*Lw[9][12] + 25096.800781f*Lw[9][13] + -12939.799805f*Lw[9][14] + 8544.309570f*Lw[9][15] + -15175.299805f*Lw[9][16] + 15355.799805f*Lw[9][17] + -7383.100098f*Lw[9][18];
#pragma endregion REGION_YLMCoeff_MUL
    }

    /**
     * @param 9-order SH Eval.
     */
    static __device__ __inline__ void SHEvalFast9(const optix::float3 &w, float *pOut) {
        const float fX = w.x;
        const float fY = w.y;
        const float fZ = w.z;

        float fC0, fS0, fC1, fS1, fPa, fPb, fPc;
        {
            float fZ2 = fZ * fZ;
            pOut[0] = 0.282094791774;
            pOut[2] = 0.488602511903f*fZ;
            pOut[6] = (0.946174695758f*fZ2) + -0.315391565253f;
            pOut[12] = fZ * ((fZ2* 1.86588166295f) + -1.11952899777f);
            pOut[20] = ((fZ* 1.9843134833f)*pOut[12]) + (-1.00623058987f*pOut[6]);
            pOut[30] = ((fZ* 1.98997487421f)*pOut[20]) + (-1.00285307284f*pOut[12]);
            pOut[42] = ((fZ* 1.99304345718f)*pOut[30]) + (-1.00154202096f*pOut[20]);
            pOut[56] = ((fZ* 1.99489143482f)*pOut[42]) + (-1.00092721392f*pOut[30]);
            pOut[72] = ((fZ* 1.99608992783f)*pOut[56]) + (-1.00060078107f*pOut[42]);
            pOut[90] = ((fZ* 1.99691119507f)*pOut[72]) + (-1.00041143799f*pOut[56]);
            fC0 = fX;
            fS0 = fY;
            fPa = -0.488602511903f;
            pOut[3] = fPa * fC0;
            pOut[1] = fPa * fS0;
            fPb = -1.09254843059f*fZ;
            pOut[7] = fPb * fC0;
            pOut[5] = fPb * fS0;
            fPc = (-2.28522899732f*fZ2) + 0.457045799464f;
            pOut[13] = fPc * fC0;
            pOut[11] = fPc * fS0;
            fPa = fZ * ((fZ2* -4.6833258049f) + 2.00713963067f);
            pOut[21] = fPa * fC0;
            pOut[19] = fPa * fS0;
            fPb = ((fZ* 2.03100960116f)*fPa) + (-0.991031208965f*fPc);
            pOut[31] = fPb * fC0;
            pOut[29] = fPb * fS0;
            fPc = ((fZ* 2.02131498924f)*fPb) + (-0.995226703056f*fPa);
            pOut[43] = fPc * fC0;
            pOut[41] = fPc * fS0;
            fPa = ((fZ* 2.01556443707f)*fPc) + (-0.997155044022f*fPb);
            pOut[57] = fPa * fC0;
            pOut[55] = fPa * fS0;
            fPb = ((fZ* 2.01186954041f)*fPa) + (-0.99816681789f*fPc);
            pOut[73] = fPb * fC0;
            pOut[71] = fPb * fS0;
            fPc = ((fZ* 2.00935312974f)*fPb) + (-0.998749217772f*fPa);
            pOut[91] = fPc * fC0;
            pOut[89] = fPc * fS0;
            fC1 = (fX*fC0) - (fY*fS0);
            fS1 = (fX*fS0) + (fY*fC0);
            fPa = 0.546274215296f;
            pOut[8] = fPa * fC1;
            pOut[4] = fPa * fS1;
            fPb = 1.44530572132f*fZ;
            pOut[14] = fPb * fC1;
            pOut[10] = fPb * fS1;
            fPc = (3.31161143515f*fZ2) + -0.473087347879f;
            pOut[22] = fPc * fC1;
            pOut[18] = fPc * fS1;
            fPa = fZ * ((fZ2* 7.19030517746f) + -2.39676839249f);
            pOut[32] = fPa * fC1;
            pOut[28] = fPa * fS1;
            fPb = ((fZ* 2.11394181566f)*fPa) + (-0.973610120462f*fPc);
            pOut[44] = fPb * fC1;
            pOut[40] = fPb * fS1;
            fPc = ((fZ* 2.08166599947f)*fPb) + (-0.984731927835f*fPa);
            pOut[58] = fPc * fC1;
            pOut[54] = fPc * fS1;
            fPa = ((fZ* 2.06155281281f)*fPc) + (-0.99033793766f*fPb);
            pOut[74] = fPa * fC1;
            pOut[70] = fPa * fS1;
            fPb = ((fZ* 2.04812235836f)*fPa) + (-0.99348527267f*fPc);
            pOut[92] = fPb * fC1;
            pOut[88] = fPb * fS1;
            fC0 = (fX*fC1) - (fY*fS1);
            fS0 = (fX*fS1) + (fY*fC1);
            fPa = -0.590043589927f;
            pOut[15] = fPa * fC0;
            pOut[9] = fPa * fS0;
            fPb = -1.77013076978f*fZ;
            pOut[23] = fPb * fC0;
            pOut[17] = fPb * fS0;
            fPc = (-4.40314469492f*fZ2) + 0.489238299435f;
            pOut[33] = fPc * fC0;
            pOut[27] = fPc * fS0;
            fPa = fZ * ((fZ2* -10.1332578547f) + 2.76361577854f);
            pOut[45] = fPa * fC0;
            pOut[39] = fPa * fS0;
            fPb = ((fZ* 2.20794021658f)*fPa) + (-0.9594032236f*fPc);
            pOut[59] = fPb * fC0;
            pOut[53] = fPb * fS0;
            fPc = ((fZ* 2.1532216877f)*fPb) + (-0.97521738656f*fPa);
            pOut[75] = fPc * fC0;
            pOut[69] = fPc * fS0;
            fPa = ((fZ* 2.11804417119f)*fPc) + (-0.983662844979f*fPb);
            pOut[93] = fPa * fC0;
            pOut[87] = fPa * fS0;
            fC1 = (fX*fC0) - (fY*fS0);
            fS1 = (fX*fS0) + (fY*fC0);
            fPa = 0.625835735449f;
            pOut[24] = fPa * fC1;
            pOut[16] = fPa * fS1;
            fPb = 2.07566231488f*fZ;
            pOut[34] = fPb * fC1;
            pOut[26] = fPb * fS1;
            fPc = (5.55021390802f*fZ2) + -0.504564900729f;
            pOut[46] = fPc * fC1;
            pOut[38] = fPc * fS1;
            fPa = fZ * ((fZ2* 13.4918050467f) + -3.11349347232f);
            pOut[60] = fPa * fC1;
            pOut[52] = fPa * fS1;
            fPb = ((fZ* 2.30488611432f)*fPa) + (-0.948176387355f*fPc);
            pOut[76] = fPb * fC1;
            pOut[68] = fPb * fS1;
            fPc = ((fZ* 2.22917715071f)*fPb) + (-0.967152839723f*fPa);
            pOut[94] = fPc * fC1;
            pOut[86] = fPc * fS1;
            fC0 = (fX*fC1) - (fY*fS1);
            fS0 = (fX*fS1) + (fY*fC1);
            fPa = -0.65638205684f;
            pOut[35] = fPa * fC0;
            pOut[25] = fPa * fS0;
            fPb = -2.36661916223f*fZ;
            pOut[47] = fPb * fC0;
            pOut[37] = fPb * fS0;
            fPc = (-6.74590252336f*fZ2) + 0.51891557872f;
            pOut[61] = fPc * fC0;
            pOut[51] = fPc * fS0;
            fPa = fZ * ((fZ2* -17.2495531105f) + 3.4499106221f);
            pOut[77] = fPa * fC0;
            pOut[67] = fPa * fS0;
            fPb = ((fZ* 2.40163634692f)*fPa) + (-0.939224604204f*fPc);
            pOut[95] = fPb * fC0;
            pOut[85] = fPb * fS0;
            fC1 = (fX*fC0) - (fY*fS0);
            fS1 = (fX*fS0) + (fY*fC0);
            fPa = 0.683184105192f;
            pOut[48] = fPa * fC1;
            pOut[36] = fPa * fS1;
            fPb = 2.6459606618f*fZ;
            pOut[62] = fPb * fC1;
            pOut[50] = fPb * fS1;
            fPc = (7.98499149089f*fZ2) + -0.53233276606f;
            pOut[78] = fPc * fC1;
            pOut[66] = fPc * fS1;
            fPa = fZ * ((fZ2* 21.3928901909f) + -3.77521591604f);
            pOut[96] = fPa * fC1;
            pOut[84] = fPa * fS1;
            fC0 = (fX*fC1) - (fY*fS1);
            fS0 = (fX*fS1) + (fY*fC1);
            fPa = -0.707162732525f;
            pOut[63] = fPa * fC0;
            pOut[49] = fPa * fS0;
            fPb = -2.9157066407f*fZ;
            pOut[79] = fPb * fC0;
            pOut[65] = fPb * fS0;
            fPc = (-9.26339318285f*fZ2) + 0.544905481344f;
            pOut[97] = fPc * fC0;
            pOut[83] = fPc * fS0;
            fC1 = (fX*fC0) - (fY*fS0);
            fS1 = (fX*fS0) + (fY*fC0);
            fPa = 0.728926660175f;
            pOut[80] = fPa * fC1;
            pOut[64] = fPa * fS1;
            fPb = 3.17731764895f*fZ;
            pOut[98] = fPb * fC1;
            pOut[82] = fPb * fS1;
            fC0 = (fX*fC1) - (fY*fS1);
            fS0 = (fX*fS1) + (fY*fC1);
            fPc = -0.748900951853f;
            pOut[99] = fPc * fC0;
            pOut[81] = fPc * fS0;
        }
    }
}

/************************************************************************/
/*         Integrating Clipped Spherical Harmonics Expansions           */
/************************************************************************/

template<>
static __device__ __inline__ float4 EstimateDirectLighting<CommonStructs::LightType::QuadLight>(
    int lightId,
    const CommonStructs::ShaderParams & shaderParams,
    const float3 & isectP, const float3 & isectDir,
    GPUSampler &localSampler)
{
#define MEASURE_TIMING_SH_COEFF 0
#define SHINTEGRATION_ANALYTIC 1
#define SHINTEGRATION_MCSAMPLING 0
    if (shaderParams.bsdfType != BSDFType::Lambert && shaderParams.bsdfType != BSDFType::Plastic)
    {
        rtPrintf("SH Integration supports lambert+plastic only.\n");
    }
    float4 L = make_float4(0.f);

    /* SH Integration */
    /* 1. Get 4 vertices of QuadLight. */
#if SHINTEGRATION_ANALYTIC
    // Todo: 1.this works for BRDF but not BTDF.
    // Todo: 2.this does not account for the "strict normal" issue.
    if(TwUtil::dot(isectDir, shaderParams.nGeometry) >= 0.f)
    {
        float3 quadShape[5];
        const CommonStructs::QuadLight &quadLight = sysLightBuffers.quadLightBuffer[lightId];

        quadShape[0] = TwUtil::xfmPoint(make_float3(-1.f, -1.f, 0.f), quadLight.lightToWorld);
        quadShape[1] = TwUtil::xfmPoint(make_float3(1.f, -1.f, 0.f), quadLight.lightToWorld);
        quadShape[2] = TwUtil::xfmPoint(make_float3(1.f, 1.f, 0.f), quadLight.lightToWorld);
        quadShape[3] = TwUtil::xfmPoint(make_float3(-1.f, 1.f, 0.f), quadLight.lightToWorld);
        quadShape[4] = make_float3(0.f);

        /* 2. Convert QuadLight from World To BSDFLocal (or just use directional information and call WorldToLocal only. */
        quadShape[0] = Cl::BSDFWorldToLocal(quadShape[0], shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn, isectP);
        quadShape[1] = Cl::BSDFWorldToLocal(quadShape[1], shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn, isectP);
        quadShape[2] = Cl::BSDFWorldToLocal(quadShape[2], shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn, isectP);
        quadShape[3] = Cl::BSDFWorldToLocal(quadShape[3], shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn, isectP);

        constexpr int lmax = 9;

        if (!Cl::CheckOrientation<3>(quadShape))
        {
#if MEASURE_TIMING_SH_COEFF
            clock_t start;
            if (sysLaunch_index == make_uint2(960, 640))
                start = clock();
#endif

            /* 3. Clipping. */
            int clippedQuadVertices = 0;
            Cl::ClipQuadToHorizon(quadShape, clippedQuadVertices);

            // Clip Quad shape but not spherical quad.
            // We need to normalize (to project) after clipping
            for (int i = 0; i < clippedQuadVertices; ++i)
            {
                quadShape[i] = safe_normalize(quadShape[i]);
            }

            /* 4. Compute Ylm projection coefficient. */
            float ylmCoeff[(lmax+1)*(lmax+1)];
            if (clippedQuadVertices == 3)
            {
                float3 ABC[4]{ make_float3(0.f),quadShape[0],quadShape[1],quadShape[2] };
                Cl::computeCoeff<3, 9>(ABC, ylmCoeff);
            }
            else if (clippedQuadVertices == 4)
            {
                float3 ABCD[5]{ make_float3(0.f),quadShape[0],quadShape[1],quadShape[2],quadShape[3] };
                Cl::computeCoeff<4, 9>(ABCD, ylmCoeff);
            }
            else if (clippedQuadVertices == 5)
            {
                float3 ABCDE[6]{ make_float3(0.f),quadShape[0],quadShape[1],quadShape[2],quadShape[3],quadShape[4] };
                Cl::computeCoeff<5, 9>(ABCDE, ylmCoeff);
            }
            else if (clippedQuadVertices == 0)
            {

            }
            else
            {
                rtPrintf("clippedQuadVertices==%d failed!\n", clippedQuadVertices);
            }

            /* 5. Dot Product of Flm and Ylm. */
            if (clippedQuadVertices != 0)
            {
                if (shaderParams.bsdfType == BSDFType::Lambert)
                {
                    for (int i = 0; i < (lmax + 1)*(lmax + 1); ++i)
                    {
                        /*if (isnan(ylmCoeff[i]))
                        {
                            rtPrintf("%d,%d] ylmCoeff[%d]=%f\n", sysLaunch_index.x, sysLaunch_index.y,i,ylmCoeff[i]);
                        }*/
                        L += make_float4(areaLightFlmVector[i] * ylmCoeff[i]);
                    }
                    if (sysLaunch_index == make_uint2(589,720-290)|| sysLaunch_index == make_uint2(590, 720 - 290))
                    {
                        //rtPrintf("%d]L:%f %f %f\n", sysLaunch_index.x, L.x, L.y, L.z);
                    }
                    L *= quadLight.intensity * shaderParams.Reflectance/* / M_PIf*/;
                }
                else if (shaderParams.bsdfType == BSDFType::Plastic)
                {
                    float ylmVector[(lmax + 1)*(lmax + 1)];

                    float3 wo_Local = TwUtil::BSDFMath::WorldToLocal(isectDir, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);
                    wo_Local = safe_normalize(wo_Local);

                    Cl::SHEvalFast9(wo_Local, ylmVector);
                    //Cl::SHEvalFast9(make_float3(0.f, 0.f, 1.f), ylmVector);
                    float FlmVector[(lmax + 1)*(lmax + 1)];
                    /* Matrix Multiplication. */
                    for (int j = 0; j < BSDFMatrix.size().y; ++j)
                    {
                        float result = 0.0f;
                        for (int i = 0; i < BSDFMatrix.size().x; ++i)
                        {
                            result += BSDFMatrix[make_uint2(i, j)] * ylmVector[i];
                        }
                        FlmVector[j] = result;
                    }
                    //if (sysLaunch_index == make_uint2(1280 / 2, 720 / 2))
                    //{
                    //    for (int i = 0; i < 100; ++i)
                    //    {
                    //        rtPrintf("wo_:%f %f %f isectDir:%f %f %f Proj[%d]:%.7f\n", wo_Local.x,wo_Local.y,wo_Local.z,isectDir.x,isectDir.y,isectDir.z,i, FlmVector[i]);
                    //    }
                    //}
                    //if (sysLaunch_index == make_uint2(694, 720 - 526)|| sysLaunch_index == make_uint2(693, 720 - 526)/*||
                    //    sysLaunch_index == make_uint2(695, 720 - 526)*/)
                    //{
                    //    float phi = TwUtil::sphericalPhi(wo_Local);
                    //    float theta = TwUtil::sphericalTheta(wo_Local);
                    //    rtPrintf("dpdu %f %f %f | dpdv %f %f %f}wo:(%f,%f)\n", shaderParams.dgShading.dpdu.x, shaderParams.dgShading.dpdu.y, shaderParams.dgShading.dpdu.z, shaderParams.dgShading.tn.x, shaderParams.dgShading.tn.y, shaderParams.dgShading.tn.z,theta,phi);
                    //    for (int i = 0; i < 100; ++i)
                    //    {
                    //        //rtPrintf("%d|wo_:%f %f %f isectDir:%f %f %f Proj[%d]:%.7f\n", sysLaunch_index.x, wo_Local.x, wo_Local.y, wo_Local.z, isectDir.x, isectDir.y, isectDir.z, i, FlmVector[i]);
                    //    }
                    //}

                    for (int i = 0; i < (lmax + 1)*(lmax + 1); ++i)
                    {
                        L += make_float4(FlmVector[i] * ylmCoeff[i]);
                    }
                    // 1.f / M_PIf is included in Flm.
                    // Remember not to multiply reflectance when using BRDFMatrix to simulate diffuse BRDF!
                    L *= quadLight.intensity/* * shaderParams.Reflectance*/;
                }

            }
#if MEASURE_TIMING_SH_COEFF
            if (sysLaunch_index == make_uint2(960, 640))
            {
                clock_t end = clock();
                rtPrintf("%lld\n", static_cast<long long>(end - start));
            }
#endif
        }
    }
#endif // SHIntegartionPart

#if SHINTEGRATION_MCSAMPLING
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
        float4 Li = Sample_Ld[toUnderlyingValue(CommonStructs::LightType::QuadLight)](p, sceneEpsilon, outWi, lightPdf, randSamples, lightId, shadowRay);
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
                if (shadow_prd.blocked)
                {
                    /* Compute Ld using MIS weight. */

                    //bsdfPdf = Pdf[toUnderlyingValue(shaderParams.bsdfType)](wo_world, outWi, shaderParams);
                    //float weight = TwUtil::MonteCarlo::PowerHeuristic(1, lightPdf, 1, bsdfPdf);
                    Ld += f * Li * (fabs(dot(outWi, shaderParams.dgShading.nn)) / lightPdf);
                    /*rtPrintf("%f %f %f Li:%f %f %f weight:%f bsdfPdf:%f,outwi:%f %f %f\n",
                        f.x, f.y, f.z,
                        Li.x, Li.y, Li.z,
                        weight,
                        bsdfPdf,
                        outWi.x, outWi.y, outWi.z);
                    rtPrintf("Ld %f %f %f\n", Ld.x, Ld.y, Ld.z);*/
                }
            }
        }
        L -= Ld;

#if 0

        /* Sample BSDF with Multiple Importance Sampling. */
        randSamples = Get2D(&localSampler);

        float4 f = Sample_f[toUnderlyingValue(shaderParams.bsdfType)](wo_world, outWi, randSamples, bsdfPdf, Get1D(&localSampler), shaderParams);
        if (!isBlack(f) && bsdfPdf > 0.)
        {
            float weight = 1.f;

            //todo:should use sampledType while we use overall bsdf type:
            if (shaderParams.bsdfType != BSDFType::SmoothGlass && shaderParams.bsdfType != BSDFType::SmoothMirror)
            {
                lightPdf = LightPdf[toUnderlyingValue(CommonStructs::LightType::QuadLight)](p, outWi, lightId, shadowRay);

                if (lightPdf == 0.f)
                    return Ld;

                weight = TwUtil::MonteCarlo::PowerHeuristic(1, bsdfPdf, 1, lightPdf);
            }


            /* Trace shadow ray to find out whether it's blocked down by object. */
            CommonStructs::PerRayData_shadow shadow_prd;
            shadow_prd.blocked = 0;

            const RTrayflags shadowRayFlags = static_cast<RTrayflags>(RTrayflags::RT_RAY_FLAG_DISABLE_ANYHIT | RTrayflags::RT_RAY_FLAG_TERMINATE_ON_FIRST_HIT);
            rtTrace<CommonStructs::PerRayData_shadow>(sysTopShadower, shadowRay, shadow_prd, RT_VISIBILITY_ALL, shadowRayFlags);

            /* In LightPdf we have confirmed that ray is able to intersect the light
             * -- surface from |p| towards |outWi| if there are no objects between
             * them. So now we need to make sure that hypothesis using shadow ray.
             * Given |shadowRay| from LightPdf, ray's |tmax| is narrowed down and
             * could lead to further optimization when tracing shadow ray. */
            if (!shadow_prd.blocked)
            {
                Li = TwUtil::Le_QuadLight(sysLightBuffers.quadLightBuffer[lightId], -outWi);
                if (!isBlack(Li))
                    Ld += f * Li * fabsf(dot(outWi, shaderParams.dgShading.nn)) * weight / bsdfPdf;
            }
        }

        
#endif //  0
    }	
#endif
    return L;
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