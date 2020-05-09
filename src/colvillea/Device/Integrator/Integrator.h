#pragma once
#ifndef COLVILLEA_DEVICE_INTEGRATOR_INTEGRATOR_H_
#define COLVILLEA_DEVICE_INTEGRATOR_INTEGRATOR_H_
/* This file is device only. */
#include "colvillea/Device/SH/AxialMoments.hpp"
//#include "colvillea/Device/SH/SphericalIntegration.hpp"
#include "colvillea/Device/SH/Test/SH.hpp"

#include <optix_world.h>
#include <optix_device.h>
#include <optixu_math_namespace.h>

#include "../Toolkit/CommonStructs.h"
#include "../Toolkit/Utility.h"
#include "../Toolkit/NvRandom.h"
#include "../Light/LightUtil.h"

#include "../Sampler/Sampler.h"

/* SH I. */
#define GLM_FORCE_CUDA
#include "glm/glm/glm.hpp"



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

/* AP Matrix for SH Integration. */
rtBuffer<float, 2> areaLightAPMatrix;

/* Flm Diffuse Matrix. */
rtBuffer<float> areaLightFlmVector;

/* Basis Directions. */
rtBuffer<float3, 1> areaLightBasisVector;

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

struct Vector : public glm::vec3 {

    __device__ Vector() : glm::vec3() {}
    __device__ Vector(const optix::float3& vec) : glm::vec3(vec.x,vec.y,vec.z){}
    __device__ Vector(float x, float y, float z) : glm::vec3(x, y, z) {}
    __device__ Vector(const glm::vec3& w) : glm::vec3(w) {}

    static inline __device__  float Dot(const glm::vec3& a, const glm::vec3& b) {
        return glm::dot(a, b);
    }

    static inline __device__ glm::vec3 Cross(const glm::vec3& a, const glm::vec3& b) {
        return glm::cross(a, b);
    }

    static inline __device__ glm::vec3 Normalize(const glm::vec3& a) {
        return glm::normalize(a);
    }

    static inline __device__ float Length(const glm::vec3& a) {
        return glm::length(a);
    }

#if 0
    Vector() {}
    Vector(float x, float y, float z)
    {
        this->vec.x = x;
        this->vec.y = y;
        this->vec.z = z;
    }
    Vector(const Vector& w) {
        this->vec.x = w.vec.x;
        this->vec.y = w.vec.y;
        this->vec.z = w.vec.z;
    }

    static inline float Dot(const Vector &a, const Vector &b) {
        return dot(a.vec, b.vec);
    }

    static inline float3 Cross(const Vector &a, const Vector &b) {
        return cross(a.vec, b.vec);
    }

    static inline float3 Normalize(const Vector &a) {
        return safe_normalize(a.vec);
    }

    static inline float Length(const Vector &a) {
        return optix::length(a.vec);
    }

    float& operator[](int idx) {
        if (idx == 0)return vec.x;
        if (idx == 1)return vec.y;
        if (idx == 2)return vec.z;
    }
    const float& operator[](int idx) const
    {
        if (idx == 0)return vec.x;
        if (idx == 1)return vec.y;
        if (idx == 2)return vec.z;
    }
    float& x()

        float3 vec;
#endif // 0
};

struct Edge {
    __device__ Edge() {}
    __device__ Edge(const Vector& a, const Vector& b) : A(a), B(b) {}
    Vector A, B;
};

struct Triangle {
    __device__ Triangle() {}
    __device__ Triangle(const Vector& A, const Vector& B, const Vector& C) :e0(Edge(A, B)), e1(Edge(B, C)), e2(Edge(C, A)) {

    }

    __device__ Edge& operator[](int idx) {
        if (idx == 0)return e0;
        if (idx == 1)return e1;
        if (idx == 2)return e2;
    }
    __device__ const Edge& operator[](int idx) const
    {
        if (idx == 0)return e0;
        if (idx == 1)return e1;
        if (idx == 2)return e2;
    }
    __device__ int size() const
    {
        return 3;
    }

    Edge e0;
    Edge e1;
    Edge e2;
};

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
    if (sysLaunch_index.y >= 340 && sysLaunch_index.y <= 360)
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

        quadShape[0] = safe_normalize(quadShape[0]);
        quadShape[1] = safe_normalize(quadShape[1]);
        quadShape[2] = safe_normalize(quadShape[2]);
        quadShape[3] = safe_normalize(quadShape[3]);

        /* 3. Clipping. */
        int clippedQuadVertices = 0;
        ClipQuadToHorizon(quadShape, clippedQuadVertices);

        /* 4. Triangulate Quad/Polygon/Triangle and integrate. */
        // Basis Vector boxing:
        if (areaLightBasisVector.size() != 5)rtPrintf("assert failed at areaLightBasisVector.size()!=5\n");
        Vector basis[5] = { Vector(areaLightBasisVector[0]), Vector(areaLightBasisVector[1]), Vector(areaLightBasisVector[2]), Vector(areaLightBasisVector[3]), Vector(areaLightBasisVector[4]) };
        // AP Matrix boxing:
        Eigen::Matrix<float, 9, 15> AP; /* Avoid dynamic allocation. */
        for (int j = 0; j < areaLightAPMatrix.size().x; ++j)
            for (int i = 0; i < areaLightAPMatrix.size().y; ++i)
                AP(j, i) = areaLightAPMatrix[make_uint2(j, i)];

        // Flm Vector boxing:
        Eigen::Matrix<float, 9, 1> flm;
        for (int i = 0; i < areaLightFlmVector.size(); ++i)
            flm(i) = areaLightFlmVector[i];


        if (clippedQuadVertices == 3)
        {
            Vector v0(quadShape[0]);
            Vector v1(quadShape[1]);
            Vector v2(quadShape[2]);
            Triangle t1(v0, v1, v2);
           // rtPrintf("Finish clippedQuadVertices 3.\n");
            Eigen::Matrix<float, 15, 1> moments = AxialMoments<Triangle, Vector>(t1, basis, areaLightBasisVector.size());

            const auto APM = (AP * moments);
            L += make_float4(flm.dot(APM));
        }
        else if (clippedQuadVertices == 4)
        {
            Vector v0(quadShape[2]);
            Vector v1(quadShape[1]);
            Vector v2(quadShape[0]);
            Triangle t1(v0, v1, v2);

            Eigen::Matrix<float, 15, 1> moments = AxialMoments<Triangle, Vector>(t1, basis, areaLightBasisVector.size());
            L += make_float4(flm.dot(AP * moments));

            /*Vector v3(quadShape[2]);
            Vector v4(quadShape[0]);
            Vector v5(quadShape[3]);
            Triangle t2(v3, v4, v5);

            moments = AxialMoments<Triangle, Vector>(t2, basis, areaLightBasisVector.size());
            L += make_float4(flm.dot(AP * moments));*/
        }
        else if (clippedQuadVertices == 5)
        {
            Vector v0(quadShape[0]);
            Vector v1(quadShape[2]);
            Vector v2(quadShape[1]);
            Triangle t1(v0, v1, v2);

            Eigen::Matrix<float, 15, 1> moments = AxialMoments<Triangle, Vector>(t1, basis, areaLightBasisVector.size());
            L += make_float4(flm.dot(AP * moments));
            rtPrintf("Finish clippedQuadVertices 5.\n");
            Vector v3(quadShape[0]);
            Vector v4(quadShape[3]);
            Vector v5(quadShape[2]);
            Triangle t2(v3, v4, v5);

            moments = AxialMoments<Triangle, Vector>(t2, basis, areaLightBasisVector.size());
            L += make_float4(flm.dot(AP * moments));

            Vector v6(quadShape[0]);
            Vector v7(quadShape[4]);
            Vector v8(quadShape[3]);
            Triangle t3(v6, v7, v8);

            moments = AxialMoments<Triangle, Vector>(t3, basis, areaLightBasisVector.size());
            L += make_float4(flm.dot(AP * moments));
        }
        L *= quadLight.intensity * shaderParams.Reflectance / M_PIf;
        return L;
    }
//#else
    else {
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
                if (!shadow_prd.blocked)
                {
                    /* Compute Ld using MIS weight. */

                    bsdfPdf = Pdf[toUnderlyingValue(shaderParams.bsdfType)](wo_world, outWi, shaderParams);
                    float weight = TwUtil::MonteCarlo::PowerHeuristic(1, lightPdf, 1, bsdfPdf);
                    Ld += f * Li * (fabs(dot(outWi, shaderParams.dgShading.nn)) * weight / lightPdf);
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

        return Ld;
    }
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