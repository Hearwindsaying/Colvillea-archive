#pragma once
#ifndef COLVILLEA_DEVICE_TOOLKIT_LIGHTUTIL_H_
#define COLVILLEA_DEVICE_TOOLKIT_LIGHTUTIL_H_

#include "../Toolkit/CommonStructs.h"
#include "../Toolkit/Utility.h"

#ifdef __CUDACC__
#include <optix_device.h>
#endif
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix.h>

namespace TwUtil
{
    /**
     * @brief Evaluate emission radiance for a direction |w| for diffuse
     * quadlight. QuadLight is single-sided so an extra parameter |n|
     * related to quadLight's normal is necessary.
     *
     * @param[in] quadLight  QuadLight
     * @param[in] w          direction w
     *
     * @return Return the intensity of quadLight when |w| and quadLight's
     * orientation is consistent.
     */
    static __device__ __inline__ optix::float4 Le_QuadLight(const CommonStructs::QuadLight &quadLight, const optix::float3 & w)
    {
        /* Note that for xfmNormal() we need to pass in inverse matrix. */
        optix::float3 n = TwUtil::safe_normalize(TwUtil::xfmNormal(
            optix::make_float3(0.f, 0.f, (quadLight.reverseOrientation ? -1.f : 1.f)), quadLight.worldToLight));
        return dot(n, w) > 0.f ? quadLight.intensity : optix::make_float4(0.f);
    }

    /**
     * @brief Evaluate emission radiance for a direction |w| for HDRILight. 
     * 
     * @param[in] w            direction w
     * @param[in] hdriEnvmap   HDRI texture id
     * @param[in] worldToLight transform matrix for applying to HDRILight
     * 
     * @note This could be used in Miss_Default() even no HDRILight is loaded.
     * 
     * @return Return intensity from HDRI texture.
     */
    static __device__ __inline__ optix::float4 Le_HDRILight(const optix::float3 &w, const int &hdriEnvmap, const optix::Matrix4x4 &worldToLight)
    {
        optix::float3 wNormalized = safe_normalize(TwUtil::xfmVector(w, worldToLight));
        return optix::rtTex2D<optix::float4>(hdriEnvmap, TwUtil::sphericalPhi(wNormalized) * M_1_PIf / 2.f, TwUtil::sphericalTheta(wNormalized) * M_1_PIf);
    }

    /**
     * @brief Make a shadow ray given start and end position.
     * 
     * @param[in] point1 start point
     * @param[in] point2 end point
     * @param[in] eps1   start point offset
     * @param[in] eps2   end point offset
     * 
     * @note The returned ray's type is CommonStructs::RayType::Shadow.
     * 
     * @return The created shadow ray.
     */
    static __device__ __inline__ optix::Ray MakeShadowRay(const optix::float3 & point1, float eps1, const optix::float3 & point2, float eps2)
    {
        float dist = distance(point1, point2);

        return optix::make_Ray(point1, (point2 - point1) / dist, toUnderlyingValue(CommonStructs::RayType::Shadow), eps1, dist * (1.f - eps2));
    }

    /**
     * @brief Make a shadow ray given start position and direction.
     *
     * @param[in] point1 start point
     * @param[in] point2 end point
     * @param[in] eps1   start point offset
     * @param[in] eps2   end point offset
     *
     * @note The returned ray's type is CommonStructs::RayType::Shadow.
     *
     * @return The created shadow ray.
     */
    static __device__ __inline__ optix::Ray MakeShadowRay(const optix::float3 & point1, float eps, const optix::float3 & wi)
    {
        return optix::make_Ray(point1, wi, toUnderlyingValue(CommonStructs::RayType::Shadow), eps, RT_DEFAULT_MAX);
    }
}

#endif // COLVILLEA_DEVICE_TOOLKIT_LIGHTUTIL_H_