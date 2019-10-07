#pragma once
#ifndef COLVILLEA_DEVICE_GEOMETRY_QUAD_H_
#define COLVILLEA_DEVICE_GEOMETRY_QUAD_H_
/* This file is device only. */

#include <optix_device.h>
#include <optix_world.h>
#include <optixu_math_namespace.h>

#include "../Toolkit/CommonStructs.h"
#include "../Toolkit/Utility.h"




/**
 * @brief Uniformly sample quadlight's underlying quad shape.
 * 
 * @param[in] quadLight
 * @param[in] sample    light sample belongs to [0,1]^2
 * @param[out] outPdf   pdf (with respect to Area) of sampled point
 * 
 * @return Return sample point in world space.
 */
static __device__ __inline__ optix::float3 Quad_Sample(const CommonStructs::QuadLight &quadLight, const optix::float2 sample, float *outPdf)
{
    /* Uniformly sample quad area in Area measure. */
    *outPdf = quadLight.invSurfaceArea;

    /* Transform samples from [0,1]*[0,1] to [-1,1]*[-1,1]. */
    return TwUtil::xfmPoint(optix::make_float3(sample.x * 2 - 1.f, sample.y * 2 - 1.f, 0.f), quadLight.lightToWorld);
}

/**
 * @brief Fast ray-quad intersection test without filling attribute.
 * This function is useful when a simple ray rectangle intersection
 * test is needed when evaluating pdf for QuadLight. 
 * It only fills t-parameter, normalized normal and intersection 
 * point information, which is faster than regularly rtTrace() calling.
 * 
 * @param[in]   ray      ray for intersection
 * 
 * @param[out]  outtHit  t parameter for result
 * @param[out]  outnn    normalized normal for quad
 * @param[out]  outpoint intersected point in world space
 * 
 * @see Intersect_Quad()
 * 
 * @return whether the ray could intersect the quad plane
 */
static __device__ __inline__ bool Quad_FastIntersect(const CommonStructs::QuadLight &quadLight, const optix::Ray &ray, float *outtHit, optix::float3 *outnn, optix::float3 *outpoint)
{
    /* Transform world scope ray into local space to perform intersecting. */
    optix::Ray localRay = TwUtil::xfmRay(ray, quadLight.worldToLight);

    /* Compute parameter |t| for intersection. 
     * -- If ray and plane is parallel, tHit is infinity and would 
     * never be a valid intersection, which will be excluded using 
     * |ray.tmax| below.
     * -- Also note that |outtHit| can be set directly as long as
     * we do not normalize |localRay| after tranformation, which is
     * indeed not needed because |ray.direction| is normalized actually.*/
    *outtHit = -localRay.origin.z / localRay.direction.z; 

    if (!(*outtHit >= localRay.tmin && *outtHit <= localRay.tmax))
        return false;

    /* Temporary storage for |localHitPoint|, |outpoint| is in local 
     * space at the moment! */
    *outpoint = localRay.origin + *outtHit * localRay.direction; 

    /* Validate area for quad. */
    if (fabsf(outpoint->x) <= 1 && fabsf(outpoint->y) <= 1)
    {
        /* Normalize transformed normal, which could be affected by |worldToLight|'s scale component. */
        *outnn = TwUtil::safe_normalize(TwUtil::xfmNormal(
            optix::make_float3(0.f, 0.f, (quadLight.reverseOrientation ? -1.f : 1.f)), quadLight.worldToLight));

        /* Calculate intersected point at world space. */
        *outpoint = ray.origin + *outtHit * ray.direction;

        return true;
    }
    return false;
}



#endif // COLVILLEA_DEVICE_GEOMETRY_QUAD_H_