#include <optix_world.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix.h>

#include "../Toolkit/CommonStructs.h"
#include "../Toolkit/Utility.h"
#include "../Geometry/Sphere.h"
#include "LightUtil.h"

using namespace optix;
using namespace TwUtil;

#ifndef TWRT_DELCARE_LIGHTBUFFER
#define TWRT_DELCARE_LIGHTBUFFER
rtDeclareVariable(CommonStructs::LightBuffers, sysLightBuffers, , );
#endif

#ifndef TWRT_DECLARE_SYSLAUNCH
#define TWRT_DECLARE_SYSLAUNCH
rtDeclareVariable(uint2, sysLaunch_Dim, rtLaunchDim, );
rtDeclareVariable(uint2, sysLaunch_index, rtLaunchIndex, );
#endif

rtDeclareVariable(float, sysSceneEpsilon, , );




/**
 * @brief Given a shading point |point|, evaluate direct lighting
 * contribution from the specific light implied by |lightBufferIndex|,
 * assuming that there is no occlusive object between the point
 * and light.
 *
 * @param[in]  point            shading point
 * @param[in]  rayEpsilon       epsilon parameter used to make shadow ray
 * @param[in]  lightSample      uniformly distributed light samples
 * @param[in]  lightBufferIndex index to spherelight buffer
 *
 * @param[out] outwi            contribution direction from |point| to the sampled point on light
 * @param[out] outpdf           sampled pdf in Solid Angle measure
 * @param[out] outShadowRay     shadow ray that could be used for testing occlusion
 *
 * @return contribution radiance from light
 */
RT_CALLABLE_PROGRAM float4 Sample_Ld_Sphere(const float3 &point, const float & rayEpsilon, float3 & outwi, float & outpdf, float2 lightSample, uint lightBufferIndex, Ray & outShadowRay)
{
    const CommonStructs::SphereLight &sphereLight = sysLightBuffers.sphereLightBuffer[lightBufferIndex];

    optix::float3 sampledPoint = Sphere_Sample(sphereLight, point, lightSample, &outpdf);
    if (outpdf == 0.f || TwUtil::sqr_length(point - sampledPoint) == 0)
    {
        outpdf = 0.f;
        return optix::make_float4(0.f);
    }

    outwi = TwUtil::safe_normalize(sampledPoint - point);
    outShadowRay = MakeShadowRay(point, rayEpsilon, sampledPoint, 1e-3f);

    return Le_SphereLight(sphereLight, -outwi, sampledPoint);
}


/**
 * @brief Compute pdf of sampling a particular point on sphereLight.
 *
 * @param[in]  p         reference point
 * @param[in]  wi        sampling direction from |p| to sampled point on sphereLight surface
 * @param[in]  lightId   index to the specific sphereLight
 * @param[out] shadowRay shadow ray for detect the occlusion from p to sampled point on light surface
 *
 * @return pdf in solid angle measure
 * @note If orientation test is failed (inconsistent normal and w), pdf=0.0 is returned.
 */
RT_CALLABLE_PROGRAM float LightPdf_Sphere(const float3 & p, const float3 & wi, const int lightId, Ray &shadowRay)
{
    const CommonStructs::SphereLight &sphereLight = sysLightBuffers.sphereLightBuffer[lightId];

    /* Spawn ray from reference point. */
    optix::Ray detectedRay = optix::make_Ray(p, wi, toUnderlyingValue(CommonStructs::RayType::Detection), 1e-3f, RT_DEFAULT_MAX); /* note that RayType::Detection is unnecessarily needed. */

    /* Perform detective intersection test manually.
      -- It's really simple to intersect one single
         specific sphere given a ray so we do not turn
         to OptiX for rtTrace. Meanwhile, it's not
         possible for OptiX to store a buffer of
         rtObject for intersection. */
    float3 sampledPoint = make_float3(0.f);
    if (!Sphere_FastIntersect(sphereLight, detectedRay, &sampledPoint))
    {/* todo: use orientation test (dot) to avoid _FastIntersect. */
        return 0.f;
    }

    // Test orientation.
    if (dot(TwUtil::safe_normalize(sampledPoint - sphereLight.center), 
            -TwUtil::safe_normalize(sampledPoint - p)) < 0.f)
        return 0.f;

    /* Make shadow ray for later detection. */
    shadowRay = MakeShadowRay(p, sysSceneEpsilon, sampledPoint, 1e-3f); // bug:need to review MakeShadowRay

    return Sphere_Pdf(sphereLight, sampledPoint);
}