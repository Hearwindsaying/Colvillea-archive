#pragma once
#ifndef COLVILLEA_DEVICE_GEOMETRY_SPHERE_H_
#define COLVILLEA_DEVICE_GEOMETRY_SPHERE_H_

/* This file could only be used by device only*/

#include <optix_device.h>
#include <optix_world.h>
#include <optixu_math_namespace.h>

#include "../Toolkit/CommonStructs.h"
#include "../Toolkit/Utility.h"

namespace TwUtil
{
    // Decls:
    __device__ __inline__ float safe_acos(float value);
    __device__ __inline__ void swap(double & lhs, double & rhs);
    __device__ __inline__ bool solveQuadraticDouble(double a, double b, double c, double &x0, double &x1);
    __device__ __inline__ optix::float3 SphericalDirection(float sinTheta, float cosTheta, float phi,
        const optix::float3 &x, const optix::float3 &y, const optix::float3 &z);

    /**
     * @note See also TwUtil::safe_normalize().
     */
    __device__ __inline__ float safe_acos(float value)
    {
        return acosf(fminf(1.0, fmaxf(-1.0, value)));
    }

    __device__ __inline__ void swap(double & lhs, double & rhs)
    {
        float tmp = lhs; lhs = rhs; rhs = tmp;
    }

    /**
     * @ref Mitsuba's util.cpp.
     */
    __device__ __inline__ bool solveQuadraticDouble(double a, double b, double c, double &x0, double &x1)
    {
        /* Linear case */
        if (a == 0) 
        {
            if (b != 0) 
            {
                x0 = x1 = -c / b;
                return true;
            }
            return false;
        }

        double discrim = b * b - 4.0f*a*c;

        /* Leave if there is no solution */
        if (discrim < 0)
            return false;

        double temp, sqrtDiscrim = sqrt(discrim);

        /* Numerically stable version of (-b (+/-) sqrtDiscrim) / (2 * a)
         *
         * Based on the observation that one solution is always
         * accurate while the other is not. Finds the solution of
         * greater magnitude which does not suffer from loss of
         * precision and then uses the identity x1 * x2 = c / a
         */
        if (b < 0)
            temp = -0.5f * (b - sqrtDiscrim);
        else
            temp = -0.5f * (b + sqrtDiscrim);

        x0 = temp / a;
        x1 = c / temp;

        /* Return the results so that x0 < x1 */
        if (x0 > x1)
            TwUtil::swap(x0, x1);

        return true;
    }

    __device__ __inline__ optix::float3 SphericalDirection(float sinTheta, float cosTheta, float phi,
        const optix::float3 &x, const optix::float3 &y, const optix::float3 &z) 
    {
        return sinTheta * cosf(phi) * x + sinTheta * sinf(phi) * y + cosTheta * z;
    }
}


/**
 * @brief Improved sampling of sphere for spherical light sources.
 *
 * @param[in] sphereLight
 * @param[in] p         isectP
 * @param[in] sample    light sample belongs to [0,1]^2
 * @param[out] outPdf   pdf (Solid Angle measure) of sampled point
 * 
 * @note Need to capture 0.0 == outPdf to detect whether |p| lies inside
 * sphere.
 *
 * @return Return sample point in world space.
 */
static __device__ __inline__ optix::float3 Sphere_Sample(const CommonStructs::SphereLight &sphereLight, const optix::float3 &p, const optix::float2 &sample, float *outPdf)
{
    optix::float3 pCenter = TwUtil::xfmPoint(optix::make_float3(0, 0, 0), sphereLight.lightToWorld);

    // Sample uniformly on sphere if $\pt{}$ is inside it
    const optix::float3 &pOrigin = p;
    if (TwUtil::sqr_length(pOrigin-pCenter) <= sphereLight.radius * sphereLight.radius)
    {
        // TODO: support shading point located in sphere
        *outPdf = 0.f;
        return make_float3(0, 0, 0);
    }

    // Sample sphere uniformly inside subtended cone

    // Compute coordinate system for sphere sampling
    float dc = optix::length(p-pCenter);
    float invDc = 1.f / dc;
    optix::float3 wc = (pCenter - p) * invDc;
    optix::float3 wcX, wcY;
    TwUtil::CoordinateSystem(wc, wcX, wcY);

    // Compute $\theta$ and $\phi$ values for sample in cone
    float sinThetaMax = sphereLight.radius * invDc;
    float sinThetaMax2 = sinThetaMax * sinThetaMax;
    float invSinThetaMax = 1 / sinThetaMax;
    float cosThetaMax = sqrtf(optix::fmaxf(0.f, 1 - sinThetaMax2));

    float cosTheta = (cosThetaMax - 1) * sample.x + 1;
    float sinTheta2 = 1 - cosTheta * cosTheta;

    if (sinThetaMax2 < 0.00068523f /* sin^2(1.5 deg) */) 
    {
        /* Fall back to a Taylor series expansion for small angles, where
           the standard approach suffers from severe cancellation errors */
        sinTheta2 = sinThetaMax2 * sample.x;
        cosTheta = sqrtf(1 - sinTheta2);
    }

    // Compute angle $\alpha$ from center of sphere to sampled point on surface
    float cosAlpha = sinTheta2 * invSinThetaMax +
        cosTheta * sqrtf(optix::fmaxf(0.f, 1.f - sinTheta2 * invSinThetaMax * invSinThetaMax));
    float sinAlpha = sqrtf(optix::fmaxf(0.f, 1.f - cosAlpha * cosAlpha));
    float phi = sample.y * 2 * M_PIf;

    // Compute surface normal and sampled point on sphere
    optix::float3 nWorld =
        TwUtil::SphericalDirection(sinAlpha, cosAlpha, phi, -wcX, -wcY, -wc);
    optix::float3 pWorld = pCenter + sphereLight.radius * optix::make_float3(nWorld.x, nWorld.y, nWorld.z);

    // Uniform cone PDF.
    *outPdf = 1 / (2 * M_PIf * (1 - cosThetaMax));

    return pWorld;
}

/**
 * @brief Pdf (Solid Angle measure) of sphere sampling.
 * @note Need to capture 0.0 == outPdf to detect whether |p| lies inside
 * sphere.
 */
static __device__ __inline__ float Sphere_Pdf(const CommonStructs::SphereLight &sphereLight, const optix::float3 &p)
{
    optix::float3 pCenter = TwUtil::xfmPoint(make_float3(0, 0, 0), sphereLight.lightToWorld);
    // Return uniform PDF if point is inside sphere
    const optix::float3 &pOrigin = p;
    // TODO: support case in which |p| lies inside sphere.
    if (TwUtil::sqr_length(pOrigin - pCenter) <= sphereLight.radius * sphereLight.radius)
        return 0.0f;

    // Compute general sphere PDF
    float sinThetaMax2 = sphereLight.radius * sphereLight.radius / TwUtil::sqr_length(p-pCenter);
    float cosThetaMax = sqrtf(optix::fmaxf(0.f, 1 - sinThetaMax2));
    return TwUtil::MonteCarlo::UniformConePdf(cosThetaMax);
}


/**
 * @brief Fast ray-sphere intersection test without filling attributes.
 * This function is useful when a simple ray sphere intersection
 * test is needed when evaluating pdf for SphereLight.
 * It only fills t-parameter, normalized normal and intersection
 * point information, which is faster than regularly rtTrace() calling.
 *
 * @param[in]   ray      ray for intersection
 *
 * @param[out]  outtHit  t parameter for result
 * @param[out]  outnn    normalized normal for quad
 * @param[out]  outpoint intersected point in world space
 *
 * @see Intersect_Sphere()
 *
 * @return whether the ray could intersect the quad plane
 */
static __device__ __inline__ bool Sphere_FastIntersect(const CommonStructs::SphereLight &sphereLight, const optix::Ray &ray, optix::float3 *outpoint)
{
    float3 o = ray.origin - sphereLight.center;
    float3 d = ray.direction;

    double A = TwUtil::sqr_length(d);
    double B = 2 * optix::dot(o, d);
    double C = TwUtil::sqr_length(o) - sphereLight.radius * sphereLight.radius;

    double nearT, farT;
    if (TwUtil::solveQuadraticDouble(A, B, C, nearT, farT))
    {
        if (nearT <= ray.tmax && farT >= ray.tmin)
        {
            if (nearT < ray.tmin)
            {
                if (farT < ray.tmax)
                {
                    // Pick farT
                    *outpoint = ray.origin + farT * ray.direction;
                    return true;
                }
            }
            else
            {
                // Pick nearT
                *outpoint = ray.origin + nearT * ray.direction;
                return true;
            }
        }
    }

    return false;
}

#endif