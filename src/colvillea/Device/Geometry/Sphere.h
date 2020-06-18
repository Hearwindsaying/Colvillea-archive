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
}

#endif