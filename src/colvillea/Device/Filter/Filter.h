#pragma once
#ifndef COLVILLEA_DEVICE_FILTER_FILTER_H_
#define COLVILLEA_DEVICE_FILTER_FILTER_H_

/*This file is device only.*/

#include <optix_device.h>
#include <optix_world.h>
#include <optixu_math_namespace.h>

#include "colvillea/Device/Toolkit/Utility.h"
#include "colvillea/Device/Toolkit/CommonStructs.h"

namespace TwUtil
{
    /*
     * @brief ceilf function for float2
     * @param val input value
     * @return the ceiling float2 number for the given value
     **/
    static __device__ __inline__ float2 ceilf2(const float2 &val)
    {
        return make_float2(ceilf(val.x), ceilf(val.y));
    }

    /*
     * @brief floorf function for float2
     * @param val input value
     * @return the floor float2 number for the given value
     **/
    static __device__ __inline__ float2 floorf2(const float2 &val)
    {
        return make_float2(floorf(val.x), floorf(val.y));
    }

    /*
     * @brief ceil function for float2 and cast to int2
     * @param val input value
     * @return the ceiling float2 number for the given value, after casting to int2
     * @see ceilf2()
     **/
    static __device__ __inline__ int2 ceilf2i(const float2 &val)
    {
        return make_int2(static_cast<int>(ceilf(val.x)), static_cast<int>(ceilf(val.y)));
    }

    /*
     * @brief floor function for float2 and cast to int2
     * @param val input value
     * @return the floor float2 number for the given value, after casting to int2
     * @see floorf2()
     **/
    static __device__ __inline__ int2 floorf2i(const float2 &val)
    {
        return make_int2(static_cast<int>(floorf(val.x)), static_cast<int>(floorf(val.y)));
    }
};

#ifndef TWRT_DECLARE_GPUFILTER
#define TWRT_DECLARE_GPUFILTER
rtDeclareVariable(CommonStructs::GPUFilter, sysGPUFilter, , );
rtDeclareVariable(int,                      sysFilterType, , );
#endif

static __device__ __inline__ float EvaulateFilter_Box(float dx, float dy, CommonStructs::BoxFilter *boxFilter)
{
    return 1.f;
}

static __device__ __inline__ float EvaulateFilter_Gaussian(float dx, float dy, CommonStructs::GaussianFilter *gaussianFilter)
{
    return fmaxf(0.f, expf(-gaussianFilter->alpha * dx * dx) - gaussianFilter->gaussianExp) *
           fmaxf(0.f, expf(-gaussianFilter->alpha * dy * dy) - gaussianFilter->gaussianExp);
}

/**
 * @brief Interface for evaluating filter. Filter type is decided by global
 * |sysGPUFilter|.
 * 
 * @param[in] dx delta position x to filter kernel, with negative value permitted
 * @param[in] dy delta position y to filter kernel, with negative value permitted
 * 
 * @return value by filter evaluation.
 */
static __device__ __inline__ float EvaluateFilter(float dx, float dy)
{
    /* Taking address on variables by rtDeclareVaraible is not supported. */
    CommonStructs::GPUFilter gpuFilter = sysGPUFilter;

    switch (static_cast<CommonStructs::FilterType>(sysFilterType))
    {
    case FilterType::BoxFilter:
        return EvaulateFilter_Box(dx, dy, &gpuFilter.boxFilter);
        break;
    case FilterType::GaussianFilter:
        return EvaulateFilter_Gaussian(dx, dy, &gpuFilter.gaussianFilter);
        break;
    default:
    {
        rtPrintf("Not supported filter type in EvaluateFilter().\n");
        return 0.f;
    } 
        break;
    }
}

/**
 * @brief Get current filter width.
 */
static __device__ __inline__ float GetFilterWidth()
{
    switch (static_cast<CommonStructs::FilterType>(sysFilterType))
    {
    case FilterType::BoxFilter:
        return sysGPUFilter.boxFilter.radius;
    case FilterType::GaussianFilter:
        return sysGPUFilter.gaussianFilter.radius;
        break;
    default:
    {
        rtPrintf("Not supported filter type in EvaluateFilter().\n");
        return 0.f;
    }
        break;
    }
}


#endif // COLVILLEA_DEVICE_FILTER_FILTER_H_