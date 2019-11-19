#pragma once
#ifndef COLVILLEA_MODULE_FILTER_FILTER_H_
#define COLVILLEA_MODULE_FILTER_FILTER_H_

#define  CL_CHECK_MEMORY_LEAKS
#ifdef CL_CHECK_MEMORY_LEAKS
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#define CL_CHECK_MEMORY_LEAKS_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
#define new CL_CHECK_MEMORY_LEAKS_NEW
#endif

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include "colvillea/Device/Toolkit/CommonStructs.h"

/**
 * @brief Filter is a base host class for all supported
 * progressive filters in renderer. 
 */
class Filter
{
public:
    Filter(CommonStructs::FilterType filterType) : m_filterType(filterType)
    {

    }

    CommonStructs::FilterType getFilterType() const
    {
        return this->m_filterType;
    }

    virtual void initFilter() = 0;

    virtual CommonStructs::GPUFilter getCommonStructsGPUFilter() const = 0;

protected:
    /// Filter type
    CommonStructs::FilterType m_filterType;
};


#endif // COLVILLEA_MODULE_FILTER_FILTER_H_