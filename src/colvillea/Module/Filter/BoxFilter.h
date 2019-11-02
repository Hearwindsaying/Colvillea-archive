#pragma once
#ifndef COLVILLEA_MODULE_FILTER_BOXFILTER_H_
#define COLVILLEA_MODULE_FILTER_BOXFILTER_H_

#include "colvillea/Module/Filter/Filter.h"

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include "colvillea/Application/TWAssert.h"
#include "colvillea/Device/Toolkit/CommonStructs.h"

class BoxFilter : public Filter
{
public:
    /**
     * @brief Factory method for creating a BoxFilter instance.
     *
     * @param[in] context
     * @param[in] radius Filter radius >= 1.0f. One sample contributes to only one pixel if radius == 1.0f
     */
    static std::unique_ptr<BoxFilter> createBoxFilter(float radius)
    {
        std::unique_ptr<BoxFilter> boxFilter = std::make_unique<BoxFilter>(radius);
        boxFilter->initFilter();
        return boxFilter;
    }

    BoxFilter(float radius) : Filter(CommonStructs::FilterType::BoxFilter)
    {
        this->setRadius(radius);
    }

    void initFilter() override
    {
        /* Note that we can't set m_csBoxFilter to context directly.
         * The host BoxFilter class holds its filter information but
         * -- does not know about what filter is being currently used
         * -- in renderer! */
    }

    CommonStructs::GPUFilter getCommonStructsGPUFilter() const override
    {
        CommonStructs::GPUFilter gpuFilter;
        gpuFilter.boxFilter = this->m_csBoxFilter;
        return gpuFilter;
    }

    void setRadius(float radius)
    {
        if (radius < 1.f)
        {
            std::cout << "[Warning] Filter radius >= 1.f is not satisfied. Setting radius to 1.f instead." << std::endl;
            this->m_csBoxFilter.radius = 1.f;
        }
        else
        {
            this->m_csBoxFilter.radius = radius;
        }
    }

private:
    CommonStructs::BoxFilter m_csBoxFilter;
};

#endif // COLVILLEA_MODULE_FILTER_BOXFILTER_H_