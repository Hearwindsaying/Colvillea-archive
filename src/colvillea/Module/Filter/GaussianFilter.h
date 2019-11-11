#pragma once
#ifndef COLVILLEA_MODULE_FILTER_GAUSSIANFILTER_H_
#define COLVILLEA_MODULE_FILTER_GAUSSIANFILTER_H_

#include "colvillea/Module/Filter/Filter.h"

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include "colvillea/Application/TWAssert.h"
#include "colvillea/Device/Toolkit/CommonStructs.h"

class GaussianFilter : public Filter
{
public:
    /**
     * @brief Factory method for creating a GaussianFilter instance.
     *
     * @param[in] context
     * @param[in] radius Filter radius >= 0.5f. One sample contributes to only one pixel if radius == 0.5f
     * @param[in] alpha  Gaussian alpha parameter controlling the fallof of the filter. Smaller values cause a slower falloff, giving a blurrier image. Note that small values would lead to floating number precision error.
     */
    static std::unique_ptr<GaussianFilter> createGaussianFilter(float radius, float alpha)
    {
        std::unique_ptr<GaussianFilter> gaussianFilter = std::make_unique<GaussianFilter>(radius, alpha);
        gaussianFilter->initFilter();
        return gaussianFilter;
    }

    GaussianFilter(float radius, float alpha) : Filter(CommonStructs::FilterType::GaussianFilter)
    {
        this->setRadius(radius);
        this->setAlpha(alpha);
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
        gpuFilter.gaussianFilter = this->m_csGaussianFilter;
        return gpuFilter;
    }

    void setRadius(float radius)
    {
        if (radius < 0.5f)
        {
            std::cout << "[Warning] Filter radius >= 0.5f is not satisfied. Setting radius to 0.5f instead." << std::endl;
            this->m_csGaussianFilter.radius = 0.5f;
        }
        else
        {
            this->m_csGaussianFilter.radius = radius;
        }
    }

    float getRadius() const noexcept
    {
        return this->m_csGaussianFilter.radius;
    }

    void setAlpha(float alpha)
    {
        if (alpha <= 1e-5f)
        {
            std::cout << "[Warning] Gaussian alpha <= 1e-5f is not satisfied. Setting to 0.1f instead." << std::endl;
            this->m_csGaussianFilter.alpha = 0.1f;
        }
        else
        {
            this->m_csGaussianFilter.alpha = alpha;
        }

        this->m_csGaussianFilter.gaussianExp = std::exp(-this->m_csGaussianFilter.alpha *
                                                         this->m_csGaussianFilter.radius * this->m_csGaussianFilter.radius);
    }

    float getAlpha() const noexcept
    {
        return this->m_csGaussianFilter.alpha;
    }

private:
    CommonStructs::GaussianFilter m_csGaussianFilter;
};

#endif // COLVILLEA_MODULE_FILTER_GAUSSIANFILTER_H_