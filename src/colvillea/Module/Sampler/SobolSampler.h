#pragma once

#include "Sampler.h"

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include "../../Application/TWAssert.h"
#include "../../Device/Toolkit/CommonStructs.h"

class Application;

class SobolSampler : public Sampler
{
public:
    SobolSampler(optix::Context context, optix::int2 filmResolution) : Sampler(context), m_filmResolution(filmResolution)
    {

    }

    void initSampler() override;

private:
    using BufferId = int;

    /**
     * @brief A simple encapsulation for aggregate class 
     * CommonStructs::GlobalSobolSampler.
     * todo:use a varidic template.
     */
    void settingSobolSamplerParameters(int resolution, int log2resolution, BufferId sobolMatrices32, BufferId vdCSobolMatrices, BufferId vdCSobolMatricesInv)
    {
        this->m_globalSobolSampler.resolution          = resolution;
        this->m_globalSobolSampler.log2Resolution      = log2resolution;
        this->m_globalSobolSampler.sobolMatrices32     = sobolMatrices32;
        this->m_globalSobolSampler.vdCSobolMatrices    = vdCSobolMatrices;
        this->m_globalSobolSampler.vdCSobolMatricesInv = vdCSobolMatricesInv;
    }

    inline int32_t roundUpPow2(int32_t v)
    {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        return v + 1;
    }

private:
    optix::int2                       m_filmResolution;    // note that this resolution is not the same as |m_globalSobolSampler.resolution|
    CommonStructs::GlobalSobolSampler m_globalSobolSampler;

    
};