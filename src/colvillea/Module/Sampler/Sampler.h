#pragma once

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include "colvillea/Device/Toolkit/CommonStructs.h"

/**
 * @brief Sampler is a base class for all supported
 * sampler including Quasi Monte-Carlo sampler, 
 * random sampler, etc. It helps initialize sampling
 * method used in GPU program.
 */
class Sampler
{
public:
    Sampler(optix::Context context, CommonStructs::SamplerType samplerType) : m_context(context), m_samplerType(samplerType)
    {

    }

    CommonStructs::SamplerType getSamplerType() const
    {
        return this->m_samplerType;
    }

    virtual void initSampler()
    {

    }

protected:
    optix::Context m_context;
    CommonStructs::SamplerType m_samplerType;
};