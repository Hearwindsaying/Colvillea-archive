#pragma once

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

/**
 * @brief Sampler is a base class for all supported
 * sampler including Quasi Monte-Carlo sampler, 
 * random sampler, etc. It helps initialize sampling
 * method used in GPU program. Unlike other fundamental
 * parts of a rendering system, we don't create but 
 * initialize sampler. todo: Consequently, changing sampler
 * is not permitted.
 */
class Sampler
{
public:
    Sampler(optix::Context context) : m_context(context)
    {

    }

    virtual void initSampler()
    {

    }

protected:
    optix::Context m_context;
};