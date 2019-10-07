#pragma once

#include "Sampler.h"

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include "../../Application/TWAssert.h"
#include "../../Device/Toolkit/CommonStructs.h"

#include <vector>

class Application;

class HaltonSampler : public Sampler
{
public:
    HaltonSampler(optix::Context context, optix::int2 filmResolution) : Sampler(context),
        m_filmResolution(filmResolution)
    {

    }

    void initSampler() override;

private:
    using BufferId = int;


    void setHaltonSamplerParameters(BufferId fastPermutationTable, BufferId hs_offsetForCurrentPixelBuffer)
    {
        this->m_globalHaltonSampler.fastPermutationTable           = fastPermutationTable;
        this->m_globalHaltonSampler.hs_offsetForCurrentPixelBuffer = hs_offsetForCurrentPixelBuffer;
    }


private:
    CommonStructs::GlobalHaltonSampler m_globalHaltonSampler;

    optix::int2 m_filmResolution; // todo: a GlobalSampler class/QMC Sampler class for intermediate.
    optix::int2 m_baseScales;
    optix::int2 m_baseExponents;
    optix::int2 m_multiInverse;
    int         m_sampleStride;

    //std::vector<uint32_t> m_permutationTable; // searches for permutation data and load when needed.
};