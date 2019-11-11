#pragma once

#include "colvillea/Module/Sampler/Sampler.h"

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include "colvillea/Application/TWAssert.h"
#include "colvillea/Device/Toolkit/CommonStructs.h"

#include <vector>

class Application;

class HaltonSampler : public Sampler
{
public:
    /**
     * @brief Factory method for creating a HaltonSampler instance.
     *
     * @param[in] context
     */
    static std::unique_ptr<HaltonSampler> createHaltonSampler(optix::Context context, optix::int2 filmResolution)
    {
        std::unique_ptr<HaltonSampler> haltonSampler = std::make_unique<HaltonSampler>(context, filmResolution);
        haltonSampler->initSampler();
        return haltonSampler;
    }

    HaltonSampler(optix::Context context, optix::int2 filmResolution) : Sampler(context, CommonStructs::SamplerType::HaltonQMCSampler),
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