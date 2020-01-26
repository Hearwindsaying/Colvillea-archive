#pragma once
#ifndef COLVILLEA_MODULE_SAMPLER_SAMPLER_H_
#define COLVILLEA_MODULE_SAMPLER_SAMPLER_H_

#include "colvillea/Module/Sampler/Sampler.h"

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include "colvillea/Application/TWAssert.h"
#include "colvillea/Device/Toolkit/CommonStructs.h"

/**
 * @brief FiniteSampler is used for tabulated sequences, such as tabulated Sobol
 * and Niederreiter sequence.
 * @note We use 6 dimension Sobol/Niederreiter(over F_5) currently.
 */
class FiniteSampler : public Sampler
{
public:
    /**
     * @brief Factory method for creating a FiniteSampler instance.
     *
     * @param[in] context
     */
    static std::unique_ptr<FiniteSampler> createFiniteSampler(optix::Context context)
    {
        std::unique_ptr<FiniteSampler> finiteSampler = std::make_unique<FiniteSampler>(context);
        finiteSampler->initSampler();
        return finiteSampler;
    }

    FiniteSampler(optix::Context context) : Sampler(context, CommonStructs::SamplerType::FiniteSequenceSampler)
    {

    }

    void initSampler() override
    {
        auto& context = this->m_context;
        auto fsBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 6, 256);
        memcpy(fsBuffer->map(), &FiniteSampler::finiteSequences[0], finiteSequencesSize * sizeof(float));
        fsBuffer->unmap();

        this->m_globalFiniteSampler.finiteSequenceBuffer = fsBuffer->getId();
        context["globalFiniteSampler"]->setUserData(sizeof(CommonStructs::GlobalFiniteSampler), &this->m_globalFiniteSampler);
    }

private:
    using BufferId = int;


    CommonStructs::GlobalFiniteSampler m_globalFiniteSampler;

    static constexpr size_t finiteSequencesSize = 256 * 6;
    static const float finiteSequences[finiteSequencesSize];
};


#endif
