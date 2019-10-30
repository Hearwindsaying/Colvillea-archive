#pragma once
#ifndef COLVILLEA_MODULE_SAMPLER_INDEPENDENTSAMPLER_H_
#define COLVILLEA_MODULE_SAMPLER_INDEPENDENTSAMPLER_H_

#include "colvillea/Module/Sampler/Sampler.h"

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include "colvillea/Application/TWAssert.h"
#include "colvillea/Device/Toolkit/CommonStructs.h"

#include <vector>

class Application;

/*
 *@brief Independent Sampler using TEA algorithm to produce
 *random samples.
 **/
class IndependentSampler : public Sampler
{
public:
    /**
     * @brief Factory method for creating a IndependentSampler instance.
     *
     * @param[in] context
     */
    static std::unique_ptr<IndependentSampler> createIndependentSampler(optix::Context context)
    {
        std::unique_ptr<IndependentSampler> independentSampler = std::make_unique<IndependentSampler>(context);
        independentSampler->initSampler();
        return independentSampler;
    }

    IndependentSampler(optix::Context context) : Sampler(context)
    {

    }
};

#endif // COLVILLEA_MODULE_SAMPLER_INDEPENDENTSAMPLER_H_