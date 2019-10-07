#pragma once
#ifndef COLVILLEA_DEVICE_SAMPLER_SAMPLER_H_
#define COLVILLEA_DEVICE_SAMPLER_SAMPLER_H_
/*This file is device only.*/

#include <optix_device.h>
#include <optix_world.h>
#include <optixu_math_namespace.h>

#include "../Toolkit/CommonStructs.h"
#include "../Toolkit/Utility.h"

#include "HaltonSampler.h"
#include "SobolSampler.h"

#if 0
//todo:use template parameterize [phase] parameter to reduce branch
enum class Phases :unsigned int
{
    RayGeneration,
    ClosestHit,

    CountOfType
};

template<typename T>
static __device__ __inline__ T makeGPUSampler(const CommonStructs::SamplerType &samplerType)
{
    switch (samplerType)
    {
    case CommonStructs::SamplerType::HaltonQMCSampler:
        return 
    default:
        break;
    }
}

template<Phases phase>
static __device__ __inline__ void makeSampler(const CommonStructs::SamplerType &samplerType);

template<Phases phase=Phases::ClosestHit>
static __device__ __inline__ Sampler makeSampler(const CommonStructs::SamplerType &samplerType)
{
    Sampler localSampler;
    StartSamplingPreprocess_CHit(localSampler);
}

template<Phases phase = Phases::RayGeneration>
static __device__ __inline__ void makeSampler()
{
    Sampler localSampler;
    StartSamplingPreprocess_RayGen(localSampler);
}
#endif // 0

#ifndef TWRT_DELCARE_SAMPLERTYPE
#define TWRT_DELCARE_SAMPLERTYPE
rtDeclareVariable(int, sysSamplerType, , );         /* Sampler type chosen in GPU program.
                                                       --need to fetch underlying value from
                                                         CommonStructs::SamplerType. */
#endif

//#define USE_HALTON_SAMPLER

/************************************************
 *               Sampler Interface              *
 ***********************************************/

/**
 * @brief Make a sampler for GPU program which could be
 * used in RayGeneration phase and ClosestHit phase,
 * respectively.
 * 
 * @note GPUSampler is a union type and we don't return
 * it directly. Take it away as a out parameter.
 * What localSampler contains depends on sysSamplerType.
 * 
 * @param phase ray tracing pipeline phase, set according
 * to where the function is invoked.
 * 
 * @param localSampler [out]GPUSampler
 */
static __device__ __inline__ void makeSampler(CommonStructs::RayTracingPipelinePhase phase, CommonStructs::GPUSampler &localSampler);


/**
 * @brief Fetch the next two dimension value of the given 
 * sample vector and advance to the next dimension.
 *
 * @param localSampler localSampler binding to current pixel
 *
 * @note samplerType parameter is not needed because it's
 * provided as a global sys_ parameter specific to current
 * launch's setting, which is shared by all programs.
 *
 * @return 2D sample value
 */
static __device__ __inline__ optix::float2 Get2D(GPUSampler *localSampler);



/**
 * @brief Fetch the dimension value of the given sample vector 
 * and advance to the next dimension.
 * 
 * @param localSampler localSampler binding to current pixel
 * 
 * @note samplerType parameter is not needed because it's 
 * provided as a global sys_ parameter specific to current 
 * launch's setting, which is shared by all programs.
 * 
 * @return 1D sample value
 */
static __device__ __inline__ float Get1D(GPUSampler *localSampler);




static __device__ __inline__ void makeSampler(CommonStructs::RayTracingPipelinePhase phase, CommonStructs::GPUSampler &localSampler)
{
    switch (static_cast<CommonStructs::SamplerType>(sysSamplerType))
    { 
#ifdef USE_HALTON_SAMPLER /* disable Halton sampler for faster JIT compilation. */
        case CommonStructs::SamplerType::HaltonQMCSampler:
        {
            (phase == RayTracingPipelinePhase::RayGeneration) ? 
                (StartSamplingPreprocess_RayGen(localSampler.haltonSampler)) :
                (StartSamplingPreprocess_CHit(localSampler.haltonSampler));  
        }
        break;
#endif
        case CommonStructs::SamplerType::SobolQMCSampler:
        {
            (phase == RayTracingPipelinePhase::RayGeneration) ?
                (StartSamplingPreprocess_RayGen(localSampler.sobolSampler)) :
                (StartSamplingPreprocess_CHit(localSampler.sobolSampler));
            
        }
        break;

        default:
        {
            rtPrintf("error in makeSampler\n");
        } 
        break;
    }
}

static __device__ __inline__ optix::float2 Get2D(GPUSampler *localSampler)
{
    switch (static_cast<CommonStructs::SamplerType>(sysSamplerType))
    {
#ifdef USE_HALTON_SAMPLER
        case CommonStructs::SamplerType::HaltonQMCSampler:
        {
            return Halton_Get2D(localSampler->haltonSampler);
        }
#endif        
        case CommonStructs::SamplerType::SobolQMCSampler:
        {
            return Sobol_Get2D(localSampler->sobolSampler); 
        }

        default:
        {
            rtPrintf("error in Get2D\n");
        }
        break;
    }
}


static __device__ __inline__ float Get1D(GPUSampler *localSampler)
{
    switch (static_cast<CommonStructs::SamplerType>(sysSamplerType))
    {
#ifdef USE_HALTON_SAMPLER
        case CommonStructs::SamplerType::HaltonQMCSampler:
        {
            return Halton_Get1D(localSampler->haltonSampler);
        }
#endif
        case CommonStructs::SamplerType::SobolQMCSampler:
        {
            return Sobol_Get1D(localSampler->sobolSampler); 
        }

        default:
        {
            rtPrintf("error in Get1D\n");
        }
        break;
    }
}


#endif // COLVILLEA_DEVICE_SAMPLER_SAMPLER_H_