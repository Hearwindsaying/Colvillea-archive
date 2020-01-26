#pragma once
#ifndef COLVILLEA_DEVICE_FINITESAMPLER_SAMPLER_H_
#define COLVILLEA_DEVICE_FINITESAMPLER_SAMPLER_H_
/*This file is device only.*/

#include <optix_device.h>
#include <optix_world.h>
#include <optixu_math_namespace.h>

#include "../Toolkit/CommonStructs.h"
#include "../Toolkit/Utility.h"

using namespace optix;
using namespace TwUtil;
using namespace CommonStructs;

rtDeclareVariable(GlobalFiniteSampler, globalFiniteSampler, , );

struct TestBufferWrapper
{
    rtBufferId<float, 2> testBuffer;
};

rtDeclareVariable(TestBufferWrapper, testBufferWrapper, , );

#ifndef TWRT_DECLARE_SYS_ITERATION_INDEX
#define TWRT_DECLARE_SYS_ITERATION_INDEX
rtDeclareVariable(uint, sysIterationIndex, , ) = 0;
#endif

#ifndef TWRT_DECLARE_SYSLAUNCH
#define TWRT_DECLARE_SYSLAUNCH
rtDeclareVariable(uint2, sysLaunch_Dim, rtLaunchDim, );
rtDeclareVariable(uint2, sysLaunch_index, rtLaunchIndex, );
#endif

static __device__ __inline__ optix::float2 Get2D_Finite(FiniteSampler &localSampler)
{
    return make_float2(globalFiniteSampler.finiteSequenceBuffer[make_uint2(localSampler.dimension++,sysIterationIndex)],
        globalFiniteSampler.finiteSequenceBuffer[make_uint2(localSampler.dimension++, sysIterationIndex)]);
}

#endif