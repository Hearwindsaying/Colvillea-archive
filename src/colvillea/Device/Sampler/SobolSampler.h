#pragma once
#ifndef COLVILLEA_DEVICE_SOBOLSAMPLER_SAMPLER_H_
#define COLVILLEA_DEVICE_SOBOLSAMPLER_SAMPLER_H_
/*This file is device only.*/

#include <optix_device.h>
#include <optix_world.h>
#include <optixu_math_namespace.h>

#include "../Toolkit/CommonStructs.h"
#include "../Toolkit/Utility.h"

using namespace optix;
using namespace TwUtil;
using namespace CommonStructs;

rtDeclareVariable(GlobalSobolSampler, globalSobolSampler, , );

#ifndef TWRT_DECLARE_SYS_ITERATION_INDEX
#define TWRT_DECLARE_SYS_ITERATION_INDEX
rtDeclareVariable(uint, sysIterationIndex, , ) = 0;
#endif

#ifndef TWRT_DECLARE_SYSLAUNCH
#define TWRT_DECLARE_SYSLAUNCH
rtDeclareVariable(uint2, sysLaunch_Dim, rtLaunchDim, );
rtDeclareVariable(uint2, sysLaunch_index, rtLaunchIndex, );
#endif

#define FloatOneMinusEpsilon 0.99999994f
#define NumSobolDimensions 1024
#define SobolMatrixSize 52

namespace TwUtil
{
	namespace Sampler
	{
		/**
          * @brief Compute Sobol' sample value for a using 32-bits Sobol Matrix.
          * @param a input parameter: index
          * @param dimension input parameter: dimension, 0,1,2...,1023
          * @param scramble scramble value
          * @return return the computed Sobol' sample value
          */
		static __device__ __inline__ float SobolSampleFloat(int64_t a, int dimension, uint32_t scramble) 
		{
			if (dimension >= NumSobolDimensions)
				rtPrintf("Integrator has consumed too many Sobol' dimensions.\n");
			uint32_t v = scramble;
			for (int i = dimension * SobolMatrixSize; a != 0; a >>= 1, i++)
				if (a & 1) v ^= globalSobolSampler.sobolMatrices32[i];

			return fminf(v * 2.3283064365386963e-10f /* 1/2^32 */,
				FloatOneMinusEpsilon);
		}

		/**
		  * @brief Return the index of the frame-th sample falling
          * into the square elementary interval (px, py),
          * without using look-up tables. Note that Sobol' QMC Sampler 
          * uses a sequence that cover the whole film plane so the index
          * could possibly go beyond 32-bit integer.
          * @param m input parameter: log2Resolution m (for one dimension)
          * @param frame input parameter: frame-th sample
          * @param p input parameter: pixel position p
          * @return return the 64-bit computed global index of Sobol' sequence
		  */
		static __device__ __inline__ uint64_t SobolIntervalToIndex(const uint32_t m, uint32_t frame,
			const uint2 &p) 
		{
			if (m == 0) return 0;

			const uint32_t m2 = m << 1;
			uint64_t index = uint64_t(frame) << m2;

			uint64_t delta = 0;
			for (int c = 0; frame; frame >>= 1, ++c)
				if (frame & 1)  // Add flipped column m + c + 1.
					delta ^= globalSobolSampler.vdCSobolMatrices[make_uint2(c, m-1)];

			// flipped b
			uint64_t b = (((uint64_t)(p.x) << m) | (p.y)) ^ delta;

			for (int c = 0; b; b >>= 1, ++c)
				if (b & 1)  // Add column 2 * m - c.
					index ^= globalSobolSampler.vdCSobolMatricesInv[make_uint2(c, m - 1)];

			return index;
		}
	}
}


/**
 * @brief Get index(of the i_th sample) in global Sobol sequence for current pixel
 * @param sampleNum i_th sample in current pixel
 * @param GlobalSobolSampler global Sobol sampler
 * @return return the computed index
 */
static __device__ __inline__ int64_t GetIndexForSample_Sobol(uint32_t sampleNum)
{
	return Sampler::SobolIntervalToIndex(globalSobolSampler.log2Resolution, sampleNum,
		sysLaunch_index);
}

/**
 * @brief compute one element of sample vector of Sobol sequence
 * @param index global index in Sobol sequence
 * @param dim the expected dimension of computed sample vector, 0,1,2...,1023
 * @return Sobol'(b_dimension)(index), without scrambling
 */
static __device__ __inline__ float SampleDimension_Sobol(uint32_t index, int dim)
{
	/*dimension check in SobolSampleFloat*/

	float s = Sampler::SobolSampleFloat(index, dim, 0);

	/*remap samples value for generating pixel samples*/
	if (dim == 0) 
	{
		s *= globalSobolSampler.resolution;
		s = optix::clamp(s - sysLaunch_index.x, 0.f, FloatOneMinusEpsilon);
	}
	else if (dim == 1)
	{
		s *= globalSobolSampler.resolution;
		s = optix::clamp(s - sysLaunch_index.y, 0.f, FloatOneMinusEpsilon);
	}
	return s;
}


/*******************************************************************/
/*                             Interfaces                          */
/*******************************************************************/

/**
 * @brief fetch the next dimension value of the given sample vector and advance dimension
 * @param localSampler localSampler binding to current pixel
 * @param globalSampler globalSampler
 * @return 1D sample value
 */
static __device__ __inline__ float Get1D_Sobol(SobolSampler &localSampler)
{
	return SampleDimension_Sobol(localSampler.intervalSampleIndex, localSampler.dimension++);
}

/**
 * @brief fetch the next two dimension value of the given sample vector and advance dimension
 * @param localSampler localSampler binding to current pixel
 * @param globalSampler globalSampler
 * @return 2D sample value
 */
static __device__ __inline__ optix::float2 Get2D_Sobol(SobolSampler &localSampler)
{
	optix::float2 p = make_float2(SampleDimension_Sobol(localSampler.intervalSampleIndex, 
		                                                localSampler.dimension),
		                          SampleDimension_Sobol(localSampler.intervalSampleIndex,
			                                            localSampler.dimension + 1));
	localSampler.dimension += 2;
	return p;
}

/**
 * @brief Start preprocess of sampling,
 *        reset the current localsampler dimension,
 *        figure out the global sample index for current iteration.
 *        Note that this is RayGen version, indicating that should be used within
 *        RayGeneration Program
 * @param localSampler localSampler binding to current pixel
 * @param globalSampler globalSampler
 */
static __device__ __inline__ void StartSamplingPreprocess_RayGen(SobolSampler &localSampler)
{
	/*next dimension of the sampled vector goes from 0*/
	localSampler.dimension = 0;

	/*retrieve the global index of Sobol sequences for current pixel*/
	uint32_t &currentPixelSampleIndex = sysIterationIndex;
	localSampler.intervalSampleIndex = GetIndexForSample_Sobol(currentPixelSampleIndex);
}

/**
 * @brief Start preprocess of sampling,
 *        reset the current localsampler dimension,
 *        figure out the global sample index for current iteration.
 *        Note that this is CHit version, indicating that should be used within
 *        ClosestHit Program
 * @param localSampler localSampler binding to current pixel
 * @param globalSampler globalSampler
 */
static __device__ __inline__ void StartSamplingPreprocess_CHit(SobolSampler &localSampler)
{
	/*next dimension of the sampled vector goes from 2,
	 *after CameraSample being consumed.*/
	localSampler.dimension = 2;

	/*retrieve the global index of Halton sequences for current pixel*/
	uint32_t &currentPixelSampleIndex = sysIterationIndex;
	localSampler.intervalSampleIndex = GetIndexForSample_Sobol(currentPixelSampleIndex);
}

#endif // COLVILLEA_DEVICE_SOBOLSAMPLER_SAMPLER_H_