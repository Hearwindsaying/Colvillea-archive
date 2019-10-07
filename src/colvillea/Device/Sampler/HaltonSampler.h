#pragma once
#ifndef COLVILLEA_DEVICE_SAMPLER_HALTONSAMPLER_H_
#define COLVILLEA_DEVICE_SAMPLER_HALTONSAMPLER_H_
/*This file is device only.*/

#include <optix_device.h>
#include <optix_world.h>
#include <optixu_math_namespace.h>

#include "../Toolkit/CommonStructs.h"
#include "../Toolkit/Utility.h"

#include "HaltonSampler_Common.h"

using namespace optix;
using namespace TwUtil;
using namespace CommonStructs;

/*system variables on Sampler:->Context*/
rtDeclareVariable(GlobalHaltonSampler, globalHaltonSampler, , );


#ifndef TWRT_DECLARE_SYS_ITERATION_INDEX
#define TWRT_DECLARE_SYS_ITERATION_INDEX
rtDeclareVariable(uint,     sysIterationIndex, , ) = 0;
#endif

#ifndef TWRT_DECLARE_SYSLAUNCH
#define TWRT_DECLARE_SYSLAUNCH
rtDeclareVariable(uint2, sysLaunch_Dim, rtLaunchDim, );
rtDeclareVariable(uint2, sysLaunch_index, rtLaunchIndex, );
#endif

namespace TwUtil
{
	namespace Sampler
	{
		static __device__ __inline__ uint32_t ReverseBits32(uint32_t n)
		{
			n = (n << 16) | (n >> 16);
			n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
			n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
			n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
			n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
			return n;
		}

		static __device__ __inline__ uint64_t ReverseBits64(uint64_t n)
		{
			uint64_t n0 = ReverseBits32((uint32_t)n);
			uint64_t n1 = ReverseBits32((uint32_t)(n >> 32));
			return (n0 << 32) | n1;
		}

		/*------------Implementation of fast radical inverse-----------*/
		static __device__ __inline__ float HaltonSpecialized3(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[0 + (index % 243u)] * 14348907u +
				globalHaltonSampler.fastPermutationTable[0 + ((index / 243u) % 243u)] * 59049u +
				globalHaltonSampler.fastPermutationTable[0 + ((index / 59049u) % 243u)] * 243u +
				globalHaltonSampler.fastPermutationTable[0 + ((index / 14348907u) % 243u)]) * float(DoubleOneMinusEpsilon / 3486784401u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[243 + (index % 125u)] * 1953125u +
				globalHaltonSampler.fastPermutationTable[243 + ((index / 125u) % 125u)] * 15625u +
				globalHaltonSampler.fastPermutationTable[243 + ((index / 15625u) % 125u)] * 125u +
				globalHaltonSampler.fastPermutationTable[243 + ((index / 1953125u) % 125u)]) * float(DoubleOneMinusEpsilon / 244140625u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[368 + (index % 343u)] * 117649u +
				globalHaltonSampler.fastPermutationTable[368 + ((index / 343u) % 343u)] * 343u +
				globalHaltonSampler.fastPermutationTable[368 + ((index / 117649u) % 343u)]) * float(DoubleOneMinusEpsilon / 40353607u); // Results in [0,1).
		}
#pragma region remainderCases_haltonx


		static __device__ __inline__ float HaltonSpecialized11(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[711 + (index % 121u)] * 1771561u +
				globalHaltonSampler.fastPermutationTable[711 + ((index / 121u) % 121u)] * 14641u +
				globalHaltonSampler.fastPermutationTable[711 + ((index / 14641u) % 121u)] * 121u +
				globalHaltonSampler.fastPermutationTable[711 + ((index / 1771561u) % 121u)]) * float(DoubleOneMinusEpsilon / 214358881u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized13(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[832 + (index % 169u)] * 4826809u +
				globalHaltonSampler.fastPermutationTable[832 + ((index / 169u) % 169u)] * 28561u +
				globalHaltonSampler.fastPermutationTable[832 + ((index / 28561u) % 169u)] * 169u +
				globalHaltonSampler.fastPermutationTable[832 + ((index / 4826809u) % 169u)]) * float(DoubleOneMinusEpsilon / 815730721u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized17(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1001 + (index % 289u)] * 83521u +
				globalHaltonSampler.fastPermutationTable[1001 + ((index / 289u) % 289u)] * 289u +
				globalHaltonSampler.fastPermutationTable[1001 + ((index / 83521u) % 289u)]) * float(DoubleOneMinusEpsilon / 24137569u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized19(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1290 + (index % 361u)] * 130321u +
				globalHaltonSampler.fastPermutationTable[1290 + ((index / 361u) % 361u)] * 361u +
				globalHaltonSampler.fastPermutationTable[1290 + ((index / 130321u) % 361u)]) * float(DoubleOneMinusEpsilon / 47045881u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized23(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1651 + (index % 23u)] * 148035889u +
				globalHaltonSampler.fastPermutationTable[1651 + ((index / 23u) % 23u)] * 6436343u +
				globalHaltonSampler.fastPermutationTable[1651 + ((index / 529u) % 23u)] * 279841u +
				globalHaltonSampler.fastPermutationTable[1651 + ((index / 12167u) % 23u)] * 12167u +
				globalHaltonSampler.fastPermutationTable[1651 + ((index / 279841u) % 23u)] * 529u +
				globalHaltonSampler.fastPermutationTable[1651 + ((index / 6436343u) % 23u)] * 23u +
				globalHaltonSampler.fastPermutationTable[1651 + ((index / 148035889u) % 23u)]) * float(DoubleOneMinusEpsilon / 3404825447u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized29(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1674 + (index % 29u)] * 20511149u +
				globalHaltonSampler.fastPermutationTable[1674 + ((index / 29u) % 29u)] * 707281u +
				globalHaltonSampler.fastPermutationTable[1674 + ((index / 841u) % 29u)] * 24389u +
				globalHaltonSampler.fastPermutationTable[1674 + ((index / 24389u) % 29u)] * 841u +
				globalHaltonSampler.fastPermutationTable[1674 + ((index / 707281u) % 29u)] * 29u +
				globalHaltonSampler.fastPermutationTable[1674 + ((index / 20511149u) % 29u)]) * float(DoubleOneMinusEpsilon / 594823321u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized31(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1703 + (index % 31u)] * 28629151u +
				globalHaltonSampler.fastPermutationTable[1703 + ((index / 31u) % 31u)] * 923521u +
				globalHaltonSampler.fastPermutationTable[1703 + ((index / 961u) % 31u)] * 29791u +
				globalHaltonSampler.fastPermutationTable[1703 + ((index / 29791u) % 31u)] * 961u +
				globalHaltonSampler.fastPermutationTable[1703 + ((index / 923521u) % 31u)] * 31u +
				globalHaltonSampler.fastPermutationTable[1703 + ((index / 28629151u) % 31u)]) * float(DoubleOneMinusEpsilon / 887503681u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized37(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1734 + (index % 37u)] * 69343957u +
				globalHaltonSampler.fastPermutationTable[1734 + ((index / 37u) % 37u)] * 1874161u +
				globalHaltonSampler.fastPermutationTable[1734 + ((index / 1369u) % 37u)] * 50653u +
				globalHaltonSampler.fastPermutationTable[1734 + ((index / 50653u) % 37u)] * 1369u +
				globalHaltonSampler.fastPermutationTable[1734 + ((index / 1874161u) % 37u)] * 37u +
				globalHaltonSampler.fastPermutationTable[1734 + ((index / 69343957u) % 37u)]) * float(DoubleOneMinusEpsilon / 2565726409u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized41(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1771 + (index % 41u)] * 2825761u +
				globalHaltonSampler.fastPermutationTable[1771 + ((index / 41u) % 41u)] * 68921u +
				globalHaltonSampler.fastPermutationTable[1771 + ((index / 1681u) % 41u)] * 1681u +
				globalHaltonSampler.fastPermutationTable[1771 + ((index / 68921u) % 41u)] * 41u +
				globalHaltonSampler.fastPermutationTable[1771 + ((index / 2825761u) % 41u)]) * float(DoubleOneMinusEpsilon / 115856201u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized43(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1812 + (index % 43u)] * 3418801u +
				globalHaltonSampler.fastPermutationTable[1812 + ((index / 43u) % 43u)] * 79507u +
				globalHaltonSampler.fastPermutationTable[1812 + ((index / 1849u) % 43u)] * 1849u +
				globalHaltonSampler.fastPermutationTable[1812 + ((index / 79507u) % 43u)] * 43u +
				globalHaltonSampler.fastPermutationTable[1812 + ((index / 3418801u) % 43u)]) * float(DoubleOneMinusEpsilon / 147008443u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized47(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1855 + (index % 47u)] * 4879681u +
				globalHaltonSampler.fastPermutationTable[1855 + ((index / 47u) % 47u)] * 103823u +
				globalHaltonSampler.fastPermutationTable[1855 + ((index / 2209u) % 47u)] * 2209u +
				globalHaltonSampler.fastPermutationTable[1855 + ((index / 103823u) % 47u)] * 47u +
				globalHaltonSampler.fastPermutationTable[1855 + ((index / 4879681u) % 47u)]) * float(DoubleOneMinusEpsilon / 229345007u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized53(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1902 + (index % 53u)] * 7890481u +
				globalHaltonSampler.fastPermutationTable[1902 + ((index / 53u) % 53u)] * 148877u +
				globalHaltonSampler.fastPermutationTable[1902 + ((index / 2809u) % 53u)] * 2809u +
				globalHaltonSampler.fastPermutationTable[1902 + ((index / 148877u) % 53u)] * 53u +
				globalHaltonSampler.fastPermutationTable[1902 + ((index / 7890481u) % 53u)]) * float(DoubleOneMinusEpsilon / 418195493u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized59(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1955 + (index % 59u)] * 12117361u +
				globalHaltonSampler.fastPermutationTable[1955 + ((index / 59u) % 59u)] * 205379u +
				globalHaltonSampler.fastPermutationTable[1955 + ((index / 3481u) % 59u)] * 3481u +
				globalHaltonSampler.fastPermutationTable[1955 + ((index / 205379u) % 59u)] * 59u +
				globalHaltonSampler.fastPermutationTable[1955 + ((index / 12117361u) % 59u)]) * float(DoubleOneMinusEpsilon / 714924299u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized61(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2014 + (index % 61u)] * 13845841u +
				globalHaltonSampler.fastPermutationTable[2014 + ((index / 61u) % 61u)] * 226981u +
				globalHaltonSampler.fastPermutationTable[2014 + ((index / 3721u) % 61u)] * 3721u +
				globalHaltonSampler.fastPermutationTable[2014 + ((index / 226981u) % 61u)] * 61u +
				globalHaltonSampler.fastPermutationTable[2014 + ((index / 13845841u) % 61u)]) * float(DoubleOneMinusEpsilon / 844596301u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized67(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2075 + (index % 67u)] * 20151121u +
				globalHaltonSampler.fastPermutationTable[2075 + ((index / 67u) % 67u)] * 300763u +
				globalHaltonSampler.fastPermutationTable[2075 + ((index / 4489u) % 67u)] * 4489u +
				globalHaltonSampler.fastPermutationTable[2075 + ((index / 300763u) % 67u)] * 67u +
				globalHaltonSampler.fastPermutationTable[2075 + ((index / 20151121u) % 67u)]) * float(DoubleOneMinusEpsilon / 1350125107u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized71(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2142 + (index % 71u)] * 25411681u +
				globalHaltonSampler.fastPermutationTable[2142 + ((index / 71u) % 71u)] * 357911u +
				globalHaltonSampler.fastPermutationTable[2142 + ((index / 5041u) % 71u)] * 5041u +
				globalHaltonSampler.fastPermutationTable[2142 + ((index / 357911u) % 71u)] * 71u +
				globalHaltonSampler.fastPermutationTable[2142 + ((index / 25411681u) % 71u)]) * float(DoubleOneMinusEpsilon / 1804229351u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized73(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2213 + (index % 73u)] * 28398241u +
				globalHaltonSampler.fastPermutationTable[2213 + ((index / 73u) % 73u)] * 389017u +
				globalHaltonSampler.fastPermutationTable[2213 + ((index / 5329u) % 73u)] * 5329u +
				globalHaltonSampler.fastPermutationTable[2213 + ((index / 389017u) % 73u)] * 73u +
				globalHaltonSampler.fastPermutationTable[2213 + ((index / 28398241u) % 73u)]) * float(DoubleOneMinusEpsilon / 2073071593u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized79(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2286 + (index % 79u)] * 38950081u +
				globalHaltonSampler.fastPermutationTable[2286 + ((index / 79u) % 79u)] * 493039u +
				globalHaltonSampler.fastPermutationTable[2286 + ((index / 6241u) % 79u)] * 6241u +
				globalHaltonSampler.fastPermutationTable[2286 + ((index / 493039u) % 79u)] * 79u +
				globalHaltonSampler.fastPermutationTable[2286 + ((index / 38950081u) % 79u)]) * float(DoubleOneMinusEpsilon / 3077056399u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized83(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2365 + (index % 83u)] * 47458321u +
				globalHaltonSampler.fastPermutationTable[2365 + ((index / 83u) % 83u)] * 571787u +
				globalHaltonSampler.fastPermutationTable[2365 + ((index / 6889u) % 83u)] * 6889u +
				globalHaltonSampler.fastPermutationTable[2365 + ((index / 571787u) % 83u)] * 83u +
				globalHaltonSampler.fastPermutationTable[2365 + ((index / 47458321u) % 83u)]) * float(DoubleOneMinusEpsilon / 3939040643u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized89(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2448 + (index % 89u)] * 704969u +
				globalHaltonSampler.fastPermutationTable[2448 + ((index / 89u) % 89u)] * 7921u +
				globalHaltonSampler.fastPermutationTable[2448 + ((index / 7921u) % 89u)] * 89u +
				globalHaltonSampler.fastPermutationTable[2448 + ((index / 704969u) % 89u)]) * float(DoubleOneMinusEpsilon / 62742241u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized97(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2537 + (index % 97u)] * 912673u +
				globalHaltonSampler.fastPermutationTable[2537 + ((index / 97u) % 97u)] * 9409u +
				globalHaltonSampler.fastPermutationTable[2537 + ((index / 9409u) % 97u)] * 97u +
				globalHaltonSampler.fastPermutationTable[2537 + ((index / 912673u) % 97u)]) * float(DoubleOneMinusEpsilon / 88529281u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized101(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2634 + (index % 101u)] * 1030301u +
				globalHaltonSampler.fastPermutationTable[2634 + ((index / 101u) % 101u)] * 10201u +
				globalHaltonSampler.fastPermutationTable[2634 + ((index / 10201u) % 101u)] * 101u +
				globalHaltonSampler.fastPermutationTable[2634 + ((index / 1030301u) % 101u)]) * float(DoubleOneMinusEpsilon / 104060401u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized103(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2735 + (index % 103u)] * 1092727u +
				globalHaltonSampler.fastPermutationTable[2735 + ((index / 103u) % 103u)] * 10609u +
				globalHaltonSampler.fastPermutationTable[2735 + ((index / 10609u) % 103u)] * 103u +
				globalHaltonSampler.fastPermutationTable[2735 + ((index / 1092727u) % 103u)]) * float(DoubleOneMinusEpsilon / 112550881u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized107(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2838 + (index % 107u)] * 1225043u +
				globalHaltonSampler.fastPermutationTable[2838 + ((index / 107u) % 107u)] * 11449u +
				globalHaltonSampler.fastPermutationTable[2838 + ((index / 11449u) % 107u)] * 107u +
				globalHaltonSampler.fastPermutationTable[2838 + ((index / 1225043u) % 107u)]) * float(DoubleOneMinusEpsilon / 131079601u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized109(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2945 + (index % 109u)] * 1295029u +
				globalHaltonSampler.fastPermutationTable[2945 + ((index / 109u) % 109u)] * 11881u +
				globalHaltonSampler.fastPermutationTable[2945 + ((index / 11881u) % 109u)] * 109u +
				globalHaltonSampler.fastPermutationTable[2945 + ((index / 1295029u) % 109u)]) * float(DoubleOneMinusEpsilon / 141158161u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized113(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3054 + (index % 113u)] * 1442897u +
				globalHaltonSampler.fastPermutationTable[3054 + ((index / 113u) % 113u)] * 12769u +
				globalHaltonSampler.fastPermutationTable[3054 + ((index / 12769u) % 113u)] * 113u +
				globalHaltonSampler.fastPermutationTable[3054 + ((index / 1442897u) % 113u)]) * float(DoubleOneMinusEpsilon / 163047361u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized127(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3167 + (index % 127u)] * 2048383u +
				globalHaltonSampler.fastPermutationTable[3167 + ((index / 127u) % 127u)] * 16129u +
				globalHaltonSampler.fastPermutationTable[3167 + ((index / 16129u) % 127u)] * 127u +
				globalHaltonSampler.fastPermutationTable[3167 + ((index / 2048383u) % 127u)]) * float(DoubleOneMinusEpsilon / 260144641u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized131(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3294 + (index % 131u)] * 2248091u +
				globalHaltonSampler.fastPermutationTable[3294 + ((index / 131u) % 131u)] * 17161u +
				globalHaltonSampler.fastPermutationTable[3294 + ((index / 17161u) % 131u)] * 131u +
				globalHaltonSampler.fastPermutationTable[3294 + ((index / 2248091u) % 131u)]) * float(DoubleOneMinusEpsilon / 294499921u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized137(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3425 + (index % 137u)] * 2571353u +
				globalHaltonSampler.fastPermutationTable[3425 + ((index / 137u) % 137u)] * 18769u +
				globalHaltonSampler.fastPermutationTable[3425 + ((index / 18769u) % 137u)] * 137u +
				globalHaltonSampler.fastPermutationTable[3425 + ((index / 2571353u) % 137u)]) * float(DoubleOneMinusEpsilon / 352275361u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized139(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3562 + (index % 139u)] * 2685619u +
				globalHaltonSampler.fastPermutationTable[3562 + ((index / 139u) % 139u)] * 19321u +
				globalHaltonSampler.fastPermutationTable[3562 + ((index / 19321u) % 139u)] * 139u +
				globalHaltonSampler.fastPermutationTable[3562 + ((index / 2685619u) % 139u)]) * float(DoubleOneMinusEpsilon / 373301041u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized149(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3701 + (index % 149u)] * 3307949u +
				globalHaltonSampler.fastPermutationTable[3701 + ((index / 149u) % 149u)] * 22201u +
				globalHaltonSampler.fastPermutationTable[3701 + ((index / 22201u) % 149u)] * 149u +
				globalHaltonSampler.fastPermutationTable[3701 + ((index / 3307949u) % 149u)]) * float(DoubleOneMinusEpsilon / 492884401u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized151(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3850 + (index % 151u)] * 3442951u +
				globalHaltonSampler.fastPermutationTable[3850 + ((index / 151u) % 151u)] * 22801u +
				globalHaltonSampler.fastPermutationTable[3850 + ((index / 22801u) % 151u)] * 151u +
				globalHaltonSampler.fastPermutationTable[3850 + ((index / 3442951u) % 151u)]) * float(DoubleOneMinusEpsilon / 519885601u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized157(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[4001 + (index % 157u)] * 3869893u +
				globalHaltonSampler.fastPermutationTable[4001 + ((index / 157u) % 157u)] * 24649u +
				globalHaltonSampler.fastPermutationTable[4001 + ((index / 24649u) % 157u)] * 157u +
				globalHaltonSampler.fastPermutationTable[4001 + ((index / 3869893u) % 157u)]) * float(DoubleOneMinusEpsilon / 607573201u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized163(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[4158 + (index % 163u)] * 4330747u +
				globalHaltonSampler.fastPermutationTable[4158 + ((index / 163u) % 163u)] * 26569u +
				globalHaltonSampler.fastPermutationTable[4158 + ((index / 26569u) % 163u)] * 163u +
				globalHaltonSampler.fastPermutationTable[4158 + ((index / 4330747u) % 163u)]) * float(DoubleOneMinusEpsilon / 705911761u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized167(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[4321 + (index % 167u)] * 4657463u +
				globalHaltonSampler.fastPermutationTable[4321 + ((index / 167u) % 167u)] * 27889u +
				globalHaltonSampler.fastPermutationTable[4321 + ((index / 27889u) % 167u)] * 167u +
				globalHaltonSampler.fastPermutationTable[4321 + ((index / 4657463u) % 167u)]) * float(DoubleOneMinusEpsilon / 777796321u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized173(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[4488 + (index % 173u)] * 5177717u +
				globalHaltonSampler.fastPermutationTable[4488 + ((index / 173u) % 173u)] * 29929u +
				globalHaltonSampler.fastPermutationTable[4488 + ((index / 29929u) % 173u)] * 173u +
				globalHaltonSampler.fastPermutationTable[4488 + ((index / 5177717u) % 173u)]) * float(DoubleOneMinusEpsilon / 895745041u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized179(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[4661 + (index % 179u)] * 5735339u +
				globalHaltonSampler.fastPermutationTable[4661 + ((index / 179u) % 179u)] * 32041u +
				globalHaltonSampler.fastPermutationTable[4661 + ((index / 32041u) % 179u)] * 179u +
				globalHaltonSampler.fastPermutationTable[4661 + ((index / 5735339u) % 179u)]) * float(DoubleOneMinusEpsilon / 1026625681u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized181(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[4840 + (index % 181u)] * 5929741u +
				globalHaltonSampler.fastPermutationTable[4840 + ((index / 181u) % 181u)] * 32761u +
				globalHaltonSampler.fastPermutationTable[4840 + ((index / 32761u) % 181u)] * 181u +
				globalHaltonSampler.fastPermutationTable[4840 + ((index / 5929741u) % 181u)]) * float(DoubleOneMinusEpsilon / 1073283121u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized191(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[5021 + (index % 191u)] * 6967871u +
				globalHaltonSampler.fastPermutationTable[5021 + ((index / 191u) % 191u)] * 36481u +
				globalHaltonSampler.fastPermutationTable[5021 + ((index / 36481u) % 191u)] * 191u +
				globalHaltonSampler.fastPermutationTable[5021 + ((index / 6967871u) % 191u)]) * float(DoubleOneMinusEpsilon / 1330863361u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized193(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[5212 + (index % 193u)] * 7189057u +
				globalHaltonSampler.fastPermutationTable[5212 + ((index / 193u) % 193u)] * 37249u +
				globalHaltonSampler.fastPermutationTable[5212 + ((index / 37249u) % 193u)] * 193u +
				globalHaltonSampler.fastPermutationTable[5212 + ((index / 7189057u) % 193u)]) * float(DoubleOneMinusEpsilon / 1387488001u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized197(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[5405 + (index % 197u)] * 7645373u +
				globalHaltonSampler.fastPermutationTable[5405 + ((index / 197u) % 197u)] * 38809u +
				globalHaltonSampler.fastPermutationTable[5405 + ((index / 38809u) % 197u)] * 197u +
				globalHaltonSampler.fastPermutationTable[5405 + ((index / 7645373u) % 197u)]) * float(DoubleOneMinusEpsilon / 1506138481u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized199(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[5602 + (index % 199u)] * 7880599u +
				globalHaltonSampler.fastPermutationTable[5602 + ((index / 199u) % 199u)] * 39601u +
				globalHaltonSampler.fastPermutationTable[5602 + ((index / 39601u) % 199u)] * 199u +
				globalHaltonSampler.fastPermutationTable[5602 + ((index / 7880599u) % 199u)]) * float(DoubleOneMinusEpsilon / 1568239201u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized211(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[5801 + (index % 211u)] * 9393931u +
				globalHaltonSampler.fastPermutationTable[5801 + ((index / 211u) % 211u)] * 44521u +
				globalHaltonSampler.fastPermutationTable[5801 + ((index / 44521u) % 211u)] * 211u +
				globalHaltonSampler.fastPermutationTable[5801 + ((index / 9393931u) % 211u)]) * float(DoubleOneMinusEpsilon / 1982119441u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized223(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[6012 + (index % 223u)] * 11089567u +
				globalHaltonSampler.fastPermutationTable[6012 + ((index / 223u) % 223u)] * 49729u +
				globalHaltonSampler.fastPermutationTable[6012 + ((index / 49729u) % 223u)] * 223u +
				globalHaltonSampler.fastPermutationTable[6012 + ((index / 11089567u) % 223u)]) * float(DoubleOneMinusEpsilon / 2472973441u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized227(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[6235 + (index % 227u)] * 11697083u +
				globalHaltonSampler.fastPermutationTable[6235 + ((index / 227u) % 227u)] * 51529u +
				globalHaltonSampler.fastPermutationTable[6235 + ((index / 51529u) % 227u)] * 227u +
				globalHaltonSampler.fastPermutationTable[6235 + ((index / 11697083u) % 227u)]) * float(DoubleOneMinusEpsilon / 2655237841u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized229(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[6462 + (index % 229u)] * 12008989u +
				globalHaltonSampler.fastPermutationTable[6462 + ((index / 229u) % 229u)] * 52441u +
				globalHaltonSampler.fastPermutationTable[6462 + ((index / 52441u) % 229u)] * 229u +
				globalHaltonSampler.fastPermutationTable[6462 + ((index / 12008989u) % 229u)]) * float(DoubleOneMinusEpsilon / 2750058481u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized233(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[6691 + (index % 233u)] * 12649337u +
				globalHaltonSampler.fastPermutationTable[6691 + ((index / 233u) % 233u)] * 54289u +
				globalHaltonSampler.fastPermutationTable[6691 + ((index / 54289u) % 233u)] * 233u +
				globalHaltonSampler.fastPermutationTable[6691 + ((index / 12649337u) % 233u)]) * float(DoubleOneMinusEpsilon / 2947295521u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized239(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[6924 + (index % 239u)] * 13651919u +
				globalHaltonSampler.fastPermutationTable[6924 + ((index / 239u) % 239u)] * 57121u +
				globalHaltonSampler.fastPermutationTable[6924 + ((index / 57121u) % 239u)] * 239u +
				globalHaltonSampler.fastPermutationTable[6924 + ((index / 13651919u) % 239u)]) * float(DoubleOneMinusEpsilon / 3262808641u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized241(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[7163 + (index % 241u)] * 13997521u +
				globalHaltonSampler.fastPermutationTable[7163 + ((index / 241u) % 241u)] * 58081u +
				globalHaltonSampler.fastPermutationTable[7163 + ((index / 58081u) % 241u)] * 241u +
				globalHaltonSampler.fastPermutationTable[7163 + ((index / 13997521u) % 241u)]) * float(DoubleOneMinusEpsilon / 3373402561u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized251(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[7404 + (index % 251u)] * 15813251u +
				globalHaltonSampler.fastPermutationTable[7404 + ((index / 251u) % 251u)] * 63001u +
				globalHaltonSampler.fastPermutationTable[7404 + ((index / 63001u) % 251u)] * 251u +
				globalHaltonSampler.fastPermutationTable[7404 + ((index / 15813251u) % 251u)]) * float(DoubleOneMinusEpsilon / 3969126001u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized257(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[7655 + (index % 257u)] * 66049u +
				globalHaltonSampler.fastPermutationTable[7655 + ((index / 257u) % 257u)] * 257u +
				globalHaltonSampler.fastPermutationTable[7655 + ((index / 66049u) % 257u)]) * float(DoubleOneMinusEpsilon / 16974593u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized263(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[7912 + (index % 263u)] * 69169u +
				globalHaltonSampler.fastPermutationTable[7912 + ((index / 263u) % 263u)] * 263u +
				globalHaltonSampler.fastPermutationTable[7912 + ((index / 69169u) % 263u)]) * float(DoubleOneMinusEpsilon / 18191447u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized269(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[8175 + (index % 269u)] * 72361u +
				globalHaltonSampler.fastPermutationTable[8175 + ((index / 269u) % 269u)] * 269u +
				globalHaltonSampler.fastPermutationTable[8175 + ((index / 72361u) % 269u)]) * float(DoubleOneMinusEpsilon / 19465109u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized271(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[8444 + (index % 271u)] * 73441u +
				globalHaltonSampler.fastPermutationTable[8444 + ((index / 271u) % 271u)] * 271u +
				globalHaltonSampler.fastPermutationTable[8444 + ((index / 73441u) % 271u)]) * float(DoubleOneMinusEpsilon / 19902511u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized277(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[8715 + (index % 277u)] * 76729u +
				globalHaltonSampler.fastPermutationTable[8715 + ((index / 277u) % 277u)] * 277u +
				globalHaltonSampler.fastPermutationTable[8715 + ((index / 76729u) % 277u)]) * float(DoubleOneMinusEpsilon / 21253933u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized281(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[8992 + (index % 281u)] * 78961u +
				globalHaltonSampler.fastPermutationTable[8992 + ((index / 281u) % 281u)] * 281u +
				globalHaltonSampler.fastPermutationTable[8992 + ((index / 78961u) % 281u)]) * float(DoubleOneMinusEpsilon / 22188041u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized283(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[9273 + (index % 283u)] * 80089u +
				globalHaltonSampler.fastPermutationTable[9273 + ((index / 283u) % 283u)] * 283u +
				globalHaltonSampler.fastPermutationTable[9273 + ((index / 80089u) % 283u)]) * float(DoubleOneMinusEpsilon / 22665187u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized293(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[9556 + (index % 293u)] * 85849u +
				globalHaltonSampler.fastPermutationTable[9556 + ((index / 293u) % 293u)] * 293u +
				globalHaltonSampler.fastPermutationTable[9556 + ((index / 85849u) % 293u)]) * float(DoubleOneMinusEpsilon / 25153757u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized307(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[9849 + (index % 307u)] * 94249u +
				globalHaltonSampler.fastPermutationTable[9849 + ((index / 307u) % 307u)] * 307u +
				globalHaltonSampler.fastPermutationTable[9849 + ((index / 94249u) % 307u)]) * float(DoubleOneMinusEpsilon / 28934443u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized311(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[10156 + (index % 311u)] * 96721u +
				globalHaltonSampler.fastPermutationTable[10156 + ((index / 311u) % 311u)] * 311u +
				globalHaltonSampler.fastPermutationTable[10156 + ((index / 96721u) % 311u)]) * float(DoubleOneMinusEpsilon / 30080231u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized313(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[10467 + (index % 313u)] * 97969u +
				globalHaltonSampler.fastPermutationTable[10467 + ((index / 313u) % 313u)] * 313u +
				globalHaltonSampler.fastPermutationTable[10467 + ((index / 97969u) % 313u)]) * float(DoubleOneMinusEpsilon / 30664297u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized317(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[10780 + (index % 317u)] * 100489u +
				globalHaltonSampler.fastPermutationTable[10780 + ((index / 317u) % 317u)] * 317u +
				globalHaltonSampler.fastPermutationTable[10780 + ((index / 100489u) % 317u)]) * float(DoubleOneMinusEpsilon / 31855013u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized331(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[11097 + (index % 331u)] * 109561u +
				globalHaltonSampler.fastPermutationTable[11097 + ((index / 331u) % 331u)] * 331u +
				globalHaltonSampler.fastPermutationTable[11097 + ((index / 109561u) % 331u)]) * float(DoubleOneMinusEpsilon / 36264691u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized337(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[11428 + (index % 337u)] * 113569u +
				globalHaltonSampler.fastPermutationTable[11428 + ((index / 337u) % 337u)] * 337u +
				globalHaltonSampler.fastPermutationTable[11428 + ((index / 113569u) % 337u)]) * float(DoubleOneMinusEpsilon / 38272753u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized347(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[11765 + (index % 347u)] * 120409u +
				globalHaltonSampler.fastPermutationTable[11765 + ((index / 347u) % 347u)] * 347u +
				globalHaltonSampler.fastPermutationTable[11765 + ((index / 120409u) % 347u)]) * float(DoubleOneMinusEpsilon / 41781923u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized349(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[12112 + (index % 349u)] * 121801u +
				globalHaltonSampler.fastPermutationTable[12112 + ((index / 349u) % 349u)] * 349u +
				globalHaltonSampler.fastPermutationTable[12112 + ((index / 121801u) % 349u)]) * float(DoubleOneMinusEpsilon / 42508549u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized353(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[12461 + (index % 353u)] * 124609u +
				globalHaltonSampler.fastPermutationTable[12461 + ((index / 353u) % 353u)] * 353u +
				globalHaltonSampler.fastPermutationTable[12461 + ((index / 124609u) % 353u)]) * float(DoubleOneMinusEpsilon / 43986977u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized359(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[12814 + (index % 359u)] * 128881u +
				globalHaltonSampler.fastPermutationTable[12814 + ((index / 359u) % 359u)] * 359u +
				globalHaltonSampler.fastPermutationTable[12814 + ((index / 128881u) % 359u)]) * float(DoubleOneMinusEpsilon / 46268279u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized367(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[13173 + (index % 367u)] * 134689u +
				globalHaltonSampler.fastPermutationTable[13173 + ((index / 367u) % 367u)] * 367u +
				globalHaltonSampler.fastPermutationTable[13173 + ((index / 134689u) % 367u)]) * float(DoubleOneMinusEpsilon / 49430863u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized373(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[13540 + (index % 373u)] * 139129u +
				globalHaltonSampler.fastPermutationTable[13540 + ((index / 373u) % 373u)] * 373u +
				globalHaltonSampler.fastPermutationTable[13540 + ((index / 139129u) % 373u)]) * float(DoubleOneMinusEpsilon / 51895117u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized379(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[13913 + (index % 379u)] * 143641u +
				globalHaltonSampler.fastPermutationTable[13913 + ((index / 379u) % 379u)] * 379u +
				globalHaltonSampler.fastPermutationTable[13913 + ((index / 143641u) % 379u)]) * float(DoubleOneMinusEpsilon / 54439939u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized383(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[14292 + (index % 383u)] * 146689u +
				globalHaltonSampler.fastPermutationTable[14292 + ((index / 383u) % 383u)] * 383u +
				globalHaltonSampler.fastPermutationTable[14292 + ((index / 146689u) % 383u)]) * float(DoubleOneMinusEpsilon / 56181887u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized389(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[14675 + (index % 389u)] * 151321u +
				globalHaltonSampler.fastPermutationTable[14675 + ((index / 389u) % 389u)] * 389u +
				globalHaltonSampler.fastPermutationTable[14675 + ((index / 151321u) % 389u)]) * float(DoubleOneMinusEpsilon / 58863869u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized397(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[15064 + (index % 397u)] * 157609u +
				globalHaltonSampler.fastPermutationTable[15064 + ((index / 397u) % 397u)] * 397u +
				globalHaltonSampler.fastPermutationTable[15064 + ((index / 157609u) % 397u)]) * float(DoubleOneMinusEpsilon / 62570773u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized401(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[15461 + (index % 401u)] * 160801u +
				globalHaltonSampler.fastPermutationTable[15461 + ((index / 401u) % 401u)] * 401u +
				globalHaltonSampler.fastPermutationTable[15461 + ((index / 160801u) % 401u)]) * float(DoubleOneMinusEpsilon / 64481201u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized409(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[15862 + (index % 409u)] * 167281u +
				globalHaltonSampler.fastPermutationTable[15862 + ((index / 409u) % 409u)] * 409u +
				globalHaltonSampler.fastPermutationTable[15862 + ((index / 167281u) % 409u)]) * float(DoubleOneMinusEpsilon / 68417929u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized419(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[16271 + (index % 419u)] * 175561u +
				globalHaltonSampler.fastPermutationTable[16271 + ((index / 419u) % 419u)] * 419u +
				globalHaltonSampler.fastPermutationTable[16271 + ((index / 175561u) % 419u)]) * float(DoubleOneMinusEpsilon / 73560059u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized421(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[16690 + (index % 421u)] * 177241u +
				globalHaltonSampler.fastPermutationTable[16690 + ((index / 421u) % 421u)] * 421u +
				globalHaltonSampler.fastPermutationTable[16690 + ((index / 177241u) % 421u)]) * float(DoubleOneMinusEpsilon / 74618461u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized431(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[17111 + (index % 431u)] * 185761u +
				globalHaltonSampler.fastPermutationTable[17111 + ((index / 431u) % 431u)] * 431u +
				globalHaltonSampler.fastPermutationTable[17111 + ((index / 185761u) % 431u)]) * float(DoubleOneMinusEpsilon / 80062991u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized433(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[17542 + (index % 433u)] * 187489u +
				globalHaltonSampler.fastPermutationTable[17542 + ((index / 433u) % 433u)] * 433u +
				globalHaltonSampler.fastPermutationTable[17542 + ((index / 187489u) % 433u)]) * float(DoubleOneMinusEpsilon / 81182737u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized439(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[17975 + (index % 439u)] * 192721u +
				globalHaltonSampler.fastPermutationTable[17975 + ((index / 439u) % 439u)] * 439u +
				globalHaltonSampler.fastPermutationTable[17975 + ((index / 192721u) % 439u)]) * float(DoubleOneMinusEpsilon / 84604519u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized443(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[18414 + (index % 443u)] * 196249u +
				globalHaltonSampler.fastPermutationTable[18414 + ((index / 443u) % 443u)] * 443u +
				globalHaltonSampler.fastPermutationTable[18414 + ((index / 196249u) % 443u)]) * float(DoubleOneMinusEpsilon / 86938307u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized449(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[18857 + (index % 449u)] * 201601u +
				globalHaltonSampler.fastPermutationTable[18857 + ((index / 449u) % 449u)] * 449u +
				globalHaltonSampler.fastPermutationTable[18857 + ((index / 201601u) % 449u)]) * float(DoubleOneMinusEpsilon / 90518849u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized457(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[19306 + (index % 457u)] * 208849u +
				globalHaltonSampler.fastPermutationTable[19306 + ((index / 457u) % 457u)] * 457u +
				globalHaltonSampler.fastPermutationTable[19306 + ((index / 208849u) % 457u)]) * float(DoubleOneMinusEpsilon / 95443993u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized461(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[19763 + (index % 461u)] * 212521u +
				globalHaltonSampler.fastPermutationTable[19763 + ((index / 461u) % 461u)] * 461u +
				globalHaltonSampler.fastPermutationTable[19763 + ((index / 212521u) % 461u)]) * float(DoubleOneMinusEpsilon / 97972181u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized463(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[20224 + (index % 463u)] * 214369u +
				globalHaltonSampler.fastPermutationTable[20224 + ((index / 463u) % 463u)] * 463u +
				globalHaltonSampler.fastPermutationTable[20224 + ((index / 214369u) % 463u)]) * float(DoubleOneMinusEpsilon / 99252847u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized467(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[20687 + (index % 467u)] * 218089u +
				globalHaltonSampler.fastPermutationTable[20687 + ((index / 467u) % 467u)] * 467u +
				globalHaltonSampler.fastPermutationTable[20687 + ((index / 218089u) % 467u)]) * float(DoubleOneMinusEpsilon / 101847563u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized479(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[21154 + (index % 479u)] * 229441u +
				globalHaltonSampler.fastPermutationTable[21154 + ((index / 479u) % 479u)] * 479u +
				globalHaltonSampler.fastPermutationTable[21154 + ((index / 229441u) % 479u)]) * float(DoubleOneMinusEpsilon / 109902239u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized487(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[21633 + (index % 487u)] * 237169u +
				globalHaltonSampler.fastPermutationTable[21633 + ((index / 487u) % 487u)] * 487u +
				globalHaltonSampler.fastPermutationTable[21633 + ((index / 237169u) % 487u)]) * float(DoubleOneMinusEpsilon / 115501303u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized491(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[22120 + (index % 491u)] * 241081u +
				globalHaltonSampler.fastPermutationTable[22120 + ((index / 491u) % 491u)] * 491u +
				globalHaltonSampler.fastPermutationTable[22120 + ((index / 241081u) % 491u)]) * float(DoubleOneMinusEpsilon / 118370771u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized499(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[22611 + (index % 499u)] * 249001u +
				globalHaltonSampler.fastPermutationTable[22611 + ((index / 499u) % 499u)] * 499u +
				globalHaltonSampler.fastPermutationTable[22611 + ((index / 249001u) % 499u)]) * float(DoubleOneMinusEpsilon / 124251499u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized503(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[23110 + (index % 503u)] * 253009u +
				globalHaltonSampler.fastPermutationTable[23110 + ((index / 503u) % 503u)] * 503u +
				globalHaltonSampler.fastPermutationTable[23110 + ((index / 253009u) % 503u)]) * float(DoubleOneMinusEpsilon / 127263527u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized509(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[23613 + (index % 509u)] * 259081u +
				globalHaltonSampler.fastPermutationTable[23613 + ((index / 509u) % 509u)] * 509u +
				globalHaltonSampler.fastPermutationTable[23613 + ((index / 259081u) % 509u)]) * float(DoubleOneMinusEpsilon / 131872229u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized521(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[24122 + (index % 521u)] * 271441u +
				globalHaltonSampler.fastPermutationTable[24122 + ((index / 521u) % 521u)] * 521u +
				globalHaltonSampler.fastPermutationTable[24122 + ((index / 271441u) % 521u)]) * float(DoubleOneMinusEpsilon / 141420761u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized523(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[24643 + (index % 523u)] * 273529u +
				globalHaltonSampler.fastPermutationTable[24643 + ((index / 523u) % 523u)] * 523u +
				globalHaltonSampler.fastPermutationTable[24643 + ((index / 273529u) % 523u)]) * float(DoubleOneMinusEpsilon / 143055667u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized541(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[25166 + (index % 541u)] * 292681u +
				globalHaltonSampler.fastPermutationTable[25166 + ((index / 541u) % 541u)] * 541u +
				globalHaltonSampler.fastPermutationTable[25166 + ((index / 292681u) % 541u)]) * float(DoubleOneMinusEpsilon / 158340421u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized547(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[25707 + (index % 547u)] * 299209u +
				globalHaltonSampler.fastPermutationTable[25707 + ((index / 547u) % 547u)] * 547u +
				globalHaltonSampler.fastPermutationTable[25707 + ((index / 299209u) % 547u)]) * float(DoubleOneMinusEpsilon / 163667323u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized557(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[26254 + (index % 557u)] * 310249u +
				globalHaltonSampler.fastPermutationTable[26254 + ((index / 557u) % 557u)] * 557u +
				globalHaltonSampler.fastPermutationTable[26254 + ((index / 310249u) % 557u)]) * float(DoubleOneMinusEpsilon / 172808693u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized563(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[26811 + (index % 563u)] * 316969u +
				globalHaltonSampler.fastPermutationTable[26811 + ((index / 563u) % 563u)] * 563u +
				globalHaltonSampler.fastPermutationTable[26811 + ((index / 316969u) % 563u)]) * float(DoubleOneMinusEpsilon / 178453547u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized569(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[27374 + (index % 569u)] * 323761u +
				globalHaltonSampler.fastPermutationTable[27374 + ((index / 569u) % 569u)] * 569u +
				globalHaltonSampler.fastPermutationTable[27374 + ((index / 323761u) % 569u)]) * float(DoubleOneMinusEpsilon / 184220009u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized571(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[27943 + (index % 571u)] * 326041u +
				globalHaltonSampler.fastPermutationTable[27943 + ((index / 571u) % 571u)] * 571u +
				globalHaltonSampler.fastPermutationTable[27943 + ((index / 326041u) % 571u)]) * float(DoubleOneMinusEpsilon / 186169411u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized577(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[28514 + (index % 577u)] * 332929u +
				globalHaltonSampler.fastPermutationTable[28514 + ((index / 577u) % 577u)] * 577u +
				globalHaltonSampler.fastPermutationTable[28514 + ((index / 332929u) % 577u)]) * float(DoubleOneMinusEpsilon / 192100033u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized587(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[29091 + (index % 587u)] * 344569u +
				globalHaltonSampler.fastPermutationTable[29091 + ((index / 587u) % 587u)] * 587u +
				globalHaltonSampler.fastPermutationTable[29091 + ((index / 344569u) % 587u)]) * float(DoubleOneMinusEpsilon / 202262003u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized593(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[29678 + (index % 593u)] * 351649u +
				globalHaltonSampler.fastPermutationTable[29678 + ((index / 593u) % 593u)] * 593u +
				globalHaltonSampler.fastPermutationTable[29678 + ((index / 351649u) % 593u)]) * float(DoubleOneMinusEpsilon / 208527857u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized599(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[30271 + (index % 599u)] * 358801u +
				globalHaltonSampler.fastPermutationTable[30271 + ((index / 599u) % 599u)] * 599u +
				globalHaltonSampler.fastPermutationTable[30271 + ((index / 358801u) % 599u)]) * float(DoubleOneMinusEpsilon / 214921799u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized601(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[30870 + (index % 601u)] * 361201u +
				globalHaltonSampler.fastPermutationTable[30870 + ((index / 601u) % 601u)] * 601u +
				globalHaltonSampler.fastPermutationTable[30870 + ((index / 361201u) % 601u)]) * float(DoubleOneMinusEpsilon / 217081801u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized607(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[31471 + (index % 607u)] * 368449u +
				globalHaltonSampler.fastPermutationTable[31471 + ((index / 607u) % 607u)] * 607u +
				globalHaltonSampler.fastPermutationTable[31471 + ((index / 368449u) % 607u)]) * float(DoubleOneMinusEpsilon / 223648543u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized613(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[32078 + (index % 613u)] * 375769u +
				globalHaltonSampler.fastPermutationTable[32078 + ((index / 613u) % 613u)] * 613u +
				globalHaltonSampler.fastPermutationTable[32078 + ((index / 375769u) % 613u)]) * float(DoubleOneMinusEpsilon / 230346397u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized617(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[32691 + (index % 617u)] * 380689u +
				globalHaltonSampler.fastPermutationTable[32691 + ((index / 617u) % 617u)] * 617u +
				globalHaltonSampler.fastPermutationTable[32691 + ((index / 380689u) % 617u)]) * float(DoubleOneMinusEpsilon / 234885113u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized619(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[33308 + (index % 619u)] * 383161u +
				globalHaltonSampler.fastPermutationTable[33308 + ((index / 619u) % 619u)] * 619u +
				globalHaltonSampler.fastPermutationTable[33308 + ((index / 383161u) % 619u)]) * float(DoubleOneMinusEpsilon / 237176659u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized631(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[33927 + (index % 631u)] * 398161u +
				globalHaltonSampler.fastPermutationTable[33927 + ((index / 631u) % 631u)] * 631u +
				globalHaltonSampler.fastPermutationTable[33927 + ((index / 398161u) % 631u)]) * float(DoubleOneMinusEpsilon / 251239591u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized641(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[34558 + (index % 641u)] * 410881u +
				globalHaltonSampler.fastPermutationTable[34558 + ((index / 641u) % 641u)] * 641u +
				globalHaltonSampler.fastPermutationTable[34558 + ((index / 410881u) % 641u)]) * float(DoubleOneMinusEpsilon / 263374721u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized643(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[35199 + (index % 643u)] * 413449u +
				globalHaltonSampler.fastPermutationTable[35199 + ((index / 643u) % 643u)] * 643u +
				globalHaltonSampler.fastPermutationTable[35199 + ((index / 413449u) % 643u)]) * float(DoubleOneMinusEpsilon / 265847707u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized647(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[35842 + (index % 647u)] * 418609u +
				globalHaltonSampler.fastPermutationTable[35842 + ((index / 647u) % 647u)] * 647u +
				globalHaltonSampler.fastPermutationTable[35842 + ((index / 418609u) % 647u)]) * float(DoubleOneMinusEpsilon / 270840023u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized653(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[36489 + (index % 653u)] * 426409u +
				globalHaltonSampler.fastPermutationTable[36489 + ((index / 653u) % 653u)] * 653u +
				globalHaltonSampler.fastPermutationTable[36489 + ((index / 426409u) % 653u)]) * float(DoubleOneMinusEpsilon / 278445077u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized659(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[37142 + (index % 659u)] * 434281u +
				globalHaltonSampler.fastPermutationTable[37142 + ((index / 659u) % 659u)] * 659u +
				globalHaltonSampler.fastPermutationTable[37142 + ((index / 434281u) % 659u)]) * float(DoubleOneMinusEpsilon / 286191179u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized661(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[37801 + (index % 661u)] * 436921u +
				globalHaltonSampler.fastPermutationTable[37801 + ((index / 661u) % 661u)] * 661u +
				globalHaltonSampler.fastPermutationTable[37801 + ((index / 436921u) % 661u)]) * float(DoubleOneMinusEpsilon / 288804781u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized673(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[38462 + (index % 673u)] * 452929u +
				globalHaltonSampler.fastPermutationTable[38462 + ((index / 673u) % 673u)] * 673u +
				globalHaltonSampler.fastPermutationTable[38462 + ((index / 452929u) % 673u)]) * float(DoubleOneMinusEpsilon / 304821217u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized677(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[39135 + (index % 677u)] * 458329u +
				globalHaltonSampler.fastPermutationTable[39135 + ((index / 677u) % 677u)] * 677u +
				globalHaltonSampler.fastPermutationTable[39135 + ((index / 458329u) % 677u)]) * float(DoubleOneMinusEpsilon / 310288733u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized683(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[39812 + (index % 683u)] * 466489u +
				globalHaltonSampler.fastPermutationTable[39812 + ((index / 683u) % 683u)] * 683u +
				globalHaltonSampler.fastPermutationTable[39812 + ((index / 466489u) % 683u)]) * float(DoubleOneMinusEpsilon / 318611987u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized691(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[40495 + (index % 691u)] * 477481u +
				globalHaltonSampler.fastPermutationTable[40495 + ((index / 691u) % 691u)] * 691u +
				globalHaltonSampler.fastPermutationTable[40495 + ((index / 477481u) % 691u)]) * float(DoubleOneMinusEpsilon / 329939371u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized701(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[41186 + (index % 701u)] * 491401u +
				globalHaltonSampler.fastPermutationTable[41186 + ((index / 701u) % 701u)] * 701u +
				globalHaltonSampler.fastPermutationTable[41186 + ((index / 491401u) % 701u)]) * float(DoubleOneMinusEpsilon / 344472101u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized709(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[41887 + (index % 709u)] * 502681u +
				globalHaltonSampler.fastPermutationTable[41887 + ((index / 709u) % 709u)] * 709u +
				globalHaltonSampler.fastPermutationTable[41887 + ((index / 502681u) % 709u)]) * float(DoubleOneMinusEpsilon / 356400829u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized719(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[42596 + (index % 719u)] * 516961u +
				globalHaltonSampler.fastPermutationTable[42596 + ((index / 719u) % 719u)] * 719u +
				globalHaltonSampler.fastPermutationTable[42596 + ((index / 516961u) % 719u)]) * float(DoubleOneMinusEpsilon / 371694959u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized727(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[43315 + (index % 727u)] * 528529u +
				globalHaltonSampler.fastPermutationTable[43315 + ((index / 727u) % 727u)] * 727u +
				globalHaltonSampler.fastPermutationTable[43315 + ((index / 528529u) % 727u)]) * float(DoubleOneMinusEpsilon / 384240583u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized733(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[44042 + (index % 733u)] * 537289u +
				globalHaltonSampler.fastPermutationTable[44042 + ((index / 733u) % 733u)] * 733u +
				globalHaltonSampler.fastPermutationTable[44042 + ((index / 537289u) % 733u)]) * float(DoubleOneMinusEpsilon / 393832837u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized739(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[44775 + (index % 739u)] * 546121u +
				globalHaltonSampler.fastPermutationTable[44775 + ((index / 739u) % 739u)] * 739u +
				globalHaltonSampler.fastPermutationTable[44775 + ((index / 546121u) % 739u)]) * float(DoubleOneMinusEpsilon / 403583419u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized743(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[45514 + (index % 743u)] * 552049u +
				globalHaltonSampler.fastPermutationTable[45514 + ((index / 743u) % 743u)] * 743u +
				globalHaltonSampler.fastPermutationTable[45514 + ((index / 552049u) % 743u)]) * float(DoubleOneMinusEpsilon / 410172407u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized751(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[46257 + (index % 751u)] * 564001u +
				globalHaltonSampler.fastPermutationTable[46257 + ((index / 751u) % 751u)] * 751u +
				globalHaltonSampler.fastPermutationTable[46257 + ((index / 564001u) % 751u)]) * float(DoubleOneMinusEpsilon / 423564751u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized757(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[47008 + (index % 757u)] * 573049u +
				globalHaltonSampler.fastPermutationTable[47008 + ((index / 757u) % 757u)] * 757u +
				globalHaltonSampler.fastPermutationTable[47008 + ((index / 573049u) % 757u)]) * float(DoubleOneMinusEpsilon / 433798093u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized761(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[47765 + (index % 761u)] * 579121u +
				globalHaltonSampler.fastPermutationTable[47765 + ((index / 761u) % 761u)] * 761u +
				globalHaltonSampler.fastPermutationTable[47765 + ((index / 579121u) % 761u)]) * float(DoubleOneMinusEpsilon / 440711081u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized769(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[48526 + (index % 769u)] * 591361u +
				globalHaltonSampler.fastPermutationTable[48526 + ((index / 769u) % 769u)] * 769u +
				globalHaltonSampler.fastPermutationTable[48526 + ((index / 591361u) % 769u)]) * float(DoubleOneMinusEpsilon / 454756609u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized773(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[49295 + (index % 773u)] * 597529u +
				globalHaltonSampler.fastPermutationTable[49295 + ((index / 773u) % 773u)] * 773u +
				globalHaltonSampler.fastPermutationTable[49295 + ((index / 597529u) % 773u)]) * float(DoubleOneMinusEpsilon / 461889917u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized787(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[50068 + (index % 787u)] * 619369u +
				globalHaltonSampler.fastPermutationTable[50068 + ((index / 787u) % 787u)] * 787u +
				globalHaltonSampler.fastPermutationTable[50068 + ((index / 619369u) % 787u)]) * float(DoubleOneMinusEpsilon / 487443403u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized797(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[50855 + (index % 797u)] * 635209u +
				globalHaltonSampler.fastPermutationTable[50855 + ((index / 797u) % 797u)] * 797u +
				globalHaltonSampler.fastPermutationTable[50855 + ((index / 635209u) % 797u)]) * float(DoubleOneMinusEpsilon / 506261573u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized809(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[51652 + (index % 809u)] * 654481u +
				globalHaltonSampler.fastPermutationTable[51652 + ((index / 809u) % 809u)] * 809u +
				globalHaltonSampler.fastPermutationTable[51652 + ((index / 654481u) % 809u)]) * float(DoubleOneMinusEpsilon / 529475129u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized811(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[52461 + (index % 811u)] * 657721u +
				globalHaltonSampler.fastPermutationTable[52461 + ((index / 811u) % 811u)] * 811u +
				globalHaltonSampler.fastPermutationTable[52461 + ((index / 657721u) % 811u)]) * float(DoubleOneMinusEpsilon / 533411731u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized821(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[53272 + (index % 821u)] * 674041u +
				globalHaltonSampler.fastPermutationTable[53272 + ((index / 821u) % 821u)] * 821u +
				globalHaltonSampler.fastPermutationTable[53272 + ((index / 674041u) % 821u)]) * float(DoubleOneMinusEpsilon / 553387661u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized823(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[54093 + (index % 823u)] * 677329u +
				globalHaltonSampler.fastPermutationTable[54093 + ((index / 823u) % 823u)] * 823u +
				globalHaltonSampler.fastPermutationTable[54093 + ((index / 677329u) % 823u)]) * float(DoubleOneMinusEpsilon / 557441767u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized827(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[54916 + (index % 827u)] * 683929u +
				globalHaltonSampler.fastPermutationTable[54916 + ((index / 827u) % 827u)] * 827u +
				globalHaltonSampler.fastPermutationTable[54916 + ((index / 683929u) % 827u)]) * float(DoubleOneMinusEpsilon / 565609283u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized829(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[55743 + (index % 829u)] * 687241u +
				globalHaltonSampler.fastPermutationTable[55743 + ((index / 829u) % 829u)] * 829u +
				globalHaltonSampler.fastPermutationTable[55743 + ((index / 687241u) % 829u)]) * float(DoubleOneMinusEpsilon / 569722789u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized839(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[56572 + (index % 839u)] * 703921u +
				globalHaltonSampler.fastPermutationTable[56572 + ((index / 839u) % 839u)] * 839u +
				globalHaltonSampler.fastPermutationTable[56572 + ((index / 703921u) % 839u)]) * float(DoubleOneMinusEpsilon / 590589719u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized853(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[57411 + (index % 853u)] * 727609u +
				globalHaltonSampler.fastPermutationTable[57411 + ((index / 853u) % 853u)] * 853u +
				globalHaltonSampler.fastPermutationTable[57411 + ((index / 727609u) % 853u)]) * float(DoubleOneMinusEpsilon / 620650477u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized857(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[58264 + (index % 857u)] * 734449u +
				globalHaltonSampler.fastPermutationTable[58264 + ((index / 857u) % 857u)] * 857u +
				globalHaltonSampler.fastPermutationTable[58264 + ((index / 734449u) % 857u)]) * float(DoubleOneMinusEpsilon / 629422793u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized859(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[59121 + (index % 859u)] * 737881u +
				globalHaltonSampler.fastPermutationTable[59121 + ((index / 859u) % 859u)] * 859u +
				globalHaltonSampler.fastPermutationTable[59121 + ((index / 737881u) % 859u)]) * float(DoubleOneMinusEpsilon / 633839779u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized863(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[59980 + (index % 863u)] * 744769u +
				globalHaltonSampler.fastPermutationTable[59980 + ((index / 863u) % 863u)] * 863u +
				globalHaltonSampler.fastPermutationTable[59980 + ((index / 744769u) % 863u)]) * float(DoubleOneMinusEpsilon / 642735647u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized877(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[60843 + (index % 877u)] * 769129u +
				globalHaltonSampler.fastPermutationTable[60843 + ((index / 877u) % 877u)] * 877u +
				globalHaltonSampler.fastPermutationTable[60843 + ((index / 769129u) % 877u)]) * float(DoubleOneMinusEpsilon / 674526133u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized881(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[61720 + (index % 881u)] * 776161u +
				globalHaltonSampler.fastPermutationTable[61720 + ((index / 881u) % 881u)] * 881u +
				globalHaltonSampler.fastPermutationTable[61720 + ((index / 776161u) % 881u)]) * float(DoubleOneMinusEpsilon / 683797841u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized883(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[62601 + (index % 883u)] * 779689u +
				globalHaltonSampler.fastPermutationTable[62601 + ((index / 883u) % 883u)] * 883u +
				globalHaltonSampler.fastPermutationTable[62601 + ((index / 779689u) % 883u)]) * float(DoubleOneMinusEpsilon / 688465387u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized887(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[63484 + (index % 887u)] * 786769u +
				globalHaltonSampler.fastPermutationTable[63484 + ((index / 887u) % 887u)] * 887u +
				globalHaltonSampler.fastPermutationTable[63484 + ((index / 786769u) % 887u)]) * float(DoubleOneMinusEpsilon / 697864103u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized907(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[64371 + (index % 907u)] * 822649u +
				globalHaltonSampler.fastPermutationTable[64371 + ((index / 907u) % 907u)] * 907u +
				globalHaltonSampler.fastPermutationTable[64371 + ((index / 822649u) % 907u)]) * float(DoubleOneMinusEpsilon / 746142643u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized911(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[65278 + (index % 911u)] * 829921u +
				globalHaltonSampler.fastPermutationTable[65278 + ((index / 911u) % 911u)] * 911u +
				globalHaltonSampler.fastPermutationTable[65278 + ((index / 829921u) % 911u)]) * float(DoubleOneMinusEpsilon / 756058031u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized919(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[66189 + (index % 919u)] * 844561u +
				globalHaltonSampler.fastPermutationTable[66189 + ((index / 919u) % 919u)] * 919u +
				globalHaltonSampler.fastPermutationTable[66189 + ((index / 844561u) % 919u)]) * float(DoubleOneMinusEpsilon / 776151559u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized929(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[67108 + (index % 929u)] * 863041u +
				globalHaltonSampler.fastPermutationTable[67108 + ((index / 929u) % 929u)] * 929u +
				globalHaltonSampler.fastPermutationTable[67108 + ((index / 863041u) % 929u)]) * float(DoubleOneMinusEpsilon / 801765089u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized937(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[68037 + (index % 937u)] * 877969u +
				globalHaltonSampler.fastPermutationTable[68037 + ((index / 937u) % 937u)] * 937u +
				globalHaltonSampler.fastPermutationTable[68037 + ((index / 877969u) % 937u)]) * float(DoubleOneMinusEpsilon / 822656953u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized941(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[68974 + (index % 941u)] * 885481u +
				globalHaltonSampler.fastPermutationTable[68974 + ((index / 941u) % 941u)] * 941u +
				globalHaltonSampler.fastPermutationTable[68974 + ((index / 885481u) % 941u)]) * float(DoubleOneMinusEpsilon / 833237621u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized947(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[69915 + (index % 947u)] * 896809u +
				globalHaltonSampler.fastPermutationTable[69915 + ((index / 947u) % 947u)] * 947u +
				globalHaltonSampler.fastPermutationTable[69915 + ((index / 896809u) % 947u)]) * float(DoubleOneMinusEpsilon / 849278123u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized953(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[70862 + (index % 953u)] * 908209u +
				globalHaltonSampler.fastPermutationTable[70862 + ((index / 953u) % 953u)] * 953u +
				globalHaltonSampler.fastPermutationTable[70862 + ((index / 908209u) % 953u)]) * float(DoubleOneMinusEpsilon / 865523177u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized967(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[71815 + (index % 967u)] * 935089u +
				globalHaltonSampler.fastPermutationTable[71815 + ((index / 967u) % 967u)] * 967u +
				globalHaltonSampler.fastPermutationTable[71815 + ((index / 935089u) % 967u)]) * float(DoubleOneMinusEpsilon / 904231063u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized971(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[72782 + (index % 971u)] * 942841u +
				globalHaltonSampler.fastPermutationTable[72782 + ((index / 971u) % 971u)] * 971u +
				globalHaltonSampler.fastPermutationTable[72782 + ((index / 942841u) % 971u)]) * float(DoubleOneMinusEpsilon / 915498611u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized977(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[73753 + (index % 977u)] * 954529u +
				globalHaltonSampler.fastPermutationTable[73753 + ((index / 977u) % 977u)] * 977u +
				globalHaltonSampler.fastPermutationTable[73753 + ((index / 954529u) % 977u)]) * float(DoubleOneMinusEpsilon / 932574833u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized983(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[74730 + (index % 983u)] * 966289u +
				globalHaltonSampler.fastPermutationTable[74730 + ((index / 983u) % 983u)] * 983u +
				globalHaltonSampler.fastPermutationTable[74730 + ((index / 966289u) % 983u)]) * float(DoubleOneMinusEpsilon / 949862087u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized991(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[75713 + (index % 991u)] * 982081u +
				globalHaltonSampler.fastPermutationTable[75713 + ((index / 991u) % 991u)] * 991u +
				globalHaltonSampler.fastPermutationTable[75713 + ((index / 982081u) % 991u)]) * float(DoubleOneMinusEpsilon / 973242271u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized997(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[76704 + (index % 997u)] * 994009u +
				globalHaltonSampler.fastPermutationTable[76704 + ((index / 997u) % 997u)] * 997u +
				globalHaltonSampler.fastPermutationTable[76704 + ((index / 994009u) % 997u)]) * float(DoubleOneMinusEpsilon / 991026973u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1009(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[77701 + (index % 1009u)] * 1018081u +
				globalHaltonSampler.fastPermutationTable[77701 + ((index / 1009u) % 1009u)] * 1009u +
				globalHaltonSampler.fastPermutationTable[77701 + ((index / 1018081u) % 1009u)]) * float(DoubleOneMinusEpsilon / 1027243729u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1013(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[78710 + (index % 1013u)] * 1026169u +
				globalHaltonSampler.fastPermutationTable[78710 + ((index / 1013u) % 1013u)] * 1013u +
				globalHaltonSampler.fastPermutationTable[78710 + ((index / 1026169u) % 1013u)]) * float(DoubleOneMinusEpsilon / 1039509197u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1019(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[79723 + (index % 1019u)] * 1038361u +
				globalHaltonSampler.fastPermutationTable[79723 + ((index / 1019u) % 1019u)] * 1019u +
				globalHaltonSampler.fastPermutationTable[79723 + ((index / 1038361u) % 1019u)]) * float(DoubleOneMinusEpsilon / 1058089859u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1021(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[80742 + (index % 1021u)] * 1042441u +
				globalHaltonSampler.fastPermutationTable[80742 + ((index / 1021u) % 1021u)] * 1021u +
				globalHaltonSampler.fastPermutationTable[80742 + ((index / 1042441u) % 1021u)]) * float(DoubleOneMinusEpsilon / 1064332261u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1031(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[81763 + (index % 1031u)] * 1062961u +
				globalHaltonSampler.fastPermutationTable[81763 + ((index / 1031u) % 1031u)] * 1031u +
				globalHaltonSampler.fastPermutationTable[81763 + ((index / 1062961u) % 1031u)]) * float(DoubleOneMinusEpsilon / 1095912791u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1033(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[82794 + (index % 1033u)] * 1067089u +
				globalHaltonSampler.fastPermutationTable[82794 + ((index / 1033u) % 1033u)] * 1033u +
				globalHaltonSampler.fastPermutationTable[82794 + ((index / 1067089u) % 1033u)]) * float(DoubleOneMinusEpsilon / 1102302937u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1039(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[83827 + (index % 1039u)] * 1079521u +
				globalHaltonSampler.fastPermutationTable[83827 + ((index / 1039u) % 1039u)] * 1039u +
				globalHaltonSampler.fastPermutationTable[83827 + ((index / 1079521u) % 1039u)]) * float(DoubleOneMinusEpsilon / 1121622319u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1049(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[84866 + (index % 1049u)] * 1100401u +
				globalHaltonSampler.fastPermutationTable[84866 + ((index / 1049u) % 1049u)] * 1049u +
				globalHaltonSampler.fastPermutationTable[84866 + ((index / 1100401u) % 1049u)]) * float(DoubleOneMinusEpsilon / 1154320649u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1051(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[85915 + (index % 1051u)] * 1104601u +
				globalHaltonSampler.fastPermutationTable[85915 + ((index / 1051u) % 1051u)] * 1051u +
				globalHaltonSampler.fastPermutationTable[85915 + ((index / 1104601u) % 1051u)]) * float(DoubleOneMinusEpsilon / 1160935651u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1061(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[86966 + (index % 1061u)] * 1125721u +
				globalHaltonSampler.fastPermutationTable[86966 + ((index / 1061u) % 1061u)] * 1061u +
				globalHaltonSampler.fastPermutationTable[86966 + ((index / 1125721u) % 1061u)]) * float(DoubleOneMinusEpsilon / 1194389981u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1063(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[88027 + (index % 1063u)] * 1129969u +
				globalHaltonSampler.fastPermutationTable[88027 + ((index / 1063u) % 1063u)] * 1063u +
				globalHaltonSampler.fastPermutationTable[88027 + ((index / 1129969u) % 1063u)]) * float(DoubleOneMinusEpsilon / 1201157047u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1069(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[89090 + (index % 1069u)] * 1142761u +
				globalHaltonSampler.fastPermutationTable[89090 + ((index / 1069u) % 1069u)] * 1069u +
				globalHaltonSampler.fastPermutationTable[89090 + ((index / 1142761u) % 1069u)]) * float(DoubleOneMinusEpsilon / 1221611509u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1087(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[90159 + (index % 1087u)] * 1181569u +
				globalHaltonSampler.fastPermutationTable[90159 + ((index / 1087u) % 1087u)] * 1087u +
				globalHaltonSampler.fastPermutationTable[90159 + ((index / 1181569u) % 1087u)]) * float(DoubleOneMinusEpsilon / 1284365503u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1091(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[91246 + (index % 1091u)] * 1190281u +
				globalHaltonSampler.fastPermutationTable[91246 + ((index / 1091u) % 1091u)] * 1091u +
				globalHaltonSampler.fastPermutationTable[91246 + ((index / 1190281u) % 1091u)]) * float(DoubleOneMinusEpsilon / 1298596571u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1093(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[92337 + (index % 1093u)] * 1194649u +
				globalHaltonSampler.fastPermutationTable[92337 + ((index / 1093u) % 1093u)] * 1093u +
				globalHaltonSampler.fastPermutationTable[92337 + ((index / 1194649u) % 1093u)]) * float(DoubleOneMinusEpsilon / 1305751357u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1097(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[93430 + (index % 1097u)] * 1203409u +
				globalHaltonSampler.fastPermutationTable[93430 + ((index / 1097u) % 1097u)] * 1097u +
				globalHaltonSampler.fastPermutationTable[93430 + ((index / 1203409u) % 1097u)]) * float(DoubleOneMinusEpsilon / 1320139673u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1103(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[94527 + (index % 1103u)] * 1216609u +
				globalHaltonSampler.fastPermutationTable[94527 + ((index / 1103u) % 1103u)] * 1103u +
				globalHaltonSampler.fastPermutationTable[94527 + ((index / 1216609u) % 1103u)]) * float(DoubleOneMinusEpsilon / 1341919727u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1109(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[95630 + (index % 1109u)] * 1229881u +
				globalHaltonSampler.fastPermutationTable[95630 + ((index / 1109u) % 1109u)] * 1109u +
				globalHaltonSampler.fastPermutationTable[95630 + ((index / 1229881u) % 1109u)]) * float(DoubleOneMinusEpsilon / 1363938029u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1117(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[96739 + (index % 1117u)] * 1247689u +
				globalHaltonSampler.fastPermutationTable[96739 + ((index / 1117u) % 1117u)] * 1117u +
				globalHaltonSampler.fastPermutationTable[96739 + ((index / 1247689u) % 1117u)]) * float(DoubleOneMinusEpsilon / 1393668613u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1123(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[97856 + (index % 1123u)] * 1261129u +
				globalHaltonSampler.fastPermutationTable[97856 + ((index / 1123u) % 1123u)] * 1123u +
				globalHaltonSampler.fastPermutationTable[97856 + ((index / 1261129u) % 1123u)]) * float(DoubleOneMinusEpsilon / 1416247867u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1129(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[98979 + (index % 1129u)] * 1274641u +
				globalHaltonSampler.fastPermutationTable[98979 + ((index / 1129u) % 1129u)] * 1129u +
				globalHaltonSampler.fastPermutationTable[98979 + ((index / 1274641u) % 1129u)]) * float(DoubleOneMinusEpsilon / 1439069689u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1151(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[100108 + (index % 1151u)] * 1324801u +
				globalHaltonSampler.fastPermutationTable[100108 + ((index / 1151u) % 1151u)] * 1151u +
				globalHaltonSampler.fastPermutationTable[100108 + ((index / 1324801u) % 1151u)]) * float(DoubleOneMinusEpsilon / 1524845951u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1153(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[101259 + (index % 1153u)] * 1329409u +
				globalHaltonSampler.fastPermutationTable[101259 + ((index / 1153u) % 1153u)] * 1153u +
				globalHaltonSampler.fastPermutationTable[101259 + ((index / 1329409u) % 1153u)]) * float(DoubleOneMinusEpsilon / 1532808577u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1163(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[102412 + (index % 1163u)] * 1352569u +
				globalHaltonSampler.fastPermutationTable[102412 + ((index / 1163u) % 1163u)] * 1163u +
				globalHaltonSampler.fastPermutationTable[102412 + ((index / 1352569u) % 1163u)]) * float(DoubleOneMinusEpsilon / 1573037747u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1171(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[103575 + (index % 1171u)] * 1371241u +
				globalHaltonSampler.fastPermutationTable[103575 + ((index / 1171u) % 1171u)] * 1171u +
				globalHaltonSampler.fastPermutationTable[103575 + ((index / 1371241u) % 1171u)]) * float(DoubleOneMinusEpsilon / 1605723211u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1181(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[104746 + (index % 1181u)] * 1394761u +
				globalHaltonSampler.fastPermutationTable[104746 + ((index / 1181u) % 1181u)] * 1181u +
				globalHaltonSampler.fastPermutationTable[104746 + ((index / 1394761u) % 1181u)]) * float(DoubleOneMinusEpsilon / 1647212741u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1187(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[105927 + (index % 1187u)] * 1408969u +
				globalHaltonSampler.fastPermutationTable[105927 + ((index / 1187u) % 1187u)] * 1187u +
				globalHaltonSampler.fastPermutationTable[105927 + ((index / 1408969u) % 1187u)]) * float(DoubleOneMinusEpsilon / 1672446203u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1193(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[107114 + (index % 1193u)] * 1423249u +
				globalHaltonSampler.fastPermutationTable[107114 + ((index / 1193u) % 1193u)] * 1193u +
				globalHaltonSampler.fastPermutationTable[107114 + ((index / 1423249u) % 1193u)]) * float(DoubleOneMinusEpsilon / 1697936057u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1201(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[108307 + (index % 1201u)] * 1442401u +
				globalHaltonSampler.fastPermutationTable[108307 + ((index / 1201u) % 1201u)] * 1201u +
				globalHaltonSampler.fastPermutationTable[108307 + ((index / 1442401u) % 1201u)]) * float(DoubleOneMinusEpsilon / 1732323601u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1213(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[109508 + (index % 1213u)] * 1471369u +
				globalHaltonSampler.fastPermutationTable[109508 + ((index / 1213u) % 1213u)] * 1213u +
				globalHaltonSampler.fastPermutationTable[109508 + ((index / 1471369u) % 1213u)]) * float(DoubleOneMinusEpsilon / 1784770597u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1217(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[110721 + (index % 1217u)] * 1481089u +
				globalHaltonSampler.fastPermutationTable[110721 + ((index / 1217u) % 1217u)] * 1217u +
				globalHaltonSampler.fastPermutationTable[110721 + ((index / 1481089u) % 1217u)]) * float(DoubleOneMinusEpsilon / 1802485313u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1223(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[111938 + (index % 1223u)] * 1495729u +
				globalHaltonSampler.fastPermutationTable[111938 + ((index / 1223u) % 1223u)] * 1223u +
				globalHaltonSampler.fastPermutationTable[111938 + ((index / 1495729u) % 1223u)]) * float(DoubleOneMinusEpsilon / 1829276567u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1229(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[113161 + (index % 1229u)] * 1510441u +
				globalHaltonSampler.fastPermutationTable[113161 + ((index / 1229u) % 1229u)] * 1229u +
				globalHaltonSampler.fastPermutationTable[113161 + ((index / 1510441u) % 1229u)]) * float(DoubleOneMinusEpsilon / 1856331989u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1231(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[114390 + (index % 1231u)] * 1515361u +
				globalHaltonSampler.fastPermutationTable[114390 + ((index / 1231u) % 1231u)] * 1231u +
				globalHaltonSampler.fastPermutationTable[114390 + ((index / 1515361u) % 1231u)]) * float(DoubleOneMinusEpsilon / 1865409391u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1237(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[115621 + (index % 1237u)] * 1530169u +
				globalHaltonSampler.fastPermutationTable[115621 + ((index / 1237u) % 1237u)] * 1237u +
				globalHaltonSampler.fastPermutationTable[115621 + ((index / 1530169u) % 1237u)]) * float(DoubleOneMinusEpsilon / 1892819053u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1249(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[116858 + (index % 1249u)] * 1560001u +
				globalHaltonSampler.fastPermutationTable[116858 + ((index / 1249u) % 1249u)] * 1249u +
				globalHaltonSampler.fastPermutationTable[116858 + ((index / 1560001u) % 1249u)]) * float(DoubleOneMinusEpsilon / 1948441249u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1259(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[118107 + (index % 1259u)] * 1585081u +
				globalHaltonSampler.fastPermutationTable[118107 + ((index / 1259u) % 1259u)] * 1259u +
				globalHaltonSampler.fastPermutationTable[118107 + ((index / 1585081u) % 1259u)]) * float(DoubleOneMinusEpsilon / 1995616979u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1277(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[119366 + (index % 1277u)] * 1630729u +
				globalHaltonSampler.fastPermutationTable[119366 + ((index / 1277u) % 1277u)] * 1277u +
				globalHaltonSampler.fastPermutationTable[119366 + ((index / 1630729u) % 1277u)]) * float(DoubleOneMinusEpsilon / 2082440933u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1279(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[120643 + (index % 1279u)] * 1635841u +
				globalHaltonSampler.fastPermutationTable[120643 + ((index / 1279u) % 1279u)] * 1279u +
				globalHaltonSampler.fastPermutationTable[120643 + ((index / 1635841u) % 1279u)]) * float(DoubleOneMinusEpsilon / 2092240639u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1283(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[121922 + (index % 1283u)] * 1646089u +
				globalHaltonSampler.fastPermutationTable[121922 + ((index / 1283u) % 1283u)] * 1283u +
				globalHaltonSampler.fastPermutationTable[121922 + ((index / 1646089u) % 1283u)]) * float(DoubleOneMinusEpsilon / 2111932187u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1289(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[123205 + (index % 1289u)] * 1661521u +
				globalHaltonSampler.fastPermutationTable[123205 + ((index / 1289u) % 1289u)] * 1289u +
				globalHaltonSampler.fastPermutationTable[123205 + ((index / 1661521u) % 1289u)]) * float(DoubleOneMinusEpsilon / 2141700569u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1291(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[124494 + (index % 1291u)] * 1666681u +
				globalHaltonSampler.fastPermutationTable[124494 + ((index / 1291u) % 1291u)] * 1291u +
				globalHaltonSampler.fastPermutationTable[124494 + ((index / 1666681u) % 1291u)]) * float(DoubleOneMinusEpsilon / 2151685171u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1297(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[125785 + (index % 1297u)] * 1682209u +
				globalHaltonSampler.fastPermutationTable[125785 + ((index / 1297u) % 1297u)] * 1297u +
				globalHaltonSampler.fastPermutationTable[125785 + ((index / 1682209u) % 1297u)]) * float(DoubleOneMinusEpsilon / 2181825073u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1301(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[127082 + (index % 1301u)] * 1692601u +
				globalHaltonSampler.fastPermutationTable[127082 + ((index / 1301u) % 1301u)] * 1301u +
				globalHaltonSampler.fastPermutationTable[127082 + ((index / 1692601u) % 1301u)]) * float(DoubleOneMinusEpsilon / 2202073901u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1303(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[128383 + (index % 1303u)] * 1697809u +
				globalHaltonSampler.fastPermutationTable[128383 + ((index / 1303u) % 1303u)] * 1303u +
				globalHaltonSampler.fastPermutationTable[128383 + ((index / 1697809u) % 1303u)]) * float(DoubleOneMinusEpsilon / 2212245127u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1307(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[129686 + (index % 1307u)] * 1708249u +
				globalHaltonSampler.fastPermutationTable[129686 + ((index / 1307u) % 1307u)] * 1307u +
				globalHaltonSampler.fastPermutationTable[129686 + ((index / 1708249u) % 1307u)]) * float(DoubleOneMinusEpsilon / 2232681443u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1319(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[130993 + (index % 1319u)] * 1739761u +
				globalHaltonSampler.fastPermutationTable[130993 + ((index / 1319u) % 1319u)] * 1319u +
				globalHaltonSampler.fastPermutationTable[130993 + ((index / 1739761u) % 1319u)]) * float(DoubleOneMinusEpsilon / 2294744759u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1321(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[132312 + (index % 1321u)] * 1745041u +
				globalHaltonSampler.fastPermutationTable[132312 + ((index / 1321u) % 1321u)] * 1321u +
				globalHaltonSampler.fastPermutationTable[132312 + ((index / 1745041u) % 1321u)]) * float(DoubleOneMinusEpsilon / 2305199161u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1327(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[133633 + (index % 1327u)] * 1760929u +
				globalHaltonSampler.fastPermutationTable[133633 + ((index / 1327u) % 1327u)] * 1327u +
				globalHaltonSampler.fastPermutationTable[133633 + ((index / 1760929u) % 1327u)]) * float(DoubleOneMinusEpsilon / 2336752783u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1361(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[134960 + (index % 1361u)] * 1852321u +
				globalHaltonSampler.fastPermutationTable[134960 + ((index / 1361u) % 1361u)] * 1361u +
				globalHaltonSampler.fastPermutationTable[134960 + ((index / 1852321u) % 1361u)]) * float(DoubleOneMinusEpsilon / 2521008881u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1367(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[136321 + (index % 1367u)] * 1868689u +
				globalHaltonSampler.fastPermutationTable[136321 + ((index / 1367u) % 1367u)] * 1367u +
				globalHaltonSampler.fastPermutationTable[136321 + ((index / 1868689u) % 1367u)]) * float(DoubleOneMinusEpsilon / 2554497863u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1373(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[137688 + (index % 1373u)] * 1885129u +
				globalHaltonSampler.fastPermutationTable[137688 + ((index / 1373u) % 1373u)] * 1373u +
				globalHaltonSampler.fastPermutationTable[137688 + ((index / 1885129u) % 1373u)]) * float(DoubleOneMinusEpsilon / 2588282117u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1381(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[139061 + (index % 1381u)] * 1907161u +
				globalHaltonSampler.fastPermutationTable[139061 + ((index / 1381u) % 1381u)] * 1381u +
				globalHaltonSampler.fastPermutationTable[139061 + ((index / 1907161u) % 1381u)]) * float(DoubleOneMinusEpsilon / 2633789341u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1399(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[140442 + (index % 1399u)] * 1957201u +
				globalHaltonSampler.fastPermutationTable[140442 + ((index / 1399u) % 1399u)] * 1399u +
				globalHaltonSampler.fastPermutationTable[140442 + ((index / 1957201u) % 1399u)]) * float(DoubleOneMinusEpsilon / 2738124199u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1409(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[141841 + (index % 1409u)] * 1985281u +
				globalHaltonSampler.fastPermutationTable[141841 + ((index / 1409u) % 1409u)] * 1409u +
				globalHaltonSampler.fastPermutationTable[141841 + ((index / 1985281u) % 1409u)]) * float(DoubleOneMinusEpsilon / 2797260929u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1423(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[143250 + (index % 1423u)] * 2024929u +
				globalHaltonSampler.fastPermutationTable[143250 + ((index / 1423u) % 1423u)] * 1423u +
				globalHaltonSampler.fastPermutationTable[143250 + ((index / 2024929u) % 1423u)]) * float(DoubleOneMinusEpsilon / 2881473967u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1427(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[144673 + (index % 1427u)] * 2036329u +
				globalHaltonSampler.fastPermutationTable[144673 + ((index / 1427u) % 1427u)] * 1427u +
				globalHaltonSampler.fastPermutationTable[144673 + ((index / 2036329u) % 1427u)]) * float(DoubleOneMinusEpsilon / 2905841483u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1429(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[146100 + (index % 1429u)] * 2042041u +
				globalHaltonSampler.fastPermutationTable[146100 + ((index / 1429u) % 1429u)] * 1429u +
				globalHaltonSampler.fastPermutationTable[146100 + ((index / 2042041u) % 1429u)]) * float(DoubleOneMinusEpsilon / 2918076589u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1433(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[147529 + (index % 1433u)] * 2053489u +
				globalHaltonSampler.fastPermutationTable[147529 + ((index / 1433u) % 1433u)] * 1433u +
				globalHaltonSampler.fastPermutationTable[147529 + ((index / 2053489u) % 1433u)]) * float(DoubleOneMinusEpsilon / 2942649737u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1439(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[148962 + (index % 1439u)] * 2070721u +
				globalHaltonSampler.fastPermutationTable[148962 + ((index / 1439u) % 1439u)] * 1439u +
				globalHaltonSampler.fastPermutationTable[148962 + ((index / 2070721u) % 1439u)]) * float(DoubleOneMinusEpsilon / 2979767519u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1447(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[150401 + (index % 1447u)] * 2093809u +
				globalHaltonSampler.fastPermutationTable[150401 + ((index / 1447u) % 1447u)] * 1447u +
				globalHaltonSampler.fastPermutationTable[150401 + ((index / 2093809u) % 1447u)]) * float(DoubleOneMinusEpsilon / 3029741623u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1451(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[151848 + (index % 1451u)] * 2105401u +
				globalHaltonSampler.fastPermutationTable[151848 + ((index / 1451u) % 1451u)] * 1451u +
				globalHaltonSampler.fastPermutationTable[151848 + ((index / 2105401u) % 1451u)]) * float(DoubleOneMinusEpsilon / 3054936851u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1453(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[153299 + (index % 1453u)] * 2111209u +
				globalHaltonSampler.fastPermutationTable[153299 + ((index / 1453u) % 1453u)] * 1453u +
				globalHaltonSampler.fastPermutationTable[153299 + ((index / 2111209u) % 1453u)]) * float(DoubleOneMinusEpsilon / 3067586677u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1459(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[154752 + (index % 1459u)] * 2128681u +
				globalHaltonSampler.fastPermutationTable[154752 + ((index / 1459u) % 1459u)] * 1459u +
				globalHaltonSampler.fastPermutationTable[154752 + ((index / 2128681u) % 1459u)]) * float(DoubleOneMinusEpsilon / 3105745579u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1471(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[156211 + (index % 1471u)] * 2163841u +
				globalHaltonSampler.fastPermutationTable[156211 + ((index / 1471u) % 1471u)] * 1471u +
				globalHaltonSampler.fastPermutationTable[156211 + ((index / 2163841u) % 1471u)]) * float(DoubleOneMinusEpsilon / 3183010111u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1481(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[157682 + (index % 1481u)] * 2193361u +
				globalHaltonSampler.fastPermutationTable[157682 + ((index / 1481u) % 1481u)] * 1481u +
				globalHaltonSampler.fastPermutationTable[157682 + ((index / 2193361u) % 1481u)]) * float(DoubleOneMinusEpsilon / 3248367641u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1483(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[159163 + (index % 1483u)] * 2199289u +
				globalHaltonSampler.fastPermutationTable[159163 + ((index / 1483u) % 1483u)] * 1483u +
				globalHaltonSampler.fastPermutationTable[159163 + ((index / 2199289u) % 1483u)]) * float(DoubleOneMinusEpsilon / 3261545587u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1487(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[160646 + (index % 1487u)] * 2211169u +
				globalHaltonSampler.fastPermutationTable[160646 + ((index / 1487u) % 1487u)] * 1487u +
				globalHaltonSampler.fastPermutationTable[160646 + ((index / 2211169u) % 1487u)]) * float(DoubleOneMinusEpsilon / 3288008303u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1489(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[162133 + (index % 1489u)] * 2217121u +
				globalHaltonSampler.fastPermutationTable[162133 + ((index / 1489u) % 1489u)] * 1489u +
				globalHaltonSampler.fastPermutationTable[162133 + ((index / 2217121u) % 1489u)]) * float(DoubleOneMinusEpsilon / 3301293169u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1493(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[163622 + (index % 1493u)] * 2229049u +
				globalHaltonSampler.fastPermutationTable[163622 + ((index / 1493u) % 1493u)] * 1493u +
				globalHaltonSampler.fastPermutationTable[163622 + ((index / 2229049u) % 1493u)]) * float(DoubleOneMinusEpsilon / 3327970157u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1499(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[165115 + (index % 1499u)] * 2247001u +
				globalHaltonSampler.fastPermutationTable[165115 + ((index / 1499u) % 1499u)] * 1499u +
				globalHaltonSampler.fastPermutationTable[165115 + ((index / 2247001u) % 1499u)]) * float(DoubleOneMinusEpsilon / 3368254499u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1511(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[166614 + (index % 1511u)] * 2283121u +
				globalHaltonSampler.fastPermutationTable[166614 + ((index / 1511u) % 1511u)] * 1511u +
				globalHaltonSampler.fastPermutationTable[166614 + ((index / 2283121u) % 1511u)]) * float(DoubleOneMinusEpsilon / 3449795831u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1523(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[168125 + (index % 1523u)] * 2319529u +
				globalHaltonSampler.fastPermutationTable[168125 + ((index / 1523u) % 1523u)] * 1523u +
				globalHaltonSampler.fastPermutationTable[168125 + ((index / 2319529u) % 1523u)]) * float(DoubleOneMinusEpsilon / 3532642667u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1531(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[169648 + (index % 1531u)] * 2343961u +
				globalHaltonSampler.fastPermutationTable[169648 + ((index / 1531u) % 1531u)] * 1531u +
				globalHaltonSampler.fastPermutationTable[169648 + ((index / 2343961u) % 1531u)]) * float(DoubleOneMinusEpsilon / 3588604291u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1543(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[171179 + (index % 1543u)] * 2380849u +
				globalHaltonSampler.fastPermutationTable[171179 + ((index / 1543u) % 1543u)] * 1543u +
				globalHaltonSampler.fastPermutationTable[171179 + ((index / 2380849u) % 1543u)]) * float(DoubleOneMinusEpsilon / 3673650007u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1549(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[172722 + (index % 1549u)] * 2399401u +
				globalHaltonSampler.fastPermutationTable[172722 + ((index / 1549u) % 1549u)] * 1549u +
				globalHaltonSampler.fastPermutationTable[172722 + ((index / 2399401u) % 1549u)]) * float(DoubleOneMinusEpsilon / 3716672149u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1553(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[174271 + (index % 1553u)] * 2411809u +
				globalHaltonSampler.fastPermutationTable[174271 + ((index / 1553u) % 1553u)] * 1553u +
				globalHaltonSampler.fastPermutationTable[174271 + ((index / 2411809u) % 1553u)]) * float(DoubleOneMinusEpsilon / 3745539377u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1559(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[175824 + (index % 1559u)] * 2430481u +
				globalHaltonSampler.fastPermutationTable[175824 + ((index / 1559u) % 1559u)] * 1559u +
				globalHaltonSampler.fastPermutationTable[175824 + ((index / 2430481u) % 1559u)]) * float(DoubleOneMinusEpsilon / 3789119879u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1567(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[177383 + (index % 1567u)] * 2455489u +
				globalHaltonSampler.fastPermutationTable[177383 + ((index / 1567u) % 1567u)] * 1567u +
				globalHaltonSampler.fastPermutationTable[177383 + ((index / 2455489u) % 1567u)]) * float(DoubleOneMinusEpsilon / 3847751263u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1571(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[178950 + (index % 1571u)] * 2468041u +
				globalHaltonSampler.fastPermutationTable[178950 + ((index / 1571u) % 1571u)] * 1571u +
				globalHaltonSampler.fastPermutationTable[178950 + ((index / 2468041u) % 1571u)]) * float(DoubleOneMinusEpsilon / 3877292411u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1579(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[180521 + (index % 1579u)] * 2493241u +
				globalHaltonSampler.fastPermutationTable[180521 + ((index / 1579u) % 1579u)] * 1579u +
				globalHaltonSampler.fastPermutationTable[180521 + ((index / 2493241u) % 1579u)]) * float(DoubleOneMinusEpsilon / 3936827539u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1583(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[182100 + (index % 1583u)] * 2505889u +
				globalHaltonSampler.fastPermutationTable[182100 + ((index / 1583u) % 1583u)] * 1583u +
				globalHaltonSampler.fastPermutationTable[182100 + ((index / 2505889u) % 1583u)]) * float(DoubleOneMinusEpsilon / 3966822287u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1597(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[183683 + (index % 1597u)] * 2550409u +
				globalHaltonSampler.fastPermutationTable[183683 + ((index / 1597u) % 1597u)] * 1597u +
				globalHaltonSampler.fastPermutationTable[183683 + ((index / 2550409u) % 1597u)]) * float(DoubleOneMinusEpsilon / 4073003173u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1601(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[185280 + (index % 1601u)] * 2563201u +
				globalHaltonSampler.fastPermutationTable[185280 + ((index / 1601u) % 1601u)] * 1601u +
				globalHaltonSampler.fastPermutationTable[185280 + ((index / 2563201u) % 1601u)]) * float(DoubleOneMinusEpsilon / 4103684801u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1607(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[186881 + (index % 1607u)] * 2582449u +
				globalHaltonSampler.fastPermutationTable[186881 + ((index / 1607u) % 1607u)] * 1607u +
				globalHaltonSampler.fastPermutationTable[186881 + ((index / 2582449u) % 1607u)]) * float(DoubleOneMinusEpsilon / 4149995543u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1609(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[188488 + (index % 1609u)] * 2588881u +
				globalHaltonSampler.fastPermutationTable[188488 + ((index / 1609u) % 1609u)] * 1609u +
				globalHaltonSampler.fastPermutationTable[188488 + ((index / 2588881u) % 1609u)]) * float(DoubleOneMinusEpsilon / 4165509529u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1613(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[190097 + (index % 1613u)] * 2601769u +
				globalHaltonSampler.fastPermutationTable[190097 + ((index / 1613u) % 1613u)] * 1613u +
				globalHaltonSampler.fastPermutationTable[190097 + ((index / 2601769u) % 1613u)]) * float(DoubleOneMinusEpsilon / 4196653397u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1619(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[191710 + (index % 1619u)] * 2621161u +
				globalHaltonSampler.fastPermutationTable[191710 + ((index / 1619u) % 1619u)] * 1619u +
				globalHaltonSampler.fastPermutationTable[191710 + ((index / 2621161u) % 1619u)]) * float(DoubleOneMinusEpsilon / 4243659659u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1621(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[193329 + (index % 1621u)] * 2627641u +
				globalHaltonSampler.fastPermutationTable[193329 + ((index / 1621u) % 1621u)] * 1621u +
				globalHaltonSampler.fastPermutationTable[193329 + ((index / 2627641u) % 1621u)]) * float(DoubleOneMinusEpsilon / 4259406061u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1627(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[194950 + (index % 1627u)] * 1627u +
				globalHaltonSampler.fastPermutationTable[194950 + ((index / 1627u) % 1627u)]) * float(DoubleOneMinusEpsilon / 2647129u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1637(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[196577 + (index % 1637u)] * 1637u +
				globalHaltonSampler.fastPermutationTable[196577 + ((index / 1637u) % 1637u)]) * float(DoubleOneMinusEpsilon / 2679769u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1657(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[198214 + (index % 1657u)] * 1657u +
				globalHaltonSampler.fastPermutationTable[198214 + ((index / 1657u) % 1657u)]) * float(DoubleOneMinusEpsilon / 2745649u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1663(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[199871 + (index % 1663u)] * 1663u +
				globalHaltonSampler.fastPermutationTable[199871 + ((index / 1663u) % 1663u)]) * float(DoubleOneMinusEpsilon / 2765569u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1667(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[201534 + (index % 1667u)] * 1667u +
				globalHaltonSampler.fastPermutationTable[201534 + ((index / 1667u) % 1667u)]) * float(DoubleOneMinusEpsilon / 2778889u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1669(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[203201 + (index % 1669u)] * 1669u +
				globalHaltonSampler.fastPermutationTable[203201 + ((index / 1669u) % 1669u)]) * float(DoubleOneMinusEpsilon / 2785561u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1693(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[204870 + (index % 1693u)] * 1693u +
				globalHaltonSampler.fastPermutationTable[204870 + ((index / 1693u) % 1693u)]) * float(DoubleOneMinusEpsilon / 2866249u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1697(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[206563 + (index % 1697u)] * 1697u +
				globalHaltonSampler.fastPermutationTable[206563 + ((index / 1697u) % 1697u)]) * float(DoubleOneMinusEpsilon / 2879809u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1699(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[208260 + (index % 1699u)] * 1699u +
				globalHaltonSampler.fastPermutationTable[208260 + ((index / 1699u) % 1699u)]) * float(DoubleOneMinusEpsilon / 2886601u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1709(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[209959 + (index % 1709u)] * 1709u +
				globalHaltonSampler.fastPermutationTable[209959 + ((index / 1709u) % 1709u)]) * float(DoubleOneMinusEpsilon / 2920681u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1721(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[211668 + (index % 1721u)] * 1721u +
				globalHaltonSampler.fastPermutationTable[211668 + ((index / 1721u) % 1721u)]) * float(DoubleOneMinusEpsilon / 2961841u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1723(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[213389 + (index % 1723u)] * 1723u +
				globalHaltonSampler.fastPermutationTable[213389 + ((index / 1723u) % 1723u)]) * float(DoubleOneMinusEpsilon / 2968729u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1733(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[215112 + (index % 1733u)] * 1733u +
				globalHaltonSampler.fastPermutationTable[215112 + ((index / 1733u) % 1733u)]) * float(DoubleOneMinusEpsilon / 3003289u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1741(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[216845 + (index % 1741u)] * 1741u +
				globalHaltonSampler.fastPermutationTable[216845 + ((index / 1741u) % 1741u)]) * float(DoubleOneMinusEpsilon / 3031081u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1747(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[218586 + (index % 1747u)] * 1747u +
				globalHaltonSampler.fastPermutationTable[218586 + ((index / 1747u) % 1747u)]) * float(DoubleOneMinusEpsilon / 3052009u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1753(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[220333 + (index % 1753u)] * 1753u +
				globalHaltonSampler.fastPermutationTable[220333 + ((index / 1753u) % 1753u)]) * float(DoubleOneMinusEpsilon / 3073009u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1759(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[222086 + (index % 1759u)] * 1759u +
				globalHaltonSampler.fastPermutationTable[222086 + ((index / 1759u) % 1759u)]) * float(DoubleOneMinusEpsilon / 3094081u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1777(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[223845 + (index % 1777u)] * 1777u +
				globalHaltonSampler.fastPermutationTable[223845 + ((index / 1777u) % 1777u)]) * float(DoubleOneMinusEpsilon / 3157729u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1783(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[225622 + (index % 1783u)] * 1783u +
				globalHaltonSampler.fastPermutationTable[225622 + ((index / 1783u) % 1783u)]) * float(DoubleOneMinusEpsilon / 3179089u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1787(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[227405 + (index % 1787u)] * 1787u +
				globalHaltonSampler.fastPermutationTable[227405 + ((index / 1787u) % 1787u)]) * float(DoubleOneMinusEpsilon / 3193369u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1789(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[229192 + (index % 1789u)] * 1789u +
				globalHaltonSampler.fastPermutationTable[229192 + ((index / 1789u) % 1789u)]) * float(DoubleOneMinusEpsilon / 3200521u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1801(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[230981 + (index % 1801u)] * 1801u +
				globalHaltonSampler.fastPermutationTable[230981 + ((index / 1801u) % 1801u)]) * float(DoubleOneMinusEpsilon / 3243601u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1811(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[232782 + (index % 1811u)] * 1811u +
				globalHaltonSampler.fastPermutationTable[232782 + ((index / 1811u) % 1811u)]) * float(DoubleOneMinusEpsilon / 3279721u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1823(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[234593 + (index % 1823u)] * 1823u +
				globalHaltonSampler.fastPermutationTable[234593 + ((index / 1823u) % 1823u)]) * float(DoubleOneMinusEpsilon / 3323329u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1831(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[236416 + (index % 1831u)] * 1831u +
				globalHaltonSampler.fastPermutationTable[236416 + ((index / 1831u) % 1831u)]) * float(DoubleOneMinusEpsilon / 3352561u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1847(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[238247 + (index % 1847u)] * 1847u +
				globalHaltonSampler.fastPermutationTable[238247 + ((index / 1847u) % 1847u)]) * float(DoubleOneMinusEpsilon / 3411409u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1861(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[240094 + (index % 1861u)] * 1861u +
				globalHaltonSampler.fastPermutationTable[240094 + ((index / 1861u) % 1861u)]) * float(DoubleOneMinusEpsilon / 3463321u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1867(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[241955 + (index % 1867u)] * 1867u +
				globalHaltonSampler.fastPermutationTable[241955 + ((index / 1867u) % 1867u)]) * float(DoubleOneMinusEpsilon / 3485689u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1871(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[243822 + (index % 1871u)] * 1871u +
				globalHaltonSampler.fastPermutationTable[243822 + ((index / 1871u) % 1871u)]) * float(DoubleOneMinusEpsilon / 3500641u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1873(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[245693 + (index % 1873u)] * 1873u +
				globalHaltonSampler.fastPermutationTable[245693 + ((index / 1873u) % 1873u)]) * float(DoubleOneMinusEpsilon / 3508129u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1877(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[247566 + (index % 1877u)] * 1877u +
				globalHaltonSampler.fastPermutationTable[247566 + ((index / 1877u) % 1877u)]) * float(DoubleOneMinusEpsilon / 3523129u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1879(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[249443 + (index % 1879u)] * 1879u +
				globalHaltonSampler.fastPermutationTable[249443 + ((index / 1879u) % 1879u)]) * float(DoubleOneMinusEpsilon / 3530641u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1889(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[251322 + (index % 1889u)] * 1889u +
				globalHaltonSampler.fastPermutationTable[251322 + ((index / 1889u) % 1889u)]) * float(DoubleOneMinusEpsilon / 3568321u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1901(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[253211 + (index % 1901u)] * 1901u +
				globalHaltonSampler.fastPermutationTable[253211 + ((index / 1901u) % 1901u)]) * float(DoubleOneMinusEpsilon / 3613801u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1907(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[255112 + (index % 1907u)] * 1907u +
				globalHaltonSampler.fastPermutationTable[255112 + ((index / 1907u) % 1907u)]) * float(DoubleOneMinusEpsilon / 3636649u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1913(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[257019 + (index % 1913u)] * 1913u +
				globalHaltonSampler.fastPermutationTable[257019 + ((index / 1913u) % 1913u)]) * float(DoubleOneMinusEpsilon / 3659569u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1931(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[258932 + (index % 1931u)] * 1931u +
				globalHaltonSampler.fastPermutationTable[258932 + ((index / 1931u) % 1931u)]) * float(DoubleOneMinusEpsilon / 3728761u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1933(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[260863 + (index % 1933u)] * 1933u +
				globalHaltonSampler.fastPermutationTable[260863 + ((index / 1933u) % 1933u)]) * float(DoubleOneMinusEpsilon / 3736489u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1949(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[262796 + (index % 1949u)] * 1949u +
				globalHaltonSampler.fastPermutationTable[262796 + ((index / 1949u) % 1949u)]) * float(DoubleOneMinusEpsilon / 3798601u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1951(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[264745 + (index % 1951u)] * 1951u +
				globalHaltonSampler.fastPermutationTable[264745 + ((index / 1951u) % 1951u)]) * float(DoubleOneMinusEpsilon / 3806401u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1973(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[266696 + (index % 1973u)] * 1973u +
				globalHaltonSampler.fastPermutationTable[266696 + ((index / 1973u) % 1973u)]) * float(DoubleOneMinusEpsilon / 3892729u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1979(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[268669 + (index % 1979u)] * 1979u +
				globalHaltonSampler.fastPermutationTable[268669 + ((index / 1979u) % 1979u)]) * float(DoubleOneMinusEpsilon / 3916441u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1987(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[270648 + (index % 1987u)] * 1987u +
				globalHaltonSampler.fastPermutationTable[270648 + ((index / 1987u) % 1987u)]) * float(DoubleOneMinusEpsilon / 3948169u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1993(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[272635 + (index % 1993u)] * 1993u +
				globalHaltonSampler.fastPermutationTable[272635 + ((index / 1993u) % 1993u)]) * float(DoubleOneMinusEpsilon / 3972049u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1997(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[274628 + (index % 1997u)] * 1997u +
				globalHaltonSampler.fastPermutationTable[274628 + ((index / 1997u) % 1997u)]) * float(DoubleOneMinusEpsilon / 3988009u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized1999(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[276625 + (index % 1999u)] * 1999u +
				globalHaltonSampler.fastPermutationTable[276625 + ((index / 1999u) % 1999u)]) * float(DoubleOneMinusEpsilon / 3996001u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2003(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[278624 + (index % 2003u)] * 2003u +
				globalHaltonSampler.fastPermutationTable[278624 + ((index / 2003u) % 2003u)]) * float(DoubleOneMinusEpsilon / 4012009u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2011(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[280627 + (index % 2011u)] * 2011u +
				globalHaltonSampler.fastPermutationTable[280627 + ((index / 2011u) % 2011u)]) * float(DoubleOneMinusEpsilon / 4044121u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2017(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[282638 + (index % 2017u)] * 2017u +
				globalHaltonSampler.fastPermutationTable[282638 + ((index / 2017u) % 2017u)]) * float(DoubleOneMinusEpsilon / 4068289u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2027(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[284655 + (index % 2027u)] * 2027u +
				globalHaltonSampler.fastPermutationTable[284655 + ((index / 2027u) % 2027u)]) * float(DoubleOneMinusEpsilon / 4108729u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2029(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[286682 + (index % 2029u)] * 2029u +
				globalHaltonSampler.fastPermutationTable[286682 + ((index / 2029u) % 2029u)]) * float(DoubleOneMinusEpsilon / 4116841u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2039(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[288711 + (index % 2039u)] * 2039u +
				globalHaltonSampler.fastPermutationTable[288711 + ((index / 2039u) % 2039u)]) * float(DoubleOneMinusEpsilon / 4157521u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2053(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[290750 + (index % 2053u)] * 2053u +
				globalHaltonSampler.fastPermutationTable[290750 + ((index / 2053u) % 2053u)]) * float(DoubleOneMinusEpsilon / 4214809u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2063(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[292803 + (index % 2063u)] * 2063u +
				globalHaltonSampler.fastPermutationTable[292803 + ((index / 2063u) % 2063u)]) * float(DoubleOneMinusEpsilon / 4255969u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2069(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[294866 + (index % 2069u)] * 2069u +
				globalHaltonSampler.fastPermutationTable[294866 + ((index / 2069u) % 2069u)]) * float(DoubleOneMinusEpsilon / 4280761u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2081(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[296935 + (index % 2081u)] * 2081u +
				globalHaltonSampler.fastPermutationTable[296935 + ((index / 2081u) % 2081u)]) * float(DoubleOneMinusEpsilon / 4330561u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2083(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[299016 + (index % 2083u)] * 2083u +
				globalHaltonSampler.fastPermutationTable[299016 + ((index / 2083u) % 2083u)]) * float(DoubleOneMinusEpsilon / 4338889u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2087(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[301099 + (index % 2087u)] * 2087u +
				globalHaltonSampler.fastPermutationTable[301099 + ((index / 2087u) % 2087u)]) * float(DoubleOneMinusEpsilon / 4355569u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2089(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[303186 + (index % 2089u)] * 2089u +
				globalHaltonSampler.fastPermutationTable[303186 + ((index / 2089u) % 2089u)]) * float(DoubleOneMinusEpsilon / 4363921u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2099(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[305275 + (index % 2099u)] * 2099u +
				globalHaltonSampler.fastPermutationTable[305275 + ((index / 2099u) % 2099u)]) * float(DoubleOneMinusEpsilon / 4405801u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2111(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[307374 + (index % 2111u)] * 2111u +
				globalHaltonSampler.fastPermutationTable[307374 + ((index / 2111u) % 2111u)]) * float(DoubleOneMinusEpsilon / 4456321u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2113(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[309485 + (index % 2113u)] * 2113u +
				globalHaltonSampler.fastPermutationTable[309485 + ((index / 2113u) % 2113u)]) * float(DoubleOneMinusEpsilon / 4464769u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2129(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[311598 + (index % 2129u)] * 2129u +
				globalHaltonSampler.fastPermutationTable[311598 + ((index / 2129u) % 2129u)]) * float(DoubleOneMinusEpsilon / 4532641u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2131(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[313727 + (index % 2131u)] * 2131u +
				globalHaltonSampler.fastPermutationTable[313727 + ((index / 2131u) % 2131u)]) * float(DoubleOneMinusEpsilon / 4541161u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2137(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[315858 + (index % 2137u)] * 2137u +
				globalHaltonSampler.fastPermutationTable[315858 + ((index / 2137u) % 2137u)]) * float(DoubleOneMinusEpsilon / 4566769u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2141(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[317995 + (index % 2141u)] * 2141u +
				globalHaltonSampler.fastPermutationTable[317995 + ((index / 2141u) % 2141u)]) * float(DoubleOneMinusEpsilon / 4583881u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2143(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[320136 + (index % 2143u)] * 2143u +
				globalHaltonSampler.fastPermutationTable[320136 + ((index / 2143u) % 2143u)]) * float(DoubleOneMinusEpsilon / 4592449u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2153(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[322279 + (index % 2153u)] * 2153u +
				globalHaltonSampler.fastPermutationTable[322279 + ((index / 2153u) % 2153u)]) * float(DoubleOneMinusEpsilon / 4635409u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2161(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[324432 + (index % 2161u)] * 2161u +
				globalHaltonSampler.fastPermutationTable[324432 + ((index / 2161u) % 2161u)]) * float(DoubleOneMinusEpsilon / 4669921u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2179(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[326593 + (index % 2179u)] * 2179u +
				globalHaltonSampler.fastPermutationTable[326593 + ((index / 2179u) % 2179u)]) * float(DoubleOneMinusEpsilon / 4748041u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2203(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[328772 + (index % 2203u)] * 2203u +
				globalHaltonSampler.fastPermutationTable[328772 + ((index / 2203u) % 2203u)]) * float(DoubleOneMinusEpsilon / 4853209u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2207(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[330975 + (index % 2207u)] * 2207u +
				globalHaltonSampler.fastPermutationTable[330975 + ((index / 2207u) % 2207u)]) * float(DoubleOneMinusEpsilon / 4870849u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2213(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[333182 + (index % 2213u)] * 2213u +
				globalHaltonSampler.fastPermutationTable[333182 + ((index / 2213u) % 2213u)]) * float(DoubleOneMinusEpsilon / 4897369u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2221(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[335395 + (index % 2221u)] * 2221u +
				globalHaltonSampler.fastPermutationTable[335395 + ((index / 2221u) % 2221u)]) * float(DoubleOneMinusEpsilon / 4932841u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2237(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[337616 + (index % 2237u)] * 2237u +
				globalHaltonSampler.fastPermutationTable[337616 + ((index / 2237u) % 2237u)]) * float(DoubleOneMinusEpsilon / 5004169u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2239(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[339853 + (index % 2239u)] * 2239u +
				globalHaltonSampler.fastPermutationTable[339853 + ((index / 2239u) % 2239u)]) * float(DoubleOneMinusEpsilon / 5013121u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2243(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[342092 + (index % 2243u)] * 2243u +
				globalHaltonSampler.fastPermutationTable[342092 + ((index / 2243u) % 2243u)]) * float(DoubleOneMinusEpsilon / 5031049u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2251(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[344335 + (index % 2251u)] * 2251u +
				globalHaltonSampler.fastPermutationTable[344335 + ((index / 2251u) % 2251u)]) * float(DoubleOneMinusEpsilon / 5067001u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2267(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[346586 + (index % 2267u)] * 2267u +
				globalHaltonSampler.fastPermutationTable[346586 + ((index / 2267u) % 2267u)]) * float(DoubleOneMinusEpsilon / 5139289u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2269(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[348853 + (index % 2269u)] * 2269u +
				globalHaltonSampler.fastPermutationTable[348853 + ((index / 2269u) % 2269u)]) * float(DoubleOneMinusEpsilon / 5148361u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2273(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[351122 + (index % 2273u)] * 2273u +
				globalHaltonSampler.fastPermutationTable[351122 + ((index / 2273u) % 2273u)]) * float(DoubleOneMinusEpsilon / 5166529u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2281(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[353395 + (index % 2281u)] * 2281u +
				globalHaltonSampler.fastPermutationTable[353395 + ((index / 2281u) % 2281u)]) * float(DoubleOneMinusEpsilon / 5202961u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2287(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[355676 + (index % 2287u)] * 2287u +
				globalHaltonSampler.fastPermutationTable[355676 + ((index / 2287u) % 2287u)]) * float(DoubleOneMinusEpsilon / 5230369u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2293(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[357963 + (index % 2293u)] * 2293u +
				globalHaltonSampler.fastPermutationTable[357963 + ((index / 2293u) % 2293u)]) * float(DoubleOneMinusEpsilon / 5257849u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2297(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[360256 + (index % 2297u)] * 2297u +
				globalHaltonSampler.fastPermutationTable[360256 + ((index / 2297u) % 2297u)]) * float(DoubleOneMinusEpsilon / 5276209u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2309(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[362553 + (index % 2309u)] * 2309u +
				globalHaltonSampler.fastPermutationTable[362553 + ((index / 2309u) % 2309u)]) * float(DoubleOneMinusEpsilon / 5331481u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2311(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[364862 + (index % 2311u)] * 2311u +
				globalHaltonSampler.fastPermutationTable[364862 + ((index / 2311u) % 2311u)]) * float(DoubleOneMinusEpsilon / 5340721u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2333(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[367173 + (index % 2333u)] * 2333u +
				globalHaltonSampler.fastPermutationTable[367173 + ((index / 2333u) % 2333u)]) * float(DoubleOneMinusEpsilon / 5442889u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2339(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[369506 + (index % 2339u)] * 2339u +
				globalHaltonSampler.fastPermutationTable[369506 + ((index / 2339u) % 2339u)]) * float(DoubleOneMinusEpsilon / 5470921u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2341(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[371845 + (index % 2341u)] * 2341u +
				globalHaltonSampler.fastPermutationTable[371845 + ((index / 2341u) % 2341u)]) * float(DoubleOneMinusEpsilon / 5480281u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2347(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[374186 + (index % 2347u)] * 2347u +
				globalHaltonSampler.fastPermutationTable[374186 + ((index / 2347u) % 2347u)]) * float(DoubleOneMinusEpsilon / 5508409u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2351(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[376533 + (index % 2351u)] * 2351u +
				globalHaltonSampler.fastPermutationTable[376533 + ((index / 2351u) % 2351u)]) * float(DoubleOneMinusEpsilon / 5527201u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2357(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[378884 + (index % 2357u)] * 2357u +
				globalHaltonSampler.fastPermutationTable[378884 + ((index / 2357u) % 2357u)]) * float(DoubleOneMinusEpsilon / 5555449u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2371(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[381241 + (index % 2371u)] * 2371u +
				globalHaltonSampler.fastPermutationTable[381241 + ((index / 2371u) % 2371u)]) * float(DoubleOneMinusEpsilon / 5621641u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2377(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[383612 + (index % 2377u)] * 2377u +
				globalHaltonSampler.fastPermutationTable[383612 + ((index / 2377u) % 2377u)]) * float(DoubleOneMinusEpsilon / 5650129u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2381(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[385989 + (index % 2381u)] * 2381u +
				globalHaltonSampler.fastPermutationTable[385989 + ((index / 2381u) % 2381u)]) * float(DoubleOneMinusEpsilon / 5669161u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2383(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[388370 + (index % 2383u)] * 2383u +
				globalHaltonSampler.fastPermutationTable[388370 + ((index / 2383u) % 2383u)]) * float(DoubleOneMinusEpsilon / 5678689u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2389(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[390753 + (index % 2389u)] * 2389u +
				globalHaltonSampler.fastPermutationTable[390753 + ((index / 2389u) % 2389u)]) * float(DoubleOneMinusEpsilon / 5707321u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2393(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[393142 + (index % 2393u)] * 2393u +
				globalHaltonSampler.fastPermutationTable[393142 + ((index / 2393u) % 2393u)]) * float(DoubleOneMinusEpsilon / 5726449u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2399(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[395535 + (index % 2399u)] * 2399u +
				globalHaltonSampler.fastPermutationTable[395535 + ((index / 2399u) % 2399u)]) * float(DoubleOneMinusEpsilon / 5755201u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2411(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[397934 + (index % 2411u)] * 2411u +
				globalHaltonSampler.fastPermutationTable[397934 + ((index / 2411u) % 2411u)]) * float(DoubleOneMinusEpsilon / 5812921u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2417(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[400345 + (index % 2417u)] * 2417u +
				globalHaltonSampler.fastPermutationTable[400345 + ((index / 2417u) % 2417u)]) * float(DoubleOneMinusEpsilon / 5841889u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2423(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[402762 + (index % 2423u)] * 2423u +
				globalHaltonSampler.fastPermutationTable[402762 + ((index / 2423u) % 2423u)]) * float(DoubleOneMinusEpsilon / 5870929u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2437(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[405185 + (index % 2437u)] * 2437u +
				globalHaltonSampler.fastPermutationTable[405185 + ((index / 2437u) % 2437u)]) * float(DoubleOneMinusEpsilon / 5938969u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2441(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[407622 + (index % 2441u)] * 2441u +
				globalHaltonSampler.fastPermutationTable[407622 + ((index / 2441u) % 2441u)]) * float(DoubleOneMinusEpsilon / 5958481u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2447(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[410063 + (index % 2447u)] * 2447u +
				globalHaltonSampler.fastPermutationTable[410063 + ((index / 2447u) % 2447u)]) * float(DoubleOneMinusEpsilon / 5987809u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2459(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[412510 + (index % 2459u)] * 2459u +
				globalHaltonSampler.fastPermutationTable[412510 + ((index / 2459u) % 2459u)]) * float(DoubleOneMinusEpsilon / 6046681u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2467(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[414969 + (index % 2467u)] * 2467u +
				globalHaltonSampler.fastPermutationTable[414969 + ((index / 2467u) % 2467u)]) * float(DoubleOneMinusEpsilon / 6086089u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2473(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[417436 + (index % 2473u)] * 2473u +
				globalHaltonSampler.fastPermutationTable[417436 + ((index / 2473u) % 2473u)]) * float(DoubleOneMinusEpsilon / 6115729u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2477(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[419909 + (index % 2477u)] * 2477u +
				globalHaltonSampler.fastPermutationTable[419909 + ((index / 2477u) % 2477u)]) * float(DoubleOneMinusEpsilon / 6135529u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2503(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[422386 + (index % 2503u)] * 2503u +
				globalHaltonSampler.fastPermutationTable[422386 + ((index / 2503u) % 2503u)]) * float(DoubleOneMinusEpsilon / 6265009u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2521(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[424889 + (index % 2521u)] * 2521u +
				globalHaltonSampler.fastPermutationTable[424889 + ((index / 2521u) % 2521u)]) * float(DoubleOneMinusEpsilon / 6355441u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2531(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[427410 + (index % 2531u)] * 2531u +
				globalHaltonSampler.fastPermutationTable[427410 + ((index / 2531u) % 2531u)]) * float(DoubleOneMinusEpsilon / 6405961u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2539(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[429941 + (index % 2539u)] * 2539u +
				globalHaltonSampler.fastPermutationTable[429941 + ((index / 2539u) % 2539u)]) * float(DoubleOneMinusEpsilon / 6446521u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2543(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[432480 + (index % 2543u)] * 2543u +
				globalHaltonSampler.fastPermutationTable[432480 + ((index / 2543u) % 2543u)]) * float(DoubleOneMinusEpsilon / 6466849u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2549(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[435023 + (index % 2549u)] * 2549u +
				globalHaltonSampler.fastPermutationTable[435023 + ((index / 2549u) % 2549u)]) * float(DoubleOneMinusEpsilon / 6497401u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2551(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[437572 + (index % 2551u)] * 2551u +
				globalHaltonSampler.fastPermutationTable[437572 + ((index / 2551u) % 2551u)]) * float(DoubleOneMinusEpsilon / 6507601u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2557(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[440123 + (index % 2557u)] * 2557u +
				globalHaltonSampler.fastPermutationTable[440123 + ((index / 2557u) % 2557u)]) * float(DoubleOneMinusEpsilon / 6538249u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2579(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[442680 + (index % 2579u)] * 2579u +
				globalHaltonSampler.fastPermutationTable[442680 + ((index / 2579u) % 2579u)]) * float(DoubleOneMinusEpsilon / 6651241u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2591(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[445259 + (index % 2591u)] * 2591u +
				globalHaltonSampler.fastPermutationTable[445259 + ((index / 2591u) % 2591u)]) * float(DoubleOneMinusEpsilon / 6713281u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2593(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[447850 + (index % 2593u)] * 2593u +
				globalHaltonSampler.fastPermutationTable[447850 + ((index / 2593u) % 2593u)]) * float(DoubleOneMinusEpsilon / 6723649u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2609(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[450443 + (index % 2609u)] * 2609u +
				globalHaltonSampler.fastPermutationTable[450443 + ((index / 2609u) % 2609u)]) * float(DoubleOneMinusEpsilon / 6806881u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2617(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[453052 + (index % 2617u)] * 2617u +
				globalHaltonSampler.fastPermutationTable[453052 + ((index / 2617u) % 2617u)]) * float(DoubleOneMinusEpsilon / 6848689u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2621(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[455669 + (index % 2621u)] * 2621u +
				globalHaltonSampler.fastPermutationTable[455669 + ((index / 2621u) % 2621u)]) * float(DoubleOneMinusEpsilon / 6869641u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2633(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[458290 + (index % 2633u)] * 2633u +
				globalHaltonSampler.fastPermutationTable[458290 + ((index / 2633u) % 2633u)]) * float(DoubleOneMinusEpsilon / 6932689u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2647(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[460923 + (index % 2647u)] * 2647u +
				globalHaltonSampler.fastPermutationTable[460923 + ((index / 2647u) % 2647u)]) * float(DoubleOneMinusEpsilon / 7006609u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2657(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[463570 + (index % 2657u)] * 2657u +
				globalHaltonSampler.fastPermutationTable[463570 + ((index / 2657u) % 2657u)]) * float(DoubleOneMinusEpsilon / 7059649u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2659(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[466227 + (index % 2659u)] * 2659u +
				globalHaltonSampler.fastPermutationTable[466227 + ((index / 2659u) % 2659u)]) * float(DoubleOneMinusEpsilon / 7070281u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2663(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[468886 + (index % 2663u)] * 2663u +
				globalHaltonSampler.fastPermutationTable[468886 + ((index / 2663u) % 2663u)]) * float(DoubleOneMinusEpsilon / 7091569u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2671(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[471549 + (index % 2671u)] * 2671u +
				globalHaltonSampler.fastPermutationTable[471549 + ((index / 2671u) % 2671u)]) * float(DoubleOneMinusEpsilon / 7134241u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2677(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[474220 + (index % 2677u)] * 2677u +
				globalHaltonSampler.fastPermutationTable[474220 + ((index / 2677u) % 2677u)]) * float(DoubleOneMinusEpsilon / 7166329u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2683(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[476897 + (index % 2683u)] * 2683u +
				globalHaltonSampler.fastPermutationTable[476897 + ((index / 2683u) % 2683u)]) * float(DoubleOneMinusEpsilon / 7198489u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2687(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[479580 + (index % 2687u)] * 2687u +
				globalHaltonSampler.fastPermutationTable[479580 + ((index / 2687u) % 2687u)]) * float(DoubleOneMinusEpsilon / 7219969u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2689(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[482267 + (index % 2689u)] * 2689u +
				globalHaltonSampler.fastPermutationTable[482267 + ((index / 2689u) % 2689u)]) * float(DoubleOneMinusEpsilon / 7230721u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2693(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[484956 + (index % 2693u)] * 2693u +
				globalHaltonSampler.fastPermutationTable[484956 + ((index / 2693u) % 2693u)]) * float(DoubleOneMinusEpsilon / 7252249u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2699(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[487649 + (index % 2699u)] * 2699u +
				globalHaltonSampler.fastPermutationTable[487649 + ((index / 2699u) % 2699u)]) * float(DoubleOneMinusEpsilon / 7284601u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2707(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[490348 + (index % 2707u)] * 2707u +
				globalHaltonSampler.fastPermutationTable[490348 + ((index / 2707u) % 2707u)]) * float(DoubleOneMinusEpsilon / 7327849u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2711(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[493055 + (index % 2711u)] * 2711u +
				globalHaltonSampler.fastPermutationTable[493055 + ((index / 2711u) % 2711u)]) * float(DoubleOneMinusEpsilon / 7349521u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2713(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[495766 + (index % 2713u)] * 2713u +
				globalHaltonSampler.fastPermutationTable[495766 + ((index / 2713u) % 2713u)]) * float(DoubleOneMinusEpsilon / 7360369u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2719(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[498479 + (index % 2719u)] * 2719u +
				globalHaltonSampler.fastPermutationTable[498479 + ((index / 2719u) % 2719u)]) * float(DoubleOneMinusEpsilon / 7392961u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2729(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[501198 + (index % 2729u)] * 2729u +
				globalHaltonSampler.fastPermutationTable[501198 + ((index / 2729u) % 2729u)]) * float(DoubleOneMinusEpsilon / 7447441u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2731(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[503927 + (index % 2731u)] * 2731u +
				globalHaltonSampler.fastPermutationTable[503927 + ((index / 2731u) % 2731u)]) * float(DoubleOneMinusEpsilon / 7458361u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2741(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[506658 + (index % 2741u)] * 2741u +
				globalHaltonSampler.fastPermutationTable[506658 + ((index / 2741u) % 2741u)]) * float(DoubleOneMinusEpsilon / 7513081u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2749(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[509399 + (index % 2749u)] * 2749u +
				globalHaltonSampler.fastPermutationTable[509399 + ((index / 2749u) % 2749u)]) * float(DoubleOneMinusEpsilon / 7557001u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2753(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[512148 + (index % 2753u)] * 2753u +
				globalHaltonSampler.fastPermutationTable[512148 + ((index / 2753u) % 2753u)]) * float(DoubleOneMinusEpsilon / 7579009u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2767(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[514901 + (index % 2767u)] * 2767u +
				globalHaltonSampler.fastPermutationTable[514901 + ((index / 2767u) % 2767u)]) * float(DoubleOneMinusEpsilon / 7656289u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2777(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[517668 + (index % 2777u)] * 2777u +
				globalHaltonSampler.fastPermutationTable[517668 + ((index / 2777u) % 2777u)]) * float(DoubleOneMinusEpsilon / 7711729u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2789(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[520445 + (index % 2789u)] * 2789u +
				globalHaltonSampler.fastPermutationTable[520445 + ((index / 2789u) % 2789u)]) * float(DoubleOneMinusEpsilon / 7778521u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2791(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[523234 + (index % 2791u)] * 2791u +
				globalHaltonSampler.fastPermutationTable[523234 + ((index / 2791u) % 2791u)]) * float(DoubleOneMinusEpsilon / 7789681u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2797(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[526025 + (index % 2797u)] * 2797u +
				globalHaltonSampler.fastPermutationTable[526025 + ((index / 2797u) % 2797u)]) * float(DoubleOneMinusEpsilon / 7823209u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2801(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[528822 + (index % 2801u)] * 2801u +
				globalHaltonSampler.fastPermutationTable[528822 + ((index / 2801u) % 2801u)]) * float(DoubleOneMinusEpsilon / 7845601u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2803(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[531623 + (index % 2803u)] * 2803u +
				globalHaltonSampler.fastPermutationTable[531623 + ((index / 2803u) % 2803u)]) * float(DoubleOneMinusEpsilon / 7856809u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2819(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[534426 + (index % 2819u)] * 2819u +
				globalHaltonSampler.fastPermutationTable[534426 + ((index / 2819u) % 2819u)]) * float(DoubleOneMinusEpsilon / 7946761u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2833(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[537245 + (index % 2833u)] * 2833u +
				globalHaltonSampler.fastPermutationTable[537245 + ((index / 2833u) % 2833u)]) * float(DoubleOneMinusEpsilon / 8025889u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2837(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[540078 + (index % 2837u)] * 2837u +
				globalHaltonSampler.fastPermutationTable[540078 + ((index / 2837u) % 2837u)]) * float(DoubleOneMinusEpsilon / 8048569u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2843(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[542915 + (index % 2843u)] * 2843u +
				globalHaltonSampler.fastPermutationTable[542915 + ((index / 2843u) % 2843u)]) * float(DoubleOneMinusEpsilon / 8082649u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2851(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[545758 + (index % 2851u)] * 2851u +
				globalHaltonSampler.fastPermutationTable[545758 + ((index / 2851u) % 2851u)]) * float(DoubleOneMinusEpsilon / 8128201u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2857(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[548609 + (index % 2857u)] * 2857u +
				globalHaltonSampler.fastPermutationTable[548609 + ((index / 2857u) % 2857u)]) * float(DoubleOneMinusEpsilon / 8162449u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2861(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[551466 + (index % 2861u)] * 2861u +
				globalHaltonSampler.fastPermutationTable[551466 + ((index / 2861u) % 2861u)]) * float(DoubleOneMinusEpsilon / 8185321u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2879(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[554327 + (index % 2879u)] * 2879u +
				globalHaltonSampler.fastPermutationTable[554327 + ((index / 2879u) % 2879u)]) * float(DoubleOneMinusEpsilon / 8288641u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2887(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[557206 + (index % 2887u)] * 2887u +
				globalHaltonSampler.fastPermutationTable[557206 + ((index / 2887u) % 2887u)]) * float(DoubleOneMinusEpsilon / 8334769u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2897(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[560093 + (index % 2897u)] * 2897u +
				globalHaltonSampler.fastPermutationTable[560093 + ((index / 2897u) % 2897u)]) * float(DoubleOneMinusEpsilon / 8392609u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2903(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[562990 + (index % 2903u)] * 2903u +
				globalHaltonSampler.fastPermutationTable[562990 + ((index / 2903u) % 2903u)]) * float(DoubleOneMinusEpsilon / 8427409u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2909(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[565893 + (index % 2909u)] * 2909u +
				globalHaltonSampler.fastPermutationTable[565893 + ((index / 2909u) % 2909u)]) * float(DoubleOneMinusEpsilon / 8462281u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2917(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[568802 + (index % 2917u)] * 2917u +
				globalHaltonSampler.fastPermutationTable[568802 + ((index / 2917u) % 2917u)]) * float(DoubleOneMinusEpsilon / 8508889u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2927(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[571719 + (index % 2927u)] * 2927u +
				globalHaltonSampler.fastPermutationTable[571719 + ((index / 2927u) % 2927u)]) * float(DoubleOneMinusEpsilon / 8567329u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2939(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[574646 + (index % 2939u)] * 2939u +
				globalHaltonSampler.fastPermutationTable[574646 + ((index / 2939u) % 2939u)]) * float(DoubleOneMinusEpsilon / 8637721u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2953(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[577585 + (index % 2953u)] * 2953u +
				globalHaltonSampler.fastPermutationTable[577585 + ((index / 2953u) % 2953u)]) * float(DoubleOneMinusEpsilon / 8720209u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2957(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[580538 + (index % 2957u)] * 2957u +
				globalHaltonSampler.fastPermutationTable[580538 + ((index / 2957u) % 2957u)]) * float(DoubleOneMinusEpsilon / 8743849u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2963(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[583495 + (index % 2963u)] * 2963u +
				globalHaltonSampler.fastPermutationTable[583495 + ((index / 2963u) % 2963u)]) * float(DoubleOneMinusEpsilon / 8779369u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2969(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[586458 + (index % 2969u)] * 2969u +
				globalHaltonSampler.fastPermutationTable[586458 + ((index / 2969u) % 2969u)]) * float(DoubleOneMinusEpsilon / 8814961u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2971(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[589427 + (index % 2971u)] * 2971u +
				globalHaltonSampler.fastPermutationTable[589427 + ((index / 2971u) % 2971u)]) * float(DoubleOneMinusEpsilon / 8826841u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized2999(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[592398 + (index % 2999u)] * 2999u +
				globalHaltonSampler.fastPermutationTable[592398 + ((index / 2999u) % 2999u)]) * float(DoubleOneMinusEpsilon / 8994001u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3001(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[595397 + (index % 3001u)] * 3001u +
				globalHaltonSampler.fastPermutationTable[595397 + ((index / 3001u) % 3001u)]) * float(DoubleOneMinusEpsilon / 9006001u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3011(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[598398 + (index % 3011u)] * 3011u +
				globalHaltonSampler.fastPermutationTable[598398 + ((index / 3011u) % 3011u)]) * float(DoubleOneMinusEpsilon / 9066121u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3019(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[601409 + (index % 3019u)] * 3019u +
				globalHaltonSampler.fastPermutationTable[601409 + ((index / 3019u) % 3019u)]) * float(DoubleOneMinusEpsilon / 9114361u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3023(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[604428 + (index % 3023u)] * 3023u +
				globalHaltonSampler.fastPermutationTable[604428 + ((index / 3023u) % 3023u)]) * float(DoubleOneMinusEpsilon / 9138529u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3037(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[607451 + (index % 3037u)] * 3037u +
				globalHaltonSampler.fastPermutationTable[607451 + ((index / 3037u) % 3037u)]) * float(DoubleOneMinusEpsilon / 9223369u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3041(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[610488 + (index % 3041u)] * 3041u +
				globalHaltonSampler.fastPermutationTable[610488 + ((index / 3041u) % 3041u)]) * float(DoubleOneMinusEpsilon / 9247681u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3049(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[613529 + (index % 3049u)] * 3049u +
				globalHaltonSampler.fastPermutationTable[613529 + ((index / 3049u) % 3049u)]) * float(DoubleOneMinusEpsilon / 9296401u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3061(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[616578 + (index % 3061u)] * 3061u +
				globalHaltonSampler.fastPermutationTable[616578 + ((index / 3061u) % 3061u)]) * float(DoubleOneMinusEpsilon / 9369721u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3067(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[619639 + (index % 3067u)] * 3067u +
				globalHaltonSampler.fastPermutationTable[619639 + ((index / 3067u) % 3067u)]) * float(DoubleOneMinusEpsilon / 9406489u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3079(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[622706 + (index % 3079u)] * 3079u +
				globalHaltonSampler.fastPermutationTable[622706 + ((index / 3079u) % 3079u)]) * float(DoubleOneMinusEpsilon / 9480241u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3083(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[625785 + (index % 3083u)] * 3083u +
				globalHaltonSampler.fastPermutationTable[625785 + ((index / 3083u) % 3083u)]) * float(DoubleOneMinusEpsilon / 9504889u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3089(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[628868 + (index % 3089u)] * 3089u +
				globalHaltonSampler.fastPermutationTable[628868 + ((index / 3089u) % 3089u)]) * float(DoubleOneMinusEpsilon / 9541921u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3109(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[631957 + (index % 3109u)] * 3109u +
				globalHaltonSampler.fastPermutationTable[631957 + ((index / 3109u) % 3109u)]) * float(DoubleOneMinusEpsilon / 9665881u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3119(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[635066 + (index % 3119u)] * 3119u +
				globalHaltonSampler.fastPermutationTable[635066 + ((index / 3119u) % 3119u)]) * float(DoubleOneMinusEpsilon / 9728161u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3121(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[638185 + (index % 3121u)] * 3121u +
				globalHaltonSampler.fastPermutationTable[638185 + ((index / 3121u) % 3121u)]) * float(DoubleOneMinusEpsilon / 9740641u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3137(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[641306 + (index % 3137u)] * 3137u +
				globalHaltonSampler.fastPermutationTable[641306 + ((index / 3137u) % 3137u)]) * float(DoubleOneMinusEpsilon / 9840769u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3163(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[644443 + (index % 3163u)] * 3163u +
				globalHaltonSampler.fastPermutationTable[644443 + ((index / 3163u) % 3163u)]) * float(DoubleOneMinusEpsilon / 10004569u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3167(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[647606 + (index % 3167u)] * 3167u +
				globalHaltonSampler.fastPermutationTable[647606 + ((index / 3167u) % 3167u)]) * float(DoubleOneMinusEpsilon / 10029889u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3169(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[650773 + (index % 3169u)] * 3169u +
				globalHaltonSampler.fastPermutationTable[650773 + ((index / 3169u) % 3169u)]) * float(DoubleOneMinusEpsilon / 10042561u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3181(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[653942 + (index % 3181u)] * 3181u +
				globalHaltonSampler.fastPermutationTable[653942 + ((index / 3181u) % 3181u)]) * float(DoubleOneMinusEpsilon / 10118761u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3187(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[657123 + (index % 3187u)] * 3187u +
				globalHaltonSampler.fastPermutationTable[657123 + ((index / 3187u) % 3187u)]) * float(DoubleOneMinusEpsilon / 10156969u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3191(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[660310 + (index % 3191u)] * 3191u +
				globalHaltonSampler.fastPermutationTable[660310 + ((index / 3191u) % 3191u)]) * float(DoubleOneMinusEpsilon / 10182481u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3203(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[663501 + (index % 3203u)] * 3203u +
				globalHaltonSampler.fastPermutationTable[663501 + ((index / 3203u) % 3203u)]) * float(DoubleOneMinusEpsilon / 10259209u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3209(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[666704 + (index % 3209u)] * 3209u +
				globalHaltonSampler.fastPermutationTable[666704 + ((index / 3209u) % 3209u)]) * float(DoubleOneMinusEpsilon / 10297681u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3217(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[669913 + (index % 3217u)] * 3217u +
				globalHaltonSampler.fastPermutationTable[669913 + ((index / 3217u) % 3217u)]) * float(DoubleOneMinusEpsilon / 10349089u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3221(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[673130 + (index % 3221u)] * 3221u +
				globalHaltonSampler.fastPermutationTable[673130 + ((index / 3221u) % 3221u)]) * float(DoubleOneMinusEpsilon / 10374841u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3229(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[676351 + (index % 3229u)] * 3229u +
				globalHaltonSampler.fastPermutationTable[676351 + ((index / 3229u) % 3229u)]) * float(DoubleOneMinusEpsilon / 10426441u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3251(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[679580 + (index % 3251u)] * 3251u +
				globalHaltonSampler.fastPermutationTable[679580 + ((index / 3251u) % 3251u)]) * float(DoubleOneMinusEpsilon / 10569001u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3253(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[682831 + (index % 3253u)] * 3253u +
				globalHaltonSampler.fastPermutationTable[682831 + ((index / 3253u) % 3253u)]) * float(DoubleOneMinusEpsilon / 10582009u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3257(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[686084 + (index % 3257u)] * 3257u +
				globalHaltonSampler.fastPermutationTable[686084 + ((index / 3257u) % 3257u)]) * float(DoubleOneMinusEpsilon / 10608049u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3259(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[689341 + (index % 3259u)] * 3259u +
				globalHaltonSampler.fastPermutationTable[689341 + ((index / 3259u) % 3259u)]) * float(DoubleOneMinusEpsilon / 10621081u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3271(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[692600 + (index % 3271u)] * 3271u +
				globalHaltonSampler.fastPermutationTable[692600 + ((index / 3271u) % 3271u)]) * float(DoubleOneMinusEpsilon / 10699441u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3299(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[695871 + (index % 3299u)] * 3299u +
				globalHaltonSampler.fastPermutationTable[695871 + ((index / 3299u) % 3299u)]) * float(DoubleOneMinusEpsilon / 10883401u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3301(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[699170 + (index % 3301u)] * 3301u +
				globalHaltonSampler.fastPermutationTable[699170 + ((index / 3301u) % 3301u)]) * float(DoubleOneMinusEpsilon / 10896601u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3307(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[702471 + (index % 3307u)] * 3307u +
				globalHaltonSampler.fastPermutationTable[702471 + ((index / 3307u) % 3307u)]) * float(DoubleOneMinusEpsilon / 10936249u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3313(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[705778 + (index % 3313u)] * 3313u +
				globalHaltonSampler.fastPermutationTable[705778 + ((index / 3313u) % 3313u)]) * float(DoubleOneMinusEpsilon / 10975969u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3319(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[709091 + (index % 3319u)] * 3319u +
				globalHaltonSampler.fastPermutationTable[709091 + ((index / 3319u) % 3319u)]) * float(DoubleOneMinusEpsilon / 11015761u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3323(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[712410 + (index % 3323u)] * 3323u +
				globalHaltonSampler.fastPermutationTable[712410 + ((index / 3323u) % 3323u)]) * float(DoubleOneMinusEpsilon / 11042329u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3329(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[715733 + (index % 3329u)] * 3329u +
				globalHaltonSampler.fastPermutationTable[715733 + ((index / 3329u) % 3329u)]) * float(DoubleOneMinusEpsilon / 11082241u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3331(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[719062 + (index % 3331u)] * 3331u +
				globalHaltonSampler.fastPermutationTable[719062 + ((index / 3331u) % 3331u)]) * float(DoubleOneMinusEpsilon / 11095561u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3343(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[722393 + (index % 3343u)] * 3343u +
				globalHaltonSampler.fastPermutationTable[722393 + ((index / 3343u) % 3343u)]) * float(DoubleOneMinusEpsilon / 11175649u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3347(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[725736 + (index % 3347u)] * 3347u +
				globalHaltonSampler.fastPermutationTable[725736 + ((index / 3347u) % 3347u)]) * float(DoubleOneMinusEpsilon / 11202409u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3359(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[729083 + (index % 3359u)] * 3359u +
				globalHaltonSampler.fastPermutationTable[729083 + ((index / 3359u) % 3359u)]) * float(DoubleOneMinusEpsilon / 11282881u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3361(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[732442 + (index % 3361u)] * 3361u +
				globalHaltonSampler.fastPermutationTable[732442 + ((index / 3361u) % 3361u)]) * float(DoubleOneMinusEpsilon / 11296321u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3371(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[735803 + (index % 3371u)] * 3371u +
				globalHaltonSampler.fastPermutationTable[735803 + ((index / 3371u) % 3371u)]) * float(DoubleOneMinusEpsilon / 11363641u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3373(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[739174 + (index % 3373u)] * 3373u +
				globalHaltonSampler.fastPermutationTable[739174 + ((index / 3373u) % 3373u)]) * float(DoubleOneMinusEpsilon / 11377129u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3389(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[742547 + (index % 3389u)] * 3389u +
				globalHaltonSampler.fastPermutationTable[742547 + ((index / 3389u) % 3389u)]) * float(DoubleOneMinusEpsilon / 11485321u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3391(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[745936 + (index % 3391u)] * 3391u +
				globalHaltonSampler.fastPermutationTable[745936 + ((index / 3391u) % 3391u)]) * float(DoubleOneMinusEpsilon / 11498881u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3407(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[749327 + (index % 3407u)] * 3407u +
				globalHaltonSampler.fastPermutationTable[749327 + ((index / 3407u) % 3407u)]) * float(DoubleOneMinusEpsilon / 11607649u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3413(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[752734 + (index % 3413u)] * 3413u +
				globalHaltonSampler.fastPermutationTable[752734 + ((index / 3413u) % 3413u)]) * float(DoubleOneMinusEpsilon / 11648569u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3433(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[756147 + (index % 3433u)] * 3433u +
				globalHaltonSampler.fastPermutationTable[756147 + ((index / 3433u) % 3433u)]) * float(DoubleOneMinusEpsilon / 11785489u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3449(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[759580 + (index % 3449u)] * 3449u +
				globalHaltonSampler.fastPermutationTable[759580 + ((index / 3449u) % 3449u)]) * float(DoubleOneMinusEpsilon / 11895601u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3457(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[763029 + (index % 3457u)] * 3457u +
				globalHaltonSampler.fastPermutationTable[763029 + ((index / 3457u) % 3457u)]) * float(DoubleOneMinusEpsilon / 11950849u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3461(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[766486 + (index % 3461u)] * 3461u +
				globalHaltonSampler.fastPermutationTable[766486 + ((index / 3461u) % 3461u)]) * float(DoubleOneMinusEpsilon / 11978521u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3463(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[769947 + (index % 3463u)] * 3463u +
				globalHaltonSampler.fastPermutationTable[769947 + ((index / 3463u) % 3463u)]) * float(DoubleOneMinusEpsilon / 11992369u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3467(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[773410 + (index % 3467u)] * 3467u +
				globalHaltonSampler.fastPermutationTable[773410 + ((index / 3467u) % 3467u)]) * float(DoubleOneMinusEpsilon / 12020089u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3469(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[776877 + (index % 3469u)] * 3469u +
				globalHaltonSampler.fastPermutationTable[776877 + ((index / 3469u) % 3469u)]) * float(DoubleOneMinusEpsilon / 12033961u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3491(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[780346 + (index % 3491u)] * 3491u +
				globalHaltonSampler.fastPermutationTable[780346 + ((index / 3491u) % 3491u)]) * float(DoubleOneMinusEpsilon / 12187081u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3499(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[783837 + (index % 3499u)] * 3499u +
				globalHaltonSampler.fastPermutationTable[783837 + ((index / 3499u) % 3499u)]) * float(DoubleOneMinusEpsilon / 12243001u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3511(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[787336 + (index % 3511u)] * 3511u +
				globalHaltonSampler.fastPermutationTable[787336 + ((index / 3511u) % 3511u)]) * float(DoubleOneMinusEpsilon / 12327121u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3517(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[790847 + (index % 3517u)] * 3517u +
				globalHaltonSampler.fastPermutationTable[790847 + ((index / 3517u) % 3517u)]) * float(DoubleOneMinusEpsilon / 12369289u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3527(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[794364 + (index % 3527u)] * 3527u +
				globalHaltonSampler.fastPermutationTable[794364 + ((index / 3527u) % 3527u)]) * float(DoubleOneMinusEpsilon / 12439729u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3529(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[797891 + (index % 3529u)] * 3529u +
				globalHaltonSampler.fastPermutationTable[797891 + ((index / 3529u) % 3529u)]) * float(DoubleOneMinusEpsilon / 12453841u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3533(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[801420 + (index % 3533u)] * 3533u +
				globalHaltonSampler.fastPermutationTable[801420 + ((index / 3533u) % 3533u)]) * float(DoubleOneMinusEpsilon / 12482089u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3539(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[804953 + (index % 3539u)] * 3539u +
				globalHaltonSampler.fastPermutationTable[804953 + ((index / 3539u) % 3539u)]) * float(DoubleOneMinusEpsilon / 12524521u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3541(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[808492 + (index % 3541u)] * 3541u +
				globalHaltonSampler.fastPermutationTable[808492 + ((index / 3541u) % 3541u)]) * float(DoubleOneMinusEpsilon / 12538681u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3547(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[812033 + (index % 3547u)] * 3547u +
				globalHaltonSampler.fastPermutationTable[812033 + ((index / 3547u) % 3547u)]) * float(DoubleOneMinusEpsilon / 12581209u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3557(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[815580 + (index % 3557u)] * 3557u +
				globalHaltonSampler.fastPermutationTable[815580 + ((index / 3557u) % 3557u)]) * float(DoubleOneMinusEpsilon / 12652249u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3559(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[819137 + (index % 3559u)] * 3559u +
				globalHaltonSampler.fastPermutationTable[819137 + ((index / 3559u) % 3559u)]) * float(DoubleOneMinusEpsilon / 12666481u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3571(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[822696 + (index % 3571u)] * 3571u +
				globalHaltonSampler.fastPermutationTable[822696 + ((index / 3571u) % 3571u)]) * float(DoubleOneMinusEpsilon / 12752041u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3581(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[826267 + (index % 3581u)] * 3581u +
				globalHaltonSampler.fastPermutationTable[826267 + ((index / 3581u) % 3581u)]) * float(DoubleOneMinusEpsilon / 12823561u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3583(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[829848 + (index % 3583u)] * 3583u +
				globalHaltonSampler.fastPermutationTable[829848 + ((index / 3583u) % 3583u)]) * float(DoubleOneMinusEpsilon / 12837889u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3593(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[833431 + (index % 3593u)] * 3593u +
				globalHaltonSampler.fastPermutationTable[833431 + ((index / 3593u) % 3593u)]) * float(DoubleOneMinusEpsilon / 12909649u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3607(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[837024 + (index % 3607u)] * 3607u +
				globalHaltonSampler.fastPermutationTable[837024 + ((index / 3607u) % 3607u)]) * float(DoubleOneMinusEpsilon / 13010449u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3613(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[840631 + (index % 3613u)] * 3613u +
				globalHaltonSampler.fastPermutationTable[840631 + ((index / 3613u) % 3613u)]) * float(DoubleOneMinusEpsilon / 13053769u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3617(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[844244 + (index % 3617u)] * 3617u +
				globalHaltonSampler.fastPermutationTable[844244 + ((index / 3617u) % 3617u)]) * float(DoubleOneMinusEpsilon / 13082689u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3623(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[847861 + (index % 3623u)] * 3623u +
				globalHaltonSampler.fastPermutationTable[847861 + ((index / 3623u) % 3623u)]) * float(DoubleOneMinusEpsilon / 13126129u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3631(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[851484 + (index % 3631u)] * 3631u +
				globalHaltonSampler.fastPermutationTable[851484 + ((index / 3631u) % 3631u)]) * float(DoubleOneMinusEpsilon / 13184161u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3637(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[855115 + (index % 3637u)] * 3637u +
				globalHaltonSampler.fastPermutationTable[855115 + ((index / 3637u) % 3637u)]) * float(DoubleOneMinusEpsilon / 13227769u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3643(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[858752 + (index % 3643u)] * 3643u +
				globalHaltonSampler.fastPermutationTable[858752 + ((index / 3643u) % 3643u)]) * float(DoubleOneMinusEpsilon / 13271449u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3659(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[862395 + (index % 3659u)] * 3659u +
				globalHaltonSampler.fastPermutationTable[862395 + ((index / 3659u) % 3659u)]) * float(DoubleOneMinusEpsilon / 13388281u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3671(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[866054 + (index % 3671u)] * 3671u +
				globalHaltonSampler.fastPermutationTable[866054 + ((index / 3671u) % 3671u)]) * float(DoubleOneMinusEpsilon / 13476241u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3673(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[869725 + (index % 3673u)] * 3673u +
				globalHaltonSampler.fastPermutationTable[869725 + ((index / 3673u) % 3673u)]) * float(DoubleOneMinusEpsilon / 13490929u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3677(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[873398 + (index % 3677u)] * 3677u +
				globalHaltonSampler.fastPermutationTable[873398 + ((index / 3677u) % 3677u)]) * float(DoubleOneMinusEpsilon / 13520329u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3691(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[877075 + (index % 3691u)] * 3691u +
				globalHaltonSampler.fastPermutationTable[877075 + ((index / 3691u) % 3691u)]) * float(DoubleOneMinusEpsilon / 13623481u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3697(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[880766 + (index % 3697u)] * 3697u +
				globalHaltonSampler.fastPermutationTable[880766 + ((index / 3697u) % 3697u)]) * float(DoubleOneMinusEpsilon / 13667809u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3701(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[884463 + (index % 3701u)] * 3701u +
				globalHaltonSampler.fastPermutationTable[884463 + ((index / 3701u) % 3701u)]) * float(DoubleOneMinusEpsilon / 13697401u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3709(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[888164 + (index % 3709u)] * 3709u +
				globalHaltonSampler.fastPermutationTable[888164 + ((index / 3709u) % 3709u)]) * float(DoubleOneMinusEpsilon / 13756681u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3719(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[891873 + (index % 3719u)] * 3719u +
				globalHaltonSampler.fastPermutationTable[891873 + ((index / 3719u) % 3719u)]) * float(DoubleOneMinusEpsilon / 13830961u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3727(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[895592 + (index % 3727u)] * 3727u +
				globalHaltonSampler.fastPermutationTable[895592 + ((index / 3727u) % 3727u)]) * float(DoubleOneMinusEpsilon / 13890529u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3733(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[899319 + (index % 3733u)] * 3733u +
				globalHaltonSampler.fastPermutationTable[899319 + ((index / 3733u) % 3733u)]) * float(DoubleOneMinusEpsilon / 13935289u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3739(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[903052 + (index % 3739u)] * 3739u +
				globalHaltonSampler.fastPermutationTable[903052 + ((index / 3739u) % 3739u)]) * float(DoubleOneMinusEpsilon / 13980121u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3761(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[906791 + (index % 3761u)] * 3761u +
				globalHaltonSampler.fastPermutationTable[906791 + ((index / 3761u) % 3761u)]) * float(DoubleOneMinusEpsilon / 14145121u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3767(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[910552 + (index % 3767u)] * 3767u +
				globalHaltonSampler.fastPermutationTable[910552 + ((index / 3767u) % 3767u)]) * float(DoubleOneMinusEpsilon / 14190289u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3769(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[914319 + (index % 3769u)] * 3769u +
				globalHaltonSampler.fastPermutationTable[914319 + ((index / 3769u) % 3769u)]) * float(DoubleOneMinusEpsilon / 14205361u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3779(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[918088 + (index % 3779u)] * 3779u +
				globalHaltonSampler.fastPermutationTable[918088 + ((index / 3779u) % 3779u)]) * float(DoubleOneMinusEpsilon / 14280841u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3793(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[921867 + (index % 3793u)] * 3793u +
				globalHaltonSampler.fastPermutationTable[921867 + ((index / 3793u) % 3793u)]) * float(DoubleOneMinusEpsilon / 14386849u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3797(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[925660 + (index % 3797u)] * 3797u +
				globalHaltonSampler.fastPermutationTable[925660 + ((index / 3797u) % 3797u)]) * float(DoubleOneMinusEpsilon / 14417209u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3803(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[929457 + (index % 3803u)] * 3803u +
				globalHaltonSampler.fastPermutationTable[929457 + ((index / 3803u) % 3803u)]) * float(DoubleOneMinusEpsilon / 14462809u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3821(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[933260 + (index % 3821u)] * 3821u +
				globalHaltonSampler.fastPermutationTable[933260 + ((index / 3821u) % 3821u)]) * float(DoubleOneMinusEpsilon / 14600041u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3823(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[937081 + (index % 3823u)] * 3823u +
				globalHaltonSampler.fastPermutationTable[937081 + ((index / 3823u) % 3823u)]) * float(DoubleOneMinusEpsilon / 14615329u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3833(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[940904 + (index % 3833u)] * 3833u +
				globalHaltonSampler.fastPermutationTable[940904 + ((index / 3833u) % 3833u)]) * float(DoubleOneMinusEpsilon / 14691889u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3847(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[944737 + (index % 3847u)] * 3847u +
				globalHaltonSampler.fastPermutationTable[944737 + ((index / 3847u) % 3847u)]) * float(DoubleOneMinusEpsilon / 14799409u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3851(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[948584 + (index % 3851u)] * 3851u +
				globalHaltonSampler.fastPermutationTable[948584 + ((index / 3851u) % 3851u)]) * float(DoubleOneMinusEpsilon / 14830201u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3853(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[952435 + (index % 3853u)] * 3853u +
				globalHaltonSampler.fastPermutationTable[952435 + ((index / 3853u) % 3853u)]) * float(DoubleOneMinusEpsilon / 14845609u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3863(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[956288 + (index % 3863u)] * 3863u +
				globalHaltonSampler.fastPermutationTable[956288 + ((index / 3863u) % 3863u)]) * float(DoubleOneMinusEpsilon / 14922769u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3877(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[960151 + (index % 3877u)] * 3877u +
				globalHaltonSampler.fastPermutationTable[960151 + ((index / 3877u) % 3877u)]) * float(DoubleOneMinusEpsilon / 15031129u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3881(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[964028 + (index % 3881u)] * 3881u +
				globalHaltonSampler.fastPermutationTable[964028 + ((index / 3881u) % 3881u)]) * float(DoubleOneMinusEpsilon / 15062161u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3889(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[967909 + (index % 3889u)] * 3889u +
				globalHaltonSampler.fastPermutationTable[967909 + ((index / 3889u) % 3889u)]) * float(DoubleOneMinusEpsilon / 15124321u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3907(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[971798 + (index % 3907u)] * 3907u +
				globalHaltonSampler.fastPermutationTable[971798 + ((index / 3907u) % 3907u)]) * float(DoubleOneMinusEpsilon / 15264649u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3911(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[975705 + (index % 3911u)] * 3911u +
				globalHaltonSampler.fastPermutationTable[975705 + ((index / 3911u) % 3911u)]) * float(DoubleOneMinusEpsilon / 15295921u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3917(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[979616 + (index % 3917u)] * 3917u +
				globalHaltonSampler.fastPermutationTable[979616 + ((index / 3917u) % 3917u)]) * float(DoubleOneMinusEpsilon / 15342889u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3919(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[983533 + (index % 3919u)] * 3919u +
				globalHaltonSampler.fastPermutationTable[983533 + ((index / 3919u) % 3919u)]) * float(DoubleOneMinusEpsilon / 15358561u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3923(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[987452 + (index % 3923u)] * 3923u +
				globalHaltonSampler.fastPermutationTable[987452 + ((index / 3923u) % 3923u)]) * float(DoubleOneMinusEpsilon / 15389929u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3929(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[991375 + (index % 3929u)] * 3929u +
				globalHaltonSampler.fastPermutationTable[991375 + ((index / 3929u) % 3929u)]) * float(DoubleOneMinusEpsilon / 15437041u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3931(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[995304 + (index % 3931u)] * 3931u +
				globalHaltonSampler.fastPermutationTable[995304 + ((index / 3931u) % 3931u)]) * float(DoubleOneMinusEpsilon / 15452761u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3943(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[999235 + (index % 3943u)] * 3943u +
				globalHaltonSampler.fastPermutationTable[999235 + ((index / 3943u) % 3943u)]) * float(DoubleOneMinusEpsilon / 15547249u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3947(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1003178 + (index % 3947u)] * 3947u +
				globalHaltonSampler.fastPermutationTable[1003178 + ((index / 3947u) % 3947u)]) * float(DoubleOneMinusEpsilon / 15578809u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3967(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1007125 + (index % 3967u)] * 3967u +
				globalHaltonSampler.fastPermutationTable[1007125 + ((index / 3967u) % 3967u)]) * float(DoubleOneMinusEpsilon / 15737089u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized3989(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1011092 + (index % 3989u)] * 3989u +
				globalHaltonSampler.fastPermutationTable[1011092 + ((index / 3989u) % 3989u)]) * float(DoubleOneMinusEpsilon / 15912121u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4001(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1015081 + (index % 4001u)] * 4001u +
				globalHaltonSampler.fastPermutationTable[1015081 + ((index / 4001u) % 4001u)]) * float(DoubleOneMinusEpsilon / 16008001u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4003(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1019082 + (index % 4003u)] * 4003u +
				globalHaltonSampler.fastPermutationTable[1019082 + ((index / 4003u) % 4003u)]) * float(DoubleOneMinusEpsilon / 16024009u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4007(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1023085 + (index % 4007u)] * 4007u +
				globalHaltonSampler.fastPermutationTable[1023085 + ((index / 4007u) % 4007u)]) * float(DoubleOneMinusEpsilon / 16056049u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4013(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1027092 + (index % 4013u)] * 4013u +
				globalHaltonSampler.fastPermutationTable[1027092 + ((index / 4013u) % 4013u)]) * float(DoubleOneMinusEpsilon / 16104169u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4019(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1031105 + (index % 4019u)] * 4019u +
				globalHaltonSampler.fastPermutationTable[1031105 + ((index / 4019u) % 4019u)]) * float(DoubleOneMinusEpsilon / 16152361u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4021(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1035124 + (index % 4021u)] * 4021u +
				globalHaltonSampler.fastPermutationTable[1035124 + ((index / 4021u) % 4021u)]) * float(DoubleOneMinusEpsilon / 16168441u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4027(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1039145 + (index % 4027u)] * 4027u +
				globalHaltonSampler.fastPermutationTable[1039145 + ((index / 4027u) % 4027u)]) * float(DoubleOneMinusEpsilon / 16216729u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4049(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1043172 + (index % 4049u)] * 4049u +
				globalHaltonSampler.fastPermutationTable[1043172 + ((index / 4049u) % 4049u)]) * float(DoubleOneMinusEpsilon / 16394401u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4051(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1047221 + (index % 4051u)] * 4051u +
				globalHaltonSampler.fastPermutationTable[1047221 + ((index / 4051u) % 4051u)]) * float(DoubleOneMinusEpsilon / 16410601u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4057(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1051272 + (index % 4057u)] * 4057u +
				globalHaltonSampler.fastPermutationTable[1051272 + ((index / 4057u) % 4057u)]) * float(DoubleOneMinusEpsilon / 16459249u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4073(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1055329 + (index % 4073u)] * 4073u +
				globalHaltonSampler.fastPermutationTable[1055329 + ((index / 4073u) % 4073u)]) * float(DoubleOneMinusEpsilon / 16589329u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4079(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1059402 + (index % 4079u)] * 4079u +
				globalHaltonSampler.fastPermutationTable[1059402 + ((index / 4079u) % 4079u)]) * float(DoubleOneMinusEpsilon / 16638241u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4091(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1063481 + (index % 4091u)] * 4091u +
				globalHaltonSampler.fastPermutationTable[1063481 + ((index / 4091u) % 4091u)]) * float(DoubleOneMinusEpsilon / 16736281u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4093(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1067572 + (index % 4093u)] * 4093u +
				globalHaltonSampler.fastPermutationTable[1067572 + ((index / 4093u) % 4093u)]) * float(DoubleOneMinusEpsilon / 16752649u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4099(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1071665 + (index % 4099u)] * 4099u +
				globalHaltonSampler.fastPermutationTable[1071665 + ((index / 4099u) % 4099u)]) * float(DoubleOneMinusEpsilon / 16801801u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4111(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1075764 + (index % 4111u)] * 4111u +
				globalHaltonSampler.fastPermutationTable[1075764 + ((index / 4111u) % 4111u)]) * float(DoubleOneMinusEpsilon / 16900321u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4127(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1079875 + (index % 4127u)] * 4127u +
				globalHaltonSampler.fastPermutationTable[1079875 + ((index / 4127u) % 4127u)]) * float(DoubleOneMinusEpsilon / 17032129u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4129(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1084002 + (index % 4129u)] * 4129u +
				globalHaltonSampler.fastPermutationTable[1084002 + ((index / 4129u) % 4129u)]) * float(DoubleOneMinusEpsilon / 17048641u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4133(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1088131 + (index % 4133u)] * 4133u +
				globalHaltonSampler.fastPermutationTable[1088131 + ((index / 4133u) % 4133u)]) * float(DoubleOneMinusEpsilon / 17081689u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4139(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1092264 + (index % 4139u)] * 4139u +
				globalHaltonSampler.fastPermutationTable[1092264 + ((index / 4139u) % 4139u)]) * float(DoubleOneMinusEpsilon / 17131321u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4153(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1096403 + (index % 4153u)] * 4153u +
				globalHaltonSampler.fastPermutationTable[1096403 + ((index / 4153u) % 4153u)]) * float(DoubleOneMinusEpsilon / 17247409u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4157(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1100556 + (index % 4157u)] * 4157u +
				globalHaltonSampler.fastPermutationTable[1100556 + ((index / 4157u) % 4157u)]) * float(DoubleOneMinusEpsilon / 17280649u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4159(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1104713 + (index % 4159u)] * 4159u +
				globalHaltonSampler.fastPermutationTable[1104713 + ((index / 4159u) % 4159u)]) * float(DoubleOneMinusEpsilon / 17297281u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4177(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1108872 + (index % 4177u)] * 4177u +
				globalHaltonSampler.fastPermutationTable[1108872 + ((index / 4177u) % 4177u)]) * float(DoubleOneMinusEpsilon / 17447329u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4201(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1113049 + (index % 4201u)] * 4201u +
				globalHaltonSampler.fastPermutationTable[1113049 + ((index / 4201u) % 4201u)]) * float(DoubleOneMinusEpsilon / 17648401u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4211(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1117250 + (index % 4211u)] * 4211u +
				globalHaltonSampler.fastPermutationTable[1117250 + ((index / 4211u) % 4211u)]) * float(DoubleOneMinusEpsilon / 17732521u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4217(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1121461 + (index % 4217u)] * 4217u +
				globalHaltonSampler.fastPermutationTable[1121461 + ((index / 4217u) % 4217u)]) * float(DoubleOneMinusEpsilon / 17783089u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4219(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1125678 + (index % 4219u)] * 4219u +
				globalHaltonSampler.fastPermutationTable[1125678 + ((index / 4219u) % 4219u)]) * float(DoubleOneMinusEpsilon / 17799961u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4229(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1129897 + (index % 4229u)] * 4229u +
				globalHaltonSampler.fastPermutationTable[1129897 + ((index / 4229u) % 4229u)]) * float(DoubleOneMinusEpsilon / 17884441u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4231(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1134126 + (index % 4231u)] * 4231u +
				globalHaltonSampler.fastPermutationTable[1134126 + ((index / 4231u) % 4231u)]) * float(DoubleOneMinusEpsilon / 17901361u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4241(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1138357 + (index % 4241u)] * 4241u +
				globalHaltonSampler.fastPermutationTable[1138357 + ((index / 4241u) % 4241u)]) * float(DoubleOneMinusEpsilon / 17986081u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4243(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1142598 + (index % 4243u)] * 4243u +
				globalHaltonSampler.fastPermutationTable[1142598 + ((index / 4243u) % 4243u)]) * float(DoubleOneMinusEpsilon / 18003049u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4253(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1146841 + (index % 4253u)] * 4253u +
				globalHaltonSampler.fastPermutationTable[1146841 + ((index / 4253u) % 4253u)]) * float(DoubleOneMinusEpsilon / 18088009u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4259(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1151094 + (index % 4259u)] * 4259u +
				globalHaltonSampler.fastPermutationTable[1151094 + ((index / 4259u) % 4259u)]) * float(DoubleOneMinusEpsilon / 18139081u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4261(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1155353 + (index % 4261u)] * 4261u +
				globalHaltonSampler.fastPermutationTable[1155353 + ((index / 4261u) % 4261u)]) * float(DoubleOneMinusEpsilon / 18156121u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4271(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1159614 + (index % 4271u)] * 4271u +
				globalHaltonSampler.fastPermutationTable[1159614 + ((index / 4271u) % 4271u)]) * float(DoubleOneMinusEpsilon / 18241441u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4273(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1163885 + (index % 4273u)] * 4273u +
				globalHaltonSampler.fastPermutationTable[1163885 + ((index / 4273u) % 4273u)]) * float(DoubleOneMinusEpsilon / 18258529u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4283(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1168158 + (index % 4283u)] * 4283u +
				globalHaltonSampler.fastPermutationTable[1168158 + ((index / 4283u) % 4283u)]) * float(DoubleOneMinusEpsilon / 18344089u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4289(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1172441 + (index % 4289u)] * 4289u +
				globalHaltonSampler.fastPermutationTable[1172441 + ((index / 4289u) % 4289u)]) * float(DoubleOneMinusEpsilon / 18395521u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4297(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1176730 + (index % 4297u)] * 4297u +
				globalHaltonSampler.fastPermutationTable[1176730 + ((index / 4297u) % 4297u)]) * float(DoubleOneMinusEpsilon / 18464209u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4327(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1181027 + (index % 4327u)] * 4327u +
				globalHaltonSampler.fastPermutationTable[1181027 + ((index / 4327u) % 4327u)]) * float(DoubleOneMinusEpsilon / 18722929u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4337(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1185354 + (index % 4337u)] * 4337u +
				globalHaltonSampler.fastPermutationTable[1185354 + ((index / 4337u) % 4337u)]) * float(DoubleOneMinusEpsilon / 18809569u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4339(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1189691 + (index % 4339u)] * 4339u +
				globalHaltonSampler.fastPermutationTable[1189691 + ((index / 4339u) % 4339u)]) * float(DoubleOneMinusEpsilon / 18826921u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4349(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1194030 + (index % 4349u)] * 4349u +
				globalHaltonSampler.fastPermutationTable[1194030 + ((index / 4349u) % 4349u)]) * float(DoubleOneMinusEpsilon / 18913801u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4357(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1198379 + (index % 4357u)] * 4357u +
				globalHaltonSampler.fastPermutationTable[1198379 + ((index / 4357u) % 4357u)]) * float(DoubleOneMinusEpsilon / 18983449u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4363(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1202736 + (index % 4363u)] * 4363u +
				globalHaltonSampler.fastPermutationTable[1202736 + ((index / 4363u) % 4363u)]) * float(DoubleOneMinusEpsilon / 19035769u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4373(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1207099 + (index % 4373u)] * 4373u +
				globalHaltonSampler.fastPermutationTable[1207099 + ((index / 4373u) % 4373u)]) * float(DoubleOneMinusEpsilon / 19123129u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4391(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1211472 + (index % 4391u)] * 4391u +
				globalHaltonSampler.fastPermutationTable[1211472 + ((index / 4391u) % 4391u)]) * float(DoubleOneMinusEpsilon / 19280881u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4397(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1215863 + (index % 4397u)] * 4397u +
				globalHaltonSampler.fastPermutationTable[1215863 + ((index / 4397u) % 4397u)]) * float(DoubleOneMinusEpsilon / 19333609u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4409(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1220260 + (index % 4409u)] * 4409u +
				globalHaltonSampler.fastPermutationTable[1220260 + ((index / 4409u) % 4409u)]) * float(DoubleOneMinusEpsilon / 19439281u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4421(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1224669 + (index % 4421u)] * 4421u +
				globalHaltonSampler.fastPermutationTable[1224669 + ((index / 4421u) % 4421u)]) * float(DoubleOneMinusEpsilon / 19545241u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4423(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1229090 + (index % 4423u)] * 4423u +
				globalHaltonSampler.fastPermutationTable[1229090 + ((index / 4423u) % 4423u)]) * float(DoubleOneMinusEpsilon / 19562929u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4441(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1233513 + (index % 4441u)] * 4441u +
				globalHaltonSampler.fastPermutationTable[1233513 + ((index / 4441u) % 4441u)]) * float(DoubleOneMinusEpsilon / 19722481u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4447(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1237954 + (index % 4447u)] * 4447u +
				globalHaltonSampler.fastPermutationTable[1237954 + ((index / 4447u) % 4447u)]) * float(DoubleOneMinusEpsilon / 19775809u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4451(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1242401 + (index % 4451u)] * 4451u +
				globalHaltonSampler.fastPermutationTable[1242401 + ((index / 4451u) % 4451u)]) * float(DoubleOneMinusEpsilon / 19811401u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4457(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1246852 + (index % 4457u)] * 4457u +
				globalHaltonSampler.fastPermutationTable[1246852 + ((index / 4457u) % 4457u)]) * float(DoubleOneMinusEpsilon / 19864849u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4463(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1251309 + (index % 4463u)] * 4463u +
				globalHaltonSampler.fastPermutationTable[1251309 + ((index / 4463u) % 4463u)]) * float(DoubleOneMinusEpsilon / 19918369u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4481(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1255772 + (index % 4481u)] * 4481u +
				globalHaltonSampler.fastPermutationTable[1255772 + ((index / 4481u) % 4481u)]) * float(DoubleOneMinusEpsilon / 20079361u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4483(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1260253 + (index % 4483u)] * 4483u +
				globalHaltonSampler.fastPermutationTable[1260253 + ((index / 4483u) % 4483u)]) * float(DoubleOneMinusEpsilon / 20097289u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4493(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1264736 + (index % 4493u)] * 4493u +
				globalHaltonSampler.fastPermutationTable[1264736 + ((index / 4493u) % 4493u)]) * float(DoubleOneMinusEpsilon / 20187049u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4507(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1269229 + (index % 4507u)] * 4507u +
				globalHaltonSampler.fastPermutationTable[1269229 + ((index / 4507u) % 4507u)]) * float(DoubleOneMinusEpsilon / 20313049u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4513(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1273736 + (index % 4513u)] * 4513u +
				globalHaltonSampler.fastPermutationTable[1273736 + ((index / 4513u) % 4513u)]) * float(DoubleOneMinusEpsilon / 20367169u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4517(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1278249 + (index % 4517u)] * 4517u +
				globalHaltonSampler.fastPermutationTable[1278249 + ((index / 4517u) % 4517u)]) * float(DoubleOneMinusEpsilon / 20403289u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4519(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1282766 + (index % 4519u)] * 4519u +
				globalHaltonSampler.fastPermutationTable[1282766 + ((index / 4519u) % 4519u)]) * float(DoubleOneMinusEpsilon / 20421361u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4523(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1287285 + (index % 4523u)] * 4523u +
				globalHaltonSampler.fastPermutationTable[1287285 + ((index / 4523u) % 4523u)]) * float(DoubleOneMinusEpsilon / 20457529u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4547(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1291808 + (index % 4547u)] * 4547u +
				globalHaltonSampler.fastPermutationTable[1291808 + ((index / 4547u) % 4547u)]) * float(DoubleOneMinusEpsilon / 20675209u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4549(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1296355 + (index % 4549u)] * 4549u +
				globalHaltonSampler.fastPermutationTable[1296355 + ((index / 4549u) % 4549u)]) * float(DoubleOneMinusEpsilon / 20693401u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4561(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1300904 + (index % 4561u)] * 4561u +
				globalHaltonSampler.fastPermutationTable[1300904 + ((index / 4561u) % 4561u)]) * float(DoubleOneMinusEpsilon / 20802721u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4567(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1305465 + (index % 4567u)] * 4567u +
				globalHaltonSampler.fastPermutationTable[1305465 + ((index / 4567u) % 4567u)]) * float(DoubleOneMinusEpsilon / 20857489u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4583(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1310032 + (index % 4583u)] * 4583u +
				globalHaltonSampler.fastPermutationTable[1310032 + ((index / 4583u) % 4583u)]) * float(DoubleOneMinusEpsilon / 21003889u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4591(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1314615 + (index % 4591u)] * 4591u +
				globalHaltonSampler.fastPermutationTable[1314615 + ((index / 4591u) % 4591u)]) * float(DoubleOneMinusEpsilon / 21077281u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4597(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1319206 + (index % 4597u)] * 4597u +
				globalHaltonSampler.fastPermutationTable[1319206 + ((index / 4597u) % 4597u)]) * float(DoubleOneMinusEpsilon / 21132409u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4603(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1323803 + (index % 4603u)] * 4603u +
				globalHaltonSampler.fastPermutationTable[1323803 + ((index / 4603u) % 4603u)]) * float(DoubleOneMinusEpsilon / 21187609u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4621(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1328406 + (index % 4621u)] * 4621u +
				globalHaltonSampler.fastPermutationTable[1328406 + ((index / 4621u) % 4621u)]) * float(DoubleOneMinusEpsilon / 21353641u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4637(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1333027 + (index % 4637u)] * 4637u +
				globalHaltonSampler.fastPermutationTable[1333027 + ((index / 4637u) % 4637u)]) * float(DoubleOneMinusEpsilon / 21501769u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4639(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1337664 + (index % 4639u)] * 4639u +
				globalHaltonSampler.fastPermutationTable[1337664 + ((index / 4639u) % 4639u)]) * float(DoubleOneMinusEpsilon / 21520321u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4643(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1342303 + (index % 4643u)] * 4643u +
				globalHaltonSampler.fastPermutationTable[1342303 + ((index / 4643u) % 4643u)]) * float(DoubleOneMinusEpsilon / 21557449u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4649(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1346946 + (index % 4649u)] * 4649u +
				globalHaltonSampler.fastPermutationTable[1346946 + ((index / 4649u) % 4649u)]) * float(DoubleOneMinusEpsilon / 21613201u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4651(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1351595 + (index % 4651u)] * 4651u +
				globalHaltonSampler.fastPermutationTable[1351595 + ((index / 4651u) % 4651u)]) * float(DoubleOneMinusEpsilon / 21631801u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4657(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1356246 + (index % 4657u)] * 4657u +
				globalHaltonSampler.fastPermutationTable[1356246 + ((index / 4657u) % 4657u)]) * float(DoubleOneMinusEpsilon / 21687649u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4663(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1360903 + (index % 4663u)] * 4663u +
				globalHaltonSampler.fastPermutationTable[1360903 + ((index / 4663u) % 4663u)]) * float(DoubleOneMinusEpsilon / 21743569u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4673(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1365566 + (index % 4673u)] * 4673u +
				globalHaltonSampler.fastPermutationTable[1365566 + ((index / 4673u) % 4673u)]) * float(DoubleOneMinusEpsilon / 21836929u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4679(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1370239 + (index % 4679u)] * 4679u +
				globalHaltonSampler.fastPermutationTable[1370239 + ((index / 4679u) % 4679u)]) * float(DoubleOneMinusEpsilon / 21893041u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4691(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1374918 + (index % 4691u)] * 4691u +
				globalHaltonSampler.fastPermutationTable[1374918 + ((index / 4691u) % 4691u)]) * float(DoubleOneMinusEpsilon / 22005481u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4703(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1379609 + (index % 4703u)] * 4703u +
				globalHaltonSampler.fastPermutationTable[1379609 + ((index / 4703u) % 4703u)]) * float(DoubleOneMinusEpsilon / 22118209u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4721(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1384312 + (index % 4721u)] * 4721u +
				globalHaltonSampler.fastPermutationTable[1384312 + ((index / 4721u) % 4721u)]) * float(DoubleOneMinusEpsilon / 22287841u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4723(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1389033 + (index % 4723u)] * 4723u +
				globalHaltonSampler.fastPermutationTable[1389033 + ((index / 4723u) % 4723u)]) * float(DoubleOneMinusEpsilon / 22306729u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4729(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1393756 + (index % 4729u)] * 4729u +
				globalHaltonSampler.fastPermutationTable[1393756 + ((index / 4729u) % 4729u)]) * float(DoubleOneMinusEpsilon / 22363441u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4733(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1398485 + (index % 4733u)] * 4733u +
				globalHaltonSampler.fastPermutationTable[1398485 + ((index / 4733u) % 4733u)]) * float(DoubleOneMinusEpsilon / 22401289u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4751(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1403218 + (index % 4751u)] * 4751u +
				globalHaltonSampler.fastPermutationTable[1403218 + ((index / 4751u) % 4751u)]) * float(DoubleOneMinusEpsilon / 22572001u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4759(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1407969 + (index % 4759u)] * 4759u +
				globalHaltonSampler.fastPermutationTable[1407969 + ((index / 4759u) % 4759u)]) * float(DoubleOneMinusEpsilon / 22648081u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4783(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1412728 + (index % 4783u)] * 4783u +
				globalHaltonSampler.fastPermutationTable[1412728 + ((index / 4783u) % 4783u)]) * float(DoubleOneMinusEpsilon / 22877089u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4787(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1417511 + (index % 4787u)] * 4787u +
				globalHaltonSampler.fastPermutationTable[1417511 + ((index / 4787u) % 4787u)]) * float(DoubleOneMinusEpsilon / 22915369u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4789(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1422298 + (index % 4789u)] * 4789u +
				globalHaltonSampler.fastPermutationTable[1422298 + ((index / 4789u) % 4789u)]) * float(DoubleOneMinusEpsilon / 22934521u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4793(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1427087 + (index % 4793u)] * 4793u +
				globalHaltonSampler.fastPermutationTable[1427087 + ((index / 4793u) % 4793u)]) * float(DoubleOneMinusEpsilon / 22972849u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4799(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1431880 + (index % 4799u)] * 4799u +
				globalHaltonSampler.fastPermutationTable[1431880 + ((index / 4799u) % 4799u)]) * float(DoubleOneMinusEpsilon / 23030401u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4801(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1436679 + (index % 4801u)] * 4801u +
				globalHaltonSampler.fastPermutationTable[1436679 + ((index / 4801u) % 4801u)]) * float(DoubleOneMinusEpsilon / 23049601u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4813(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1441480 + (index % 4813u)] * 4813u +
				globalHaltonSampler.fastPermutationTable[1441480 + ((index / 4813u) % 4813u)]) * float(DoubleOneMinusEpsilon / 23164969u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4817(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1446293 + (index % 4817u)] * 4817u +
				globalHaltonSampler.fastPermutationTable[1446293 + ((index / 4817u) % 4817u)]) * float(DoubleOneMinusEpsilon / 23203489u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4831(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1451110 + (index % 4831u)] * 4831u +
				globalHaltonSampler.fastPermutationTable[1451110 + ((index / 4831u) % 4831u)]) * float(DoubleOneMinusEpsilon / 23338561u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4861(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1455941 + (index % 4861u)] * 4861u +
				globalHaltonSampler.fastPermutationTable[1455941 + ((index / 4861u) % 4861u)]) * float(DoubleOneMinusEpsilon / 23629321u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4871(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1460802 + (index % 4871u)] * 4871u +
				globalHaltonSampler.fastPermutationTable[1460802 + ((index / 4871u) % 4871u)]) * float(DoubleOneMinusEpsilon / 23726641u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4877(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1465673 + (index % 4877u)] * 4877u +
				globalHaltonSampler.fastPermutationTable[1465673 + ((index / 4877u) % 4877u)]) * float(DoubleOneMinusEpsilon / 23785129u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4889(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1470550 + (index % 4889u)] * 4889u +
				globalHaltonSampler.fastPermutationTable[1470550 + ((index / 4889u) % 4889u)]) * float(DoubleOneMinusEpsilon / 23902321u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4903(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1475439 + (index % 4903u)] * 4903u +
				globalHaltonSampler.fastPermutationTable[1475439 + ((index / 4903u) % 4903u)]) * float(DoubleOneMinusEpsilon / 24039409u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4909(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1480342 + (index % 4909u)] * 4909u +
				globalHaltonSampler.fastPermutationTable[1480342 + ((index / 4909u) % 4909u)]) * float(DoubleOneMinusEpsilon / 24098281u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4919(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1485251 + (index % 4919u)] * 4919u +
				globalHaltonSampler.fastPermutationTable[1485251 + ((index / 4919u) % 4919u)]) * float(DoubleOneMinusEpsilon / 24196561u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4931(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1490170 + (index % 4931u)] * 4931u +
				globalHaltonSampler.fastPermutationTable[1490170 + ((index / 4931u) % 4931u)]) * float(DoubleOneMinusEpsilon / 24314761u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4933(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1495101 + (index % 4933u)] * 4933u +
				globalHaltonSampler.fastPermutationTable[1495101 + ((index / 4933u) % 4933u)]) * float(DoubleOneMinusEpsilon / 24334489u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4937(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1500034 + (index % 4937u)] * 4937u +
				globalHaltonSampler.fastPermutationTable[1500034 + ((index / 4937u) % 4937u)]) * float(DoubleOneMinusEpsilon / 24373969u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4943(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1504971 + (index % 4943u)] * 4943u +
				globalHaltonSampler.fastPermutationTable[1504971 + ((index / 4943u) % 4943u)]) * float(DoubleOneMinusEpsilon / 24433249u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4951(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1509914 + (index % 4951u)] * 4951u +
				globalHaltonSampler.fastPermutationTable[1509914 + ((index / 4951u) % 4951u)]) * float(DoubleOneMinusEpsilon / 24512401u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4957(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1514865 + (index % 4957u)] * 4957u +
				globalHaltonSampler.fastPermutationTable[1514865 + ((index / 4957u) % 4957u)]) * float(DoubleOneMinusEpsilon / 24571849u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4967(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1519822 + (index % 4967u)] * 4967u +
				globalHaltonSampler.fastPermutationTable[1519822 + ((index / 4967u) % 4967u)]) * float(DoubleOneMinusEpsilon / 24671089u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4969(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1524789 + (index % 4969u)] * 4969u +
				globalHaltonSampler.fastPermutationTable[1524789 + ((index / 4969u) % 4969u)]) * float(DoubleOneMinusEpsilon / 24690961u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4973(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1529758 + (index % 4973u)] * 4973u +
				globalHaltonSampler.fastPermutationTable[1529758 + ((index / 4973u) % 4973u)]) * float(DoubleOneMinusEpsilon / 24730729u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4987(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1534731 + (index % 4987u)] * 4987u +
				globalHaltonSampler.fastPermutationTable[1534731 + ((index / 4987u) % 4987u)]) * float(DoubleOneMinusEpsilon / 24870169u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4993(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1539718 + (index % 4993u)] * 4993u +
				globalHaltonSampler.fastPermutationTable[1539718 + ((index / 4993u) % 4993u)]) * float(DoubleOneMinusEpsilon / 24930049u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized4999(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1544711 + (index % 4999u)] * 4999u +
				globalHaltonSampler.fastPermutationTable[1544711 + ((index / 4999u) % 4999u)]) * float(DoubleOneMinusEpsilon / 24990001u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5003(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1549710 + (index % 5003u)] * 5003u +
				globalHaltonSampler.fastPermutationTable[1549710 + ((index / 5003u) % 5003u)]) * float(DoubleOneMinusEpsilon / 25030009u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5009(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1554713 + (index % 5009u)] * 5009u +
				globalHaltonSampler.fastPermutationTable[1554713 + ((index / 5009u) % 5009u)]) * float(DoubleOneMinusEpsilon / 25090081u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5011(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1559722 + (index % 5011u)] * 5011u +
				globalHaltonSampler.fastPermutationTable[1559722 + ((index / 5011u) % 5011u)]) * float(DoubleOneMinusEpsilon / 25110121u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5021(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1564733 + (index % 5021u)] * 5021u +
				globalHaltonSampler.fastPermutationTable[1564733 + ((index / 5021u) % 5021u)]) * float(DoubleOneMinusEpsilon / 25210441u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5023(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1569754 + (index % 5023u)] * 5023u +
				globalHaltonSampler.fastPermutationTable[1569754 + ((index / 5023u) % 5023u)]) * float(DoubleOneMinusEpsilon / 25230529u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5039(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1574777 + (index % 5039u)] * 5039u +
				globalHaltonSampler.fastPermutationTable[1574777 + ((index / 5039u) % 5039u)]) * float(DoubleOneMinusEpsilon / 25391521u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5051(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1579816 + (index % 5051u)] * 5051u +
				globalHaltonSampler.fastPermutationTable[1579816 + ((index / 5051u) % 5051u)]) * float(DoubleOneMinusEpsilon / 25512601u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5059(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1584867 + (index % 5059u)] * 5059u +
				globalHaltonSampler.fastPermutationTable[1584867 + ((index / 5059u) % 5059u)]) * float(DoubleOneMinusEpsilon / 25593481u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5077(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1589926 + (index % 5077u)] * 5077u +
				globalHaltonSampler.fastPermutationTable[1589926 + ((index / 5077u) % 5077u)]) * float(DoubleOneMinusEpsilon / 25775929u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5081(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1595003 + (index % 5081u)] * 5081u +
				globalHaltonSampler.fastPermutationTable[1595003 + ((index / 5081u) % 5081u)]) * float(DoubleOneMinusEpsilon / 25816561u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5087(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1600084 + (index % 5087u)] * 5087u +
				globalHaltonSampler.fastPermutationTable[1600084 + ((index / 5087u) % 5087u)]) * float(DoubleOneMinusEpsilon / 25877569u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5099(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1605171 + (index % 5099u)] * 5099u +
				globalHaltonSampler.fastPermutationTable[1605171 + ((index / 5099u) % 5099u)]) * float(DoubleOneMinusEpsilon / 25999801u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5101(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1610270 + (index % 5101u)] * 5101u +
				globalHaltonSampler.fastPermutationTable[1610270 + ((index / 5101u) % 5101u)]) * float(DoubleOneMinusEpsilon / 26020201u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5107(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1615371 + (index % 5107u)] * 5107u +
				globalHaltonSampler.fastPermutationTable[1615371 + ((index / 5107u) % 5107u)]) * float(DoubleOneMinusEpsilon / 26081449u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5113(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1620478 + (index % 5113u)] * 5113u +
				globalHaltonSampler.fastPermutationTable[1620478 + ((index / 5113u) % 5113u)]) * float(DoubleOneMinusEpsilon / 26142769u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5119(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1625591 + (index % 5119u)] * 5119u +
				globalHaltonSampler.fastPermutationTable[1625591 + ((index / 5119u) % 5119u)]) * float(DoubleOneMinusEpsilon / 26204161u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5147(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1630710 + (index % 5147u)] * 5147u +
				globalHaltonSampler.fastPermutationTable[1630710 + ((index / 5147u) % 5147u)]) * float(DoubleOneMinusEpsilon / 26491609u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5153(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1635857 + (index % 5153u)] * 5153u +
				globalHaltonSampler.fastPermutationTable[1635857 + ((index / 5153u) % 5153u)]) * float(DoubleOneMinusEpsilon / 26553409u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5167(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1641010 + (index % 5167u)] * 5167u +
				globalHaltonSampler.fastPermutationTable[1641010 + ((index / 5167u) % 5167u)]) * float(DoubleOneMinusEpsilon / 26697889u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5171(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1646177 + (index % 5171u)] * 5171u +
				globalHaltonSampler.fastPermutationTable[1646177 + ((index / 5171u) % 5171u)]) * float(DoubleOneMinusEpsilon / 26739241u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5179(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1651348 + (index % 5179u)] * 5179u +
				globalHaltonSampler.fastPermutationTable[1651348 + ((index / 5179u) % 5179u)]) * float(DoubleOneMinusEpsilon / 26822041u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5189(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1656527 + (index % 5189u)] * 5189u +
				globalHaltonSampler.fastPermutationTable[1656527 + ((index / 5189u) % 5189u)]) * float(DoubleOneMinusEpsilon / 26925721u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5197(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1661716 + (index % 5197u)] * 5197u +
				globalHaltonSampler.fastPermutationTable[1661716 + ((index / 5197u) % 5197u)]) * float(DoubleOneMinusEpsilon / 27008809u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5209(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1666913 + (index % 5209u)] * 5209u +
				globalHaltonSampler.fastPermutationTable[1666913 + ((index / 5209u) % 5209u)]) * float(DoubleOneMinusEpsilon / 27133681u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5227(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1672122 + (index % 5227u)] * 5227u +
				globalHaltonSampler.fastPermutationTable[1672122 + ((index / 5227u) % 5227u)]) * float(DoubleOneMinusEpsilon / 27321529u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5231(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1677349 + (index % 5231u)] * 5231u +
				globalHaltonSampler.fastPermutationTable[1677349 + ((index / 5231u) % 5231u)]) * float(DoubleOneMinusEpsilon / 27363361u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5233(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1682580 + (index % 5233u)] * 5233u +
				globalHaltonSampler.fastPermutationTable[1682580 + ((index / 5233u) % 5233u)]) * float(DoubleOneMinusEpsilon / 27384289u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5237(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1687813 + (index % 5237u)] * 5237u +
				globalHaltonSampler.fastPermutationTable[1687813 + ((index / 5237u) % 5237u)]) * float(DoubleOneMinusEpsilon / 27426169u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5261(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1693050 + (index % 5261u)] * 5261u +
				globalHaltonSampler.fastPermutationTable[1693050 + ((index / 5261u) % 5261u)]) * float(DoubleOneMinusEpsilon / 27678121u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5273(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1698311 + (index % 5273u)] * 5273u +
				globalHaltonSampler.fastPermutationTable[1698311 + ((index / 5273u) % 5273u)]) * float(DoubleOneMinusEpsilon / 27804529u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5279(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1703584 + (index % 5279u)] * 5279u +
				globalHaltonSampler.fastPermutationTable[1703584 + ((index / 5279u) % 5279u)]) * float(DoubleOneMinusEpsilon / 27867841u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5281(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1708863 + (index % 5281u)] * 5281u +
				globalHaltonSampler.fastPermutationTable[1708863 + ((index / 5281u) % 5281u)]) * float(DoubleOneMinusEpsilon / 27888961u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5297(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1714144 + (index % 5297u)] * 5297u +
				globalHaltonSampler.fastPermutationTable[1714144 + ((index / 5297u) % 5297u)]) * float(DoubleOneMinusEpsilon / 28058209u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5303(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1719441 + (index % 5303u)] * 5303u +
				globalHaltonSampler.fastPermutationTable[1719441 + ((index / 5303u) % 5303u)]) * float(DoubleOneMinusEpsilon / 28121809u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5309(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1724744 + (index % 5309u)] * 5309u +
				globalHaltonSampler.fastPermutationTable[1724744 + ((index / 5309u) % 5309u)]) * float(DoubleOneMinusEpsilon / 28185481u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5323(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1730053 + (index % 5323u)] * 5323u +
				globalHaltonSampler.fastPermutationTable[1730053 + ((index / 5323u) % 5323u)]) * float(DoubleOneMinusEpsilon / 28334329u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5333(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1735376 + (index % 5333u)] * 5333u +
				globalHaltonSampler.fastPermutationTable[1735376 + ((index / 5333u) % 5333u)]) * float(DoubleOneMinusEpsilon / 28440889u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5347(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1740709 + (index % 5347u)] * 5347u +
				globalHaltonSampler.fastPermutationTable[1740709 + ((index / 5347u) % 5347u)]) * float(DoubleOneMinusEpsilon / 28590409u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5351(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1746056 + (index % 5351u)] * 5351u +
				globalHaltonSampler.fastPermutationTable[1746056 + ((index / 5351u) % 5351u)]) * float(DoubleOneMinusEpsilon / 28633201u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5381(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1751407 + (index % 5381u)] * 5381u +
				globalHaltonSampler.fastPermutationTable[1751407 + ((index / 5381u) % 5381u)]) * float(DoubleOneMinusEpsilon / 28955161u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5387(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1756788 + (index % 5387u)] * 5387u +
				globalHaltonSampler.fastPermutationTable[1756788 + ((index / 5387u) % 5387u)]) * float(DoubleOneMinusEpsilon / 29019769u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5393(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1762175 + (index % 5393u)] * 5393u +
				globalHaltonSampler.fastPermutationTable[1762175 + ((index / 5393u) % 5393u)]) * float(DoubleOneMinusEpsilon / 29084449u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5399(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1767568 + (index % 5399u)] * 5399u +
				globalHaltonSampler.fastPermutationTable[1767568 + ((index / 5399u) % 5399u)]) * float(DoubleOneMinusEpsilon / 29149201u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5407(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1772967 + (index % 5407u)] * 5407u +
				globalHaltonSampler.fastPermutationTable[1772967 + ((index / 5407u) % 5407u)]) * float(DoubleOneMinusEpsilon / 29235649u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5413(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1778374 + (index % 5413u)] * 5413u +
				globalHaltonSampler.fastPermutationTable[1778374 + ((index / 5413u) % 5413u)]) * float(DoubleOneMinusEpsilon / 29300569u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5417(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1783787 + (index % 5417u)] * 5417u +
				globalHaltonSampler.fastPermutationTable[1783787 + ((index / 5417u) % 5417u)]) * float(DoubleOneMinusEpsilon / 29343889u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5419(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1789204 + (index % 5419u)] * 5419u +
				globalHaltonSampler.fastPermutationTable[1789204 + ((index / 5419u) % 5419u)]) * float(DoubleOneMinusEpsilon / 29365561u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5431(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1794623 + (index % 5431u)] * 5431u +
				globalHaltonSampler.fastPermutationTable[1794623 + ((index / 5431u) % 5431u)]) * float(DoubleOneMinusEpsilon / 29495761u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5437(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1800054 + (index % 5437u)] * 5437u +
				globalHaltonSampler.fastPermutationTable[1800054 + ((index / 5437u) % 5437u)]) * float(DoubleOneMinusEpsilon / 29560969u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5441(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1805491 + (index % 5441u)] * 5441u +
				globalHaltonSampler.fastPermutationTable[1805491 + ((index / 5441u) % 5441u)]) * float(DoubleOneMinusEpsilon / 29604481u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5443(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1810932 + (index % 5443u)] * 5443u +
				globalHaltonSampler.fastPermutationTable[1810932 + ((index / 5443u) % 5443u)]) * float(DoubleOneMinusEpsilon / 29626249u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5449(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1816375 + (index % 5449u)] * 5449u +
				globalHaltonSampler.fastPermutationTable[1816375 + ((index / 5449u) % 5449u)]) * float(DoubleOneMinusEpsilon / 29691601u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5471(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1821824 + (index % 5471u)] * 5471u +
				globalHaltonSampler.fastPermutationTable[1821824 + ((index / 5471u) % 5471u)]) * float(DoubleOneMinusEpsilon / 29931841u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5477(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1827295 + (index % 5477u)] * 5477u +
				globalHaltonSampler.fastPermutationTable[1827295 + ((index / 5477u) % 5477u)]) * float(DoubleOneMinusEpsilon / 29997529u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5479(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1832772 + (index % 5479u)] * 5479u +
				globalHaltonSampler.fastPermutationTable[1832772 + ((index / 5479u) % 5479u)]) * float(DoubleOneMinusEpsilon / 30019441u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5483(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1838251 + (index % 5483u)] * 5483u +
				globalHaltonSampler.fastPermutationTable[1838251 + ((index / 5483u) % 5483u)]) * float(DoubleOneMinusEpsilon / 30063289u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5501(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1843734 + (index % 5501u)] * 5501u +
				globalHaltonSampler.fastPermutationTable[1843734 + ((index / 5501u) % 5501u)]) * float(DoubleOneMinusEpsilon / 30261001u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5503(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1849235 + (index % 5503u)] * 5503u +
				globalHaltonSampler.fastPermutationTable[1849235 + ((index / 5503u) % 5503u)]) * float(DoubleOneMinusEpsilon / 30283009u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5507(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1854738 + (index % 5507u)] * 5507u +
				globalHaltonSampler.fastPermutationTable[1854738 + ((index / 5507u) % 5507u)]) * float(DoubleOneMinusEpsilon / 30327049u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5519(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1860245 + (index % 5519u)] * 5519u +
				globalHaltonSampler.fastPermutationTable[1860245 + ((index / 5519u) % 5519u)]) * float(DoubleOneMinusEpsilon / 30459361u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5521(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1865764 + (index % 5521u)] * 5521u +
				globalHaltonSampler.fastPermutationTable[1865764 + ((index / 5521u) % 5521u)]) * float(DoubleOneMinusEpsilon / 30481441u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5527(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1871285 + (index % 5527u)] * 5527u +
				globalHaltonSampler.fastPermutationTable[1871285 + ((index / 5527u) % 5527u)]) * float(DoubleOneMinusEpsilon / 30547729u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5531(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1876812 + (index % 5531u)] * 5531u +
				globalHaltonSampler.fastPermutationTable[1876812 + ((index / 5531u) % 5531u)]) * float(DoubleOneMinusEpsilon / 30591961u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5557(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1882343 + (index % 5557u)] * 5557u +
				globalHaltonSampler.fastPermutationTable[1882343 + ((index / 5557u) % 5557u)]) * float(DoubleOneMinusEpsilon / 30880249u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5563(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1887900 + (index % 5563u)] * 5563u +
				globalHaltonSampler.fastPermutationTable[1887900 + ((index / 5563u) % 5563u)]) * float(DoubleOneMinusEpsilon / 30946969u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5569(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1893463 + (index % 5569u)] * 5569u +
				globalHaltonSampler.fastPermutationTable[1893463 + ((index / 5569u) % 5569u)]) * float(DoubleOneMinusEpsilon / 31013761u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5573(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1899032 + (index % 5573u)] * 5573u +
				globalHaltonSampler.fastPermutationTable[1899032 + ((index / 5573u) % 5573u)]) * float(DoubleOneMinusEpsilon / 31058329u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5581(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1904605 + (index % 5581u)] * 5581u +
				globalHaltonSampler.fastPermutationTable[1904605 + ((index / 5581u) % 5581u)]) * float(DoubleOneMinusEpsilon / 31147561u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5591(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1910186 + (index % 5591u)] * 5591u +
				globalHaltonSampler.fastPermutationTable[1910186 + ((index / 5591u) % 5591u)]) * float(DoubleOneMinusEpsilon / 31259281u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5623(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1915777 + (index % 5623u)] * 5623u +
				globalHaltonSampler.fastPermutationTable[1915777 + ((index / 5623u) % 5623u)]) * float(DoubleOneMinusEpsilon / 31618129u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5639(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1921400 + (index % 5639u)] * 5639u +
				globalHaltonSampler.fastPermutationTable[1921400 + ((index / 5639u) % 5639u)]) * float(DoubleOneMinusEpsilon / 31798321u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5641(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1927039 + (index % 5641u)] * 5641u +
				globalHaltonSampler.fastPermutationTable[1927039 + ((index / 5641u) % 5641u)]) * float(DoubleOneMinusEpsilon / 31820881u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5647(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1932680 + (index % 5647u)] * 5647u +
				globalHaltonSampler.fastPermutationTable[1932680 + ((index / 5647u) % 5647u)]) * float(DoubleOneMinusEpsilon / 31888609u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5651(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1938327 + (index % 5651u)] * 5651u +
				globalHaltonSampler.fastPermutationTable[1938327 + ((index / 5651u) % 5651u)]) * float(DoubleOneMinusEpsilon / 31933801u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5653(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1943978 + (index % 5653u)] * 5653u +
				globalHaltonSampler.fastPermutationTable[1943978 + ((index / 5653u) % 5653u)]) * float(DoubleOneMinusEpsilon / 31956409u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5657(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1949631 + (index % 5657u)] * 5657u +
				globalHaltonSampler.fastPermutationTable[1949631 + ((index / 5657u) % 5657u)]) * float(DoubleOneMinusEpsilon / 32001649u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5659(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1955288 + (index % 5659u)] * 5659u +
				globalHaltonSampler.fastPermutationTable[1955288 + ((index / 5659u) % 5659u)]) * float(DoubleOneMinusEpsilon / 32024281u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5669(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1960947 + (index % 5669u)] * 5669u +
				globalHaltonSampler.fastPermutationTable[1960947 + ((index / 5669u) % 5669u)]) * float(DoubleOneMinusEpsilon / 32137561u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5683(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1966616 + (index % 5683u)] * 5683u +
				globalHaltonSampler.fastPermutationTable[1966616 + ((index / 5683u) % 5683u)]) * float(DoubleOneMinusEpsilon / 32296489u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5689(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1972299 + (index % 5689u)] * 5689u +
				globalHaltonSampler.fastPermutationTable[1972299 + ((index / 5689u) % 5689u)]) * float(DoubleOneMinusEpsilon / 32364721u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5693(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1977988 + (index % 5693u)] * 5693u +
				globalHaltonSampler.fastPermutationTable[1977988 + ((index / 5693u) % 5693u)]) * float(DoubleOneMinusEpsilon / 32410249u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5701(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1983681 + (index % 5701u)] * 5701u +
				globalHaltonSampler.fastPermutationTable[1983681 + ((index / 5701u) % 5701u)]) * float(DoubleOneMinusEpsilon / 32501401u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5711(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1989382 + (index % 5711u)] * 5711u +
				globalHaltonSampler.fastPermutationTable[1989382 + ((index / 5711u) % 5711u)]) * float(DoubleOneMinusEpsilon / 32615521u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5717(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[1995093 + (index % 5717u)] * 5717u +
				globalHaltonSampler.fastPermutationTable[1995093 + ((index / 5717u) % 5717u)]) * float(DoubleOneMinusEpsilon / 32684089u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5737(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2000810 + (index % 5737u)] * 5737u +
				globalHaltonSampler.fastPermutationTable[2000810 + ((index / 5737u) % 5737u)]) * float(DoubleOneMinusEpsilon / 32913169u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5741(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2006547 + (index % 5741u)] * 5741u +
				globalHaltonSampler.fastPermutationTable[2006547 + ((index / 5741u) % 5741u)]) * float(DoubleOneMinusEpsilon / 32959081u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5743(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2012288 + (index % 5743u)] * 5743u +
				globalHaltonSampler.fastPermutationTable[2012288 + ((index / 5743u) % 5743u)]) * float(DoubleOneMinusEpsilon / 32982049u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5749(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2018031 + (index % 5749u)] * 5749u +
				globalHaltonSampler.fastPermutationTable[2018031 + ((index / 5749u) % 5749u)]) * float(DoubleOneMinusEpsilon / 33051001u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5779(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2023780 + (index % 5779u)] * 5779u +
				globalHaltonSampler.fastPermutationTable[2023780 + ((index / 5779u) % 5779u)]) * float(DoubleOneMinusEpsilon / 33396841u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5783(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2029559 + (index % 5783u)] * 5783u +
				globalHaltonSampler.fastPermutationTable[2029559 + ((index / 5783u) % 5783u)]) * float(DoubleOneMinusEpsilon / 33443089u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5791(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2035342 + (index % 5791u)] * 5791u +
				globalHaltonSampler.fastPermutationTable[2035342 + ((index / 5791u) % 5791u)]) * float(DoubleOneMinusEpsilon / 33535681u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5801(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2041133 + (index % 5801u)] * 5801u +
				globalHaltonSampler.fastPermutationTable[2041133 + ((index / 5801u) % 5801u)]) * float(DoubleOneMinusEpsilon / 33651601u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5807(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2046934 + (index % 5807u)] * 5807u +
				globalHaltonSampler.fastPermutationTable[2046934 + ((index / 5807u) % 5807u)]) * float(DoubleOneMinusEpsilon / 33721249u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5813(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2052741 + (index % 5813u)] * 5813u +
				globalHaltonSampler.fastPermutationTable[2052741 + ((index / 5813u) % 5813u)]) * float(DoubleOneMinusEpsilon / 33790969u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5821(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2058554 + (index % 5821u)] * 5821u +
				globalHaltonSampler.fastPermutationTable[2058554 + ((index / 5821u) % 5821u)]) * float(DoubleOneMinusEpsilon / 33884041u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5827(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2064375 + (index % 5827u)] * 5827u +
				globalHaltonSampler.fastPermutationTable[2064375 + ((index / 5827u) % 5827u)]) * float(DoubleOneMinusEpsilon / 33953929u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5839(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2070202 + (index % 5839u)] * 5839u +
				globalHaltonSampler.fastPermutationTable[2070202 + ((index / 5839u) % 5839u)]) * float(DoubleOneMinusEpsilon / 34093921u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5843(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2076041 + (index % 5843u)] * 5843u +
				globalHaltonSampler.fastPermutationTable[2076041 + ((index / 5843u) % 5843u)]) * float(DoubleOneMinusEpsilon / 34140649u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5849(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2081884 + (index % 5849u)] * 5849u +
				globalHaltonSampler.fastPermutationTable[2081884 + ((index / 5849u) % 5849u)]) * float(DoubleOneMinusEpsilon / 34210801u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5851(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2087733 + (index % 5851u)] * 5851u +
				globalHaltonSampler.fastPermutationTable[2087733 + ((index / 5851u) % 5851u)]) * float(DoubleOneMinusEpsilon / 34234201u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5857(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2093584 + (index % 5857u)] * 5857u +
				globalHaltonSampler.fastPermutationTable[2093584 + ((index / 5857u) % 5857u)]) * float(DoubleOneMinusEpsilon / 34304449u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5861(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2099441 + (index % 5861u)] * 5861u +
				globalHaltonSampler.fastPermutationTable[2099441 + ((index / 5861u) % 5861u)]) * float(DoubleOneMinusEpsilon / 34351321u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5867(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2105302 + (index % 5867u)] * 5867u +
				globalHaltonSampler.fastPermutationTable[2105302 + ((index / 5867u) % 5867u)]) * float(DoubleOneMinusEpsilon / 34421689u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5869(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2111169 + (index % 5869u)] * 5869u +
				globalHaltonSampler.fastPermutationTable[2111169 + ((index / 5869u) % 5869u)]) * float(DoubleOneMinusEpsilon / 34445161u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5879(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2117038 + (index % 5879u)] * 5879u +
				globalHaltonSampler.fastPermutationTable[2117038 + ((index / 5879u) % 5879u)]) * float(DoubleOneMinusEpsilon / 34562641u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5881(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2122917 + (index % 5881u)] * 5881u +
				globalHaltonSampler.fastPermutationTable[2122917 + ((index / 5881u) % 5881u)]) * float(DoubleOneMinusEpsilon / 34586161u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5897(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2128798 + (index % 5897u)] * 5897u +
				globalHaltonSampler.fastPermutationTable[2128798 + ((index / 5897u) % 5897u)]) * float(DoubleOneMinusEpsilon / 34774609u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5903(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2134695 + (index % 5903u)] * 5903u +
				globalHaltonSampler.fastPermutationTable[2134695 + ((index / 5903u) % 5903u)]) * float(DoubleOneMinusEpsilon / 34845409u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5923(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2140598 + (index % 5923u)] * 5923u +
				globalHaltonSampler.fastPermutationTable[2140598 + ((index / 5923u) % 5923u)]) * float(DoubleOneMinusEpsilon / 35081929u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5927(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2146521 + (index % 5927u)] * 5927u +
				globalHaltonSampler.fastPermutationTable[2146521 + ((index / 5927u) % 5927u)]) * float(DoubleOneMinusEpsilon / 35129329u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5939(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2152448 + (index % 5939u)] * 5939u +
				globalHaltonSampler.fastPermutationTable[2152448 + ((index / 5939u) % 5939u)]) * float(DoubleOneMinusEpsilon / 35271721u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5953(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2158387 + (index % 5953u)] * 5953u +
				globalHaltonSampler.fastPermutationTable[2158387 + ((index / 5953u) % 5953u)]) * float(DoubleOneMinusEpsilon / 35438209u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5981(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2164340 + (index % 5981u)] * 5981u +
				globalHaltonSampler.fastPermutationTable[2164340 + ((index / 5981u) % 5981u)]) * float(DoubleOneMinusEpsilon / 35772361u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized5987(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2170321 + (index % 5987u)] * 5987u +
				globalHaltonSampler.fastPermutationTable[2170321 + ((index / 5987u) % 5987u)]) * float(DoubleOneMinusEpsilon / 35844169u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6007(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2176308 + (index % 6007u)] * 6007u +
				globalHaltonSampler.fastPermutationTable[2176308 + ((index / 6007u) % 6007u)]) * float(DoubleOneMinusEpsilon / 36084049u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6011(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2182315 + (index % 6011u)] * 6011u +
				globalHaltonSampler.fastPermutationTable[2182315 + ((index / 6011u) % 6011u)]) * float(DoubleOneMinusEpsilon / 36132121u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6029(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2188326 + (index % 6029u)] * 6029u +
				globalHaltonSampler.fastPermutationTable[2188326 + ((index / 6029u) % 6029u)]) * float(DoubleOneMinusEpsilon / 36348841u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6037(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2194355 + (index % 6037u)] * 6037u +
				globalHaltonSampler.fastPermutationTable[2194355 + ((index / 6037u) % 6037u)]) * float(DoubleOneMinusEpsilon / 36445369u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6043(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2200392 + (index % 6043u)] * 6043u +
				globalHaltonSampler.fastPermutationTable[2200392 + ((index / 6043u) % 6043u)]) * float(DoubleOneMinusEpsilon / 36517849u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6047(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2206435 + (index % 6047u)] * 6047u +
				globalHaltonSampler.fastPermutationTable[2206435 + ((index / 6047u) % 6047u)]) * float(DoubleOneMinusEpsilon / 36566209u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6053(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2212482 + (index % 6053u)] * 6053u +
				globalHaltonSampler.fastPermutationTable[2212482 + ((index / 6053u) % 6053u)]) * float(DoubleOneMinusEpsilon / 36638809u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6067(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2218535 + (index % 6067u)] * 6067u +
				globalHaltonSampler.fastPermutationTable[2218535 + ((index / 6067u) % 6067u)]) * float(DoubleOneMinusEpsilon / 36808489u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6073(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2224602 + (index % 6073u)] * 6073u +
				globalHaltonSampler.fastPermutationTable[2224602 + ((index / 6073u) % 6073u)]) * float(DoubleOneMinusEpsilon / 36881329u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6079(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2230675 + (index % 6079u)] * 6079u +
				globalHaltonSampler.fastPermutationTable[2230675 + ((index / 6079u) % 6079u)]) * float(DoubleOneMinusEpsilon / 36954241u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6089(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2236754 + (index % 6089u)] * 6089u +
				globalHaltonSampler.fastPermutationTable[2236754 + ((index / 6089u) % 6089u)]) * float(DoubleOneMinusEpsilon / 37075921u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6091(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2242843 + (index % 6091u)] * 6091u +
				globalHaltonSampler.fastPermutationTable[2242843 + ((index / 6091u) % 6091u)]) * float(DoubleOneMinusEpsilon / 37100281u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6101(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2248934 + (index % 6101u)] * 6101u +
				globalHaltonSampler.fastPermutationTable[2248934 + ((index / 6101u) % 6101u)]) * float(DoubleOneMinusEpsilon / 37222201u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6113(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2255035 + (index % 6113u)] * 6113u +
				globalHaltonSampler.fastPermutationTable[2255035 + ((index / 6113u) % 6113u)]) * float(DoubleOneMinusEpsilon / 37368769u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6121(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2261148 + (index % 6121u)] * 6121u +
				globalHaltonSampler.fastPermutationTable[2261148 + ((index / 6121u) % 6121u)]) * float(DoubleOneMinusEpsilon / 37466641u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6131(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2267269 + (index % 6131u)] * 6131u +
				globalHaltonSampler.fastPermutationTable[2267269 + ((index / 6131u) % 6131u)]) * float(DoubleOneMinusEpsilon / 37589161u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6133(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2273400 + (index % 6133u)] * 6133u +
				globalHaltonSampler.fastPermutationTable[2273400 + ((index / 6133u) % 6133u)]) * float(DoubleOneMinusEpsilon / 37613689u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6143(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2279533 + (index % 6143u)] * 6143u +
				globalHaltonSampler.fastPermutationTable[2279533 + ((index / 6143u) % 6143u)]) * float(DoubleOneMinusEpsilon / 37736449u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6151(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2285676 + (index % 6151u)] * 6151u +
				globalHaltonSampler.fastPermutationTable[2285676 + ((index / 6151u) % 6151u)]) * float(DoubleOneMinusEpsilon / 37834801u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6163(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2291827 + (index % 6163u)] * 6163u +
				globalHaltonSampler.fastPermutationTable[2291827 + ((index / 6163u) % 6163u)]) * float(DoubleOneMinusEpsilon / 37982569u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6173(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2297990 + (index % 6173u)] * 6173u +
				globalHaltonSampler.fastPermutationTable[2297990 + ((index / 6173u) % 6173u)]) * float(DoubleOneMinusEpsilon / 38105929u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6197(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2304163 + (index % 6197u)] * 6197u +
				globalHaltonSampler.fastPermutationTable[2304163 + ((index / 6197u) % 6197u)]) * float(DoubleOneMinusEpsilon / 38402809u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6199(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2310360 + (index % 6199u)] * 6199u +
				globalHaltonSampler.fastPermutationTable[2310360 + ((index / 6199u) % 6199u)]) * float(DoubleOneMinusEpsilon / 38427601u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6203(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2316559 + (index % 6203u)] * 6203u +
				globalHaltonSampler.fastPermutationTable[2316559 + ((index / 6203u) % 6203u)]) * float(DoubleOneMinusEpsilon / 38477209u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6211(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2322762 + (index % 6211u)] * 6211u +
				globalHaltonSampler.fastPermutationTable[2322762 + ((index / 6211u) % 6211u)]) * float(DoubleOneMinusEpsilon / 38576521u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6217(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2328973 + (index % 6217u)] * 6217u +
				globalHaltonSampler.fastPermutationTable[2328973 + ((index / 6217u) % 6217u)]) * float(DoubleOneMinusEpsilon / 38651089u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6221(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2335190 + (index % 6221u)] * 6221u +
				globalHaltonSampler.fastPermutationTable[2335190 + ((index / 6221u) % 6221u)]) * float(DoubleOneMinusEpsilon / 38700841u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6229(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2341411 + (index % 6229u)] * 6229u +
				globalHaltonSampler.fastPermutationTable[2341411 + ((index / 6229u) % 6229u)]) * float(DoubleOneMinusEpsilon / 38800441u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6247(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2347640 + (index % 6247u)] * 6247u +
				globalHaltonSampler.fastPermutationTable[2347640 + ((index / 6247u) % 6247u)]) * float(DoubleOneMinusEpsilon / 39025009u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6257(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2353887 + (index % 6257u)] * 6257u +
				globalHaltonSampler.fastPermutationTable[2353887 + ((index / 6257u) % 6257u)]) * float(DoubleOneMinusEpsilon / 39150049u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6263(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2360144 + (index % 6263u)] * 6263u +
				globalHaltonSampler.fastPermutationTable[2360144 + ((index / 6263u) % 6263u)]) * float(DoubleOneMinusEpsilon / 39225169u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6269(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2366407 + (index % 6269u)] * 6269u +
				globalHaltonSampler.fastPermutationTable[2366407 + ((index / 6269u) % 6269u)]) * float(DoubleOneMinusEpsilon / 39300361u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6271(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2372676 + (index % 6271u)] * 6271u +
				globalHaltonSampler.fastPermutationTable[2372676 + ((index / 6271u) % 6271u)]) * float(DoubleOneMinusEpsilon / 39325441u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6277(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2378947 + (index % 6277u)] * 6277u +
				globalHaltonSampler.fastPermutationTable[2378947 + ((index / 6277u) % 6277u)]) * float(DoubleOneMinusEpsilon / 39400729u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6287(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2385224 + (index % 6287u)] * 6287u +
				globalHaltonSampler.fastPermutationTable[2385224 + ((index / 6287u) % 6287u)]) * float(DoubleOneMinusEpsilon / 39526369u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6299(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2391511 + (index % 6299u)] * 6299u +
				globalHaltonSampler.fastPermutationTable[2391511 + ((index / 6299u) % 6299u)]) * float(DoubleOneMinusEpsilon / 39677401u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6301(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2397810 + (index % 6301u)] * 6301u +
				globalHaltonSampler.fastPermutationTable[2397810 + ((index / 6301u) % 6301u)]) * float(DoubleOneMinusEpsilon / 39702601u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6311(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2404111 + (index % 6311u)] * 6311u +
				globalHaltonSampler.fastPermutationTable[2404111 + ((index / 6311u) % 6311u)]) * float(DoubleOneMinusEpsilon / 39828721u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6317(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2410422 + (index % 6317u)] * 6317u +
				globalHaltonSampler.fastPermutationTable[2410422 + ((index / 6317u) % 6317u)]) * float(DoubleOneMinusEpsilon / 39904489u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6323(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2416739 + (index % 6323u)] * 6323u +
				globalHaltonSampler.fastPermutationTable[2416739 + ((index / 6323u) % 6323u)]) * float(DoubleOneMinusEpsilon / 39980329u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6329(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2423062 + (index % 6329u)] * 6329u +
				globalHaltonSampler.fastPermutationTable[2423062 + ((index / 6329u) % 6329u)]) * float(DoubleOneMinusEpsilon / 40056241u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6337(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2429391 + (index % 6337u)] * 6337u +
				globalHaltonSampler.fastPermutationTable[2429391 + ((index / 6337u) % 6337u)]) * float(DoubleOneMinusEpsilon / 40157569u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6343(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2435728 + (index % 6343u)] * 6343u +
				globalHaltonSampler.fastPermutationTable[2435728 + ((index / 6343u) % 6343u)]) * float(DoubleOneMinusEpsilon / 40233649u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6353(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2442071 + (index % 6353u)] * 6353u +
				globalHaltonSampler.fastPermutationTable[2442071 + ((index / 6353u) % 6353u)]) * float(DoubleOneMinusEpsilon / 40360609u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6359(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2448424 + (index % 6359u)] * 6359u +
				globalHaltonSampler.fastPermutationTable[2448424 + ((index / 6359u) % 6359u)]) * float(DoubleOneMinusEpsilon / 40436881u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6361(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2454783 + (index % 6361u)] * 6361u +
				globalHaltonSampler.fastPermutationTable[2454783 + ((index / 6361u) % 6361u)]) * float(DoubleOneMinusEpsilon / 40462321u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6367(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2461144 + (index % 6367u)] * 6367u +
				globalHaltonSampler.fastPermutationTable[2461144 + ((index / 6367u) % 6367u)]) * float(DoubleOneMinusEpsilon / 40538689u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6373(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2467511 + (index % 6373u)] * 6373u +
				globalHaltonSampler.fastPermutationTable[2467511 + ((index / 6373u) % 6373u)]) * float(DoubleOneMinusEpsilon / 40615129u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6379(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2473884 + (index % 6379u)] * 6379u +
				globalHaltonSampler.fastPermutationTable[2473884 + ((index / 6379u) % 6379u)]) * float(DoubleOneMinusEpsilon / 40691641u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6389(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2480263 + (index % 6389u)] * 6389u +
				globalHaltonSampler.fastPermutationTable[2480263 + ((index / 6389u) % 6389u)]) * float(DoubleOneMinusEpsilon / 40819321u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6397(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2486652 + (index % 6397u)] * 6397u +
				globalHaltonSampler.fastPermutationTable[2486652 + ((index / 6397u) % 6397u)]) * float(DoubleOneMinusEpsilon / 40921609u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6421(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2493049 + (index % 6421u)] * 6421u +
				globalHaltonSampler.fastPermutationTable[2493049 + ((index / 6421u) % 6421u)]) * float(DoubleOneMinusEpsilon / 41229241u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6427(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2499470 + (index % 6427u)] * 6427u +
				globalHaltonSampler.fastPermutationTable[2499470 + ((index / 6427u) % 6427u)]) * float(DoubleOneMinusEpsilon / 41306329u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6449(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2505897 + (index % 6449u)] * 6449u +
				globalHaltonSampler.fastPermutationTable[2505897 + ((index / 6449u) % 6449u)]) * float(DoubleOneMinusEpsilon / 41589601u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6451(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2512346 + (index % 6451u)] * 6451u +
				globalHaltonSampler.fastPermutationTable[2512346 + ((index / 6451u) % 6451u)]) * float(DoubleOneMinusEpsilon / 41615401u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6469(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2518797 + (index % 6469u)] * 6469u +
				globalHaltonSampler.fastPermutationTable[2518797 + ((index / 6469u) % 6469u)]) * float(DoubleOneMinusEpsilon / 41847961u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6473(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2525266 + (index % 6473u)] * 6473u +
				globalHaltonSampler.fastPermutationTable[2525266 + ((index / 6473u) % 6473u)]) * float(DoubleOneMinusEpsilon / 41899729u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6481(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2531739 + (index % 6481u)] * 6481u +
				globalHaltonSampler.fastPermutationTable[2531739 + ((index / 6481u) % 6481u)]) * float(DoubleOneMinusEpsilon / 42003361u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6491(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2538220 + (index % 6491u)] * 6491u +
				globalHaltonSampler.fastPermutationTable[2538220 + ((index / 6491u) % 6491u)]) * float(DoubleOneMinusEpsilon / 42133081u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6521(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2544711 + (index % 6521u)] * 6521u +
				globalHaltonSampler.fastPermutationTable[2544711 + ((index / 6521u) % 6521u)]) * float(DoubleOneMinusEpsilon / 42523441u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6529(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2551232 + (index % 6529u)] * 6529u +
				globalHaltonSampler.fastPermutationTable[2551232 + ((index / 6529u) % 6529u)]) * float(DoubleOneMinusEpsilon / 42627841u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6547(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2557761 + (index % 6547u)] * 6547u +
				globalHaltonSampler.fastPermutationTable[2557761 + ((index / 6547u) % 6547u)]) * float(DoubleOneMinusEpsilon / 42863209u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6551(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2564308 + (index % 6551u)] * 6551u +
				globalHaltonSampler.fastPermutationTable[2564308 + ((index / 6551u) % 6551u)]) * float(DoubleOneMinusEpsilon / 42915601u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6553(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2570859 + (index % 6553u)] * 6553u +
				globalHaltonSampler.fastPermutationTable[2570859 + ((index / 6553u) % 6553u)]) * float(DoubleOneMinusEpsilon / 42941809u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6563(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2577412 + (index % 6563u)] * 6563u +
				globalHaltonSampler.fastPermutationTable[2577412 + ((index / 6563u) % 6563u)]) * float(DoubleOneMinusEpsilon / 43072969u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6569(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2583975 + (index % 6569u)] * 6569u +
				globalHaltonSampler.fastPermutationTable[2583975 + ((index / 6569u) % 6569u)]) * float(DoubleOneMinusEpsilon / 43151761u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6571(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2590544 + (index % 6571u)] * 6571u +
				globalHaltonSampler.fastPermutationTable[2590544 + ((index / 6571u) % 6571u)]) * float(DoubleOneMinusEpsilon / 43178041u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6577(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2597115 + (index % 6577u)] * 6577u +
				globalHaltonSampler.fastPermutationTable[2597115 + ((index / 6577u) % 6577u)]) * float(DoubleOneMinusEpsilon / 43256929u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6581(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2603692 + (index % 6581u)] * 6581u +
				globalHaltonSampler.fastPermutationTable[2603692 + ((index / 6581u) % 6581u)]) * float(DoubleOneMinusEpsilon / 43309561u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6599(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2610273 + (index % 6599u)] * 6599u +
				globalHaltonSampler.fastPermutationTable[2610273 + ((index / 6599u) % 6599u)]) * float(DoubleOneMinusEpsilon / 43546801u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6607(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2616872 + (index % 6607u)] * 6607u +
				globalHaltonSampler.fastPermutationTable[2616872 + ((index / 6607u) % 6607u)]) * float(DoubleOneMinusEpsilon / 43652449u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6619(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2623479 + (index % 6619u)] * 6619u +
				globalHaltonSampler.fastPermutationTable[2623479 + ((index / 6619u) % 6619u)]) * float(DoubleOneMinusEpsilon / 43811161u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6637(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2630098 + (index % 6637u)] * 6637u +
				globalHaltonSampler.fastPermutationTable[2630098 + ((index / 6637u) % 6637u)]) * float(DoubleOneMinusEpsilon / 44049769u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6653(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2636735 + (index % 6653u)] * 6653u +
				globalHaltonSampler.fastPermutationTable[2636735 + ((index / 6653u) % 6653u)]) * float(DoubleOneMinusEpsilon / 44262409u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6659(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2643388 + (index % 6659u)] * 6659u +
				globalHaltonSampler.fastPermutationTable[2643388 + ((index / 6659u) % 6659u)]) * float(DoubleOneMinusEpsilon / 44342281u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6661(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2650047 + (index % 6661u)] * 6661u +
				globalHaltonSampler.fastPermutationTable[2650047 + ((index / 6661u) % 6661u)]) * float(DoubleOneMinusEpsilon / 44368921u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6673(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2656708 + (index % 6673u)] * 6673u +
				globalHaltonSampler.fastPermutationTable[2656708 + ((index / 6673u) % 6673u)]) * float(DoubleOneMinusEpsilon / 44528929u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6679(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2663381 + (index % 6679u)] * 6679u +
				globalHaltonSampler.fastPermutationTable[2663381 + ((index / 6679u) % 6679u)]) * float(DoubleOneMinusEpsilon / 44609041u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6689(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2670060 + (index % 6689u)] * 6689u +
				globalHaltonSampler.fastPermutationTable[2670060 + ((index / 6689u) % 6689u)]) * float(DoubleOneMinusEpsilon / 44742721u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6691(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2676749 + (index % 6691u)] * 6691u +
				globalHaltonSampler.fastPermutationTable[2676749 + ((index / 6691u) % 6691u)]) * float(DoubleOneMinusEpsilon / 44769481u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6701(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2683440 + (index % 6701u)] * 6701u +
				globalHaltonSampler.fastPermutationTable[2683440 + ((index / 6701u) % 6701u)]) * float(DoubleOneMinusEpsilon / 44903401u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6703(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2690141 + (index % 6703u)] * 6703u +
				globalHaltonSampler.fastPermutationTable[2690141 + ((index / 6703u) % 6703u)]) * float(DoubleOneMinusEpsilon / 44930209u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6709(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2696844 + (index % 6709u)] * 6709u +
				globalHaltonSampler.fastPermutationTable[2696844 + ((index / 6709u) % 6709u)]) * float(DoubleOneMinusEpsilon / 45010681u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6719(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2703553 + (index % 6719u)] * 6719u +
				globalHaltonSampler.fastPermutationTable[2703553 + ((index / 6719u) % 6719u)]) * float(DoubleOneMinusEpsilon / 45144961u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6733(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2710272 + (index % 6733u)] * 6733u +
				globalHaltonSampler.fastPermutationTable[2710272 + ((index / 6733u) % 6733u)]) * float(DoubleOneMinusEpsilon / 45333289u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6737(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2717005 + (index % 6737u)] * 6737u +
				globalHaltonSampler.fastPermutationTable[2717005 + ((index / 6737u) % 6737u)]) * float(DoubleOneMinusEpsilon / 45387169u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6761(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2723742 + (index % 6761u)] * 6761u +
				globalHaltonSampler.fastPermutationTable[2723742 + ((index / 6761u) % 6761u)]) * float(DoubleOneMinusEpsilon / 45711121u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6763(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2730503 + (index % 6763u)] * 6763u +
				globalHaltonSampler.fastPermutationTable[2730503 + ((index / 6763u) % 6763u)]) * float(DoubleOneMinusEpsilon / 45738169u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6779(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2737266 + (index % 6779u)] * 6779u +
				globalHaltonSampler.fastPermutationTable[2737266 + ((index / 6779u) % 6779u)]) * float(DoubleOneMinusEpsilon / 45954841u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6781(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2744045 + (index % 6781u)] * 6781u +
				globalHaltonSampler.fastPermutationTable[2744045 + ((index / 6781u) % 6781u)]) * float(DoubleOneMinusEpsilon / 45981961u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6791(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2750826 + (index % 6791u)] * 6791u +
				globalHaltonSampler.fastPermutationTable[2750826 + ((index / 6791u) % 6791u)]) * float(DoubleOneMinusEpsilon / 46117681u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6793(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2757617 + (index % 6793u)] * 6793u +
				globalHaltonSampler.fastPermutationTable[2757617 + ((index / 6793u) % 6793u)]) * float(DoubleOneMinusEpsilon / 46144849u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6803(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2764410 + (index % 6803u)] * 6803u +
				globalHaltonSampler.fastPermutationTable[2764410 + ((index / 6803u) % 6803u)]) * float(DoubleOneMinusEpsilon / 46280809u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6823(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2771213 + (index % 6823u)] * 6823u +
				globalHaltonSampler.fastPermutationTable[2771213 + ((index / 6823u) % 6823u)]) * float(DoubleOneMinusEpsilon / 46553329u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6827(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2778036 + (index % 6827u)] * 6827u +
				globalHaltonSampler.fastPermutationTable[2778036 + ((index / 6827u) % 6827u)]) * float(DoubleOneMinusEpsilon / 46607929u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6829(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2784863 + (index % 6829u)] * 6829u +
				globalHaltonSampler.fastPermutationTable[2784863 + ((index / 6829u) % 6829u)]) * float(DoubleOneMinusEpsilon / 46635241u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6833(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2791692 + (index % 6833u)] * 6833u +
				globalHaltonSampler.fastPermutationTable[2791692 + ((index / 6833u) % 6833u)]) * float(DoubleOneMinusEpsilon / 46689889u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6841(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2798525 + (index % 6841u)] * 6841u +
				globalHaltonSampler.fastPermutationTable[2798525 + ((index / 6841u) % 6841u)]) * float(DoubleOneMinusEpsilon / 46799281u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6857(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2805366 + (index % 6857u)] * 6857u +
				globalHaltonSampler.fastPermutationTable[2805366 + ((index / 6857u) % 6857u)]) * float(DoubleOneMinusEpsilon / 47018449u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6863(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2812223 + (index % 6863u)] * 6863u +
				globalHaltonSampler.fastPermutationTable[2812223 + ((index / 6863u) % 6863u)]) * float(DoubleOneMinusEpsilon / 47100769u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6869(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2819086 + (index % 6869u)] * 6869u +
				globalHaltonSampler.fastPermutationTable[2819086 + ((index / 6869u) % 6869u)]) * float(DoubleOneMinusEpsilon / 47183161u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6871(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2825955 + (index % 6871u)] * 6871u +
				globalHaltonSampler.fastPermutationTable[2825955 + ((index / 6871u) % 6871u)]) * float(DoubleOneMinusEpsilon / 47210641u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6883(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2832826 + (index % 6883u)] * 6883u +
				globalHaltonSampler.fastPermutationTable[2832826 + ((index / 6883u) % 6883u)]) * float(DoubleOneMinusEpsilon / 47375689u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6899(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2839709 + (index % 6899u)] * 6899u +
				globalHaltonSampler.fastPermutationTable[2839709 + ((index / 6899u) % 6899u)]) * float(DoubleOneMinusEpsilon / 47596201u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6907(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2846608 + (index % 6907u)] * 6907u +
				globalHaltonSampler.fastPermutationTable[2846608 + ((index / 6907u) % 6907u)]) * float(DoubleOneMinusEpsilon / 47706649u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6911(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2853515 + (index % 6911u)] * 6911u +
				globalHaltonSampler.fastPermutationTable[2853515 + ((index / 6911u) % 6911u)]) * float(DoubleOneMinusEpsilon / 47761921u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6917(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2860426 + (index % 6917u)] * 6917u +
				globalHaltonSampler.fastPermutationTable[2860426 + ((index / 6917u) % 6917u)]) * float(DoubleOneMinusEpsilon / 47844889u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6947(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2867343 + (index % 6947u)] * 6947u +
				globalHaltonSampler.fastPermutationTable[2867343 + ((index / 6947u) % 6947u)]) * float(DoubleOneMinusEpsilon / 48260809u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6949(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2874290 + (index % 6949u)] * 6949u +
				globalHaltonSampler.fastPermutationTable[2874290 + ((index / 6949u) % 6949u)]) * float(DoubleOneMinusEpsilon / 48288601u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6959(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2881239 + (index % 6959u)] * 6959u +
				globalHaltonSampler.fastPermutationTable[2881239 + ((index / 6959u) % 6959u)]) * float(DoubleOneMinusEpsilon / 48427681u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6961(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2888198 + (index % 6961u)] * 6961u +
				globalHaltonSampler.fastPermutationTable[2888198 + ((index / 6961u) % 6961u)]) * float(DoubleOneMinusEpsilon / 48455521u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6967(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2895159 + (index % 6967u)] * 6967u +
				globalHaltonSampler.fastPermutationTable[2895159 + ((index / 6967u) % 6967u)]) * float(DoubleOneMinusEpsilon / 48539089u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6971(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2902126 + (index % 6971u)] * 6971u +
				globalHaltonSampler.fastPermutationTable[2902126 + ((index / 6971u) % 6971u)]) * float(DoubleOneMinusEpsilon / 48594841u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6977(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2909097 + (index % 6977u)] * 6977u +
				globalHaltonSampler.fastPermutationTable[2909097 + ((index / 6977u) % 6977u)]) * float(DoubleOneMinusEpsilon / 48678529u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6983(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2916074 + (index % 6983u)] * 6983u +
				globalHaltonSampler.fastPermutationTable[2916074 + ((index / 6983u) % 6983u)]) * float(DoubleOneMinusEpsilon / 48762289u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6991(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2923057 + (index % 6991u)] * 6991u +
				globalHaltonSampler.fastPermutationTable[2923057 + ((index / 6991u) % 6991u)]) * float(DoubleOneMinusEpsilon / 48874081u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized6997(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2930048 + (index % 6997u)] * 6997u +
				globalHaltonSampler.fastPermutationTable[2930048 + ((index / 6997u) % 6997u)]) * float(DoubleOneMinusEpsilon / 48958009u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7001(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2937045 + (index % 7001u)] * 7001u +
				globalHaltonSampler.fastPermutationTable[2937045 + ((index / 7001u) % 7001u)]) * float(DoubleOneMinusEpsilon / 49014001u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7013(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2944046 + (index % 7013u)] * 7013u +
				globalHaltonSampler.fastPermutationTable[2944046 + ((index / 7013u) % 7013u)]) * float(DoubleOneMinusEpsilon / 49182169u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7019(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2951059 + (index % 7019u)] * 7019u +
				globalHaltonSampler.fastPermutationTable[2951059 + ((index / 7019u) % 7019u)]) * float(DoubleOneMinusEpsilon / 49266361u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7027(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2958078 + (index % 7027u)] * 7027u +
				globalHaltonSampler.fastPermutationTable[2958078 + ((index / 7027u) % 7027u)]) * float(DoubleOneMinusEpsilon / 49378729u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7039(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2965105 + (index % 7039u)] * 7039u +
				globalHaltonSampler.fastPermutationTable[2965105 + ((index / 7039u) % 7039u)]) * float(DoubleOneMinusEpsilon / 49547521u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7043(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2972144 + (index % 7043u)] * 7043u +
				globalHaltonSampler.fastPermutationTable[2972144 + ((index / 7043u) % 7043u)]) * float(DoubleOneMinusEpsilon / 49603849u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7057(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2979187 + (index % 7057u)] * 7057u +
				globalHaltonSampler.fastPermutationTable[2979187 + ((index / 7057u) % 7057u)]) * float(DoubleOneMinusEpsilon / 49801249u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7069(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2986244 + (index % 7069u)] * 7069u +
				globalHaltonSampler.fastPermutationTable[2986244 + ((index / 7069u) % 7069u)]) * float(DoubleOneMinusEpsilon / 49970761u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7079(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[2993313 + (index % 7079u)] * 7079u +
				globalHaltonSampler.fastPermutationTable[2993313 + ((index / 7079u) % 7079u)]) * float(DoubleOneMinusEpsilon / 50112241u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7103(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3000392 + (index % 7103u)] * 7103u +
				globalHaltonSampler.fastPermutationTable[3000392 + ((index / 7103u) % 7103u)]) * float(DoubleOneMinusEpsilon / 50452609u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7109(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3007495 + (index % 7109u)] * 7109u +
				globalHaltonSampler.fastPermutationTable[3007495 + ((index / 7109u) % 7109u)]) * float(DoubleOneMinusEpsilon / 50537881u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7121(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3014604 + (index % 7121u)] * 7121u +
				globalHaltonSampler.fastPermutationTable[3014604 + ((index / 7121u) % 7121u)]) * float(DoubleOneMinusEpsilon / 50708641u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7127(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3021725 + (index % 7127u)] * 7127u +
				globalHaltonSampler.fastPermutationTable[3021725 + ((index / 7127u) % 7127u)]) * float(DoubleOneMinusEpsilon / 50794129u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7129(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3028852 + (index % 7129u)] * 7129u +
				globalHaltonSampler.fastPermutationTable[3028852 + ((index / 7129u) % 7129u)]) * float(DoubleOneMinusEpsilon / 50822641u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7151(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3035981 + (index % 7151u)] * 7151u +
				globalHaltonSampler.fastPermutationTable[3035981 + ((index / 7151u) % 7151u)]) * float(DoubleOneMinusEpsilon / 51136801u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7159(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3043132 + (index % 7159u)] * 7159u +
				globalHaltonSampler.fastPermutationTable[3043132 + ((index / 7159u) % 7159u)]) * float(DoubleOneMinusEpsilon / 51251281u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7177(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3050291 + (index % 7177u)] * 7177u +
				globalHaltonSampler.fastPermutationTable[3050291 + ((index / 7177u) % 7177u)]) * float(DoubleOneMinusEpsilon / 51509329u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7187(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3057468 + (index % 7187u)] * 7187u +
				globalHaltonSampler.fastPermutationTable[3057468 + ((index / 7187u) % 7187u)]) * float(DoubleOneMinusEpsilon / 51652969u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7193(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3064655 + (index % 7193u)] * 7193u +
				globalHaltonSampler.fastPermutationTable[3064655 + ((index / 7193u) % 7193u)]) * float(DoubleOneMinusEpsilon / 51739249u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7207(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3071848 + (index % 7207u)] * 7207u +
				globalHaltonSampler.fastPermutationTable[3071848 + ((index / 7207u) % 7207u)]) * float(DoubleOneMinusEpsilon / 51940849u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7211(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3079055 + (index % 7211u)] * 7211u +
				globalHaltonSampler.fastPermutationTable[3079055 + ((index / 7211u) % 7211u)]) * float(DoubleOneMinusEpsilon / 51998521u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7213(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3086266 + (index % 7213u)] * 7213u +
				globalHaltonSampler.fastPermutationTable[3086266 + ((index / 7213u) % 7213u)]) * float(DoubleOneMinusEpsilon / 52027369u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7219(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3093479 + (index % 7219u)] * 7219u +
				globalHaltonSampler.fastPermutationTable[3093479 + ((index / 7219u) % 7219u)]) * float(DoubleOneMinusEpsilon / 52113961u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7229(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3100698 + (index % 7229u)] * 7229u +
				globalHaltonSampler.fastPermutationTable[3100698 + ((index / 7229u) % 7229u)]) * float(DoubleOneMinusEpsilon / 52258441u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7237(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3107927 + (index % 7237u)] * 7237u +
				globalHaltonSampler.fastPermutationTable[3107927 + ((index / 7237u) % 7237u)]) * float(DoubleOneMinusEpsilon / 52374169u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7243(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3115164 + (index % 7243u)] * 7243u +
				globalHaltonSampler.fastPermutationTable[3115164 + ((index / 7243u) % 7243u)]) * float(DoubleOneMinusEpsilon / 52461049u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7247(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3122407 + (index % 7247u)] * 7247u +
				globalHaltonSampler.fastPermutationTable[3122407 + ((index / 7247u) % 7247u)]) * float(DoubleOneMinusEpsilon / 52519009u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7253(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3129654 + (index % 7253u)] * 7253u +
				globalHaltonSampler.fastPermutationTable[3129654 + ((index / 7253u) % 7253u)]) * float(DoubleOneMinusEpsilon / 52606009u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7283(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3136907 + (index % 7283u)] * 7283u +
				globalHaltonSampler.fastPermutationTable[3136907 + ((index / 7283u) % 7283u)]) * float(DoubleOneMinusEpsilon / 53042089u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7297(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3144190 + (index % 7297u)] * 7297u +
				globalHaltonSampler.fastPermutationTable[3144190 + ((index / 7297u) % 7297u)]) * float(DoubleOneMinusEpsilon / 53246209u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7307(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3151487 + (index % 7307u)] * 7307u +
				globalHaltonSampler.fastPermutationTable[3151487 + ((index / 7307u) % 7307u)]) * float(DoubleOneMinusEpsilon / 53392249u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7309(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3158794 + (index % 7309u)] * 7309u +
				globalHaltonSampler.fastPermutationTable[3158794 + ((index / 7309u) % 7309u)]) * float(DoubleOneMinusEpsilon / 53421481u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7321(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3166103 + (index % 7321u)] * 7321u +
				globalHaltonSampler.fastPermutationTable[3166103 + ((index / 7321u) % 7321u)]) * float(DoubleOneMinusEpsilon / 53597041u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7331(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3173424 + (index % 7331u)] * 7331u +
				globalHaltonSampler.fastPermutationTable[3173424 + ((index / 7331u) % 7331u)]) * float(DoubleOneMinusEpsilon / 53743561u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7333(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3180755 + (index % 7333u)] * 7333u +
				globalHaltonSampler.fastPermutationTable[3180755 + ((index / 7333u) % 7333u)]) * float(DoubleOneMinusEpsilon / 53772889u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7349(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3188088 + (index % 7349u)] * 7349u +
				globalHaltonSampler.fastPermutationTable[3188088 + ((index / 7349u) % 7349u)]) * float(DoubleOneMinusEpsilon / 54007801u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7351(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3195437 + (index % 7351u)] * 7351u +
				globalHaltonSampler.fastPermutationTable[3195437 + ((index / 7351u) % 7351u)]) * float(DoubleOneMinusEpsilon / 54037201u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7369(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3202788 + (index % 7369u)] * 7369u +
				globalHaltonSampler.fastPermutationTable[3202788 + ((index / 7369u) % 7369u)]) * float(DoubleOneMinusEpsilon / 54302161u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7393(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3210157 + (index % 7393u)] * 7393u +
				globalHaltonSampler.fastPermutationTable[3210157 + ((index / 7393u) % 7393u)]) * float(DoubleOneMinusEpsilon / 54656449u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7411(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3217550 + (index % 7411u)] * 7411u +
				globalHaltonSampler.fastPermutationTable[3217550 + ((index / 7411u) % 7411u)]) * float(DoubleOneMinusEpsilon / 54922921u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7417(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3224961 + (index % 7417u)] * 7417u +
				globalHaltonSampler.fastPermutationTable[3224961 + ((index / 7417u) % 7417u)]) * float(DoubleOneMinusEpsilon / 55011889u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7433(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3232378 + (index % 7433u)] * 7433u +
				globalHaltonSampler.fastPermutationTable[3232378 + ((index / 7433u) % 7433u)]) * float(DoubleOneMinusEpsilon / 55249489u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7451(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3239811 + (index % 7451u)] * 7451u +
				globalHaltonSampler.fastPermutationTable[3239811 + ((index / 7451u) % 7451u)]) * float(DoubleOneMinusEpsilon / 55517401u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7457(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3247262 + (index % 7457u)] * 7457u +
				globalHaltonSampler.fastPermutationTable[3247262 + ((index / 7457u) % 7457u)]) * float(DoubleOneMinusEpsilon / 55606849u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7459(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3254719 + (index % 7459u)] * 7459u +
				globalHaltonSampler.fastPermutationTable[3254719 + ((index / 7459u) % 7459u)]) * float(DoubleOneMinusEpsilon / 55636681u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7477(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3262178 + (index % 7477u)] * 7477u +
				globalHaltonSampler.fastPermutationTable[3262178 + ((index / 7477u) % 7477u)]) * float(DoubleOneMinusEpsilon / 55905529u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7481(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3269655 + (index % 7481u)] * 7481u +
				globalHaltonSampler.fastPermutationTable[3269655 + ((index / 7481u) % 7481u)]) * float(DoubleOneMinusEpsilon / 55965361u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7487(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3277136 + (index % 7487u)] * 7487u +
				globalHaltonSampler.fastPermutationTable[3277136 + ((index / 7487u) % 7487u)]) * float(DoubleOneMinusEpsilon / 56055169u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7489(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3284623 + (index % 7489u)] * 7489u +
				globalHaltonSampler.fastPermutationTable[3284623 + ((index / 7489u) % 7489u)]) * float(DoubleOneMinusEpsilon / 56085121u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7499(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3292112 + (index % 7499u)] * 7499u +
				globalHaltonSampler.fastPermutationTable[3292112 + ((index / 7499u) % 7499u)]) * float(DoubleOneMinusEpsilon / 56235001u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7507(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3299611 + (index % 7507u)] * 7507u +
				globalHaltonSampler.fastPermutationTable[3299611 + ((index / 7507u) % 7507u)]) * float(DoubleOneMinusEpsilon / 56355049u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7517(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3307118 + (index % 7517u)] * 7517u +
				globalHaltonSampler.fastPermutationTable[3307118 + ((index / 7517u) % 7517u)]) * float(DoubleOneMinusEpsilon / 56505289u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7523(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3314635 + (index % 7523u)] * 7523u +
				globalHaltonSampler.fastPermutationTable[3314635 + ((index / 7523u) % 7523u)]) * float(DoubleOneMinusEpsilon / 56595529u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7529(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3322158 + (index % 7529u)] * 7529u +
				globalHaltonSampler.fastPermutationTable[3322158 + ((index / 7529u) % 7529u)]) * float(DoubleOneMinusEpsilon / 56685841u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7537(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3329687 + (index % 7537u)] * 7537u +
				globalHaltonSampler.fastPermutationTable[3329687 + ((index / 7537u) % 7537u)]) * float(DoubleOneMinusEpsilon / 56806369u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7541(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3337224 + (index % 7541u)] * 7541u +
				globalHaltonSampler.fastPermutationTable[3337224 + ((index / 7541u) % 7541u)]) * float(DoubleOneMinusEpsilon / 56866681u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7547(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3344765 + (index % 7547u)] * 7547u +
				globalHaltonSampler.fastPermutationTable[3344765 + ((index / 7547u) % 7547u)]) * float(DoubleOneMinusEpsilon / 56957209u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7549(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3352312 + (index % 7549u)] * 7549u +
				globalHaltonSampler.fastPermutationTable[3352312 + ((index / 7549u) % 7549u)]) * float(DoubleOneMinusEpsilon / 56987401u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7559(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3359861 + (index % 7559u)] * 7559u +
				globalHaltonSampler.fastPermutationTable[3359861 + ((index / 7559u) % 7559u)]) * float(DoubleOneMinusEpsilon / 57138481u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7561(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3367420 + (index % 7561u)] * 7561u +
				globalHaltonSampler.fastPermutationTable[3367420 + ((index / 7561u) % 7561u)]) * float(DoubleOneMinusEpsilon / 57168721u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7573(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3374981 + (index % 7573u)] * 7573u +
				globalHaltonSampler.fastPermutationTable[3374981 + ((index / 7573u) % 7573u)]) * float(DoubleOneMinusEpsilon / 57350329u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7577(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3382554 + (index % 7577u)] * 7577u +
				globalHaltonSampler.fastPermutationTable[3382554 + ((index / 7577u) % 7577u)]) * float(DoubleOneMinusEpsilon / 57410929u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7583(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3390131 + (index % 7583u)] * 7583u +
				globalHaltonSampler.fastPermutationTable[3390131 + ((index / 7583u) % 7583u)]) * float(DoubleOneMinusEpsilon / 57501889u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7589(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3397714 + (index % 7589u)] * 7589u +
				globalHaltonSampler.fastPermutationTable[3397714 + ((index / 7589u) % 7589u)]) * float(DoubleOneMinusEpsilon / 57592921u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7591(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3405303 + (index % 7591u)] * 7591u +
				globalHaltonSampler.fastPermutationTable[3405303 + ((index / 7591u) % 7591u)]) * float(DoubleOneMinusEpsilon / 57623281u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7603(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3412894 + (index % 7603u)] * 7603u +
				globalHaltonSampler.fastPermutationTable[3412894 + ((index / 7603u) % 7603u)]) * float(DoubleOneMinusEpsilon / 57805609u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7607(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3420497 + (index % 7607u)] * 7607u +
				globalHaltonSampler.fastPermutationTable[3420497 + ((index / 7607u) % 7607u)]) * float(DoubleOneMinusEpsilon / 57866449u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7621(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3428104 + (index % 7621u)] * 7621u +
				globalHaltonSampler.fastPermutationTable[3428104 + ((index / 7621u) % 7621u)]) * float(DoubleOneMinusEpsilon / 58079641u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7639(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3435725 + (index % 7639u)] * 7639u +
				globalHaltonSampler.fastPermutationTable[3435725 + ((index / 7639u) % 7639u)]) * float(DoubleOneMinusEpsilon / 58354321u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7643(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3443364 + (index % 7643u)] * 7643u +
				globalHaltonSampler.fastPermutationTable[3443364 + ((index / 7643u) % 7643u)]) * float(DoubleOneMinusEpsilon / 58415449u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7649(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3451007 + (index % 7649u)] * 7649u +
				globalHaltonSampler.fastPermutationTable[3451007 + ((index / 7649u) % 7649u)]) * float(DoubleOneMinusEpsilon / 58507201u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7669(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3458656 + (index % 7669u)] * 7669u +
				globalHaltonSampler.fastPermutationTable[3458656 + ((index / 7669u) % 7669u)]) * float(DoubleOneMinusEpsilon / 58813561u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7673(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3466325 + (index % 7673u)] * 7673u +
				globalHaltonSampler.fastPermutationTable[3466325 + ((index / 7673u) % 7673u)]) * float(DoubleOneMinusEpsilon / 58874929u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7681(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3473998 + (index % 7681u)] * 7681u +
				globalHaltonSampler.fastPermutationTable[3473998 + ((index / 7681u) % 7681u)]) * float(DoubleOneMinusEpsilon / 58997761u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7687(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3481679 + (index % 7687u)] * 7687u +
				globalHaltonSampler.fastPermutationTable[3481679 + ((index / 7687u) % 7687u)]) * float(DoubleOneMinusEpsilon / 59089969u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7691(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3489366 + (index % 7691u)] * 7691u +
				globalHaltonSampler.fastPermutationTable[3489366 + ((index / 7691u) % 7691u)]) * float(DoubleOneMinusEpsilon / 59151481u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7699(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3497057 + (index % 7699u)] * 7699u +
				globalHaltonSampler.fastPermutationTable[3497057 + ((index / 7699u) % 7699u)]) * float(DoubleOneMinusEpsilon / 59274601u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7703(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3504756 + (index % 7703u)] * 7703u +
				globalHaltonSampler.fastPermutationTable[3504756 + ((index / 7703u) % 7703u)]) * float(DoubleOneMinusEpsilon / 59336209u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7717(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3512459 + (index % 7717u)] * 7717u +
				globalHaltonSampler.fastPermutationTable[3512459 + ((index / 7717u) % 7717u)]) * float(DoubleOneMinusEpsilon / 59552089u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7723(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3520176 + (index % 7723u)] * 7723u +
				globalHaltonSampler.fastPermutationTable[3520176 + ((index / 7723u) % 7723u)]) * float(DoubleOneMinusEpsilon / 59644729u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7727(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3527899 + (index % 7727u)] * 7727u +
				globalHaltonSampler.fastPermutationTable[3527899 + ((index / 7727u) % 7727u)]) * float(DoubleOneMinusEpsilon / 59706529u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7741(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3535626 + (index % 7741u)] * 7741u +
				globalHaltonSampler.fastPermutationTable[3535626 + ((index / 7741u) % 7741u)]) * float(DoubleOneMinusEpsilon / 59923081u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7753(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3543367 + (index % 7753u)] * 7753u +
				globalHaltonSampler.fastPermutationTable[3543367 + ((index / 7753u) % 7753u)]) * float(DoubleOneMinusEpsilon / 60109009u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7757(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3551120 + (index % 7757u)] * 7757u +
				globalHaltonSampler.fastPermutationTable[3551120 + ((index / 7757u) % 7757u)]) * float(DoubleOneMinusEpsilon / 60171049u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7759(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3558877 + (index % 7759u)] * 7759u +
				globalHaltonSampler.fastPermutationTable[3558877 + ((index / 7759u) % 7759u)]) * float(DoubleOneMinusEpsilon / 60202081u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7789(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3566636 + (index % 7789u)] * 7789u +
				globalHaltonSampler.fastPermutationTable[3566636 + ((index / 7789u) % 7789u)]) * float(DoubleOneMinusEpsilon / 60668521u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7793(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3574425 + (index % 7793u)] * 7793u +
				globalHaltonSampler.fastPermutationTable[3574425 + ((index / 7793u) % 7793u)]) * float(DoubleOneMinusEpsilon / 60730849u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7817(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3582218 + (index % 7817u)] * 7817u +
				globalHaltonSampler.fastPermutationTable[3582218 + ((index / 7817u) % 7817u)]) * float(DoubleOneMinusEpsilon / 61105489u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7823(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3590035 + (index % 7823u)] * 7823u +
				globalHaltonSampler.fastPermutationTable[3590035 + ((index / 7823u) % 7823u)]) * float(DoubleOneMinusEpsilon / 61199329u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7829(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3597858 + (index % 7829u)] * 7829u +
				globalHaltonSampler.fastPermutationTable[3597858 + ((index / 7829u) % 7829u)]) * float(DoubleOneMinusEpsilon / 61293241u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7841(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3605687 + (index % 7841u)] * 7841u +
				globalHaltonSampler.fastPermutationTable[3605687 + ((index / 7841u) % 7841u)]) * float(DoubleOneMinusEpsilon / 61481281u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7853(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3613528 + (index % 7853u)] * 7853u +
				globalHaltonSampler.fastPermutationTable[3613528 + ((index / 7853u) % 7853u)]) * float(DoubleOneMinusEpsilon / 61669609u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7867(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3621381 + (index % 7867u)] * 7867u +
				globalHaltonSampler.fastPermutationTable[3621381 + ((index / 7867u) % 7867u)]) * float(DoubleOneMinusEpsilon / 61889689u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7873(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3629248 + (index % 7873u)] * 7873u +
				globalHaltonSampler.fastPermutationTable[3629248 + ((index / 7873u) % 7873u)]) * float(DoubleOneMinusEpsilon / 61984129u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7877(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3637121 + (index % 7877u)] * 7877u +
				globalHaltonSampler.fastPermutationTable[3637121 + ((index / 7877u) % 7877u)]) * float(DoubleOneMinusEpsilon / 62047129u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7879(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3644998 + (index % 7879u)] * 7879u +
				globalHaltonSampler.fastPermutationTable[3644998 + ((index / 7879u) % 7879u)]) * float(DoubleOneMinusEpsilon / 62078641u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7883(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3652877 + (index % 7883u)] * 7883u +
				globalHaltonSampler.fastPermutationTable[3652877 + ((index / 7883u) % 7883u)]) * float(DoubleOneMinusEpsilon / 62141689u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7901(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3660760 + (index % 7901u)] * 7901u +
				globalHaltonSampler.fastPermutationTable[3660760 + ((index / 7901u) % 7901u)]) * float(DoubleOneMinusEpsilon / 62425801u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7907(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3668661 + (index % 7907u)] * 7907u +
				globalHaltonSampler.fastPermutationTable[3668661 + ((index / 7907u) % 7907u)]) * float(DoubleOneMinusEpsilon / 62520649u); // Results in [0,1).
		}

		static __device__ __inline__ float HaltonSpecialized7919(const uint32_t index)
		{
			return (globalHaltonSampler.fastPermutationTable[3676568 + (index % 7919u)] * 7919u +
				globalHaltonSampler.fastPermutationTable[3676568 + ((index / 7919u) % 7919u)]) * float(DoubleOneMinusEpsilon / 62710561u); // Results in [0,1).
		}
#pragma endregion remainderCases_haltonx


		/**
          * @brief Compute RadicalInverse of given index, with respect to dimension.
		  * Starting from the 4th(value 3) dimension, random scrambling is employed 
		  * to break up correlation pattern. This function is fast, due to the cost
		  * of precomputation table of scrambling digits.
          * @param dimension the requested dimension, 0,1,2,3,...,999
          * @param index input parameter: 32bits unsigned index
          * @return return the radical inverse
          */
		static __device__ __noinline__ float ScrambledRadicalInverse_Fast(uint32_t dimension, uint32_t index)
		{
			switch (dimension)
			{
			case 0: return ReverseBits32(index) * 2.328306437e-10f;/*32 bits used for base 2*/
			case 1: return HaltonSpecialized3(index);
			case 2: return HaltonSpecialized5(index);
			case 3: return HaltonSpecialized7(index);
			case 4: return HaltonSpecialized11(index);
#pragma region remainderCases_sample
			case 5: return HaltonSpecialized13(index);
			case 6: return HaltonSpecialized17(index);
			case 7: return HaltonSpecialized19(index);
			case 8: return HaltonSpecialized23(index);
			case 9: return HaltonSpecialized29(index);
			case 10: return HaltonSpecialized31(index);
			case 11: return HaltonSpecialized37(index);
			case 12: return HaltonSpecialized41(index);
			case 13: return HaltonSpecialized43(index);
			case 14: return HaltonSpecialized47(index);
			case 15: return HaltonSpecialized53(index);
			case 16: return HaltonSpecialized59(index);
			case 17: return HaltonSpecialized61(index);
			case 18: return HaltonSpecialized67(index);
			case 19: return HaltonSpecialized71(index);
			case 20: return HaltonSpecialized73(index);
#if 1
			case 21: return HaltonSpecialized79(index);
			case 22: return HaltonSpecialized83(index);
			case 23: return HaltonSpecialized89(index);
			case 24: return HaltonSpecialized97(index);
			case 25: return HaltonSpecialized101(index);
			case 26: return HaltonSpecialized103(index);
			case 27: return HaltonSpecialized107(index);
			case 28: return HaltonSpecialized109(index);
			case 29: return HaltonSpecialized113(index);
			case 30: return HaltonSpecialized127(index);
			case 31: return HaltonSpecialized131(index);
			case 32: return HaltonSpecialized137(index);
			case 33: return HaltonSpecialized139(index);
			case 34: return HaltonSpecialized149(index);
			case 35: return HaltonSpecialized151(index);
			case 36: return HaltonSpecialized157(index);
			case 37: return HaltonSpecialized163(index);
			case 38: return HaltonSpecialized167(index);
			case 39: return HaltonSpecialized173(index);
			case 40: return HaltonSpecialized179(index);
			case 41: return HaltonSpecialized181(index);
			case 42: return HaltonSpecialized191(index);
			case 43: return HaltonSpecialized193(index);
			case 44: return HaltonSpecialized197(index);
			case 45: return HaltonSpecialized199(index);
			case 46: return HaltonSpecialized211(index);
			case 47: return HaltonSpecialized223(index);
			case 48: return HaltonSpecialized227(index);
			case 49: return HaltonSpecialized229(index);
			case 50: return HaltonSpecialized233(index);
			case 51: return HaltonSpecialized239(index);
			case 52: return HaltonSpecialized241(index);
			case 53: return HaltonSpecialized251(index);
			case 54: return HaltonSpecialized257(index);
			case 55: return HaltonSpecialized263(index);
			case 56: return HaltonSpecialized269(index);
			case 57: return HaltonSpecialized271(index);
			case 58: return HaltonSpecialized277(index);
			case 59: return HaltonSpecialized281(index);
			case 60: return HaltonSpecialized283(index);
			case 61: return HaltonSpecialized293(index);
			case 62: return HaltonSpecialized307(index);
			case 63: return HaltonSpecialized311(index);
			case 64: return HaltonSpecialized313(index);
			case 65: return HaltonSpecialized317(index);
			case 66: return HaltonSpecialized331(index);
			case 67: return HaltonSpecialized337(index);
			case 68: return HaltonSpecialized347(index);
			case 69: return HaltonSpecialized349(index);
			case 70: return HaltonSpecialized353(index);
			case 71: return HaltonSpecialized359(index);
			case 72: return HaltonSpecialized367(index);
			case 73: return HaltonSpecialized373(index);
			case 74: return HaltonSpecialized379(index);
			case 75: return HaltonSpecialized383(index);
			case 76: return HaltonSpecialized389(index);
			case 77: return HaltonSpecialized397(index);
			case 78: return HaltonSpecialized401(index);
			case 79: return HaltonSpecialized409(index);
			case 80: return HaltonSpecialized419(index);
			case 81: return HaltonSpecialized421(index);
			case 82: return HaltonSpecialized431(index);
			case 83: return HaltonSpecialized433(index);
			case 84: return HaltonSpecialized439(index);
			case 85: return HaltonSpecialized443(index);
			case 86: return HaltonSpecialized449(index);
			case 87: return HaltonSpecialized457(index);
			case 88: return HaltonSpecialized461(index);
			case 89: return HaltonSpecialized463(index);
			case 90: return HaltonSpecialized467(index);
			case 91: return HaltonSpecialized479(index);
			case 92: return HaltonSpecialized487(index);
			case 93: return HaltonSpecialized491(index);
			case 94: return HaltonSpecialized499(index);
			case 95: return HaltonSpecialized503(index);
			case 96: return HaltonSpecialized509(index);
			case 97: return HaltonSpecialized521(index);
			case 98: return HaltonSpecialized523(index);
			case 99: return HaltonSpecialized541(index);
			case 100: return HaltonSpecialized547(index);
			case 101: return HaltonSpecialized557(index);
			case 102: return HaltonSpecialized563(index);
			case 103: return HaltonSpecialized569(index);
			case 104: return HaltonSpecialized571(index);
			case 105: return HaltonSpecialized577(index);
			case 106: return HaltonSpecialized587(index);
			case 107: return HaltonSpecialized593(index);
			case 108: return HaltonSpecialized599(index);
			case 109: return HaltonSpecialized601(index);
			case 110: return HaltonSpecialized607(index);
			case 111: return HaltonSpecialized613(index);
			case 112: return HaltonSpecialized617(index);
			case 113: return HaltonSpecialized619(index);
			case 114: return HaltonSpecialized631(index);
			case 115: return HaltonSpecialized641(index);
			case 116: return HaltonSpecialized643(index);
			case 117: return HaltonSpecialized647(index);
			case 118: return HaltonSpecialized653(index);
			case 119: return HaltonSpecialized659(index);
			case 120: return HaltonSpecialized661(index);
			case 121: return HaltonSpecialized673(index);
			case 122: return HaltonSpecialized677(index);
			case 123: return HaltonSpecialized683(index);
			case 124: return HaltonSpecialized691(index);
			case 125: return HaltonSpecialized701(index);
			case 126: return HaltonSpecialized709(index);
			case 127: return HaltonSpecialized719(index);
			case 128: return HaltonSpecialized727(index);
			case 129: return HaltonSpecialized733(index);
			case 130: return HaltonSpecialized739(index);
			case 131: return HaltonSpecialized743(index);
			case 132: return HaltonSpecialized751(index);
			case 133: return HaltonSpecialized757(index);
			case 134: return HaltonSpecialized761(index);
			case 135: return HaltonSpecialized769(index);
			case 136: return HaltonSpecialized773(index);
			case 137: return HaltonSpecialized787(index);
			case 138: return HaltonSpecialized797(index);
			case 139: return HaltonSpecialized809(index);
			case 140: return HaltonSpecialized811(index);
			case 141: return HaltonSpecialized821(index);
			case 142: return HaltonSpecialized823(index);
			case 143: return HaltonSpecialized827(index);
			case 144: return HaltonSpecialized829(index);
			case 145: return HaltonSpecialized839(index);
			case 146: return HaltonSpecialized853(index);
			case 147: return HaltonSpecialized857(index);
			case 148: return HaltonSpecialized859(index);
			case 149: return HaltonSpecialized863(index);
			case 150: return HaltonSpecialized877(index);
			case 151: return HaltonSpecialized881(index);
			case 152: return HaltonSpecialized883(index);
			case 153: return HaltonSpecialized887(index);
			case 154: return HaltonSpecialized907(index);
			case 155: return HaltonSpecialized911(index);
			case 156: return HaltonSpecialized919(index);
			case 157: return HaltonSpecialized929(index);
			case 158: return HaltonSpecialized937(index);
			case 159: return HaltonSpecialized941(index);
			case 160: return HaltonSpecialized947(index);
			case 161: return HaltonSpecialized953(index);
			case 162: return HaltonSpecialized967(index);
			case 163: return HaltonSpecialized971(index);
			case 164: return HaltonSpecialized977(index);
			case 165: return HaltonSpecialized983(index);
			case 166: return HaltonSpecialized991(index);
			case 167: return HaltonSpecialized997(index);
			case 168: return HaltonSpecialized1009(index);
			case 169: return HaltonSpecialized1013(index);
			case 170: return HaltonSpecialized1019(index);
			case 171: return HaltonSpecialized1021(index);
			case 172: return HaltonSpecialized1031(index);
			case 173: return HaltonSpecialized1033(index);
			case 174: return HaltonSpecialized1039(index);
			case 175: return HaltonSpecialized1049(index);
			case 176: return HaltonSpecialized1051(index);
			case 177: return HaltonSpecialized1061(index);
			case 178: return HaltonSpecialized1063(index);
			case 179: return HaltonSpecialized1069(index);
			case 180: return HaltonSpecialized1087(index);
			case 181: return HaltonSpecialized1091(index);
			case 182: return HaltonSpecialized1093(index);
			case 183: return HaltonSpecialized1097(index);
			case 184: return HaltonSpecialized1103(index);
			case 185: return HaltonSpecialized1109(index);
			case 186: return HaltonSpecialized1117(index);
			case 187: return HaltonSpecialized1123(index);
			case 188: return HaltonSpecialized1129(index);
			case 189: return HaltonSpecialized1151(index);
			case 190: return HaltonSpecialized1153(index);
			case 191: return HaltonSpecialized1163(index);
			case 192: return HaltonSpecialized1171(index);
			case 193: return HaltonSpecialized1181(index);
			case 194: return HaltonSpecialized1187(index);
			case 195: return HaltonSpecialized1193(index);
			case 196: return HaltonSpecialized1201(index);
			case 197: return HaltonSpecialized1213(index);
			case 198: return HaltonSpecialized1217(index);
			case 199: return HaltonSpecialized1223(index);
			case 200: return HaltonSpecialized1229(index);
			case 201: return HaltonSpecialized1231(index);
			case 202: return HaltonSpecialized1237(index);
			case 203: return HaltonSpecialized1249(index);
			case 204: return HaltonSpecialized1259(index);
			case 205: return HaltonSpecialized1277(index);
			case 206: return HaltonSpecialized1279(index);
			case 207: return HaltonSpecialized1283(index);
			case 208: return HaltonSpecialized1289(index);
			case 209: return HaltonSpecialized1291(index);
			case 210: return HaltonSpecialized1297(index);
			case 211: return HaltonSpecialized1301(index);
			case 212: return HaltonSpecialized1303(index);
			case 213: return HaltonSpecialized1307(index);
			case 214: return HaltonSpecialized1319(index);
			case 215: return HaltonSpecialized1321(index);
			case 216: return HaltonSpecialized1327(index);
			case 217: return HaltonSpecialized1361(index);
			case 218: return HaltonSpecialized1367(index);
			case 219: return HaltonSpecialized1373(index);
			case 220: return HaltonSpecialized1381(index);
			case 221: return HaltonSpecialized1399(index);
			case 222: return HaltonSpecialized1409(index);
			case 223: return HaltonSpecialized1423(index);
			case 224: return HaltonSpecialized1427(index);
			case 225: return HaltonSpecialized1429(index);
			case 226: return HaltonSpecialized1433(index);
			case 227: return HaltonSpecialized1439(index);
			case 228: return HaltonSpecialized1447(index);
			case 229: return HaltonSpecialized1451(index);
			case 230: return HaltonSpecialized1453(index);
			case 231: return HaltonSpecialized1459(index);
			case 232: return HaltonSpecialized1471(index);
			case 233: return HaltonSpecialized1481(index);
			case 234: return HaltonSpecialized1483(index);
			case 235: return HaltonSpecialized1487(index);
			case 236: return HaltonSpecialized1489(index);
			case 237: return HaltonSpecialized1493(index);
			case 238: return HaltonSpecialized1499(index);
			case 239: return HaltonSpecialized1511(index);
			case 240: return HaltonSpecialized1523(index);
			case 241: return HaltonSpecialized1531(index);
			case 242: return HaltonSpecialized1543(index);
			case 243: return HaltonSpecialized1549(index);
			case 244: return HaltonSpecialized1553(index);
			case 245: return HaltonSpecialized1559(index);
			case 246: return HaltonSpecialized1567(index);
			case 247: return HaltonSpecialized1571(index);
			case 248: return HaltonSpecialized1579(index);
			case 249: return HaltonSpecialized1583(index);
			case 250: return HaltonSpecialized1597(index);

#endif
#if 1
			case 251: return HaltonSpecialized1601(index);
			case 252: return HaltonSpecialized1607(index);
			case 253: return HaltonSpecialized1609(index);
			case 254: return HaltonSpecialized1613(index);
			case 255: return HaltonSpecialized1619(index);
			case 256: return HaltonSpecialized1621(index);
			case 257: return HaltonSpecialized1627(index);
			case 258: return HaltonSpecialized1637(index);
			case 259: return HaltonSpecialized1657(index);
			case 260: return HaltonSpecialized1663(index);
			case 261: return HaltonSpecialized1667(index);
			case 262: return HaltonSpecialized1669(index);
			case 263: return HaltonSpecialized1693(index);
			case 264: return HaltonSpecialized1697(index);
			case 265: return HaltonSpecialized1699(index);
			case 266: return HaltonSpecialized1709(index);
			case 267: return HaltonSpecialized1721(index);
			case 268: return HaltonSpecialized1723(index);
			case 269: return HaltonSpecialized1733(index);
			case 270: return HaltonSpecialized1741(index);
			case 271: return HaltonSpecialized1747(index);
			case 272: return HaltonSpecialized1753(index);
			case 273: return HaltonSpecialized1759(index);
			case 274: return HaltonSpecialized1777(index);
			case 275: return HaltonSpecialized1783(index);
			case 276: return HaltonSpecialized1787(index);
			case 277: return HaltonSpecialized1789(index);
			case 278: return HaltonSpecialized1801(index);
			case 279: return HaltonSpecialized1811(index);
			case 280: return HaltonSpecialized1823(index);
			case 281: return HaltonSpecialized1831(index);
			case 282: return HaltonSpecialized1847(index);
			case 283: return HaltonSpecialized1861(index);
			case 284: return HaltonSpecialized1867(index);
			case 285: return HaltonSpecialized1871(index);
			case 286: return HaltonSpecialized1873(index);
			case 287: return HaltonSpecialized1877(index);
			case 288: return HaltonSpecialized1879(index);
			case 289: return HaltonSpecialized1889(index);
			case 290: return HaltonSpecialized1901(index);
			case 291: return HaltonSpecialized1907(index);
			case 292: return HaltonSpecialized1913(index);
			case 293: return HaltonSpecialized1931(index);
			case 294: return HaltonSpecialized1933(index);
			case 295: return HaltonSpecialized1949(index);
			case 296: return HaltonSpecialized1951(index);
			case 297: return HaltonSpecialized1973(index);
			case 298: return HaltonSpecialized1979(index);
			case 299: return HaltonSpecialized1987(index);
			case 300: return HaltonSpecialized1993(index);
			case 301: return HaltonSpecialized1997(index);
			case 302: return HaltonSpecialized1999(index);
			case 303: return HaltonSpecialized2003(index);
			case 304: return HaltonSpecialized2011(index);
			case 305: return HaltonSpecialized2017(index);
			case 306: return HaltonSpecialized2027(index);
			case 307: return HaltonSpecialized2029(index);
			case 308: return HaltonSpecialized2039(index);
			case 309: return HaltonSpecialized2053(index);
			case 310: return HaltonSpecialized2063(index);
			case 311: return HaltonSpecialized2069(index);
			case 312: return HaltonSpecialized2081(index);
			case 313: return HaltonSpecialized2083(index);
			case 314: return HaltonSpecialized2087(index);
			case 315: return HaltonSpecialized2089(index);
			case 316: return HaltonSpecialized2099(index);
			case 317: return HaltonSpecialized2111(index);
			case 318: return HaltonSpecialized2113(index);
			case 319: return HaltonSpecialized2129(index);
			case 320: return HaltonSpecialized2131(index);
			case 321: return HaltonSpecialized2137(index);
			case 322: return HaltonSpecialized2141(index);
			case 323: return HaltonSpecialized2143(index);
			case 324: return HaltonSpecialized2153(index);
			case 325: return HaltonSpecialized2161(index);
			case 326: return HaltonSpecialized2179(index);
			case 327: return HaltonSpecialized2203(index);
			case 328: return HaltonSpecialized2207(index);
			case 329: return HaltonSpecialized2213(index);
			case 330: return HaltonSpecialized2221(index);
			case 331: return HaltonSpecialized2237(index);
			case 332: return HaltonSpecialized2239(index);
			case 333: return HaltonSpecialized2243(index);
			case 334: return HaltonSpecialized2251(index);
			case 335: return HaltonSpecialized2267(index);
			case 336: return HaltonSpecialized2269(index);
			case 337: return HaltonSpecialized2273(index);
			case 338: return HaltonSpecialized2281(index);
			case 339: return HaltonSpecialized2287(index);
			case 340: return HaltonSpecialized2293(index);
			case 341: return HaltonSpecialized2297(index);
			case 342: return HaltonSpecialized2309(index);
			case 343: return HaltonSpecialized2311(index);
			case 344: return HaltonSpecialized2333(index);
			case 345: return HaltonSpecialized2339(index);
			case 346: return HaltonSpecialized2341(index);
			case 347: return HaltonSpecialized2347(index);
			case 348: return HaltonSpecialized2351(index);
			case 349: return HaltonSpecialized2357(index);
			case 350: return HaltonSpecialized2371(index);
			case 351: return HaltonSpecialized2377(index);
			case 352: return HaltonSpecialized2381(index);
			case 353: return HaltonSpecialized2383(index);
			case 354: return HaltonSpecialized2389(index);
			case 355: return HaltonSpecialized2393(index);
			case 356: return HaltonSpecialized2399(index);
			case 357: return HaltonSpecialized2411(index);
			case 358: return HaltonSpecialized2417(index);
			case 359: return HaltonSpecialized2423(index);
			case 360: return HaltonSpecialized2437(index);
			case 361: return HaltonSpecialized2441(index);
			case 362: return HaltonSpecialized2447(index);
			case 363: return HaltonSpecialized2459(index);
			case 364: return HaltonSpecialized2467(index);
			case 365: return HaltonSpecialized2473(index);
			case 366: return HaltonSpecialized2477(index);
			case 367: return HaltonSpecialized2503(index);
			case 368: return HaltonSpecialized2521(index);
			case 369: return HaltonSpecialized2531(index);
			case 370: return HaltonSpecialized2539(index);
			case 371: return HaltonSpecialized2543(index);
			case 372: return HaltonSpecialized2549(index);
			case 373: return HaltonSpecialized2551(index);
			case 374: return HaltonSpecialized2557(index);
			case 375: return HaltonSpecialized2579(index);
			case 376: return HaltonSpecialized2591(index);
			case 377: return HaltonSpecialized2593(index);
			case 378: return HaltonSpecialized2609(index);
			case 379: return HaltonSpecialized2617(index);
			case 380: return HaltonSpecialized2621(index);
			case 381: return HaltonSpecialized2633(index);
			case 382: return HaltonSpecialized2647(index);
			case 383: return HaltonSpecialized2657(index);
			case 384: return HaltonSpecialized2659(index);
			case 385: return HaltonSpecialized2663(index);
			case 386: return HaltonSpecialized2671(index);
			case 387: return HaltonSpecialized2677(index);
			case 388: return HaltonSpecialized2683(index);
			case 389: return HaltonSpecialized2687(index);
			case 390: return HaltonSpecialized2689(index);
			case 391: return HaltonSpecialized2693(index);
			case 392: return HaltonSpecialized2699(index);
			case 393: return HaltonSpecialized2707(index);
			case 394: return HaltonSpecialized2711(index);
			case 395: return HaltonSpecialized2713(index);
			case 396: return HaltonSpecialized2719(index);
			case 397: return HaltonSpecialized2729(index);
			case 398: return HaltonSpecialized2731(index);
			case 399: return HaltonSpecialized2741(index);
			case 400: return HaltonSpecialized2749(index);
			case 401: return HaltonSpecialized2753(index);
			case 402: return HaltonSpecialized2767(index);
			case 403: return HaltonSpecialized2777(index);
			case 404: return HaltonSpecialized2789(index);
			case 405: return HaltonSpecialized2791(index);
			case 406: return HaltonSpecialized2797(index);
			case 407: return HaltonSpecialized2801(index);
			case 408: return HaltonSpecialized2803(index);
			case 409: return HaltonSpecialized2819(index);
			case 410: return HaltonSpecialized2833(index);
			case 411: return HaltonSpecialized2837(index);
			case 412: return HaltonSpecialized2843(index);
			case 413: return HaltonSpecialized2851(index);
			case 414: return HaltonSpecialized2857(index);
			case 415: return HaltonSpecialized2861(index);
			case 416: return HaltonSpecialized2879(index);
			case 417: return HaltonSpecialized2887(index);
			case 418: return HaltonSpecialized2897(index);
			case 419: return HaltonSpecialized2903(index);
			case 420: return HaltonSpecialized2909(index);
			case 421: return HaltonSpecialized2917(index);
			case 422: return HaltonSpecialized2927(index);
			case 423: return HaltonSpecialized2939(index);
			case 424: return HaltonSpecialized2953(index);
			case 425: return HaltonSpecialized2957(index);
			case 426: return HaltonSpecialized2963(index);
			case 427: return HaltonSpecialized2969(index);
			case 428: return HaltonSpecialized2971(index);
			case 429: return HaltonSpecialized2999(index);
			case 430: return HaltonSpecialized3001(index);
			case 431: return HaltonSpecialized3011(index);
			case 432: return HaltonSpecialized3019(index);
			case 433: return HaltonSpecialized3023(index);
			case 434: return HaltonSpecialized3037(index);
			case 435: return HaltonSpecialized3041(index);
			case 436: return HaltonSpecialized3049(index);
			case 437: return HaltonSpecialized3061(index);
			case 438: return HaltonSpecialized3067(index);
			case 439: return HaltonSpecialized3079(index);
			case 440: return HaltonSpecialized3083(index);
			case 441: return HaltonSpecialized3089(index);
			case 442: return HaltonSpecialized3109(index);
			case 443: return HaltonSpecialized3119(index);
			case 444: return HaltonSpecialized3121(index);
			case 445: return HaltonSpecialized3137(index);
			case 446: return HaltonSpecialized3163(index);
			case 447: return HaltonSpecialized3167(index);
			case 448: return HaltonSpecialized3169(index);
			case 449: return HaltonSpecialized3181(index);
			case 450: return HaltonSpecialized3187(index);
			case 451: return HaltonSpecialized3191(index);
			case 452: return HaltonSpecialized3203(index);
			case 453: return HaltonSpecialized3209(index);
			case 454: return HaltonSpecialized3217(index);
			case 455: return HaltonSpecialized3221(index);
			case 456: return HaltonSpecialized3229(index);
			case 457: return HaltonSpecialized3251(index);
			case 458: return HaltonSpecialized3253(index);
			case 459: return HaltonSpecialized3257(index);
			case 460: return HaltonSpecialized3259(index);
			case 461: return HaltonSpecialized3271(index);
			case 462: return HaltonSpecialized3299(index);
			case 463: return HaltonSpecialized3301(index);
			case 464: return HaltonSpecialized3307(index);
			case 465: return HaltonSpecialized3313(index);
			case 466: return HaltonSpecialized3319(index);
			case 467: return HaltonSpecialized3323(index);
			case 468: return HaltonSpecialized3329(index);
			case 469: return HaltonSpecialized3331(index);
			case 470: return HaltonSpecialized3343(index);
			case 471: return HaltonSpecialized3347(index);
			case 472: return HaltonSpecialized3359(index);
			case 473: return HaltonSpecialized3361(index);
			case 474: return HaltonSpecialized3371(index);
			case 475: return HaltonSpecialized3373(index);
			case 476: return HaltonSpecialized3389(index);
			case 477: return HaltonSpecialized3391(index);
			case 478: return HaltonSpecialized3407(index);
			case 479: return HaltonSpecialized3413(index);
			case 480: return HaltonSpecialized3433(index);
			case 481: return HaltonSpecialized3449(index);
			case 482: return HaltonSpecialized3457(index);
			case 483: return HaltonSpecialized3461(index);
			case 484: return HaltonSpecialized3463(index);
			case 485: return HaltonSpecialized3467(index);
			case 486: return HaltonSpecialized3469(index);
			case 487: return HaltonSpecialized3491(index);
			case 488: return HaltonSpecialized3499(index);
			case 489: return HaltonSpecialized3511(index);
			case 490: return HaltonSpecialized3517(index);
			case 491: return HaltonSpecialized3527(index);
			case 492: return HaltonSpecialized3529(index);
			case 493: return HaltonSpecialized3533(index);
			case 494: return HaltonSpecialized3539(index);
			case 495: return HaltonSpecialized3541(index);
			case 496: return HaltonSpecialized3547(index);
			case 497: return HaltonSpecialized3557(index);
			case 498: return HaltonSpecialized3559(index);
			case 499: return HaltonSpecialized3571(index);
			case 500: return HaltonSpecialized3581(index);
			case 501: return HaltonSpecialized3583(index);
			case 502: return HaltonSpecialized3593(index);
			case 503: return HaltonSpecialized3607(index);
			case 504: return HaltonSpecialized3613(index);
			case 505: return HaltonSpecialized3617(index);
			case 506: return HaltonSpecialized3623(index);
			case 507: return HaltonSpecialized3631(index);
			case 508: return HaltonSpecialized3637(index);
			case 509: return HaltonSpecialized3643(index);
			case 510: return HaltonSpecialized3659(index);
			case 511: return HaltonSpecialized3671(index);
			case 512: return HaltonSpecialized3673(index);
			case 513: return HaltonSpecialized3677(index);
			case 514: return HaltonSpecialized3691(index);
			case 515: return HaltonSpecialized3697(index);
			case 516: return HaltonSpecialized3701(index);
			case 517: return HaltonSpecialized3709(index);
			case 518: return HaltonSpecialized3719(index);
			case 519: return HaltonSpecialized3727(index);
			case 520: return HaltonSpecialized3733(index);
			case 521: return HaltonSpecialized3739(index);
			case 522: return HaltonSpecialized3761(index);
			case 523: return HaltonSpecialized3767(index);
			case 524: return HaltonSpecialized3769(index);
			case 525: return HaltonSpecialized3779(index);
			case 526: return HaltonSpecialized3793(index);
			case 527: return HaltonSpecialized3797(index);
			case 528: return HaltonSpecialized3803(index);
			case 529: return HaltonSpecialized3821(index);
			case 530: return HaltonSpecialized3823(index);
			case 531: return HaltonSpecialized3833(index);
			case 532: return HaltonSpecialized3847(index);
			case 533: return HaltonSpecialized3851(index);
			case 534: return HaltonSpecialized3853(index);
			case 535: return HaltonSpecialized3863(index);
			case 536: return HaltonSpecialized3877(index);
			case 537: return HaltonSpecialized3881(index);
			case 538: return HaltonSpecialized3889(index);
			case 539: return HaltonSpecialized3907(index);
			case 540: return HaltonSpecialized3911(index);
			case 541: return HaltonSpecialized3917(index);
			case 542: return HaltonSpecialized3919(index);
			case 543: return HaltonSpecialized3923(index);
			case 544: return HaltonSpecialized3929(index);
			case 545: return HaltonSpecialized3931(index);
			case 546: return HaltonSpecialized3943(index);
			case 547: return HaltonSpecialized3947(index);
			case 548: return HaltonSpecialized3967(index);
			case 549: return HaltonSpecialized3989(index);
			case 550: return HaltonSpecialized4001(index);
			case 551: return HaltonSpecialized4003(index);
			case 552: return HaltonSpecialized4007(index);
			case 553: return HaltonSpecialized4013(index);
			case 554: return HaltonSpecialized4019(index);
			case 555: return HaltonSpecialized4021(index);
			case 556: return HaltonSpecialized4027(index);
			case 557: return HaltonSpecialized4049(index);
			case 558: return HaltonSpecialized4051(index);
			case 559: return HaltonSpecialized4057(index);
			case 560: return HaltonSpecialized4073(index);
			case 561: return HaltonSpecialized4079(index);
			case 562: return HaltonSpecialized4091(index);
			case 563: return HaltonSpecialized4093(index);
			case 564: return HaltonSpecialized4099(index);
			case 565: return HaltonSpecialized4111(index);
			case 566: return HaltonSpecialized4127(index);
			case 567: return HaltonSpecialized4129(index);
			case 568: return HaltonSpecialized4133(index);
			case 569: return HaltonSpecialized4139(index);
			case 570: return HaltonSpecialized4153(index);
			case 571: return HaltonSpecialized4157(index);
			case 572: return HaltonSpecialized4159(index);
			case 573: return HaltonSpecialized4177(index);
			case 574: return HaltonSpecialized4201(index);
			case 575: return HaltonSpecialized4211(index);
			case 576: return HaltonSpecialized4217(index);
			case 577: return HaltonSpecialized4219(index);
			case 578: return HaltonSpecialized4229(index);
			case 579: return HaltonSpecialized4231(index);
			case 580: return HaltonSpecialized4241(index);
			case 581: return HaltonSpecialized4243(index);
			case 582: return HaltonSpecialized4253(index);
			case 583: return HaltonSpecialized4259(index);
			case 584: return HaltonSpecialized4261(index);
			case 585: return HaltonSpecialized4271(index);
			case 586: return HaltonSpecialized4273(index);
			case 587: return HaltonSpecialized4283(index);
			case 588: return HaltonSpecialized4289(index);
			case 589: return HaltonSpecialized4297(index);
			case 590: return HaltonSpecialized4327(index);
			case 591: return HaltonSpecialized4337(index);
			case 592: return HaltonSpecialized4339(index);
			case 593: return HaltonSpecialized4349(index);
			case 594: return HaltonSpecialized4357(index);
			case 595: return HaltonSpecialized4363(index);
			case 596: return HaltonSpecialized4373(index);
			case 597: return HaltonSpecialized4391(index);
			case 598: return HaltonSpecialized4397(index);
			case 599: return HaltonSpecialized4409(index);
			case 600: return HaltonSpecialized4421(index);
			case 601: return HaltonSpecialized4423(index);
			case 602: return HaltonSpecialized4441(index);
			case 603: return HaltonSpecialized4447(index);
			case 604: return HaltonSpecialized4451(index);
			case 605: return HaltonSpecialized4457(index);
			case 606: return HaltonSpecialized4463(index);
			case 607: return HaltonSpecialized4481(index);
			case 608: return HaltonSpecialized4483(index);
			case 609: return HaltonSpecialized4493(index);
			case 610: return HaltonSpecialized4507(index);
			case 611: return HaltonSpecialized4513(index);
			case 612: return HaltonSpecialized4517(index);
			case 613: return HaltonSpecialized4519(index);
			case 614: return HaltonSpecialized4523(index);
			case 615: return HaltonSpecialized4547(index);
			case 616: return HaltonSpecialized4549(index);
			case 617: return HaltonSpecialized4561(index);
			case 618: return HaltonSpecialized4567(index);
			case 619: return HaltonSpecialized4583(index);
			case 620: return HaltonSpecialized4591(index);
			case 621: return HaltonSpecialized4597(index);
			case 622: return HaltonSpecialized4603(index);
			case 623: return HaltonSpecialized4621(index);
			case 624: return HaltonSpecialized4637(index);
			case 625: return HaltonSpecialized4639(index);
			case 626: return HaltonSpecialized4643(index);
			case 627: return HaltonSpecialized4649(index);
			case 628: return HaltonSpecialized4651(index);
			case 629: return HaltonSpecialized4657(index);
			case 630: return HaltonSpecialized4663(index);
			case 631: return HaltonSpecialized4673(index);
			case 632: return HaltonSpecialized4679(index);
			case 633: return HaltonSpecialized4691(index);
			case 634: return HaltonSpecialized4703(index);
			case 635: return HaltonSpecialized4721(index);
			case 636: return HaltonSpecialized4723(index);
			case 637: return HaltonSpecialized4729(index);
			case 638: return HaltonSpecialized4733(index);
			case 639: return HaltonSpecialized4751(index);
			case 640: return HaltonSpecialized4759(index);
			case 641: return HaltonSpecialized4783(index);
			case 642: return HaltonSpecialized4787(index);
			case 643: return HaltonSpecialized4789(index);
			case 644: return HaltonSpecialized4793(index);
			case 645: return HaltonSpecialized4799(index);
			case 646: return HaltonSpecialized4801(index);
			case 647: return HaltonSpecialized4813(index);
			case 648: return HaltonSpecialized4817(index);
			case 649: return HaltonSpecialized4831(index);
			case 650: return HaltonSpecialized4861(index);
			case 651: return HaltonSpecialized4871(index);
			case 652: return HaltonSpecialized4877(index);
			case 653: return HaltonSpecialized4889(index);
			case 654: return HaltonSpecialized4903(index);
			case 655: return HaltonSpecialized4909(index);
			case 656: return HaltonSpecialized4919(index);
			case 657: return HaltonSpecialized4931(index);
			case 658: return HaltonSpecialized4933(index);
			case 659: return HaltonSpecialized4937(index);
			case 660: return HaltonSpecialized4943(index);
			case 661: return HaltonSpecialized4951(index);
			case 662: return HaltonSpecialized4957(index);
			case 663: return HaltonSpecialized4967(index);
			case 664: return HaltonSpecialized4969(index);
			case 665: return HaltonSpecialized4973(index);
			case 666: return HaltonSpecialized4987(index);
			case 667: return HaltonSpecialized4993(index);
			case 668: return HaltonSpecialized4999(index);
			case 669: return HaltonSpecialized5003(index);
			case 670: return HaltonSpecialized5009(index);
			case 671: return HaltonSpecialized5011(index);
			case 672: return HaltonSpecialized5021(index);
			case 673: return HaltonSpecialized5023(index);
			case 674: return HaltonSpecialized5039(index);
			case 675: return HaltonSpecialized5051(index);
			case 676: return HaltonSpecialized5059(index);
			case 677: return HaltonSpecialized5077(index);
			case 678: return HaltonSpecialized5081(index);
			case 679: return HaltonSpecialized5087(index);
			case 680: return HaltonSpecialized5099(index);
			case 681: return HaltonSpecialized5101(index);
			case 682: return HaltonSpecialized5107(index);
			case 683: return HaltonSpecialized5113(index);
			case 684: return HaltonSpecialized5119(index);
			case 685: return HaltonSpecialized5147(index);
			case 686: return HaltonSpecialized5153(index);
			case 687: return HaltonSpecialized5167(index);
			case 688: return HaltonSpecialized5171(index);
			case 689: return HaltonSpecialized5179(index);
			case 690: return HaltonSpecialized5189(index);
			case 691: return HaltonSpecialized5197(index);
			case 692: return HaltonSpecialized5209(index);
			case 693: return HaltonSpecialized5227(index);
			case 694: return HaltonSpecialized5231(index);
			case 695: return HaltonSpecialized5233(index);
			case 696: return HaltonSpecialized5237(index);
			case 697: return HaltonSpecialized5261(index);
			case 698: return HaltonSpecialized5273(index);
			case 699: return HaltonSpecialized5279(index);
			case 700: return HaltonSpecialized5281(index);
			case 701: return HaltonSpecialized5297(index);
			case 702: return HaltonSpecialized5303(index);
			case 703: return HaltonSpecialized5309(index);
			case 704: return HaltonSpecialized5323(index);
			case 705: return HaltonSpecialized5333(index);
			case 706: return HaltonSpecialized5347(index);
			case 707: return HaltonSpecialized5351(index);
			case 708: return HaltonSpecialized5381(index);
			case 709: return HaltonSpecialized5387(index);
			case 710: return HaltonSpecialized5393(index);
			case 711: return HaltonSpecialized5399(index);
			case 712: return HaltonSpecialized5407(index);
			case 713: return HaltonSpecialized5413(index);
			case 714: return HaltonSpecialized5417(index);
			case 715: return HaltonSpecialized5419(index);
			case 716: return HaltonSpecialized5431(index);
			case 717: return HaltonSpecialized5437(index);
			case 718: return HaltonSpecialized5441(index);
			case 719: return HaltonSpecialized5443(index);
			case 720: return HaltonSpecialized5449(index);
			case 721: return HaltonSpecialized5471(index);
			case 722: return HaltonSpecialized5477(index);
			case 723: return HaltonSpecialized5479(index);
			case 724: return HaltonSpecialized5483(index);
			case 725: return HaltonSpecialized5501(index);
			case 726: return HaltonSpecialized5503(index);
			case 727: return HaltonSpecialized5507(index);
			case 728: return HaltonSpecialized5519(index);
			case 729: return HaltonSpecialized5521(index);
			case 730: return HaltonSpecialized5527(index);
			case 731: return HaltonSpecialized5531(index);
			case 732: return HaltonSpecialized5557(index);
			case 733: return HaltonSpecialized5563(index);
			case 734: return HaltonSpecialized5569(index);
			case 735: return HaltonSpecialized5573(index);
			case 736: return HaltonSpecialized5581(index);
			case 737: return HaltonSpecialized5591(index);
			case 738: return HaltonSpecialized5623(index);
			case 739: return HaltonSpecialized5639(index);
			case 740: return HaltonSpecialized5641(index);
			case 741: return HaltonSpecialized5647(index);
			case 742: return HaltonSpecialized5651(index);
			case 743: return HaltonSpecialized5653(index);
			case 744: return HaltonSpecialized5657(index);
			case 745: return HaltonSpecialized5659(index);
			case 746: return HaltonSpecialized5669(index);
			case 747: return HaltonSpecialized5683(index);
			case 748: return HaltonSpecialized5689(index);
			case 749: return HaltonSpecialized5693(index);
			case 750: return HaltonSpecialized5701(index);
			case 751: return HaltonSpecialized5711(index);
			case 752: return HaltonSpecialized5717(index);
			case 753: return HaltonSpecialized5737(index);
			case 754: return HaltonSpecialized5741(index);
			case 755: return HaltonSpecialized5743(index);
			case 756: return HaltonSpecialized5749(index);
			case 757: return HaltonSpecialized5779(index);
			case 758: return HaltonSpecialized5783(index);
			case 759: return HaltonSpecialized5791(index);
			case 760: return HaltonSpecialized5801(index);
			case 761: return HaltonSpecialized5807(index);
			case 762: return HaltonSpecialized5813(index);
			case 763: return HaltonSpecialized5821(index);
			case 764: return HaltonSpecialized5827(index);
			case 765: return HaltonSpecialized5839(index);
			case 766: return HaltonSpecialized5843(index);
			case 767: return HaltonSpecialized5849(index);
			case 768: return HaltonSpecialized5851(index);
			case 769: return HaltonSpecialized5857(index);
			case 770: return HaltonSpecialized5861(index);
			case 771: return HaltonSpecialized5867(index);
			case 772: return HaltonSpecialized5869(index);
			case 773: return HaltonSpecialized5879(index);
			case 774: return HaltonSpecialized5881(index);
			case 775: return HaltonSpecialized5897(index);
			case 776: return HaltonSpecialized5903(index);
			case 777: return HaltonSpecialized5923(index);
			case 778: return HaltonSpecialized5927(index);
			case 779: return HaltonSpecialized5939(index);
			case 780: return HaltonSpecialized5953(index);
			case 781: return HaltonSpecialized5981(index);
			case 782: return HaltonSpecialized5987(index);
			case 783: return HaltonSpecialized6007(index);
			case 784: return HaltonSpecialized6011(index);
			case 785: return HaltonSpecialized6029(index);
			case 786: return HaltonSpecialized6037(index);
			case 787: return HaltonSpecialized6043(index);
			case 788: return HaltonSpecialized6047(index);
			case 789: return HaltonSpecialized6053(index);
			case 790: return HaltonSpecialized6067(index);
			case 791: return HaltonSpecialized6073(index);
			case 792: return HaltonSpecialized6079(index);
			case 793: return HaltonSpecialized6089(index);
			case 794: return HaltonSpecialized6091(index);
			case 795: return HaltonSpecialized6101(index);
			case 796: return HaltonSpecialized6113(index);
			case 797: return HaltonSpecialized6121(index);
			case 798: return HaltonSpecialized6131(index);
			case 799: return HaltonSpecialized6133(index);
			case 800: return HaltonSpecialized6143(index);
			case 801: return HaltonSpecialized6151(index);
			case 802: return HaltonSpecialized6163(index);
			case 803: return HaltonSpecialized6173(index);
			case 804: return HaltonSpecialized6197(index);
			case 805: return HaltonSpecialized6199(index);
			case 806: return HaltonSpecialized6203(index);
			case 807: return HaltonSpecialized6211(index);
			case 808: return HaltonSpecialized6217(index);
			case 809: return HaltonSpecialized6221(index);
			case 810: return HaltonSpecialized6229(index);
			case 811: return HaltonSpecialized6247(index);
			case 812: return HaltonSpecialized6257(index);
			case 813: return HaltonSpecialized6263(index);
			case 814: return HaltonSpecialized6269(index);
			case 815: return HaltonSpecialized6271(index);
			case 816: return HaltonSpecialized6277(index);
			case 817: return HaltonSpecialized6287(index);
			case 818: return HaltonSpecialized6299(index);
			case 819: return HaltonSpecialized6301(index);
			case 820: return HaltonSpecialized6311(index);
			case 821: return HaltonSpecialized6317(index);
			case 822: return HaltonSpecialized6323(index);
			case 823: return HaltonSpecialized6329(index);
			case 824: return HaltonSpecialized6337(index);
			case 825: return HaltonSpecialized6343(index);
			case 826: return HaltonSpecialized6353(index);
			case 827: return HaltonSpecialized6359(index);
			case 828: return HaltonSpecialized6361(index);
			case 829: return HaltonSpecialized6367(index);
			case 830: return HaltonSpecialized6373(index);
			case 831: return HaltonSpecialized6379(index);
			case 832: return HaltonSpecialized6389(index);
			case 833: return HaltonSpecialized6397(index);
			case 834: return HaltonSpecialized6421(index);
			case 835: return HaltonSpecialized6427(index);
			case 836: return HaltonSpecialized6449(index);
			case 837: return HaltonSpecialized6451(index);
			case 838: return HaltonSpecialized6469(index);
			case 839: return HaltonSpecialized6473(index);
			case 840: return HaltonSpecialized6481(index);
			case 841: return HaltonSpecialized6491(index);
			case 842: return HaltonSpecialized6521(index);
			case 843: return HaltonSpecialized6529(index);
			case 844: return HaltonSpecialized6547(index);
			case 845: return HaltonSpecialized6551(index);
			case 846: return HaltonSpecialized6553(index);
			case 847: return HaltonSpecialized6563(index);
			case 848: return HaltonSpecialized6569(index);
			case 849: return HaltonSpecialized6571(index);
			case 850: return HaltonSpecialized6577(index);
			case 851: return HaltonSpecialized6581(index);
			case 852: return HaltonSpecialized6599(index);
			case 853: return HaltonSpecialized6607(index);
			case 854: return HaltonSpecialized6619(index);
			case 855: return HaltonSpecialized6637(index);
			case 856: return HaltonSpecialized6653(index);
			case 857: return HaltonSpecialized6659(index);
			case 858: return HaltonSpecialized6661(index);
			case 859: return HaltonSpecialized6673(index);
			case 860: return HaltonSpecialized6679(index);
			case 861: return HaltonSpecialized6689(index);
			case 862: return HaltonSpecialized6691(index);
			case 863: return HaltonSpecialized6701(index);
			case 864: return HaltonSpecialized6703(index);
			case 865: return HaltonSpecialized6709(index);
			case 866: return HaltonSpecialized6719(index);
			case 867: return HaltonSpecialized6733(index);
			case 868: return HaltonSpecialized6737(index);
			case 869: return HaltonSpecialized6761(index);
			case 870: return HaltonSpecialized6763(index);
			case 871: return HaltonSpecialized6779(index);
			case 872: return HaltonSpecialized6781(index);
			case 873: return HaltonSpecialized6791(index);
			case 874: return HaltonSpecialized6793(index);
			case 875: return HaltonSpecialized6803(index);
			case 876: return HaltonSpecialized6823(index);
			case 877: return HaltonSpecialized6827(index);
			case 878: return HaltonSpecialized6829(index);
			case 879: return HaltonSpecialized6833(index);
			case 880: return HaltonSpecialized6841(index);
			case 881: return HaltonSpecialized6857(index);
			case 882: return HaltonSpecialized6863(index);
			case 883: return HaltonSpecialized6869(index);
			case 884: return HaltonSpecialized6871(index);
			case 885: return HaltonSpecialized6883(index);
			case 886: return HaltonSpecialized6899(index);
			case 887: return HaltonSpecialized6907(index);
			case 888: return HaltonSpecialized6911(index);
			case 889: return HaltonSpecialized6917(index);
			case 890: return HaltonSpecialized6947(index);
			case 891: return HaltonSpecialized6949(index);
			case 892: return HaltonSpecialized6959(index);
			case 893: return HaltonSpecialized6961(index);
			case 894: return HaltonSpecialized6967(index);
			case 895: return HaltonSpecialized6971(index);
			case 896: return HaltonSpecialized6977(index);
			case 897: return HaltonSpecialized6983(index);
			case 898: return HaltonSpecialized6991(index);
			case 899: return HaltonSpecialized6997(index);
			case 900: return HaltonSpecialized7001(index);
			case 901: return HaltonSpecialized7013(index);
			case 902: return HaltonSpecialized7019(index);
			case 903: return HaltonSpecialized7027(index);
			case 904: return HaltonSpecialized7039(index);
			case 905: return HaltonSpecialized7043(index);
			case 906: return HaltonSpecialized7057(index);
			case 907: return HaltonSpecialized7069(index);
			case 908: return HaltonSpecialized7079(index);
			case 909: return HaltonSpecialized7103(index);
			case 910: return HaltonSpecialized7109(index);
			case 911: return HaltonSpecialized7121(index);
			case 912: return HaltonSpecialized7127(index);
			case 913: return HaltonSpecialized7129(index);
			case 914: return HaltonSpecialized7151(index);
			case 915: return HaltonSpecialized7159(index);
			case 916: return HaltonSpecialized7177(index);
			case 917: return HaltonSpecialized7187(index);
			case 918: return HaltonSpecialized7193(index);
			case 919: return HaltonSpecialized7207(index);
			case 920: return HaltonSpecialized7211(index);
			case 921: return HaltonSpecialized7213(index);
			case 922: return HaltonSpecialized7219(index);
			case 923: return HaltonSpecialized7229(index);
			case 924: return HaltonSpecialized7237(index);
			case 925: return HaltonSpecialized7243(index);
			case 926: return HaltonSpecialized7247(index);
			case 927: return HaltonSpecialized7253(index);
			case 928: return HaltonSpecialized7283(index);
			case 929: return HaltonSpecialized7297(index);
			case 930: return HaltonSpecialized7307(index);
			case 931: return HaltonSpecialized7309(index);
			case 932: return HaltonSpecialized7321(index);
			case 933: return HaltonSpecialized7331(index);
			case 934: return HaltonSpecialized7333(index);
			case 935: return HaltonSpecialized7349(index);
			case 936: return HaltonSpecialized7351(index);
			case 937: return HaltonSpecialized7369(index);
			case 938: return HaltonSpecialized7393(index);
			case 939: return HaltonSpecialized7411(index);
			case 940: return HaltonSpecialized7417(index);
			case 941: return HaltonSpecialized7433(index);
			case 942: return HaltonSpecialized7451(index);
			case 943: return HaltonSpecialized7457(index);
			case 944: return HaltonSpecialized7459(index);
			case 945: return HaltonSpecialized7477(index);
			case 946: return HaltonSpecialized7481(index);
			case 947: return HaltonSpecialized7487(index);
			case 948: return HaltonSpecialized7489(index);
			case 949: return HaltonSpecialized7499(index);
			case 950: return HaltonSpecialized7507(index);
			case 951: return HaltonSpecialized7517(index);
			case 952: return HaltonSpecialized7523(index);
			case 953: return HaltonSpecialized7529(index);
			case 954: return HaltonSpecialized7537(index);
			case 955: return HaltonSpecialized7541(index);
			case 956: return HaltonSpecialized7547(index);
			case 957: return HaltonSpecialized7549(index);
			case 958: return HaltonSpecialized7559(index);
			case 959: return HaltonSpecialized7561(index);
			case 960: return HaltonSpecialized7573(index);
			case 961: return HaltonSpecialized7577(index);
			case 962: return HaltonSpecialized7583(index);
			case 963: return HaltonSpecialized7589(index);
			case 964: return HaltonSpecialized7591(index);
			case 965: return HaltonSpecialized7603(index);
			case 966: return HaltonSpecialized7607(index);
			case 967: return HaltonSpecialized7621(index);
			case 968: return HaltonSpecialized7639(index);
			case 969: return HaltonSpecialized7643(index);
			case 970: return HaltonSpecialized7649(index);
			case 971: return HaltonSpecialized7669(index);
			case 972: return HaltonSpecialized7673(index);
			case 973: return HaltonSpecialized7681(index);
			case 974: return HaltonSpecialized7687(index);
			case 975: return HaltonSpecialized7691(index);
			case 976: return HaltonSpecialized7699(index);
			case 977: return HaltonSpecialized7703(index);
			case 978: return HaltonSpecialized7717(index);
			case 979: return HaltonSpecialized7723(index);
			case 980: return HaltonSpecialized7727(index);
			case 981: return HaltonSpecialized7741(index);
			case 982: return HaltonSpecialized7753(index);
			case 983: return HaltonSpecialized7757(index);
			case 984: return HaltonSpecialized7759(index);
			case 985: return HaltonSpecialized7789(index);
			case 986: return HaltonSpecialized7793(index);
			case 987: return HaltonSpecialized7817(index);
			case 988: return HaltonSpecialized7823(index);
			case 989: return HaltonSpecialized7829(index);
			case 990: return HaltonSpecialized7841(index);
			case 991: return HaltonSpecialized7853(index);
			case 992: return HaltonSpecialized7867(index);
			case 993: return HaltonSpecialized7873(index);
			case 994: return HaltonSpecialized7877(index);
			case 995: return HaltonSpecialized7879(index);
			case 996: return HaltonSpecialized7883(index);
			case 997: return HaltonSpecialized7901(index);
			case 998: return HaltonSpecialized7907(index);
			case 999: return HaltonSpecialized7919(index);
#endif
#pragma endregion remainderCases_sample
			}
			rtPrintf("[Error]Dimension >= 1000 in HaltonSampler.\n");
			return 0.f;
		}
	}
}

/**
 * @brief Get index(of the i_th sample) in global Halton sequence for current pixel
 * @param sampleNum i_th sample in current pixel
 * @return return the computed index
 */
static __device__ __inline__ uint32_t Halton_GetIndexForSample(uint32_t sampleNum)
{
	uint32_t &offsetForCurrentPixel = globalHaltonSampler.hs_offsetForCurrentPixelBuffer[sysLaunch_index];

	/*subsequent samples are computed using addition*/
	return offsetForCurrentPixel + sampleNum * 31104;/*sampleStride 31104=128*243*/
}

/**
 * @brief compute one element of sample vector of halton sequence
 * @param index global index in halton sequence
 * @param dim the expected dimension of computed sample vector
 * @return phi(b_dimension)(index), where random permutation is employed
 * when dimension goes beyond 1(start from 0)
 */
static __device__ __inline__ float Halton_SampleDimension(uint32_t index, int dim)
{
	/*see mitsuba halton.cpp L375 for another equivalent implementation*/
	if (dim == 0)
		return Sampler::ScrambledRadicalInverse_Fast(dim, index >> 7);//baseExponents.x
	else if (dim == 1)
		return Sampler::ScrambledRadicalInverse_Fast(dim, index / 243);//baseScales.y
	else
		return Sampler::ScrambledRadicalInverse_Fast(dim, index);
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
static __device__ __inline__ float Halton_Get1D(HaltonSampler &localSampler)
{
	return Halton_SampleDimension(localSampler.intervalSampleIndex, localSampler.dimension++);
}

/**
 * @brief fetch the next two dimension value of the given sample vector and advance dimension
 * @param localSampler localSampler binding to current pixel
 * @param globalSampler globalSampler
 * @return 2D sample value
 */
static __device__ __inline__ optix::float2 Halton_Get2D(HaltonSampler &localSampler)
{
	optix::float2 p = make_float2(Halton_SampleDimension(localSampler.intervalSampleIndex,
		                                                 localSampler.dimension),
		                          Halton_SampleDimension(localSampler.intervalSampleIndex,
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
static __device__ __inline__ void StartSamplingPreprocess_RayGen(HaltonSampler &localSampler)
{
	/*next dimension of the sampled vector goes from 0*/
	localSampler.dimension = 0;

	/*retrieve the global index of Halton sequences for current pixel*/
	uint32_t &currentPixelSampleIndex = sysIterationIndex;
	localSampler.intervalSampleIndex = Halton_GetIndexForSample(currentPixelSampleIndex);
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
static __device__ __inline__ void StartSamplingPreprocess_CHit(HaltonSampler &localSampler)
{
	/*next dimension of the sampled vector goes from 2,
	 *after CameraSample being consumed.*/
	localSampler.dimension = 2;

	/*retrieve the global index of Halton sequences for current pixel*/
	uint32_t &currentPixelSampleIndex = sysIterationIndex;
	localSampler.intervalSampleIndex = Halton_GetIndexForSample(currentPixelSampleIndex);
}

#endif // COLVILLEA_DEVICE_SAMPLER_HALTONSAMPLER_H_