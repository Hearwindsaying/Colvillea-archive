#pragma once
#ifndef COLVILLEA_DEVICE_SAMPLER_HALTONSAMPLER_COMMON_H_
#define COLVILLEA_DEVICE_SAMPLER_HALTONSAMPLER_COMMON_H_

/*Common part of HaltonSampler for host and device*/

#include "../Toolkit/CommonStructs.h"
#include "../Toolkit/Utility.h"

#include <optix_world.h>
#include <optixu_math_namespace.h>



/*Some constants defined in macro to sync Host and Device codes
 *todo:enclose in namespace to prevent name collision*/

/*maximum resolution for sufficent halton sequences precision,
	   --a tile of samples would spread across the whole image film possibly.*/
#define kMaxResolution 128u
#define DoubleOneMinusEpsilon 0.99999999999999989

namespace TwUtil
{
	namespace Sampler
	{
		/**
		 * @brief compute a mod b, where a and b are both integer.
		 * @param a integer a
		 * @param b integer b
		 * @return a mod b and ensure the result is positive
		 */
		template <typename T>
		static __device__ __host__ __inline__ T mod(T a, T b)
		{
			T result = a - (a / b) * b;
			return (T)((result < 0) ? result + b : result);
		}

		/**
		 * @brief Evaluate the GCD(greatest common divisor) and also compute
		 *        the Bezout's identity x and y such that ax+by=gcd(a,b).
		 *        Note that x or y could possibly be negative.
		 * @param a integer a
		 * @param b integer b
		 * @param x(result) Bezout's identity x
		 * @param y(result) Bezout's identity y
		 */
		static __device__ __host__ __inline__ void extendedGCD(uint64_t a, uint64_t b, int64_t *x, int64_t *y)
		{
			if (b == 0)
			{
				*x = 1;
				*y = 0;
				return;
			}
			int64_t d = a / b, xp, yp;
			extendedGCD(b, a % b, &xp, &yp);
			*x = yp;
			*y = xp - (d * yp);
		}

		/**
		 * @brief Compute the modular multiplicative inverse of the integer a with respect to the modular n.
		 * @param a integer a
		 * @param n modular n, note that a and n are coprime
		 * @return the modular multiplicative inverse belongs to [1,n-1]
		 */
		static __host__ __device__ __inline__ uint64_t multiplicativeInverse(int64_t a, int64_t n)
		{
			int64_t x, y;
			extendedGCD(a, n, &x, &y);
			return mod(x, n);
		}

		/**
		 * @brief convert the given integer radicalInverse*base^dj(usually indicates pixel)
		 * with nDigits digits to the original index
		 * @param inverse input integer number
		 * @param nDigits number of digits(adding some zeros to the result number to make
		 * up nDigits digits
		 * @return the inverse version of radicalInverse
		 */
		template <int base>
		static __host__ __inline__ uint64_t InverseRadicalInverse(uint64_t inverse, int nDigits)
		{
			uint64_t index = 0;
			for (int i = 0; i < nDigits; ++i)
			{
				uint64_t digit = inverse % base;
				inverse /= base;
				index = index * base + digit;
			}
			return index;
		}

#ifndef __CUDACC__
		/**
		 * @brief Compute ScrambledRadicalInverse from Leonhard Gruenschloss 2012, used for init_random to implement fast radical inverse only
		 * @param base prime base number
		 * @param digits number of digits
		 * @param index input parameter for radicalInverse
		 * @param perm permutation table
		 * @return ScrambledRadicalInverse of index
		 */
		static __host__ __inline__ unsigned short RadicalInverse_invert(const unsigned short base, const unsigned short digits,
			unsigned short index, const std::vector<unsigned short>& perm)
		{
			unsigned short result = 0;
			for (unsigned short i = 0; i < digits; ++i)
			{
				result = result * base + perm[index % base];
				index /= base;
			}
			return result;
		}
#endif
	}
}

#endif // COLVILLEA_DEVICE_SAMPLER_HALTONSAMPLER_COMMON_H_