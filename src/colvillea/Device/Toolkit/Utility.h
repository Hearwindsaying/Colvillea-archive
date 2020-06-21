#pragma once
#ifndef COLVILLEA_DEVICE_TOOLKIT_UTILITY_H_
#define COLVILLEA_DEVICE_TOOLKIT_UTILITY_H_
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix.h>
#ifndef __cplusplus
#include <optix_device.h>
#endif

namespace TwUtil
{
    /* From GlobalDefs.h */
    template<typename E>
    static __device__ __inline__ constexpr auto toUnderlyingValue(E enumerator) noexcept
    {
        return static_cast<std::underlying_type_t<E>>(enumerator);
    }


	/*Auxiliary mathematical methods declarations:*/

	/**
	 * @brief return the sign of the given value
	 * @param value input value
	 * @return return 1 if value>=0, -1 otherwise.
	 */
	static __host__ __device__ __inline__ float signum(float value);


	//////////////////////////////////////////////////////////////////////////
	//Forward decls:
	static __host__ __device__ __inline__ float dot(const optix::float3& a, const optix::float4& b);

	namespace MonteCarlo
	{
		//////////////////////////////////////////////////////////////////////////
		//Forward declarations:
		static __device__ __inline__ optix::float3 CosineSampleHemisphere(optix::float2 urand);
		static __device__ __inline__ void ConcentricSampleDisk(const optix::float2 & urand, float * dx, float * dy);

		static __host__ __device__ __inline__ float PowerHeuristic(int nf, float fPdf, int ng, float gPdf);
        static __host__ __device__ __inline__ float UniformConePdf(float cosThetaMax);

		//MCSampling
		static __device__ __inline__ optix::float3 CosineSampleHemisphere(optix::float2 urand)
		{
			optix::float3 ret = optix::make_float3(0.f);
			ConcentricSampleDisk(urand, &ret.x, &ret.y);
			ret.z = sqrtf(fmaxf(0.f, 1.f - ret.x*ret.x - ret.y*ret.y));
			return ret;
		}

		__device__ __inline__ void TwUtil::MonteCarlo::ConcentricSampleDisk(const optix::float2 & urand, float * dx, float * dy)
		{
			float r = 0.f, theta = 0.f;
			// Map uniform random numbers to $[-1,1]^2$
			const float & u1 = urand.x;
			const float & u2 = urand.y;
			float sx = 2 * u1 - 1;
			float sy = 2 * u2 - 1;

			// Map square to $(r,\theta)$

			// Handle degeneracy at the origin
			if (sx == 0.0 && sy == 0.0) 
			{
				*dx = 0.0;
				*dy = 0.0;
				return;
			}
			if (sx >= -sy) 
			{
				if (sx > sy) 
				{
					// Handle first region of disk
					r = sx;
					if (sy > 0.0) theta = sy / r;
					else          theta = 8.0f + sy / r;
				}
				else 
				{
					// Handle second region of disk
					r = sy;
					theta = 2.0f - sx / r;
				}
			}
			else 
			{
				if (sx <= sy) 
				{
					// Handle third region of disk
					r = -sx;
					theta = 4.0f - sy / r;
				}
				else 
				{
					// Handle fourth region of disk
					r = -sy;
					theta = 6.0f + sx / r;
				}
			}
			theta *= M_PIf / 4.f;
			*dx = r * cosf(theta);
			*dy = r * sinf(theta);
		}
	
	
		//Multiple importance sampling:
		static __host__ __device__ __inline__ float PowerHeuristic(int nf, float fPdf, int ng, float gPdf)
		{
			float f = nf * fPdf, g = ng * gPdf;
			return (f*f) / (f*f + g * g);
		}

        /**
         * @brief Pdf (w.r.t Solid Angle measure) for uniform sampling cone.
         */
        static __host__ __device__ __inline__ float UniformConePdf(float cosThetaMax) 
        {
            return 1 / (2 * M_PIf * (1 - cosThetaMax));
        }
	}

	namespace BSDFMath
	{
		//////////////////////////////////////////////////////////////////////////
		//Forward declarations:
		//BSDFMath:
		static __device__ __inline__ float AbsCosTheta(const optix::float3 & w_local);
		static __device__ __inline__ float CosTheta(const optix::float3 & w_local);
		static __device__ __inline__ float Cos2Theta(const optix::float3 & w_local);
		static __device__ __inline__ float Sin2Theta(const optix::float3 & w_local);
		static __device__ __inline__ float SinTheta(const optix::float3 &w_local);
		static __device__ __inline__ float Tan2Theta(const optix::float3 &w_local);
		static __device__ __inline__ float CosPhi(const optix::float3 &w_local);
		static __device__ __inline__ float SinPhi(const optix::float3 &w_local);
		static __device__ __inline__ float TanTheta(const optix::float3 &w_local);
		static __device__ __inline__ float Cos2Phi(const optix::float3 &w_local);
		static __device__ __inline__ float Sin2Phi(const optix::float3 &w_local);

		//BSDF Space conversion:
		static __device__ __inline__ optix::float3 WorldToLocal(const optix::float3 & v, const optix::float3 & sn, const optix::float3 & tn, const optix::float3 & nn);
		static __device__ __inline__ optix::float3 WorldToLocal(const optix::float3 & v, const optix::float3 & sn, const optix::float3 & tn, const optix::float4 & nn);
		static __device__ __inline__ optix::float3 LocalToWorld(const optix::float3 & v, const optix::float3 & sn, const optix::float3 & tn, const optix::float3 & nn);
		static __device__ __inline__ optix::float3 LocalToWorld(const optix::float3 & v, const optix::float3 & sn, const optix::float3 & tn, const optix::float4 & nn);

		//////////////////////////////////////////////////////////////////////////
		//BSDFMath:
		static __device__ __inline__ float AbsCosTheta(const optix::float3 & w_local)
		{
			return fabsf(w_local.z);
		}

		static __device__ __inline__ float CosTheta(const optix::float3 & w_local)
		{
			return w_local.z;
		}

		static __device__ __inline__ float Cos2Theta(const optix::float3 & w_local)
		{
			return w_local.z * w_local.z;
		}

		static __device__ __inline__ float Sin2Theta(const optix::float3 & w_local)
		{
			return fmaxf(0.f, 1 - Cos2Theta(w_local));
		}

		static __device__ __inline__ float SinTheta(const optix::float3 &w_local)
		{
			return sqrtf(Sin2Theta(w_local));
		}

		static __device__ __inline__ float Tan2Theta(const optix::float3 &w_local)
		{
			return Sin2Theta(w_local) / Cos2Theta(w_local);
		}

		static __device__ __inline__ float CosPhi(const optix::float3 &w_local)
		{
			float sinTheta = SinTheta(w_local);
			return (sinTheta == 0) ? 1 : clamp(w_local.x / sinTheta, -1.f, 1.f);
		}

		static __device__ __inline__ float SinPhi(const optix::float3 &w_local)
		{
			float sinTheta = SinTheta(w_local);
			return (sinTheta == 0) ? 0 : clamp(w_local.y / sinTheta, -1.f, 1.f);
		}

		static __device__ __inline__ float TanTheta(const optix::float3 &w_local)
		{
			return SinTheta(w_local) / CosTheta(w_local);
		}

		static __device__ __inline__ float Cos2Phi(const optix::float3 &w_local)
		{
			return CosPhi(w_local) * CosPhi(w_local);
		}

		static __device__ __inline__ float Sin2Phi(const optix::float3 &w_local)
		{
			return SinPhi(w_local) * SinPhi(w_local);
		}


		//BSDF space conversion:
		static __device__ __inline__ optix::float3 WorldToLocal(const optix::float3 & v, const optix::float3 & sn, const optix::float3 & tn, const optix::float3 & nn)
		{
			return optix::make_float3(dot(v, sn), dot(v, tn), dot(v, nn));
		}
		static __device__ __inline__ optix::float3 WorldToLocal(const optix::float3 & v, const optix::float3 & sn, const optix::float3 & tn, const optix::float4 & nn)
		{
			return optix::make_float3(dot(v, sn), dot(v, tn), dot(v, nn));
		}
		static __device__ __inline__ optix::float3 LocalToWorld(const optix::float3 & v, const optix::float3 & sn, const optix::float3 & tn, const optix::float3 & nn)
		{
			return optix::make_float3(sn.x * v.x + tn.x * v.y + nn.x * v.z,
									  sn.y * v.x + tn.y * v.y + nn.y * v.z,
									  sn.z * v.x + tn.z * v.y + nn.z * v.z);
		}
		static __device__ __inline__ optix::float3 LocalToWorld(const optix::float3 & v, const optix::float3 & sn, const optix::float3 & tn, const optix::float4 & nn)
		{
			return optix::make_float3(sn.x * v.x + tn.x * v.y + nn.x * v.z,
									  sn.y * v.x + tn.y * v.y + nn.y * v.z,
									  sn.z * v.x + tn.z * v.y + nn.z * v.z);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//Forward declarations:
	static __device__ __inline__ void GenerateRay(const optix::float2 & samplePos, optix::float3 & out_rayOrigin, optix::float3 & out_rayDirection, const optix::Matrix4x4 & RasterToCamera, const optix::Matrix4x4 & CameraToWorld, const float lensRadius, const float focalDistance, const optix::float2 *lensSample);
	static __host__ __inline__ optix::Matrix4x4 GetPerspectiveMatrix(float fov, float n, float f);

	static __host__ __device__ __inline__ optix::float3 xfmPoint(const optix::float3 & pt, const optix::Matrix4x4 & matrix);
	static __host__ __device__ __inline__ optix::float3 xfmVector(const optix::float3 & v, const optix::Matrix4x4 & matrix);
	static __host__ __device__ __inline__ optix::float3 xfmNormal(const optix::float3 & n, const optix::Matrix4x4 & matrixInv);
    static __device__ __inline__ optix::Ray xfmRay(const optix::Ray &ray, const optix::Matrix4x4 &matrix);

	static __device__ __inline__ float sphericalTheta(const optix::float3& v);
	static __device__ __inline__ float sphericalPhi(const optix::float3& v);
	static __device__ __inline__ bool SameHemisphere(const optix::float3 & w, const optix::float3 & wp);
	static __host__ __device__ __inline__ void CoordinateSystem(const optix::float3 & v1, optix::float3 & outv2, optix::float3 & outv3);
	template<typename T>
	static __host__ __device__ __inline__ T deg2rad(const T& x);

	static __host__ __device__ __inline__ optix::float3 safe_normalize(const optix::float3& v);
	//static __host__ __device__ __inline__ optix::float4 safe_normalize(const optix::float4& v);
	static __host__ __device__ __inline__ float dot(const optix::float3& a, const optix::float4& b);
	static __host__ __device__ __inline__ float sqr_length(const optix::float3 & a);
	static __host__ __device__ __inline__ float distance(const optix::float3 & a, const optix::float3 & b);
	static __host__ __device__ __inline__ bool isBlack(const optix::float3 &col);
	static __host__ __device__ __inline__ bool isBlack(const optix::float4 &col);
	static __host__ __device__ __inline__ float luminance(const optix::float3 &col);

	static __host__ __device__ __inline__ bool hasScale(const optix::Matrix4x4 & m);

	static __host__ __device__ __inline__ optix::float3 reflect(const optix::float3 &wo, const optix::float3 & n);
	static __host__ __device__ __inline__ bool refract(const optix::float3 &wi, const optix::float3 &n, float eta, optix::float3 & outWt);
	static __device__ __inline__ optix::float3 faceforward(const optix::float3 &vDest, const optix::float3 &vRef);

	template <typename Predicate>
	static __host__ __device__ __inline__ int FindInterval(int size, const Predicate &pred);

	static __host__ __device__ __inline__ void swap(float &lhs, float &rhs);


	//////////////////////////////////////////////////////////////////////////
	//Camera related
	static __device__ __inline__ void GenerateRay(const optix::float2 & samplePos, optix::float3 & out_rayOrigin, optix::float3 & out_rayDirection, const optix::Matrix4x4 & RasterToCamera, const optix::Matrix4x4 & CameraToWorld, const float lensRadius, const float focalDistance, const optix::float2 *lensSample)
	{
		//Pras in RasterSpace
		optix::float3 Pras = optix::make_float3(samplePos.x, samplePos.y, 0);
		//Apply transform RasterToCamera to Pras
		optix::float3 Pcamera = xfmPoint(Pras, RasterToCamera);
		//Construct Ray in CameraSpace
		optix::float3 rayDir = safe_normalize(Pcamera);
		
        // Account for Depth of Field effect
        if (lensRadius > 0.f)
        {
            // Assert
#ifndef __cplusplus
            if (!lensSample)
                rtPrintf("[Error] lensSample is null but lensRadius>0.!\n");
#endif 
            // Sample point on lens
            optix::float2 lensUV = optix::make_float2(0.f);
            TwUtil::MonteCarlo::ConcentricSampleDisk(*lensSample, &lensUV.x, &lensUV.y);
            lensUV *= lensRadius;

            // Compute point on plane of focus
            float ft = focalDistance / rayDir.z;
            optix::float3 pFocus = ft * rayDir;

            // Update ray for effect of lens
            out_rayOrigin = optix::make_float3(lensUV.x, lensUV.y, 0.f);
            rayDir = TwUtil::safe_normalize(pFocus - out_rayOrigin);
        }
        else
        {
            out_rayOrigin = optix::make_float3(0.f);
        }

        // Transform Ray to world space.
		out_rayOrigin = xfmPoint(out_rayOrigin, CameraToWorld);
		out_rayDirection = xfmVector(rayDir, CameraToWorld);
	}

	//Creates an perspective projective transform
	static __host__ __inline__ optix::Matrix4x4 GetPerspectiveMatrix(float fov, float n, float f)
	{
		// Perform projective divide
		float perspMatrixdata[] = { 1, 0,       0,          0,
									0, 1,       0,			0,
									0, 0, f/(f-n), -f*n/(f-n),
									0, 0,       1,          0 };
		optix::Matrix4x4 persp = optix::Matrix4x4(perspMatrixdata);

		// Scale to canonical viewing volume
		float invTanAng = 1.f / tanf(deg2rad(fov) / 2.f);
		return optix::Matrix4x4::scale(optix::make_float3(invTanAng, invTanAng, 1)) * optix::Matrix4x4(persp);
	}

	//////////////////////////////////////////////////////////////////////////
	//Matrix & Transform related

	static __host__ __device__ __inline__ optix::float3 xfmPoint(const optix::float3 & pt, const optix::Matrix4x4 & matrix)
	{
		float x = pt.x, y = pt.y, z = pt.z;
		//[x y z 1]T
		const float *m = matrix.getData();
		float xp = m[0] * x + m[1] * y + m[2] * z + m[3];
		float yp = m[4] * x + m[5] * y + m[6] * z + m[7];
		float zp = m[8] * x + m[9] * y + m[10] * z + m[11];
		float wp = m[12] * x + m[13] * y + m[14] * z + m[15];
		if (wp != 1.f)
			return optix::make_float3(xp, yp, zp);
		else
			return optix::make_float3(xp, yp, zp) / wp;
	}

	static __host__ __device__ __inline__ optix::float3 xfmVector(const optix::float3 & v, const optix::Matrix4x4 & matrix)
	{
		float x = v.x, y = v.y, z = v.z;
		//[x y z 0]T
		const float *m = matrix.getData();
		return optix::make_float3(m[0] * x + m[1] * y + m[2] * z,
								  m[4] * x + m[5] * y + m[6] * z,
								  m[8] * x + m[9] * y + m[10] * z);
	}

	static __host__ __device__ __inline__ optix::float3 xfmNormal(const optix::float3 & n, const optix::Matrix4x4 & matrixInv)
	{//matrix passed into the function should be inverse matrix
		float x = n.x, y = n.y, z = n.z;
		const float *m = matrixInv.getData();
		return optix::make_float3(m[0] * x + m[4] * y + m[8] * z,
								  m[1] * x + m[5] * y + m[9] * z,
								  m[2] * x + m[6] * y + m[10] * z);
	}

    /**
     * @brief Transform a ray using affine matrix. Except from
     * |ray.origin| and |ray.direction| being transformed, other
     * properties are preserved.
     *
     * @param[in] ray    source ray
     * @param[in] matrix affine matrix, could be worldToObject or objectToWorld
     *
     * @return [nodiscard] transformed ray.
     */
    //[[nodiscard]]
    static __device__ __inline__ optix::Ray xfmRay(const optix::Ray &ray, const optix::Matrix4x4 &matrix)
    {
        optix::Ray dRay;
        dRay.tmin = ray.tmin;
        dRay.tmax = ray.tmax;
        dRay.ray_type = ray.ray_type;
        dRay.origin = xfmPoint(ray.origin, matrix);
        dRay.direction = xfmVector(ray.direction, matrix);

        return dRay;
    }

	//Coordinates transformation
	static __device__ __inline__ float sphericalTheta(const optix::float3& v)
	{
		return acosf(clamp(v.z, -1.f, 1.f));
	}
	static __device__ __inline__ float sphericalPhi(const optix::float3& v)
	{
		float p = atan2f(v.y, v.x);
		return (p < 0.f) ? p + 2.f*M_PIf : p;
	}

	__device__ __inline__ bool SameHemisphere(const optix::float3 & w, const optix::float3 & wp)
	{
		return w.z * wp.z > 0.f;
	}

	static __host__ __device__ __inline__ void CoordinateSystem(const optix::float3 & v1, optix::float3 & outv2, optix::float3 & outv3)
	{
		if (fabsf(v1.x) > fabsf(v1.y))
		{
			float invLen = 1.0f / sqrtf(v1.x * v1.x + v1.z * v1.z);
			outv2 = optix::make_float3(-v1.z * invLen, 0.0f, v1.x * invLen);
		}
		else
		{
			float invLen = 1.0f / sqrtf(v1.y * v1.y + v1.z * v1.z);
			outv2 = optix::make_float3(0.0f, v1.z * invLen, -v1.y * invLen);
		}
		outv3 = optix::cross(v1, outv2);
	}


	template<typename T> 
	static __host__ __device__ __inline__ T deg2rad(const T& x) 
	{ 
		return x * static_cast<T>(1.74532925199432957692e-2f); 
	}

	//////////////////////////////////////////////////////////////////////////
	//Vector geometry
	static __host__ __device__ __inline__ optix::float3 safe_normalize(const optix::float3& v)
	{
		float d = optix::dot(v, v);
		if (d == 0.0f)
		{
			return v;
		}
		else
		{
			float invLen = 1.0f / sqrtf(d);
			return v * invLen;
		}
	}

	/*static __host__ __device__ __inline__ optix::float4 safe_normalize(const optix::float4& v)
	{
		float d = optix::dot(v, v);
		if (d == 0.0f)
		{
			return v;
		}
		else
		{
			float invLen = 1.0f / sqrtf(d);
			return v * invLen;
		}
	}*/

	static __host__ __device__ __inline__ float dot(const optix::float3& a, const optix::float4& b)
	{
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	static __host__ __device__ __inline__ float sqr_length(const optix::float3 & a)
	{
		return optix::dot(a, a);
	}

	static __host__ __device__ __inline__ float distance(const optix::float3 & a, const optix::float3 & b)
	{
		return length(a - b);
	}

	static __host__ __device__ __inline__ bool isBlack(const optix::float3 &col)
	{
		return (col.x == 0 && col.y == 0 && col.z == 0);
	}

	//note that this is a fake isBlack function, ignore col.w == 0 judgement
	static __host__ __device__ __inline__ bool isBlack(const optix::float4 &col)
	{
		return (col.x == 0 && col.y == 0 && col.z == 0);
	}

	__host__ __device__ __inline__ float luminance(const optix::float3 & col)
	{
		const optix::float3 cie_luminance = { 0.212671f, 0.715160f, 0.072169f };
		return  dot(col, cie_luminance);
	}

	//////////////////////////////////////////////////////////////////////////
	//Matrix related:
    static __host__ __device__ __inline__ bool hasXScale(const optix::Matrix4x4 & m)
    {
        float la2 = sqr_length(xfmVector(optix::make_float3(1, 0, 0), m));
        return ((la2) < .999f || (la2) > 1.001f);
    }

    static __host__ __device__ __inline__ bool hasYScale(const optix::Matrix4x4 & m)
    {
        float lb2 = sqr_length(xfmVector(optix::make_float3(0, 1, 0), m));
        return ((lb2) < .999f || (lb2) > 1.001f);
    }

    static __host__ __device__ __inline__ bool hasZScale(const optix::Matrix4x4 & m)
    {
        float lc2 = sqr_length(xfmVector(optix::make_float3(0, 0, 1), m));
        return ((lc2) < .999f || (lc2) > 1.001f);
    }

	static __host__ __device__ __inline__ bool hasScale(const optix::Matrix4x4 & m)
	{
        return hasXScale(m) || hasYScale(m) || hasZScale(m);
	}

    static __host__ __device__ __inline__ float getXScale(const optix::Matrix4x4 &m)
    {
        return length(xfmVector(optix::make_float3(1, 0, 0), m));
    }

    static __host__ __device__ __inline__ float getYScale(const optix::Matrix4x4 &m)
    {
        return length(xfmVector(optix::make_float3(0, 1, 0), m));
    }

    static __host__ __device__ __inline__ float getZScale(const optix::Matrix4x4 &m)
    {
        return length(xfmVector(optix::make_float3(0, 0, 1), m));
    }

	
	static __host__ __device__ __inline__ optix::float3 reflect(const optix::float3 & wo, const optix::float3 & n)
	{
		return -wo + 2 * optix::dot(wo, n) * n;
	}

	//deprecated
	static __host__ __device__ __inline__ bool refract(const optix::float3 &wi, const optix::float3 &n, float eta, optix::float3 & outWt) 
	{
		//rtPrintf("deprecated!\n");
		// Compute $\cos \theta_\roman{t}$ using Snell's law
		float cosThetaI = dot(n, wi);
		float sin2ThetaI = fmaxf(0.f, 1 - cosThetaI * cosThetaI);
		float sin2ThetaT = eta * eta * sin2ThetaI;

		// Handle total internal reflection for transmission
		if (sin2ThetaT >= 1) 
			return false;
		float cosThetaT = sqrtf(1 - sin2ThetaT);
		outWt = eta * -wi + (eta * cosThetaI - cosThetaT) * n;

		return true;
	}

	__device__ __inline__ optix::float3 faceforward(const optix::float3 & vDest, const optix::float3 & vRef)
	{
		return (optix::dot(vDest, vRef) < 0.f) ? -vDest : vDest;
	}

	//////////////////////////////////////////////////////////////////////////
	//Algorithm:
	template <typename Predicate> 
	static __host__ __device__ __inline__ int FindInterval(int size, const Predicate &pred) 
	{
		int first = 0, len = size;
		while (len > 0) 
		{
			int half = len >> 1, middle = first + half;
			if (pred(middle))
			{
				first = middle + 1;
				len -= half + 1;
			}
			else
				len = half;
		}
		return optix::clamp(first - 1, 0, size - 2);
	}
	__host__ __device__ __inline__ void swap(float & lhs, float & rhs)
	{
		float tmp = lhs; lhs = rhs; rhs = tmp;
	}

	/**
	 * @brief return the sign of the given value
	 * @param value input value
	 * @return return 1 if value>=0, -1 otherwise.
	 */
	static __host__ __device__ __inline__ float signum(float value)
	{
		return value >= 0.f ? 1.f : -1.f;
	}
}

#endif // COLVILLEA_DEVICE_TOOLKIT_UTILITY_H_