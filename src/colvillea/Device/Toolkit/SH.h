#pragma once
#ifndef COLVILLEA_DEVICE_TOOLKIT_SH_H_
#define COLVILLEA_DEVICE_TOOLKIT_SH_H_

#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix.h>

#include "../Toolkit/Utility.h"

#ifndef TW_RT_DECLARE_AREALIGHTCOEFF
#define TW_RT_DECLARE_AREALIGHTCOEFF
/* Flm Diffuse Matrix. */
rtBuffer<float> areaLightFlmVector;

/* Basis Directions. */
rtBuffer<optix::float3, 1> areaLightBasisVector;

rtBuffer<float, 2> areaLightAlphaCoeff;
#endif

namespace Cl
{
    static __device__ __host__ __inline__ void swap(optix::float3 & lhs, optix::float3 & rhs)
    {
        optix::float3 tmp = lhs; lhs = rhs; rhs = tmp;
    }

    /************************************************************************/
    /*                              Legendre Polynomial                     */
    /************************************************************************/
    template<int N>
    static __device__ __host__ float LegendreP(float x);

    template<>
    static __device__ __host__ float LegendreP<0>(float x)
    {
        return 1.0f;
    }
    template<>
    static __device__ __host__ float LegendreP<1>(float x)
    {
        return x;
    }
    template<>
    static __device__ __host__ float LegendreP<2>(float x)
    {
        return 0.5f*(3.f*x*x - 1);
    }
    template<>
    static __device__ __host__ float LegendreP<3>(float x)
    {
        return 0.5f*(5.f*x*x*x - 3.f*x);
    }
    template<>
    static __device__ __host__ float LegendreP<4>(float x)
    {
        return 0.125f*(35.f*x*x*x*x - 30.f*x*x + 3);
    }
    template<>
    static __device__ __host__ float LegendreP<5>(float x)
    {
        return 0.125f*(63.f*x*x*x*x*x - 70.f*x*x*x + 15.f*x);
    }
    template<>
    static __device__ __host__ float LegendreP<6>(float x)
    {
        return (231.f*x*x*x*x*x*x - 315.f*x*x*x*x + 105.f*x*x - 5.f) / 16.f;
    }
    template<>
    static __device__ __host__ float LegendreP<7>(float x)
    {
        return (429.f*x*x*x*x*x*x*x - 693.f*x*x*x*x*x + 315.f*x*x*x - 35.f*x) / 16.f;
    }
    template<>
    static __device__ __host__ float LegendreP<8>(float x)
    {
        return (6435.f*pow(x, 8) - 12012.f*pow(x, 6) + 6930.f*x*x*x*x - 1260.f*x*x + 35.f) / 128.f;
    }
    template<>
    static __device__ __host__ float LegendreP<9>(float x)
    {
        return (12155.f*pow(x, 9) - 25740.f*pow(x, 7) + 18018.f*pow(x, 5) - 4620.f*x*x*x + 315.f*x) / 128.f;
    }
    template<>
    static __device__ __host__ float LegendreP<10>(float x)
    {
        return (46189.f*pow(x, 10) - 109395.f*pow(x, 8) + 90090.f*pow(x, 6) - 30030.f*x*x*x*x + 3465.f*x*x - 63.f) / 256.f;
    }

    template<int M>
    static __device__ __inline__ bool CheckOrientation(const optix::float3 P[]);

    template<>
    static __device__ __inline__ bool CheckOrientation<3>(const optix::float3 P[])
    {
        const auto D = (P[1] + P[2] + P[3]) / 3.0f;
        const auto N = optix::cross(P[2] - P[1], P[3] - P[1]);
        return optix::dot(D, N) <= 0.0f;
    }

    /* Clipping Algorithm. */
    static __device__ __inline__ void ClipQuadToHorizon(optix::float3 L[5], int &n)
    {
        /* Make a copy of L[]. */
        optix::float3 Lorg[4];

        //memcpy(&Lorg[0], &L[0], sizeof(optix::float3) * 4);
        for (int i = 0; i <= 3; ++i)
            Lorg[i] = L[i];

        auto IntersectRayZ0 = [](const optix::float3 &A, const optix::float3 &B)->optix::float3
        {
            optix::float3 o = A;
            optix::float3 d = TwUtil::safe_normalize(B - A);
            float t = -A.z * (optix::length(B - A) / (B - A).z);
            if (!(t >= 0.f))rtPrintf("error in IntersectRayZ0.\n");
            return o + t * d;
        };

        n = 0;
        for (int i = 1; i <= 4; ++i)
        {
            const optix::float3& A = Lorg[i - 1];
            const optix::float3& B = i == 4 ? Lorg[0] : Lorg[i]; // Loop back to zero index
            if (A.z <= 0 && B.z <= 0)
                continue;
            else if (A.z >= 0 && B.z >= 0)
            {
                L[n++] = A;
            }
            else if (A.z >= 0 && B.z <= 0)
            {
                L[n++] = A;
                L[n++] = IntersectRayZ0(A, B);
            }
            else if (A.z <= 0 && B.z >= 0)
            {
                L[n++] = IntersectRayZ0(A, B);
            }
            else
            {
                rtPrintf("ClipQuadToHorizon A B.z.\n");
            }
        }
        if (!(n == 0 || n == 3 || n == 4 || n == 5))
            rtPrintf("ClipQuadToHorizon n.\n");
    }

    /**
     * @brief GPU version compute Solid Angle.
     * @param we spherical projection of polygon, index starting from 1
     */
    template<int M>
    static __device__ __inline__ float computeSolidAngle(const optix::float3 we[])
    {
        float S0 = 0.0f;
        for (int e = 1; e <= M; ++e)
        {
            const optix::float3& we_minus_1 = (e == 1 ? we[M] : we[e - 1]);
            const optix::float3& we_plus_1 = (e == M ? we[1] : we[e + 1]);

            float3 tmpa = optix::cross(we[e], we_minus_1);
            float3 tmpb = optix::cross(we[e], we_plus_1);
            S0 += acosf(optix::dot(tmpa, tmpb) / (optix::length(tmpa)*optix::length(tmpb))); // Typo in Wang's paper, length is inside acos evaluation!
        }
        S0 -= (M - 2)*M_PIf;
        return S0;
    }


    static __device__ __inline__ optix::float3 BSDFWorldToLocal(const optix::float3 & v, const optix::float3 & sn, const optix::float3 & tn, const optix::float4 & nn, const optix::float3 &worldPoint)
    {
        optix::float3 pt;
        pt.x = optix::dot(v, sn) - optix::dot(worldPoint, sn);
        pt.y = optix::dot(v, tn) - optix::dot(worldPoint, tn);
        pt.z = TwUtil::dot(v, nn) - TwUtil::dot(worldPoint, nn);
        return pt;
    }

    /************************************************************************/
    /*   Analytic Spherical Harmonic Coefficients for Polygonal Area Light  */
    /************************************************************************/

    // Unrolling loops by recursive template:
    template<int l, int M>
    static __device__ __inline__ void computeLw_unroll(float Lw[][19], float ae[], float gammae[], float be[], float ce[], float D1e[], float B0e[], float D2e[], float B1e[], float D0e[], int i, float Bl_1, float S0, float S1)
    {
        float Bl = 0;
        float Cl_1e[M + 1];
        float B2_e[M + 1];

        for (int e = 1; e <= M; ++e)
        {
            Cl_1e[e] = 1.f / l * ((ae[e] * sinf(gammae[e]) - be[e] * cosf(gammae[e]))*
                LegendreP<l - 1>(ae[e] * cosf(gammae[e]) + be[e] * sinf(gammae[e])) +
                be[e] * LegendreP<l - 1>(ae[e]) + (ae[e] * ae[e] + be[e] * be[e] - 1.f)*D1e[e] +
                (l - 1.f)*B0e[e]);
            B2_e[e] = ((2.f*l - 1.f) / l)*(Cl_1e[e]) - (l - 1.f) / l * B0e[e];
            Bl = Bl + ce[e] * B2_e[e];
            D2e[e] = (2.f*l - 1.f) * B1e[e] + D0e[e];

            D0e[e] = D1e[e];
            D1e[e] = D2e[e];
            B0e[e] = B1e[e];
            B1e[e] = B2_e[e];
        }

        // Optimal storage for S (recurrence relation so that only three terms are kept).
        // S2 is not represented as S2 really (Sl).
        if (l % 2 == 0)
        {
            float S2 = ((2.f*l - 1) / (l*(l + 1))*Bl_1) + ((l - 2.f)*(l - 1.f) / ((l)*(l + 1.f)))*S0;
            S0 = S2;
            Lw[l][i] = sqrtf((2.f * l + 1) / (4.f*M_PIf))*S2;
        }
        else
        {
            float S2 = ((2.f*l - 1) / (l*(l + 1))*Bl_1) + ((l - 2.f)*(l - 1.f) / ((l)*(l + 1.f)))*S1;
            S1 = S2;
            Lw[l][i] = sqrtf((2.f * l + 1) / (4.f*M_PIf))*S2;
        }

        Bl_1 = Bl;

        computeLw_unroll<l + 1, M>(Lw, ae, gammae, be, ce, D1e, B0e, D2e, B1e, D0e, i, Bl_1, S0, S1);
    }

    // Partial specialization is not supported in function templates:
    template<>
    static __device__ __inline__ void computeLw_unroll<9, 5>(float Lw[][19], float ae[], float gammae[], float be[], float ce[], float D1e[], float B0e[], float D2e[], float B1e[], float D0e[], int i, float Bl_1, float S0, float S1)
    {
        float Bl = 0;
        float Cl_1e[5 + 1];
        float B2_e[5 + 1];

        for (int e = 1; e <= 5; ++e)
        {
            Cl_1e[e] = 1.f / 9 * ((ae[e] * sinf(gammae[e]) - be[e] * cosf(gammae[e]))*
                LegendreP<9 - 1>(ae[e] * cosf(gammae[e]) + be[e] * sinf(gammae[e])) +
                be[e] * LegendreP<9 - 1>(ae[e]) + (ae[e] * ae[e] + be[e] * be[e] - 1.f)*D1e[e] +
                (9 - 1.f)*B0e[e]);
            B2_e[e] = ((2.f * 9 - 1.f) / 9)*(Cl_1e[e]) - (9 - 1.f) / 9 * B0e[e];
            Bl = Bl + ce[e] * B2_e[e];
            D2e[e] = (2.f * 9 - 1.f) * B1e[e] + D0e[e];

            D0e[e] = D1e[e];
            D1e[e] = D2e[e];
            B0e[e] = B1e[e];
            B1e[e] = B2_e[e];
        }

        // Optimal storage for S (recurrence relation so that only three terms are kept).
        // S2 is not represented as S2 really (Sl).
        float S2 = ((2.f * 9 - 1) / (9 * (9 + 1))*Bl_1) + ((9 - 2.f)*(9 - 1.f) / ((9)*(9 + 1.f)))*S1;
        S1 = S2;
        Lw[9][i] = sqrtf((2.f * 9 + 1) / (4.f*M_PIf))*S2;

        Bl_1 = Bl;
    }

    template<>
    static __device__ __inline__ void computeLw_unroll<9, 4>(float Lw[][19], float ae[], float gammae[], float be[], float ce[], float D1e[], float B0e[], float D2e[], float B1e[], float D0e[], int i, float Bl_1, float S0, float S1)
    {
        float Bl = 0;
        float Cl_1e[4 + 1];
        float B2_e[4 + 1];

        for (int e = 1; e <= 4; ++e)
        {
            Cl_1e[e] = 1.f / 9 * ((ae[e] * sinf(gammae[e]) - be[e] * cosf(gammae[e]))*
                LegendreP<9 - 1>(ae[e] * cosf(gammae[e]) + be[e] * sinf(gammae[e])) +
                be[e] * LegendreP<9 - 1>(ae[e]) + (ae[e] * ae[e] + be[e] * be[e] - 1.f)*D1e[e] +
                (9 - 1.f)*B0e[e]);
            B2_e[e] = ((2.f * 9 - 1.f) / 9)*(Cl_1e[e]) - (9 - 1.f) / 9 * B0e[e];
            Bl = Bl + ce[e] * B2_e[e];
            D2e[e] = (2.f * 9 - 1.f) * B1e[e] + D0e[e];

            D0e[e] = D1e[e];
            D1e[e] = D2e[e];
            B0e[e] = B1e[e];
            B1e[e] = B2_e[e];
        }

        // Optimal storage for S (recurrence relation so that only three terms are kept).
        // S2 is not represented as S2 really (Sl).
        float S2 = ((2.f * 9 - 1) / (9 * (9 + 1))*Bl_1) + ((9 - 2.f)*(9 - 1.f) / ((9)*(9 + 1.f)))*S1;
        S1 = S2;
        Lw[9][i] = sqrtf((2.f * 9 + 1) / (4.f*M_PIf))*S2;

        Bl_1 = Bl;
    }

    template<>
    static __device__ __inline__ void computeLw_unroll<9, 3>(float Lw[][19], float ae[], float gammae[], float be[], float ce[], float D1e[], float B0e[], float D2e[], float B1e[], float D0e[], int i, float Bl_1, float S0, float S1)
    {
        float Bl = 0;
        float Cl_1e[3 + 1];
        float B2_e[3 + 1];

        for (int e = 1; e <= 3; ++e)
        {
            Cl_1e[e] = 1.f / 9 * ((ae[e] * sinf(gammae[e]) - be[e] * cosf(gammae[e]))*
                LegendreP<9 - 1>(ae[e] * cosf(gammae[e]) + be[e] * sinf(gammae[e])) +
                be[e] * LegendreP<9 - 1>(ae[e]) + (ae[e] * ae[e] + be[e] * be[e] - 1.f)*D1e[e] +
                (9 - 1.f)*B0e[e]);
            B2_e[e] = ((2.f * 9 - 1.f) / 9)*(Cl_1e[e]) - (9 - 1.f) / 9 * B0e[e];
            Bl = Bl + ce[e] * B2_e[e];
            D2e[e] = (2.f * 9 - 1.f) * B1e[e] + D0e[e];

            D0e[e] = D1e[e];
            D1e[e] = D2e[e];
            B0e[e] = B1e[e];
            B1e[e] = B2_e[e];
        }

        // Optimal storage for S (recurrence relation so that only three terms are kept).
        // S2 is not represented as S2 really (Sl).
        float S2 = ((2.f * 9 - 1) / (9 * (9 + 1))*Bl_1) + ((9 - 2.f)*(9 - 1.f) / ((9)*(9 + 1.f)))*S1;
        S1 = S2;
        Lw[9][i] = sqrtf((2.f * 9 + 1) / (4.f*M_PIf))*S2;

        Bl_1 = Bl;
    }


    template<int l>
    static __device__ __inline__ void computeYlm_unroll(float *ylmCoeff, float Lw[9 + 1][2 * 9 + 1])
    {
        for (int i = 0; i < 2 * l + 1; ++i)
        {
            float coeff = 0.0f;
            for (int k = 0; k < 2 * l + 1; ++k)
            {
                /* Differ from CPU version! access buffer like a coordinates (need a transpose) */
                coeff += areaLightAlphaCoeff[make_uint2(k, l * l + i)] * Lw[l][k];
            }
            ylmCoeff[l * l + i] = coeff;
        }
        computeYlm_unroll<l + 1>(ylmCoeff, Lw);
    }

    template<>
    static __device__ __inline__ void computeYlm_unroll<9>(float *ylmCoeff, float Lw[9 + 1][2 * 9 + 1])
    {
        //TW_ASSERT(2 * j + 1 == 2*lmax+1); // redundant storage
        for (int i = 0; i < 2 * 9 + 1; ++i)
        {
            float coeff = 0.0f;
            for (int k = 0; k < 2 * 9 + 1; ++k)
            {
                /* Differ from CPU version! access buffer like a coordinates (need a transpose) */
                coeff += areaLightAlphaCoeff[make_uint2(k, 9 * 9 + i)] * Lw[9][k];
            }
            ylmCoeff[9 * 9 + i] = coeff;
        }
    }

    /**
     * @brief GPU Version computeCoeff
     * @param[in] we Area Light vertices in local shading space.
     *  Note that this parameter will be modified and should
     *  not be used after calling this function. The vertices
     *  indices starts from 1, to M.
     *  They could be not normalized.
     * @note Since it's assumed that |we| are in local shading space.
     *  The original |x| parameter is set by default to (0,0,0)
     */
    template<int M, int lmax>
    static __device__ __inline__ void computeCoeff(optix::float3 we[], float ylmCoeff[(lmax + 1)*(lmax + 1)])
    {
#ifdef __CUDACC__
#undef TW_ASSERT
#define TW_ASSERT(expr) TW_ASSERT_INFO(expr, ##expr)
#define TW_ASSERT_INFO(expr, str)    if (!(expr)) {rtPrintf(str); rtPrintf("Above at Line%d:\n",__LINE__);}
#endif
        //TW_ASSERT(v.size() == M + 1);
        //TW_ASSERT(n == 2);
        // for all edges:

        for (int e = 1; e <= M; ++e)
        {
            we[e] = TwUtil::safe_normalize(we[e]);
        }

        float3 lambdae[M + 1];
        float3 ue[M + 1];
        float gammae[M + 1];
        for (int e = 1; e <= M; ++e)
        {
            // Incorrect modular arthmetic: we[(e + 1) % (M+1)] or we[(e + 1) % (M)]
            const optix::float3& we_plus_1 = (e == M ? we[1] : we[e + 1]);
            lambdae[e] = optix::cross(TwUtil::safe_normalize(cross(we[e], we_plus_1)), we[e]);
            ue[e] = optix::cross(we[e], lambdae[e]);
            gammae[e] = acosf(optix::dot(we[e], we_plus_1));
        }
        // Solid angle computation
        float solidAngle = computeSolidAngle<M>(we);

        float Lw[lmax + 1][2 * lmax + 1];

        for (int i = 0; i < 2 * lmax + 1; ++i)
        {
            float ae[M + 1];
            float be[M + 1];
            float ce[M + 1];
            float B0e[M + 1];
            float B1e[M + 1];
            float D0e[M + 1];
            float D1e[M + 1];
            float D2e[M + 1];


            const float3 &wi = areaLightBasisVector[i];
            float S0 = solidAngle;
            float S1 = 0;
            for (int e = 1; e <= M; ++e)
            {
                ae[e] = optix::dot(wi, we[e]); be[e] = optix::dot(wi, lambdae[e]); ce[e] = optix::dot(wi, ue[e]);
                S1 += 0.5f*ce[e] * gammae[e];

                B0e[e] = gammae[e];
                B1e[e] = ae[e] * sinf(gammae[e]) - be[e] * cosf(gammae[e]) + be[e];
                D0e[e] = 0; D1e[e] = gammae[e]; D2e[e] = 3 * B1e[e];
            }

            // my code for B1
            float Bl_1 = 0.f;
            for (int e = 1; e <= M; ++e)
            {
                Bl_1 += ce[e] * B1e[e];
            }

            // Initial Bands l=0, l=1:
            Lw[0][i] = sqrtf(1.f / (4.f*M_PIf))*S0;
            Lw[1][i] = sqrtf(3.f / (4.f*M_PIf))*S1;

            computeLw_unroll<2, M>(Lw, ae, gammae, be, ce, D1e, B0e, D2e, B1e, D0e, i, Bl_1, S0, S1);
        }


        //TW_ASSERT(9 == a.size());
        //for (int j = 0; j <= lmax; ++j)
        //{
        //    //TW_ASSERT(2 * j + 1 == 2*lmax+1); // redundant storage
        //    for (int i = 0; i < 2 * j + 1; ++i)
        //    {
        //        float coeff = 0.0f;
        //        for (int k = 0; k < 2 * j + 1; ++k)
        //        {
        //            /* Differ from CPU version! access buffer like a coordinates (need a transpose) */
        //            coeff += areaLightAlphaCoeff[make_uint2(k, j*j + i)] * Lw[j][k];
        //        }
        //        ylmCoeff[j*j + i] = coeff;
        //    }
        //}
        computeYlm_unroll<0>(ylmCoeff, Lw);
    }
}

#endif