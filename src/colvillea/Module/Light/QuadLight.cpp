// Include Eigen before STL and after OptiX (for macro support like __host__)
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix.h>

#include "colvillea/Device/SH/Test/SH.hpp"

// STL include
#include <random>
#include <utility>
#include <cstdio>
#include <fstream>
#include <limits>
#include <algorithm>

#include "colvillea/Module/Light/QuadLight.h"

#include "colvillea/Application/Application.h" // For LightPool::createLightPool()
#include "colvillea/Module/Light/LightPool.h"

using namespace optix;

typedef std::numeric_limits<float> dbl;

void QuadLight::setPosition(const optix::float3 &position)
{
    /* Update underlying Quad shape. */
    this->m_quadShape->setPosition(position);

    /* Update QuadLight Struct for GPU program. */
    this->updateMatrixParameter();

    //update all quadlights
    this->m_lightPool->updateAllQuadLights(true);
}

void QuadLight::setRotation(const optix::float3 &rotation)
{
    /* Update underlying Quad shape. */
    this->m_quadShape->setRotation(rotation);

    /* Update QuadLight Struct for GPU program. */
    this->updateMatrixParameter();

    //update all quadlights
    this->m_lightPool->updateAllQuadLights(true);
}

void QuadLight::setScale(const optix::float3 &scale)
{
    /* Update underlying Quad shape. */
    this->m_quadShape->setScale(scale);

    /* Update QuadLight Struct for GPU program. */
    this->updateMatrixParameter();

    //update all quadlights
    this->m_lightPool->updateAllQuadLights(true);
}

void QuadLight::setLightIntensity(float intensity)
{
    this->m_intensity = intensity;
    optix::float3 csIntensity = this->m_intensity * this->m_color;
    this->m_csQuadLight.intensity = optix::make_float4(csIntensity.x, csIntensity.y, csIntensity.z, 1.0f);

    this->m_lightPool->updateAllQuadLights(false);
}

void QuadLight::setLightColor(const optix::float3 &color)
{
    this->m_color = color;
    optix::float3 csIntensity = this->m_intensity * this->m_color;
    this->m_csQuadLight.intensity = optix::make_float4(csIntensity.x, csIntensity.y, csIntensity.z, 1.0f);

    this->m_lightPool->updateAllQuadLights(false);
}

namespace CLTest
{
    namespace CommonStruct
    {
        /**
         * A internal struct used for QuadLight testing (SH Integration).
         * Similar to optix::float3 but support index accessor.
         */
        struct QuadLight_float3 {
            float x, y, z;

            float& operator[](int idx) {
                if (idx == 0)return x;
                if (idx == 1)return y;
                if (idx == 2)return z;
                TW_ASSERT(false);
            }
            const float& operator[](int idx) const
            {
                if (idx == 0)return x;
                if (idx == 1)return y;
                if (idx == 2)return z;
                TW_ASSERT(false);
            }
        };
        QuadLight_float3 make_QuadLight_float3(float x, float y, float z)
        {
            QuadLight_float3 fl;
            fl.x = x; fl.y = y; fl.z = z;
            return fl;
        }
    }
}

void QuadLight::ClipQuadToHorizon(optix::float3 L[5], int &n)
{
    /* Make a copy of L[]. */
    optix::float3 Lorg[4];
    memcpy(&Lorg[0], &L[0], sizeof(optix::float3) * 4);

    auto IntersectRayZ0 = [](const optix::float3 &A, const optix::float3 &B)->optix::float3
    {
        float3 o = A;
        float3 d = TwUtil::safe_normalize(B - A);
        float t = -A.z * (length(B - A) / (B - A).z);
        TW_ASSERT(t >= 0.f);
        return o + t * d;
    };

    n = 0;
    for (int i = 1; i <= 4; ++i)
    {
        const float3& A = Lorg[i - 1];
        const float3& B = i == 4 ? Lorg[0] : Lorg[i]; // Loop back to zero index
        if(A.z<=0 && B.z<=0)
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
            TW_ASSERT(false);
        }
    }
    TW_ASSERT(n == 0 || n == 3 || n == 4 || n == 5);
}

template<int N>
float LegendreP(float x);

template<>
float LegendreP<0>(float x)
{
    return 1.0f;
}
template<>
float LegendreP<1>(float x)
{
    return x;
}
template<>
float LegendreP<2>(float x)
{
    return 0.5f*(3.f*x*x - 1);
}
template<>
float LegendreP<3>(float x)
{
    return 0.5f*(5.f*x*x*x - 3.f*x);
}
template<>
float LegendreP<4>(float x)
{
    return 0.125f*(35.f*x*x*x*x-30.f*x*x+3);
}
template<>
float LegendreP<5>(float x)
{
    return 0.125f*(63.f*powf(x,5)-70.f*x*x*x+15.f*x);
}
template<>
float LegendreP<6>(float x)
{
    return (231.f*powf(x, 6) - 315.f*x*x*x*x + 105.f*x*x-5.f) / 16.f;
}
template<>
float LegendreP<7>(float x)
{
    return (429.f*powf(x, 7) - 693.f*powf(x,5) + 315.f*powf(x,3)-35.f*x) / 16.f;
}
template<>
float LegendreP<8>(float x)
{
    return (6435.f*powf(x, 8) - 12012.f*powf(x, 6) + 6930.f*powf(x, 4) - 1260.f*powf(x, 2) + 35.f) / 128.f;
}
template<>
float LegendreP<9>(float x)
{
    return (12155.f*powf(x, 9) - 25740.f*powf(x, 7) + 18018.f*powf(x, 5) - 4620.f*powf(x, 3) + 315.f*x) / 128.f;
}
template<>
float LegendreP<10>(float x)
{
    return (46189.f*powf(x, 10) - 109395.f*powf(x, 8) + 90090.f*powf(x, 6) - 30030.f*powf(x, 4) + 3465.f*powf(x,2)-63.f) / 256.f;
}

float LegendreP(int l, float x)
{
    TW_ASSERT(l <= 10 && l >= 0);
    switch (l)
    {
    case 0:return LegendreP<0>(x);
    case 1:return LegendreP<1>(x);
    case 2:return LegendreP<2>(x);
    case 3:return LegendreP<3>(x);
    case 4:return LegendreP<4>(x);
    case 5:return LegendreP<5>(x);
    case 6:return LegendreP<6>(x);
    case 7:return LegendreP<7>(x);
    case 8:return LegendreP<8>(x);
    case 9:return LegendreP<9>(x);
    case 10:return LegendreP<10>(x);
    default:return 0.0f;
    }
}

/**
 * @ref code adapted from Wang's glsl.
 */
float solid_angle(optix::float3 verts[5], int numVerts) {
    float sa = 0;
    float3 tmp1 = cross(verts[0], verts[numVerts - 1]);
    float3 tmp2 = cross(verts[0], verts[1]);
    sa += acos(dot(tmp1, tmp2) / (length(tmp1) * length(tmp2)));

    // Polygon will be at least a triangle
    // i = 1
    tmp1 = cross(verts[1], verts[0]);
    tmp2 = cross(verts[1], verts[2]);
    sa += acos(dot(tmp1, tmp2) / (length(tmp1) * length(tmp2)));

    // i = 2
    tmp1 = cross(verts[2], verts[1]);
    tmp2 = cross(verts[2], verts[3 % numVerts]);
    sa += acos(dot(tmp1, tmp2) / (length(tmp1) * length(tmp2)));

    if (numVerts >= 4) {
        tmp1 = cross(verts[3], verts[2]);
        tmp2 = cross(verts[3], verts[4 % numVerts]);
        sa += acos(dot(tmp1, tmp2) / (length(tmp1) * length(tmp2)));
    }
    if (numVerts >= 5) {
        tmp1 = cross(verts[4], verts[3]);
        tmp2 = cross(verts[4], verts[0]);   // for now let max vertices be 5
        sa += acos(dot(tmp1, tmp2) / (length(tmp1) * length(tmp2)));
    }

    sa -= (numVerts - 2) * M_PIf;
    return sa;
}

/**
 * @param v: vertices of quad/polygonal light projected into unit hemisphere. index starting from 1
 */
template<int M>
float computeSolidAngle(std::vector<float3> const& v)
{
    TW_ASSERT(v.size() == M + 1);
    std::vector<float3> const& we = v;
    float S0 = 0.0f;
    for (int e = 1; e <= M; ++e)
    {
        const optix::float3& we_minus_1 = (e == 1 ? we[M] : we[e - 1]);
        const optix::float3& we_plus_1 = (e == M ? we[1] : we[e + 1]);

        float3 tmpa = cross(we[e], we_minus_1);
        float3 tmpb = cross(we[e], we_plus_1);
        S0 += acos(dot(tmpa, tmpb) / (length(tmpa)*length(tmpb))); // Typo in Wang's paper, length is inside acos evaluation!
    }
    S0 -= (M - 2)*M_PIf;
    return S0;
}

/**
 * @brief GPU version compute Solid Angle.
 * @param we spherical projection of polygon, index starting from 1
 */
template<int M>
float computeSolidAngle(const float3 we[])
{
    float S0 = 0.0f;
    for (int e = 1; e <= M; ++e)
    {
        const optix::float3& we_minus_1 = (e == 1 ? we[M] : we[e - 1]);
        const optix::float3& we_plus_1 = (e == M ? we[1] : we[e + 1]);

        float3 tmpa = cross(we[e], we_minus_1);
        float3 tmpb = cross(we[e], we_plus_1);
        S0 += acosf(dot(tmpa, tmpb) / (length(tmpa)*length(tmpb))); // Typo in Wang's paper, length is inside acos evaluation!
    }
    S0 -= (M - 2)*M_PIf;
    return S0;
}

template<int l>
void computeYlm_unroll(std::vector<float> & ylmCoeff, std::vector<std::vector<float>> const& Lw, std::vector<std::vector<float>> const& areaLightAlphaCoeff)
{
    for (int i = 0; i < 2 * l + 1; ++i)
    {
        float coeff = 0.0f;
        for (int k = 0; k < 2 * l + 1; ++k)
        {
            /* Differ from CPU version! access buffer like a coordinates (need a transpose) */
            coeff += areaLightAlphaCoeff[l * l + i][k] * Lw[l][k];
        }
        ylmCoeff[l * l + i] = coeff;
    }
    computeYlm_unroll<l + 1>(ylmCoeff, Lw, areaLightAlphaCoeff);
}
template<>
void computeYlm_unroll<9>(std::vector<float> & ylmCoeff, std::vector<std::vector<float>> const& Lw, std::vector<std::vector<float>> const& areaLightAlphaCoeff)
{
    constexpr int l = 9;
    for (int i = 0; i < 2 * l + 1; ++i)
    {
        float coeff = 0.0f;
        for (int k = 0; k < 2 * l + 1; ++k)
        {
            /* Differ from CPU version! access buffer like a coordinates (need a transpose) */
            coeff += areaLightAlphaCoeff[l * l + i][k] * Lw[l][k];
        }
        ylmCoeff[l * l + i] = coeff;
    }
}

/**
 * @ x:shading point in world space
 * @ v:vertices of quad/polygonal light in world space, size==M+1, index starting from 1
 * @ n:l_max 
 * @ a:alpha_l,d,m
 * @ w:lobe direction
 */
template<int M, int lmax>
std::vector<float> computeCoeff(float3 x, std::vector<float3> & v, /*int n, std::vector<std::vector<float>> const& a,*/ std::vector<float3> const& w, bool vIsProjected, std::vector<std::vector<float>> const& a)
{
#define computeCoeff_DBGINFO 0
    std::vector<float> ylmCoeff; ylmCoeff.resize((lmax+1)*(lmax+1));

    TW_ASSERT(v.size() == M + 1);
    //TW_ASSERT(n == 2);
    // for all edges:
    std::vector<float3> we; we.resize(v.size());
    
    if (!vIsProjected)
        for (int e = 1; e <= M; ++e)
        {
            v[e] = v[e] - x;
            we[e] = TwUtil::safe_normalize(v[e]);
        }
    else
        we = v;

    std::vector<float3> lambdae; lambdae.resize(v.size());
    std::vector<float3> ue; ue.resize(v.size());
    std::vector<float> gammae; gammae.resize(v.size());
    for (int e = 1; e <= M; ++e)
    {
        // Incorrect modular arthmetic: we[(e + 1) % (M+1)] or we[(e + 1) % (M)]
        const optix::float3& we_plus_1 = (e == M ? we[1] : we[e+1]);
        lambdae[e] = cross(TwUtil::safe_normalize(cross(we[e], we_plus_1)), we[e]);
        ue[e] = cross(we[e], lambdae[e]);
        gammae[e] = acos(dot(we[e], we_plus_1));
    }
    // Solid angle computation
    float solidAngle = computeSolidAngle<M>(we);

    std::vector<std::vector<float>> Lw; Lw.resize(lmax+1);
    for (auto&& Lxw : Lw)
    {
        //todo:redundant computation and storage.
        Lxw.resize(w.size());
    }
//     std::vector<float> L0w; L0w.resize(w.size());
//     std::vector<float> L1w; L1w.resize(w.size());
//     std::vector<float> L2w; L2w.resize(w.size());
    for (int i = 0; i < w.size(); ++i)
    {
        std::vector<float> ae; ae.resize(v.size());
        std::vector<float> be; be.resize(v.size());
        std::vector<float> ce; ce.resize(v.size());
        std::vector<float> B0e; B0e.resize(v.size());
        std::vector<float> B1e; B1e.resize(v.size());
        std::vector<float> D0e; D0e.resize(v.size());
        std::vector<float> D1e; D1e.resize(v.size());
        std::vector<float> D2e; D2e.resize(v.size());


        const float3 &wi = w[i];
        float S0 = solidAngle; // Initialize S0 before computation on each lobe direction!
        float S1 = 0;
        for (int e = 1; e <= M; ++e)
        {
            ae[e] = dot(wi, we[e]); be[e] = dot(wi, lambdae[e]); ce[e] = dot(wi, ue[e]);
            S1 = S1 + 0.5*ce[e] * gammae[e];

            B0e[e] = gammae[e];
            B1e[e] = ae[e] * sin(gammae[e]) - be[e] * cos(gammae[e]) + be[e];
            D0e[e] = 0; D1e[e] = gammae[e]; D2e[e] = 3 * B1e[e];
        }

        // my code for B1
        float Bl_1 = 0.f;
        for (int e = 1; e <= M; ++e)
        {
            Bl_1 += ce[e] * B1e[e];
        }

        // Initial Bands l=0, l=1:
        Lw[0][i] = sqrt(1.f / (4.f*M_PIf))*S0;
        Lw[1][i] = sqrt(3.f / (4.f*M_PIf))*S1;

        // Bands starting from l=2:
        for (int l = 2; l <= lmax; ++l)
        {
            float Bl = 0;
            std::vector<float> Cl_1e; Cl_1e.resize(v.size());
            std::vector<float> B2_e; B2_e.resize(v.size());
            for (int e = 1; e <= M; ++e)
            {
                Cl_1e[e] = 1.f / l * ((ae[e] * sin(gammae[e]) - be[e] * cos(gammae[e]))*
                                       LegendreP(l-1, ae[e] * cos(gammae[e]) + be[e] * sin(gammae[e])) +
                                       be[e] * LegendreP(l-1, ae[e]) + (ae[e] * ae[e] + be[e] * be[e] - 1.f)*D1e[e] +
                                       (l - 1.f)*B0e[e]);
                B2_e[e] = ((2.f*l-1.f)/l)*(Cl_1e[e]) - (l-1.f)/l*B0e[e];
                Bl = Bl + ce[e] * B2_e[e];
                D2e[e] = (2.f*l-1.f) * B1e[e] + D0e[e];

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
                Lw[l][i] = sqrt((2.f * l + 1) / (4.f*M_PIf))*S2;
            }
            else
            {
                float S2 = ((2.f*l - 1) / (l*(l + 1))*Bl_1) + ((l - 2.f)*(l - 1.f) / ((l)*(l + 1.f)))*S1;
                S1 = S2;
                Lw[l][i] = sqrt((2.f * l + 1) / (4.f*M_PIf))*S2;
            }

            Bl_1 = Bl;
        }
    }

#if computeCoeff_DBGINFO
    std::cout << "\n C++ Version" << std::endl;
    int coutx = 1;
    for (const auto& l0wi : Lw[0])
    {
        if (coutx == 2)
            break;
        std::cout << "l0wi:   " << l0wi << std::endl;
        ++coutx;
    }
    std::cout << "--------------end l1wi" << std::endl;

    coutx = 1;
    for (const auto& l1wi : Lw[1])
    {
        if (coutx == 4)break;
        std::cout << "l1wi:   " << l1wi << std::endl;
        ++coutx;
    }
    std::cout << "--------------end l1wi" << std::endl;

    coutx = 1;
    for (const auto& l2wi : Lw[2])
    {
        if (coutx == 6)break;
        std::cout << "l2wi:   " << l2wi << std::endl;
        ++coutx;
    }
    std::cout << "--------------end l2wi" << std::endl;

    if (lmax >= 3)
    {
        coutx = 1;
        for (const auto& l3wi : Lw[3])
        {
            if (coutx == 8)break;
            std::cout << "l3wi:   " << l3wi << std::endl;
            ++coutx;
        }
        std::cout << "--------------end l3wi" << std::endl;

        coutx = 1;
        for (const auto& l4wi : Lw[4])
        {
            if (coutx == 10)break;
            std::cout << "l4wi:   " << l4wi << std::endl;
            ++coutx;
        }
        std::cout << "--------------end l4wi" << std::endl;

        coutx = 1;
        for (const auto& l5wi : Lw[5])
        {
            if (coutx == 12)break;
            std::cout << "l5wi:   " << l5wi << std::endl;
            ++coutx;
        }
        std::cout << "--------------end l5wi" << std::endl;

        coutx = 1;
        for (const auto& l6wi : Lw[6])
        {
            if (coutx == 14)break;
            std::cout << "l6wi:   " << l6wi << std::endl;
            ++coutx;
        }
        std::cout << "--------------end l6wi" << std::endl;

        coutx = 1;
        for (const auto& l7wi : Lw[7])
        {
            if (coutx == 16)break;
            std::cout << "l7wi:   " << l7wi << std::endl;
            ++coutx;
        }
        std::cout << "--------------end l7wi" << std::endl;

        coutx = 1;
        for (const auto& l8wi : Lw[8])
        {
            if (coutx == 18)break;
            std::cout << "l8wi:   " << l8wi << std::endl;
            ++coutx;
        }
        std::cout << "--------------end l8wi" << std::endl;

        coutx = 1;
        for (const auto& l9wi : Lw[9])
        {
            if (coutx == 20)break;
            std::cout << "l9wi:   " << l9wi << std::endl;
            ++coutx;
        }
        std::cout << "--------------end l9wi" << std::endl;
    }
#endif   

    ////TW_ASSERT(9 == a.size());
    ////printf("------\n");
    for (int j = 0; j <= lmax; ++j)
    {
        if(lmax == 2)
            TW_ASSERT(2 * j + 1 == a[j*j+0].size());
        for (int i = 0; i < 2 * j + 1; ++i)
        {
            float coeff = 0.0f;
            //printf("ylmCoeff[%d]=", j*j + i);
            for (int k = 0; k < 2 * j + 1; ++k)
            {
                //printf("%ff*Lw[%d][%d]",a[j*j+i][k],j,k);
                //if(k!=2*j)
                //    printf("+");
                coeff += a[j*j+i][k] * Lw[j][k];
            }
            //printf(";\n");
            ylmCoeff[j*j + i] = coeff;
        }
    }
#if computeCoeff_DBGINFO
     for (const auto& ylmC : ylmCoeff)
     {
         printf("%f\n", ylmC);
     }
#endif
    return ylmCoeff;
}

/**
 * @brief GPU Version computeCoeff
 */
template<int M, int lmax>
void computeCoeff(float3 x, float3 v[], const float3 w[], const float a[][2*lmax+1], float ylmCoeff[(lmax + 1)*(lmax + 1)])
{
    //constexpr int lmax = 2;
#define computeCoeff_DBGINFO 0
    //auto P1 = [](float z)->float {return z; };
#ifdef __CUDACC__
#undef TW_ASSERT
#define TW_ASSERT(expr) TW_ASSERT_INFO(expr, ##expr)
#define TW_ASSERT_INFO(expr, str)    if (!(expr)) {rtPrintf(str); rtPrintf("Above at Line%d:\n",__LINE__);}
#endif
    //TW_ASSERT(v.size() == M + 1);
    //TW_ASSERT(n == 2);
    // for all edges:
    float3 we[M+1];

    for (int e = 1; e <= M; ++e)
    {
        v[e] = v[e] - x;
        we[e] = TwUtil::safe_normalize(v[e]);
    }

    float3 lambdae[M+1];
    float3 ue[M + 1];
    float gammae[M + 1];
    for (int e = 1; e <= M; ++e)
    {
        // Incorrect modular arthmetic: we[(e + 1) % (M+1)] or we[(e + 1) % (M)]
        const optix::float3& we_plus_1 = (e == M ? we[1] : we[e + 1]);
        lambdae[e] = cross(TwUtil::safe_normalize(cross(we[e], we_plus_1)), we[e]);
        ue[e] = cross(we[e], lambdae[e]);
        gammae[e] = acosf(dot(we[e], we_plus_1));
    }
    // Solid angle computation
    float solidAngle = computeSolidAngle<M>(we);

    float Lw[lmax+1][2 * lmax + 1];

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


        const float3 &wi = w[i];
        float S0 = solidAngle;
        float S1 = 0;
        for (int e = 1; e <= M; ++e)
        {
            ae[e] = dot(wi, we[e]); be[e] = dot(wi, lambdae[e]); ce[e] = dot(wi, ue[e]);
            S1 = S1 + 0.5*ce[e] * gammae[e];

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

        // Bands starting from l=2:
        for (int l = 2; l <= lmax; ++l)
        {
            float Bl = 0;
            float Cl_1e[M + 1];
            float B2_e[M + 1];

            for (int e = 1; e <= M; ++e)
            {
                Cl_1e[e] = 1.f / l * ((ae[e] * sin(gammae[e]) - be[e] * cos(gammae[e]))*
                    LegendreP(l - 1, ae[e] * cos(gammae[e]) + be[e] * sin(gammae[e])) +
                    be[e] * LegendreP(l - 1, ae[e]) + (ae[e] * ae[e] + be[e] * be[e] - 1.f)*D1e[e] +
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
                Lw[l][i] = sqrt((2.f * l + 1) / (4.f*M_PIf))*S2;
            }
            else
            {
                float S2 = ((2.f*l - 1) / (l*(l + 1))*Bl_1) + ((l - 2.f)*(l - 1.f) / ((l)*(l + 1.f)))*S1;
                S1 = S2;
                Lw[l][i] = sqrt((2.f * l + 1) / (4.f*M_PIf))*S2;
            }

            Bl_1 = Bl;
        }
    }

#if computeCoeff_DBGINFO
    int coutx = 1;
    for (const auto& l0wi : Lw[0])
    {
        if (coutx == 2)
            break;
        std::cout << "l0wi:   " << l0wi << std::endl;
        ++coutx;
    }
    std::cout << "--------------end l1wi" << std::endl;

    coutx = 1;
    for (const auto& l1wi : Lw[1])
    {
        if (coutx == 4)break;
        std::cout << "l1wi:   " << l1wi << std::endl;
        ++coutx;
    }
    std::cout << "--------------end l1wi" << std::endl;

    coutx = 1;
    for (const auto& l2wi : Lw[2])
    {
        if (coutx == 6)break;
        std::cout << "l2wi:   " << l2wi << std::endl;
        ++coutx;
    }
    std::cout << "--------------end l2wi" << std::endl;

    if (lmax >= 3)
    {
        coutx = 1;
        for (const auto& l3wi : Lw[3])
        {
            if (coutx == 8)break;
            std::cout << "l3wi:   " << l3wi << std::endl;
            ++coutx;
        }
        std::cout << "--------------end l3wi" << std::endl;

        coutx = 1;
        for (const auto& l4wi : Lw[4])
        {
            if (coutx == 10)break;
            std::cout << "l4wi:   " << l4wi << std::endl;
            ++coutx;
        }
        std::cout << "--------------end l4wi" << std::endl;

        coutx = 1;
        for (const auto& l5wi : Lw[5])
        {
            if (coutx == 12)break;
            std::cout << "l5wi:   " << l5wi << std::endl;
            ++coutx;
        }
        std::cout << "--------------end l5wi" << std::endl;

        coutx = 1;
        for (const auto& l6wi : Lw[6])
        {
            if (coutx == 14)break;
            std::cout << "l6wi:   " << l6wi << std::endl;
            ++coutx;
        }
        std::cout << "--------------end l6wi" << std::endl;

        coutx = 1;
        for (const auto& l7wi : Lw[7])
        {
            if (coutx == 16)break;
            std::cout << "l7wi:   " << l7wi << std::endl;
            ++coutx;
        }
        std::cout << "--------------end l7wi" << std::endl;

        coutx = 1;
        for (const auto& l8wi : Lw[8])
        {
            if (coutx == 18)break;
            std::cout << "l8wi:   " << l8wi << std::endl;
            ++coutx;
        }
        std::cout << "--------------end l8wi" << std::endl;

        coutx = 1;
        for (const auto& l9wi : Lw[9])
        {
            if (coutx == 20)break;
            std::cout << "l9wi:   " << l9wi << std::endl;
            ++coutx;
        }
        std::cout << "--------------end l9wi" << std::endl;
    }
#endif

    //TW_ASSERT(9 == a.size());
    //printf("------\n");
    for (int j = 0; j <= lmax; ++j)
    {
        //TW_ASSERT(2 * j + 1 == 2*lmax+1); // redundant storage
        for (int i = 0; i < 2 * j + 1; ++i)
        {
            float coeff = 0.0f;
            for (int k = 0; k < 2 * j + 1; ++k)
            {
                //printf("gpuVersion Lw[%d][%d]=%f\n", j, k, Lw[j][k]);
                coeff += a[j*j + i][k] * Lw[j][k];
            }
            ylmCoeff[j*j + i] = coeff;
            //printf("\n");
        }
    }
}

void QuadLight::TestSolidAngle()
{
    int nfails = 0;
    int ntests = 0;
    float epsilon = 1e-5f;
#define EQ(x,y) (std::abs(x-y)<=epsilon)
    auto sphToCartesian = [](const float theta, const float phi)->float3
    {
        return make_float3(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
    };
    {
        auto A1 = make_float3(0.0f, 1.0f, 0.0f);
        auto B1 = make_float3(-1.0f, 0.0f, 0.0f);
        auto C1 = make_float3(0.0f, -1.0f, 0.0f);
        auto D1 = make_float3(1.0f, 0.0f, 0.0f);
        std::vector<float3> v2{ A1,sphToCartesian(0.f,0.f),D1 };
        float t1 = solid_angle(v2.data(), 3);

        std::vector<float3> v22{ make_float3(0.f), A1,sphToCartesian(0.f,0.f),D1 };
        float t2 = computeSolidAngle<3>(v22);
        float t3 = computeSolidAngle<3>(v22.data());
        if (!(EQ(t1,t2) && EQ(t2,t3) && EQ(t3, 0.5f*M_PIf)))
        {
            ++nfails;
            printf("Test failed at %d: t1==%f t2==%f t3==%f != %f\n", __LINE__, t1, t2, t3, 0.5f*M_PIf);
        }
        ++ntests;

        std::vector<float3> v3{ A1,D1,sphToCartesian(0.f,0.f) };
        t1 = solid_angle(v3.data(), 3);

        std::vector<float3> v33{ make_float3(0.f), A1,D1,sphToCartesian(0.f,0.f) };
        t2 = computeSolidAngle<3>(v33);
        t3 = computeSolidAngle<3>(v33.data());
        if (!(EQ(t1, t2) && EQ(t2, t3) && EQ(t3, 0.5f*M_PIf)))
        {
            ++nfails;
            printf("Test failed at %d: t1==%f t2==%f t3==%f != %f\n", __LINE__, t1, t2, t3, 0.5f*M_PIf);
        }
        ++ntests;
    }
    {
        auto A1 = make_float3(0.0f, 1.0f, 0.0f);
        auto C1 = make_float3(0.0f, -1.0f, 0.0f);
        auto D1 = make_float3(1.0f, 0.0f, 0.0f);
        std::vector<float3> v2{ A1,C1,D1 };
        float t1 = solid_angle(v2.data(), 3);

        std::vector<float3> v22{ make_float3(0.f), A1,C1,D1 };
        float t2 = computeSolidAngle<3>(v22);
        float t3 = computeSolidAngle<3>(v22.data());
        if (!(EQ(t1, t2) && EQ(t2, t3)))
        {
            ++nfails;
            printf("Test failed at %d: t1==%f t2==%f t3==%f \n", __LINE__, t1, t2, t3);
        }
        ++ntests;
    }
    {
        auto A1 = make_float3(0.0f, 1.0f, 0.0f);
        auto B1 = make_float3(-1.0f, 0.0f, 0.0f);
        auto C1 = make_float3(0.0f, -1.0f, 0.0f);
        auto D1 = make_float3(1.0f, 0.0f, 0.0f);
        std::vector<float3> v2{ A1,B1,C1,D1 };
        float t1 = solid_angle(v2.data(), 4);

        std::vector<float3> v22{ make_float3(0.f), A1,B1,C1,D1 };
        float t2 = computeSolidAngle<4>(v22);
        float t3 = computeSolidAngle<4>(v22.data());
        if (!(EQ(t1, t2) && EQ(t2, t3) && EQ(t3, 2.f*M_PIf)))
        {
            ++nfails;
            printf("Test failed at %d: t1==%f t2==%f t3==%f != %f\n", __LINE__, t1, t2, t3, 2.f*M_PIf);
        }
        ++ntests;

        std::vector<float3> v3{ A1,D1,C1,B1 };
        t1 = solid_angle(v3.data(), 4);

        std::vector<float3> v33{ make_float3(0.f), A1,D1,C1,B1 };
        t2 = computeSolidAngle<4>(v33);
        t3 = computeSolidAngle<4>(v33.data());
        if (!(EQ(t1, t2) && EQ(t2, t3) && EQ(t3, 2.f*M_PIf)))
        {
            ++nfails;
            printf("Test failed at %d: t1==%f t2==%f t3==%f != %f\n", __LINE__, t1, t2, t3, 2.f*M_PIf);
        }
        ++ntests;
    }
    {
        auto A1 = (sphToCartesian(M_PI / 2.f, M_PI / 2.f));
        auto B1 = (sphToCartesian(M_PI / 4.f, M_PI / 2.f));
        auto C1 = (sphToCartesian(M_PI / 4.f, 0.f));
        auto D1 = (sphToCartesian(M_PI / 2.f, 0));
        std::vector<float3> v2{ A1,B1,C1,D1 };
        float t1 = solid_angle(v2.data(), 4);

        std::vector<float3> v22{ make_float3(0.f), A1,B1,C1,D1 };
        float t2 = computeSolidAngle<4>(v22);
        float t3 = computeSolidAngle<4>(v22.data());
        if (!(EQ(t1, t2) && EQ(t2, t3) && EQ(t3, M_PIf / sqrt(8))))
        {
            ++nfails;
            printf("Test failed at %d: t1==%f t2==%f t3==%f != %f\n", __LINE__, t1, t2, t3, M_PIf / sqrt(8));
        }
        ++ntests;
    }
    {
        std::random_device rd;  //Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> dis(0.0, 1.0);

        auto uniformSamplingHemisphere = [](float x, float y)->float3
        {
            float z = x;
            float r = std::sqrt(std::max(0.0, 1.0 - z * z));
            float phi = 2.0 * M_PI * y;
            return make_float3(r * std::cos(phi), r * std::sin(phi), z);
        };

        // Random vertices
        auto A1 = (uniformSamplingHemisphere(dis(gen),dis(gen)));
        auto B1 = (uniformSamplingHemisphere(dis(gen), dis(gen)));
        auto C1 = (uniformSamplingHemisphere(dis(gen), dis(gen)));
        std::vector<float3> v2{ A1,B1,C1 };
        float t1 = solid_angle(v2.data(), 3);

        std::vector<float3> v22{ make_float3(0.f), A1,B1,C1 };
        float t2 = computeSolidAngle<3>(v22);
        float t3 = computeSolidAngle<3>(v22.data());
        if (!(EQ(t1, t2) && EQ(t2, t3)))
        {
            ++nfails;
            printf("Test failed at %d: t1==%f t2==%f t3==%f", __LINE__, t1, t2, t3);
        }
        ++ntests;
    }
    // todo:add MC validation
    printf("\nTest coverage:%f%%(%d/%d) passed!\n", 100.f*static_cast<float>(ntests-nfails)/ntests, ntests - nfails, ntests);
}

/**
 * @brief Check polygon. P starts at index 1.
 */
bool CheckOrientation(const float3 P[])
{
    const auto D = (P[1] + P[2] + P[3]) / 3.0f;
    const auto N = cross(P[2] - P[1], P[3] - P[1]);
    return dot(D, N) <= 0.0f;
}

void SHEvalFast9(const optix::float3 &w, float *pOut) {
    const float fX = w.x;
    const float fY = w.y;
    const float fZ = w.z;

    float fC0, fS0, fC1, fS1, fPa, fPb, fPc;
    {
        float fZ2 = fZ * fZ;
        pOut[0] = 0.282094791774;
        pOut[2] = 0.488602511903f*fZ;
        pOut[6] = (0.946174695758f*fZ2) + -0.315391565253f;
        pOut[12] = fZ * ((fZ2* 1.86588166295f) + -1.11952899777f);
        pOut[20] = ((fZ* 1.9843134833f)*pOut[12]) + (-1.00623058987f*pOut[6]);
        pOut[30] = ((fZ* 1.98997487421f)*pOut[20]) + (-1.00285307284f*pOut[12]);
        pOut[42] = ((fZ* 1.99304345718f)*pOut[30]) + (-1.00154202096f*pOut[20]);
        pOut[56] = ((fZ* 1.99489143482f)*pOut[42]) + (-1.00092721392f*pOut[30]);
        pOut[72] = ((fZ* 1.99608992783f)*pOut[56]) + (-1.00060078107f*pOut[42]);
        pOut[90] = ((fZ* 1.99691119507f)*pOut[72]) + (-1.00041143799f*pOut[56]);
        fC0 = fX;
        fS0 = fY;
        fPa = -0.488602511903f;
        pOut[3] = fPa * fC0;
        pOut[1] = fPa * fS0;
        fPb = -1.09254843059f*fZ;
        pOut[7] = fPb * fC0;
        pOut[5] = fPb * fS0;
        fPc = (-2.28522899732f*fZ2) + 0.457045799464f;
        pOut[13] = fPc * fC0;
        pOut[11] = fPc * fS0;
        fPa = fZ * ((fZ2* -4.6833258049f) + 2.00713963067f);
        pOut[21] = fPa * fC0;
        pOut[19] = fPa * fS0;
        fPb = ((fZ* 2.03100960116f)*fPa) + (-0.991031208965f*fPc);
        pOut[31] = fPb * fC0;
        pOut[29] = fPb * fS0;
        fPc = ((fZ* 2.02131498924f)*fPb) + (-0.995226703056f*fPa);
        pOut[43] = fPc * fC0;
        pOut[41] = fPc * fS0;
        fPa = ((fZ* 2.01556443707f)*fPc) + (-0.997155044022f*fPb);
        pOut[57] = fPa * fC0;
        pOut[55] = fPa * fS0;
        fPb = ((fZ* 2.01186954041f)*fPa) + (-0.99816681789f*fPc);
        pOut[73] = fPb * fC0;
        pOut[71] = fPb * fS0;
        fPc = ((fZ* 2.00935312974f)*fPb) + (-0.998749217772f*fPa);
        pOut[91] = fPc * fC0;
        pOut[89] = fPc * fS0;
        fC1 = (fX*fC0) - (fY*fS0);
        fS1 = (fX*fS0) + (fY*fC0);
        fPa = 0.546274215296f;
        pOut[8] = fPa * fC1;
        pOut[4] = fPa * fS1;
        fPb = 1.44530572132f*fZ;
        pOut[14] = fPb * fC1;
        pOut[10] = fPb * fS1;
        fPc = (3.31161143515f*fZ2) + -0.473087347879f;
        pOut[22] = fPc * fC1;
        pOut[18] = fPc * fS1;
        fPa = fZ * ((fZ2* 7.19030517746f) + -2.39676839249f);
        pOut[32] = fPa * fC1;
        pOut[28] = fPa * fS1;
        fPb = ((fZ* 2.11394181566f)*fPa) + (-0.973610120462f*fPc);
        pOut[44] = fPb * fC1;
        pOut[40] = fPb * fS1;
        fPc = ((fZ* 2.08166599947f)*fPb) + (-0.984731927835f*fPa);
        pOut[58] = fPc * fC1;
        pOut[54] = fPc * fS1;
        fPa = ((fZ* 2.06155281281f)*fPc) + (-0.99033793766f*fPb);
        pOut[74] = fPa * fC1;
        pOut[70] = fPa * fS1;
        fPb = ((fZ* 2.04812235836f)*fPa) + (-0.99348527267f*fPc);
        pOut[92] = fPb * fC1;
        pOut[88] = fPb * fS1;
        fC0 = (fX*fC1) - (fY*fS1);
        fS0 = (fX*fS1) + (fY*fC1);
        fPa = -0.590043589927f;
        pOut[15] = fPa * fC0;
        pOut[9] = fPa * fS0;
        fPb = -1.77013076978f*fZ;
        pOut[23] = fPb * fC0;
        pOut[17] = fPb * fS0;
        fPc = (-4.40314469492f*fZ2) + 0.489238299435f;
        pOut[33] = fPc * fC0;
        pOut[27] = fPc * fS0;
        fPa = fZ * ((fZ2* -10.1332578547f) + 2.76361577854f);
        pOut[45] = fPa * fC0;
        pOut[39] = fPa * fS0;
        fPb = ((fZ* 2.20794021658f)*fPa) + (-0.9594032236f*fPc);
        pOut[59] = fPb * fC0;
        pOut[53] = fPb * fS0;
        fPc = ((fZ* 2.1532216877f)*fPb) + (-0.97521738656f*fPa);
        pOut[75] = fPc * fC0;
        pOut[69] = fPc * fS0;
        fPa = ((fZ* 2.11804417119f)*fPc) + (-0.983662844979f*fPb);
        pOut[93] = fPa * fC0;
        pOut[87] = fPa * fS0;
        fC1 = (fX*fC0) - (fY*fS0);
        fS1 = (fX*fS0) + (fY*fC0);
        fPa = 0.625835735449f;
        pOut[24] = fPa * fC1;
        pOut[16] = fPa * fS1;
        fPb = 2.07566231488f*fZ;
        pOut[34] = fPb * fC1;
        pOut[26] = fPb * fS1;
        fPc = (5.55021390802f*fZ2) + -0.504564900729f;
        pOut[46] = fPc * fC1;
        pOut[38] = fPc * fS1;
        fPa = fZ * ((fZ2* 13.4918050467f) + -3.11349347232f);
        pOut[60] = fPa * fC1;
        pOut[52] = fPa * fS1;
        fPb = ((fZ* 2.30488611432f)*fPa) + (-0.948176387355f*fPc);
        pOut[76] = fPb * fC1;
        pOut[68] = fPb * fS1;
        fPc = ((fZ* 2.22917715071f)*fPb) + (-0.967152839723f*fPa);
        pOut[94] = fPc * fC1;
        pOut[86] = fPc * fS1;
        fC0 = (fX*fC1) - (fY*fS1);
        fS0 = (fX*fS1) + (fY*fC1);
        fPa = -0.65638205684f;
        pOut[35] = fPa * fC0;
        pOut[25] = fPa * fS0;
        fPb = -2.36661916223f*fZ;
        pOut[47] = fPb * fC0;
        pOut[37] = fPb * fS0;
        fPc = (-6.74590252336f*fZ2) + 0.51891557872f;
        pOut[61] = fPc * fC0;
        pOut[51] = fPc * fS0;
        fPa = fZ * ((fZ2* -17.2495531105f) + 3.4499106221f);
        pOut[77] = fPa * fC0;
        pOut[67] = fPa * fS0;
        fPb = ((fZ* 2.40163634692f)*fPa) + (-0.939224604204f*fPc);
        pOut[95] = fPb * fC0;
        pOut[85] = fPb * fS0;
        fC1 = (fX*fC0) - (fY*fS0);
        fS1 = (fX*fS0) + (fY*fC0);
        fPa = 0.683184105192f;
        pOut[48] = fPa * fC1;
        pOut[36] = fPa * fS1;
        fPb = 2.6459606618f*fZ;
        pOut[62] = fPb * fC1;
        pOut[50] = fPb * fS1;
        fPc = (7.98499149089f*fZ2) + -0.53233276606f;
        pOut[78] = fPc * fC1;
        pOut[66] = fPc * fS1;
        fPa = fZ * ((fZ2* 21.3928901909f) + -3.77521591604f);
        pOut[96] = fPa * fC1;
        pOut[84] = fPa * fS1;
        fC0 = (fX*fC1) - (fY*fS1);
        fS0 = (fX*fS1) + (fY*fC1);
        fPa = -0.707162732525f;
        pOut[63] = fPa * fC0;
        pOut[49] = fPa * fS0;
        fPb = -2.9157066407f*fZ;
        pOut[79] = fPb * fC0;
        pOut[65] = fPb * fS0;
        fPc = (-9.26339318285f*fZ2) + 0.544905481344f;
        pOut[97] = fPc * fC0;
        pOut[83] = fPc * fS0;
        fC1 = (fX*fC0) - (fY*fS0);
        fS1 = (fX*fS0) + (fY*fC0);
        fPa = 0.728926660175f;
        pOut[80] = fPa * fC1;
        pOut[64] = fPa * fS1;
        fPb = 3.17731764895f*fZ;
        pOut[98] = fPb * fC1;
        pOut[82] = fPb * fS1;
        fC0 = (fX*fC1) - (fY*fS1);
        fS0 = (fX*fS1) + (fY*fC1);
        fPc = -0.748900951853f;
        pOut[99] = fPc * fC0;
        pOut[81] = fPc * fS0;
    }
}

void QuadLight::TestYlmCoeff()
{
#define VALIDATE_SH_ORDER3 0
#define VALIDATE_SH_ORDER9_MC 0
    float epsilon = 8e-5f;
    int nfails = 0;
    int ntests = 0;

    auto sphToCartesian = [](const float theta, const float phi)->float3
    {
        return make_float3(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
    };

    printf("\n");

#if VALIDATE_SH_ORDER3
    {
        float rawA[9][5] = { 1, 0, 0, 0, 0,
                             0.04762, -0.0952401, -1.06303, 0, 0,
                             0.843045, 0.813911, 0.505827, 0, 0,
                             -0.542607, 1.08521, 0.674436, 0, 0,
                             2.61289, -0.196102, 0.056974, -1.11255, -3.29064,
                             -4.46838, 0.540528, 0.0802047, -0.152141, 4.77508,
                             -3.36974, -6.50662, -1.43347, -6.50662, -3.36977,
                             -2.15306, -2.18249, -0.913913, -2.24328, -1.34185,
                             2.43791, 3.78023, -0.322086, 3.61812, 1.39367, };

        std::vector<float3> basisData{ make_float3(0.6, 0, 0.8),
                                       make_float3(-0.67581, -0.619097, 0.4),
                                       make_float3(0.0874255, 0.996171, 0),
                                       make_float3(0.557643, -0.727347, -0.4),
                                       make_float3(-0.590828, 0.104509, -0.8), };

        std::vector<std::vector<float>> a{ {1},
                                               {0.04762, -0.0952401, -1.06303},
                                               {0.843045, 0.813911, 0.505827},
                                               {-0.542607, 1.08521, 0.674436},
                                                       {2.61289, -0.196102, 0.056974, -1.11255, -3.29064},
                                                       {-4.46838, 0.540528, 0.0802047, -0.152141, 4.77508},
                                                       {-3.36974, -6.50662, -1.43347, -6.50662, -3.36977},
                                                       {-2.15306, -2.18249, -0.913913, -2.24328, -1.34185},
                                                       {2.43791, 3.78023, -0.322086, 3.61812, 1.39367} };
        // Test statistics against computeCoeff(vector version), computeCoeff(gpu version), AxialMoments.
        {
            auto A1 = (sphToCartesian(M_PI / 2.f, M_PI / 2.f));
            auto B1 = (sphToCartesian(M_PI / 4.f, M_PI / 2.f));
            auto C1 = (sphToCartesian(M_PI / 4.f, 0.f));
            auto D1 = (sphToCartesian(M_PI / 2.f, 0));

            //std::vector<float3> v{ make_float3(0.f),A1,B1,C1,D1 };
            std::vector<float3> v{ make_float3(0.f),A1,D1,C1 };
            //std::vector<float3> v{ make_float3(0.f),A1,B1,D1 };
            std::vector<float> t1 = computeCoeff<3, 2>(make_float3(0.f), v, basisData, true, a);
            //computeCoeff<4>(make_float3(0.f), v, basisData, true, a);
            std::vector<float> t2{ 0.221557, -0.191874,0.112397,-0.271351,0.257516,-0.106667,-0.157696,-0.182091,0.0910456 };
            if (!equal(t1.begin(), t1.end(), t2.begin(), [epsilon](const float& x, const float& y)
            {
                return std::abs(x - y) <= epsilon;
            }))
            {
                ++nfails;
                TW_ASSERT(t1.size() == t2.size());
                printf("{\n");
                for (int i = 0; i < t1.size(); ++i)
                {
                    printf("Test failed at Line:%d t1[%d]:%.7f t2[%d]:%.7f\n", __LINE__, i, t1[i], i, t2[i]);
                }
                printf("}\n");
            }
            ++ntests;

            float ylmCoeff[9];
            computeCoeff<3, 2>(make_float3(0.f), v.data(), basisData.data(), rawA, ylmCoeff);
            if (!equal(t2.begin(), t2.end(), &ylmCoeff[0], [epsilon](const float& x, const float& y)
            {
                return std::abs(x - y) <= epsilon;
            }))
            {
                ++nfails;
                TW_ASSERT(t2.size() == 9);
                printf("{\n");
                for (int i = 0; i < t1.size(); ++i)
                {
                    printf("Test failed at Line:%d t1[%d]:%.7f ylmCoeff(GPU)[%d]:%.7f\n", __LINE__, i, t1[i], i, ylmCoeff[i]);
                }
                printf("}\n");
            }
            ++ntests;
        }

        {
            auto A1 = (sphToCartesian(M_PI / 2.f, M_PI / 2.f));
            auto B1 = (sphToCartesian(M_PI / 4.f, M_PI / 2.f));
            auto C1 = (sphToCartesian(M_PI / 4.f, 0.f));
            auto D1 = (sphToCartesian(M_PI / 2.f, 0));

            std::vector<float3> v{ make_float3(0.f),A1,B1,C1,D1 };
            //std::vector<float3> v{ make_float3(0.f),A1,B1,D1 };
            std::vector<float> t1 = computeCoeff<4, 2>(make_float3(0.f), v, basisData, true, a);
            std::vector<float> t2{ 0.347247,-0.339578,0.236043,-0.339578,0.343355,-0.278344,-0.148677,-0.278344,0 };
            if (!equal(t1.begin(), t1.end(), t2.begin(), [epsilon](const float& x, const float& y)
            {
                return std::abs(x - y) <= epsilon;
            }))
            {
                ++nfails;
                TW_ASSERT(t1.size() == t2.size());
                for (int i = 0; i < t1.size(); ++i)
                {
                    printf("Test failed at Line:%d t1[%d]:%f t2[%d]:%f\n", __LINE__, i, t1[i], i, t2[i]);
                }
            }
            ++ntests;

            float ylmCoeff[9];
            computeCoeff<4, 2>(make_float3(0.f), v.data(), basisData.data(), rawA, ylmCoeff);
            if (!equal(t2.begin(), t2.end(), &ylmCoeff[0], [epsilon](const float& x, const float& y)
            {
                return std::abs(x - y) <= epsilon;
            }))
            {
                ++nfails;
                TW_ASSERT(t2.size() == 9);
                for (int i = 0; i < t1.size(); ++i)
                {
                    printf("Test failed at Line:%d t2[%d]:%f ylmCoeff(GPU)[%d]:%f\n", __LINE__, i, t1[i], i, ylmCoeff[i]);
                }
            }
            ++ntests;
        }

        {
            // Well projected on hemisphere
            auto A1 = make_float3(0.0f, 1.0f, 0.0f);
            auto B1 = make_float3(-1.0f, 0.0f, 0.0f);
            auto C1 = make_float3(0.0f, -1.0f, 0.0f);
            auto D1 = make_float3(1.0f, 0.0f, 0.0f);

            std::vector<float3> v{ make_float3(0.f),A1,B1,C1,D1 };
            //std::vector<float3> v{ make_float3(0.f),A1,B1,D1 };
            std::vector<float> t1 = computeCoeff<4, 2>(make_float3(0.f), v, basisData, true, a);
            std::vector<float> t2{ 0,-3.72529e-09,1.53499,0,-7.25683e-08,8.81259e-08,-1.43062e-07,-2.99029e-08,7.10429e-08 };
            if (!equal(t1.begin(), t1.end(), t2.begin(), [epsilon](const float& x, const float& y)
            {
                return std::abs(x - y) <= epsilon;
            }))
            {
                ++nfails;
                TW_ASSERT(t1.size() == t2.size());
                for (int i = 0; i < t1.size(); ++i)
                {
                    printf("Test failed at Line:%d t1[%d]:%f t2[%d]:%f\n", __LINE__, i, t1[i], i, t2[i]);
                }
            }
            ++ntests;

            float ylmCoeff[9];
            computeCoeff<4, 2>(make_float3(0.f), v.data(), basisData.data(), rawA, ylmCoeff);
            if (!equal(t2.begin(), t2.end(), &ylmCoeff[0], [epsilon](const float& x, const float& y)
            {
                return std::abs(x - y) <= epsilon;
            }))
            {
                ++nfails;
                TW_ASSERT(t2.size() == 9);
                for (int i = 0; i < t1.size(); ++i)
                {
                    printf("Test failed at Line:%d t2[%d]:%f ylmCoeff(GPU)[%d]:%f\n", __LINE__, i, t2[i], i, ylmCoeff[i]);
                }
            }
            ++ntests;
        }
    }
#endif

    // Order = 9
    {
        float rawA[100][19]{ 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0.684451,0.359206,-1.29053,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0.491764,0.430504,0.232871,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
-1.66836,1.27394,0.689107,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0.330376,0.163767,-0.973178,-1.58573,-1.06532,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
-0.154277,0.466127,-0.597988,0.289095,-0.324048,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0.651874,0.53183,0.467255,0.225622,0.243216,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
-0.675183,0.736365,0.558372,0.231532,0.497581,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
1.05513,-0.712438,-1.15021,-0.780808,0.626025,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
-3.59672,-4.00147,-1.96217,-1.38707,-4.62964,1.83853,-4.19686,0,0,0,0,0,0,0,0,0,0,0,0,
-1.02631,-1.14148,-0.328964,-0.581013,-2.09467,1.84828,-1.76873,0,0,0,0,0,0,0,0,0,0,0,0,
-0.276614,0.125182,-0.74475,0.647966,-0.943961,0.578956,-0.379926,0,0,0,0,0,0,0,0,0,0,0,0,
2.8152,2.99951,1.55905,0.530353,2.94505,-1.80575,2.25959,0,0,0,0,0,0,0,0,0,0,0,0,
0.933848,2.36693,1.06564,0.138454,2.52347,-1.63423,1.77388,0,0,0,0,0,0,0,0,0,0,0,0,
-1.53162,-2.26699,-1.63944,-1.27998,-1.49923,0.802996,-1.87734,0,0,0,0,0,0,0,0,0,0,0,0,
2.20369,2.65718,2.3773,1.47866,2.61718,-0.154934,1.01416,0,0,0,0,0,0,0,0,0,0,0,0,
0.386892,-0.234194,0.425573,-2.07542,-0.99878,0.087789,-1.49088,-2.10113,-2.21767,0,0,0,0,0,0,0,0,0,0,
0.670532,1.11525,0.0213808,0.213706,-0.775345,-0.870597,-0.243734,-1.07331,-0.921999,0,0,0,0,0,0,0,0,0,0,
3.34363,2.88333,1.5528,-0.690988,-1.21196,1.38443,0.248891,-3.33599,-2.18742,0,0,0,0,0,0,0,0,0,0,
-4.43576,-3.10679,-1.50732,1.14355,3.05976,-1.32176,-0.279068,6.36581,4.37729,0,0,0,0,0,0,0,0,0,0,
0.801925,-0.589618,-0.498002,-1.99003,-1.17152,0.0225333,-1.83288,-2.48201,-0.800297,0,0,0,0,0,0,0,0,0,0,
-0.0741623,0.463091,-0.490407,-1.33794,-1.19302,-0.0671063,-0.991982,-2.579,-1.40303,0,0,0,0,0,0,0,0,0,0,
2.30898,2.59505,-0.245139,-0.143466,-1.28047,0.0225859,0.402924,-3.23301,-2.68613,0,0,0,0,0,0,0,0,0,0,
2.45217,1.30712,0.624695,-1.20453,-0.992719,0.8808,-1.4293,-4.16149,-2.09886,0,0,0,0,0,0,0,0,0,0,
-3.29553,-2.75451,-0.707257,-3.38379,1.15948,-1.85432,-2.29433,0.282948,-0.104368,0,0,0,0,0,0,0,0,0,0,
0.901645,0.385552,0.881322,0.762582,0.0627355,0.500188,0.815467,0.0169501,0.29148,1.15498,0.629604,0,0,0,0,0,0,0,0,
1.48735,0.327641,1.03651,0.605442,1.299,1.28978,0.0118243,0.774944,-1.53547,1.58578,0.857565,0,0,0,0,0,0,0,0,
-0.139832,-0.767387,0.690406,-0.647648,-1.73758,-0.953175,-0.415786,0.357295,0.342909,-0.860505,-1.00317,0,0,0,0,0,0,0,0,
-1.30254,-0.0299509,-0.923929,-1.09153,-0.484701,-1.1409,-1.01218,0.732852,0.567873,-1.39764,-0.58814,0,0,0,0,0,0,0,0,
1.3415,0.479091,0.816822,0.0875707,0.305267,0.492711,-0.38267,-0.252676,-0.294921,1.50257,-0.944971,0,0,0,0,0,0,0,0,
-3.18545,-3.58245,-1.57054,-3.91511,-4.13576,-3.40871,-3.03685,1.47586,1.64368,-3.67964,-3.48077,0,0,0,0,0,0,0,0,
-1.49758,-0.795641,-0.477492,-0.9121,-0.961176,-0.978628,-0.587473,0.514521,-0.10312,-0.437121,-0.99984,0,0,0,0,0,0,0,0,
2.04199,1.20223,0.339812,1.55051,1.45889,0.618371,1.08261,-0.0765456,-1.4675,2.22075,0.53648,0,0,0,0,0,0,0,0,
0.0453747,-0.721773,0.127358,0.344248,0.0228017,-0.923741,-0.898898,0.594424,0.0211068,0.407756,-1.21018,0,0,0,0,0,0,0,0,
0.369802,-0.429225,0.962211,-0.428983,1.22736,0.0473707,0.177308,0.884197,-1.56972,1.2173,0.321895,0,0,0,0,0,0,0,0,
0.838529,1.88589,0.571345,1.3097,1.89246,2.36037,2.18634,-0.36959,-0.305823,0.624519,1.9482,0,0,0,0,0,0,0,0,
0.130954,-0.516826,-0.0390864,-0.0555911,-0.483086,0.549499,-0.425825,0.274285,-0.624874,0.704007,0.68713,-0.507504,0.16394,0,0,0,0,0,0,
-0.319237,0.457155,0.0503858,0.0374498,-0.0900405,-0.0600437,-0.0621607,1.7398,0.379183,-0.33379,-0.700205,-1.53056,0.827214,0,0,0,0,0,0,
-1.42509,0.341737,-2.29356,0.486814,3.32227,0.521771,0.22662,2.09383,3.62748,1.29747,0.113476,-4.24123,-1.57304,0,0,0,0,0,0,
0.110626,0.570583,0.681116,0.393754,0.0764495,0.47705,0.0317332,1.01107,0.132282,-0.207397,-0.607639,-0.912909,-0.0276892,0,0,0,0,0,0,
-0.871339,0.953107,-1.23588,0.951312,2.71071,-0.676999,0.417402,1.64249,2.11142,0.667482,-0.64461,-2.83809,-0.224166,0,0,0,0,0,0,
1.02862,-0.207223,1.93275,-0.537461,-2.93969,-1.08259,0.74633,-3.07593,-2.64397,-2.02553,0.324457,3.92295,1.43658,0,0,0,0,0,0,
-0.596174,0.139976,-1.07057,0.92516,2.08283,-0.639657,1.00171,0.956261,1.18423,-0.420206,-1.38542,-2.30077,-0.244787,0,0,0,0,0,0,
-2.21381,0.563199,-2.75559,-0.108763,3.37868,-0.351411,1.51645,0.93244,3.01257,1.26995,-0.682257,-2.69755,-2.34132,0,0,0,0,0,0,
-0.803653,0.486332,-2.71571,0.00486127,3.2686,-0.556502,1.78242,0.779195,3.27404,1.49038,-0.610542,-2.35778,-2.57451,0,0,0,0,0,0,
0.847329,-0.834072,2.37916,1.24146,-1.98473,-0.52079,-0.48859,-0.115073,-1.32164,-0.886867,-0.0804312,1.37765,1.79765,0,0,0,0,0,0,
-2.44787,0.39709,-3.7761,0.015054,6.59958,0.329229,0.798877,3.76058,5.89172,3.02555,-0.532234,-6.77026,-3.27629,0,0,0,0,0,0,
-0.658281,0.529111,-1.75975,-0.618552,1.90219,0.644707,0.912778,0.531902,1.53514,1.07795,0.382837,-0.831314,-1.17779,0,0,0,0,0,0,
0.90254,-0.00195845,0.416233,-0.534067,-1.33826,-0.406687,0.157748,-0.934546,-1.60637,-1.04187,0.646452,1.97913,0.0958256,0,0,0,0,0,0,
1.41138,-1.49318,-0.267813,0.118793,0.0667789,1.1114,1.21832,-0.364957,0.996974,-1.37857,-0.590952,-0.990955,1.44545,0.300644,0.521915,0,0,0,0,
-0.980688,0.364658,0.617238,0.970392,-0.845702,0.32811,-0.286463,0.866263,-0.592107,0.645209,-0.224906,0.547207,-0.936674,-0.788088,-0.536917,0,0,0,0,
-0.168275,0.53673,-0.365787,-1.25915,0.0107433,-0.413228,-0.0320747,-0.366879,-0.353566,0.0366365,0.302125,0.738732,0.35326,0.0523419,-0.221827,0,0,0,0,
-0.584207,0.430717,-0.130176,-0.274328,0.382646,-0.992711,0.0961735,-0.261535,0.0946536,0.772936,-0.148429,0.808034,-0.98955,0.367983,-0.497198,0,0,0,0,
-0.655831,0.734315,0.474604,-0.242935,-0.174109,0.226868,0.216102,0.234184,0.0758351,0.312709,0.304648,0.691801,0.132165,0.248373,-0.771094,0,0,0,0,
0.0795894,1.07338,-0.292855,-1.0238,0.581984,-0.873444,-0.632578,-0.599404,-0.774384,0.293745,0.164963,0.878368,0.574305,0.0938578,1.00816,0,0,0,0,
2.66202,-2.02176,0.195195,0.551417,0.618997,1.44304,2.92024,-0.450233,1.02399,-3.37295,-0.106694,-1.96011,1.64395,0.940143,0.462851,0,0,0,0,
-0.105968,-1.25284,0.864732,2.02985,-0.311623,1.51714,0.530248,-0.186852,-0.190595,-1.46531,-0.509711,-0.848307,-0.040913,0.517662,-1.19258,0,0,0,0,
-1.87156,1.57585,0.171384,-1.07235,0.0795015,-1.88109,-1.77911,-0.466125,-0.225306,2.0612,0.746487,1.15275,-0.836341,-1.0385,-0.0588058,0,0,0,0,
-0.231926,1.37785,-0.192581,-1.36978,-0.125444,-1.93895,-1.58626,-0.52281,-0.00773775,1.94619,1.14006,1.36407,-0.205571,-0.710586,0.220972,0,0,0,0,
0.33655,-0.574124,-0.732785,-0.764633,-0.384849,-0.0135144,0.504584,0.0967235,0.278052,-0.246882,0.53561,0.588689,1.36747,0.94626,0.718744,0,0,0,0,
-1.47714,1.41647,0.480085,-1.61308,0.495642,-1.87418,-1.98503,0.0255505,-1.03677,2.6324,-0.0743271,1.9304,-1.19671,-0.655958,0.10449,0,0,0,0,
0.804917,0.149715,-1.2958,-1.7613,1.1501,-0.56573,0.34409,-0.14935,0.177333,0.810151,0.991728,0.996871,0.634889,-0.423213,0.898464,0,0,0,0,
1.26666,-0.647383,-0.70616,-0.628073,0.550705,-0.287921,1.01286,0.604584,0.565855,-0.58263,0.00775118,0.532163,1.49201,0.565321,0.325189,0,0,0,0,
1.11794,-1.13155,-0.282903,0.617965,-0.0717177,1.57804,1.29605,0.671933,0.738547,-2.33639,-0.274473,-0.262591,1.11294,0.807418,0.257607,0,0,0,0,
-0.377914,-0.225818,-0.429096,0.987763,0.193171,0.714889,-0.666905,-0.929931,-0.588023,-0.0435213,0.465649,1.11136,0.810951,-0.312313,-0.0704491,-0.539556,0.159737,0,0,
-2.32767,1.34197,0.36476,0.48528,-2.29735,-0.107763,-0.157041,2.38077,-1.10103,0.9694,-0.118429,-1.97709,-0.678479,2.7164,1.27078,1.77615,-1.76046,0,0,
-0.506219,0.36112,-0.772923,0.637045,-1.9832,-1.24995,0.187018,1.29697,0.753882,-0.780084,-0.108084,-1.1623,0.228745,0.782582,0.190188,1.46219,-1.24104,0,0,
-0.261945,-0.134153,-0.861752,-0.32569,-1.08022,-0.635845,0.108112,0.980172,0.272034,-0.176725,-0.170833,-0.771681,-0.31043,0.87253,0.529705,1.48879,-0.608076,0,0,
-0.652701,0.343429,-0.860292,1.39669,-1.21608,-0.217333,0.624246,0.513427,-0.448237,0.419166,-0.201683,-0.834232,0.63071,0.541281,-0.198191,1.73257,-1.33826,0,0,
-0.143953,1.26514,0.252472,-0.406242,-0.671232,-0.463832,-0.187793,-0.0536602,0.755577,0.0418132,-0.613325,0.185727,-0.582403,0.168035,-0.114024,0.891265,-0.929824,0,0,
2.01231,-1.57626,-0.800351,0.856102,2.55656,1.95036,0.395023,-3.5701,0.742491,-0.329472,-0.0741527,2.63708,0.83174,-2.53329,-1.54782,-1.52773,1.88953,0,0,
-1.01344,0.222599,0.0148034,0.204784,-0.807036,0.182928,-0.523892,1.60103,-0.937233,0.743981,-0.674546,-0.0547825,-0.667966,1.43427,0.187707,0.861661,-0.698571,0,0,
-0.496894,0.258762,0.294853,0.568549,-0.587026,-0.761855,-0.250601,0.208739,0.283704,0.0268767,0.470202,-0.815505,-0.244517,-0.188146,0.19042,0.823236,-0.0702735,0,0,
-0.400609,-0.530642,-0.0301947,-0.01536,0.655302,-0.239775,0.572657,-0.241502,0.26003,-0.401339,0.12529,-0.0177895,0.198477,0.419563,-0.149376,0.522912,-0.248691,0,0,
3.02225,-1.04811,0.382163,-0.814561,2.24272,-0.140416,0.693969,-2.79055,1.04339,-0.215989,-0.0298695,1.39015,0.197856,-1.48015,-1.53468,-1.01782,1.39767,0,0,
3.50719,-1.32831,0.82969,-1.76717,3.12907,0.64418,0.485468,-4.61659,1.32673,0.264079,-0.585126,2.83722,0.276637,-3.21029,-2.21937,-3.05265,2.95631,0,0,
-0.687074,-0.364741,0.182821,0.36512,-0.775456,0.474574,-0.0408075,0.633208,-0.0875694,-0.0766544,-0.14942,-0.318291,0.280064,0.234616,0.977562,0.441624,-0.662151,0,0,
0.0898836,0.0633354,-1.49628,1.36927,-0.473625,0.208693,-0.458777,-0.25294,0.156376,-0.349746,0.342975,0.425743,-0.28819,-0.386056,-1.10283,0.639174,-1.61187,0,0,
0.683439,-0.256975,0.853269,-1.25306,0.689052,-0.205386,-0.250166,-0.0950969,0.375352,0.789996,-0.948669,-0.12304,-0.222474,0.474984,1.02151,-1.0293,1.25793,0,0,
-1.32926,0.386258,-0.413633,0.452075,-1.29237,0.123832,-0.775261,2.05353,-0.438136,0.371959,-0.196067,-1.72413,0.537271,1.33648,0.961259,0.902856,-0.412672,0,0,
-2.26639,1.17612,0.583651,0.185289,-1.79347,-0.720326,-0.414004,2.51146,-1.16678,-0.257522,-0.307256,-2.13279,-0.37188,1.88216,1.74421,1.33016,-1.07328,0,0,
18392,-38257.4,37806.7,-21284.7,32237,-62523.1,-41790.9,110133,92665.1,4731.58,92667,110134,-41792.6,-62522.7,32237,-21286.5,37806.1,-38255.3,18392.3,
-75.8607,157.288,-155.575,87.4328,-132.743,256.299,172.322,-453.639,-381.202,-18.6079,-381.246,-453.673,172.329,256.294,-132.735,87.4507,-155.557,157.267,-75.8604,
-2503.15,5205.17,-5145.71,2897.2,-4387.52,8508.29,5688.74,-14988.2,-12612.2,-644.042,-12610.3,-14988.5,5687.31,8508.34,-4386.49,2896.59,-5145.41,5206.82,-2502.99,
-12750.3,26524.9,-26210.7,14755.7,-22349.6,43346.7,28971.1,-76353.7,-64242,-3280.42,-64245.4,-76354.1,28975.1,43346,-22348.9,14758.9,-26210.4,26520.3,-12751.3,
3672.85,-7639.39,7547.34,-4249.92,6436.45,-12483.9,-8343.34,21989,18501.1,944.875,18501.9,21989.2,-8344.08,-12483.2,6436.8,-4250.16,7547.37,-7638.16,3672.5,
-14546,30256.1,-29899.7,16833.2,-25494.3,49446.8,33050.4,-87099,-73285,-3742.06,-73285.7,-87100.7,33052.3,49446.3,-25495.2,16834.4,-29898.9,30254.3,-14545.2,
2599.25,-5405.28,5340.93,-3007.28,4554.29,-8833.02,-5905.07,15560.5,13091.6,666.984,13092.1,15560.6,-5905.66,-8832.45,4554.75,-3007.63,5340.95,-5405.05,2598.8,
9713.83,-20206.3,19968.3,-11241.4,17025.7,-33021.3,-22071.7,58167.3,48941,2499.18,48942.9,58167.7,-22073.9,-33020.6,17025.1,-11242.7,19967.2,-20203.1,9713.36,
-15217.6,31652.3,-31280,17611,-26671,51729.2,34576,-91119.5,-76667,-3914.73,-76668.9,-91120.6,34578.1,51727.6,-26671.3,17610.6,-31278.9,31650.4,-15216.8,
22110.1,-45994.4,45450.8,-25586.7,38754.6,-75164.7,-50238.3,132400,111398,5688.3,111403,132400,-50244.2,-75162.2,38754.3,-25591.2,45448.8,-45987.3,22111.3,
6281,-13066.7,12911.9,-7269.6,11010.4,-21354.4,-14272,37613.1,31647.3,1616.11,31648.1,37613.6,-14272.9,-21353.7,11010.5,-7269.43,12911.4,-13066,6280.7,
9762.08,-20306.4,20065.7,-11296.4,17110.4,-33184.5,-22179.3,58452.4,49180.7,2511.16,49182.5,58452.7,-22181.6,-33183.8,17109.8,-11297.7,20064.6,-20303.1,9761.61,
-6209.93,12916.3,-12764.4,7185.95,-10883.3,21110,14108.8,-37183.1,-31285.4,-1598.15,-31286.7,-37183.3,14110.2,21108.6,-10884.4,7186.79,-12764.4,12915.8,-6208.84,
-70.5743,147.913,-146.15,82.1403,-125.439,243.072,161.074,-425.993,-358.776,-19.0758,-358.764,-426.003,161.08,243.066,-125.446,82.1452,-146.143,147.9,-70.5705,
9020.58,-18763.9,18542.3,-10439,15809.9,-30664.7,-20495.9,54014.2,45446.5,2320.56,45448.5,54014.7,-20497.7,-30663,15810.7,-10439.6,18542.3,-18760.9,9019.74,
12565.3,-26138.9,25829.3,-14540.3,22024.7,-42715.6,-28550.8,75243.1,63307.8,3232.64,63311.2,75243.5,-28554.8,-42715,22024,-14543.4,25829,-26134.3,12566.3,
-1062.07,2209.78,-2182.31,1229.07,-1862.27,3611.85,2412.59,-6359.96,-5351.25,-273.013,-5350.44,-6360.09,2411.99,3611.87,-1861.83,1228.81,-2182.18,2210.48,-1062.01,
7764.91,-16152.3,15962.1,-8985.76,13610.3,-26396.8,-17643.6,46496.1,39121.1,1997.65,39123.3,46497.6,-17644.3,-26395.7,13609.6,-8987.13,15960.5,-16150.2,7764.96,
-7382.98,15356.7,-15175.6,8543.61,-12939.8,25096.9,16775.5,-44208.7,-37195.9,-1899.55,-37196.7,-44209,16776.2,25096.8,-12939.8,8544.31,-15175.3,15355.8,-7383.1 };

        std::vector<std::vector<float>> a;
        for (int i = 0; i < 100; ++i)
        {
            std::vector<float> row;
            for (int j = 0; j < 19; ++j)
            {
                row.push_back(rawA[i][j]);
            }
            a.push_back(std::move(row));
        }
        std::vector<float3> basisData{ make_float3(0.320145, 0.000000, 0.947368),
                                   make_float3(-0.397673, -0.364301, 0.842105),
                                   make_float3(0.059105, 0.673476, 0.736842),
                                   make_float3(0.471729, -0.615288, 0.631579),
                                   make_float3(-0.837291, 0.148105, 0.526316),
                                   make_float3(0.765317, 0.486832, 0.421053),
                                   make_float3(-0.246321, -0.916299, 0.315789),
                                   make_float3(-0.450576, 0.867560, 0.210526),
                                   make_float3(0.934103, -0.341132, 0.105263),
                                   make_float3(-0.924346, -0.381556, 0.000000),
                                   make_float3(0.421492, 0.900702, -0.105263),
                                   make_float3(0.292575, -0.932780, -0.210526),
                                   make_float3(-0.820937, 0.475752, -0.315789),
                                   make_float3(0.885880, 0.194759, -0.421053),
                                   make_float3(-0.489028, -0.695588, -0.526316),
                                   make_float3(-0.099635, 0.768883, -0.631579),
                                   make_float3(0.516953, -0.435687, -0.736842),
                                   make_float3(-0.538853, -0.022282, -0.842105),
                                   make_float3(0.226929, 0.225824, -0.947368), };

        // Unit Test Polygon with edge = 4. UnitHemisphere case.
        {
            // Well projected on hemisphere
            auto A1 = make_float3(0.0f, 1.0f, 0.0f);
            auto B1 = make_float3(-1.0f, 0.0f, 0.0f);
            auto C1 = make_float3(0.0f, -1.0f, 0.0f);
            auto D1 = make_float3(1.0f, 0.0f, 0.0f);

            /* MC Validation of canonical cases: */
#if VALIDATE_SH_ORDER9_MC
            auto uniformSamplingHemisphere = [](float x, float y)->CLTest::CommonStruct::QuadLight_float3
            {
                float z = x;
                float r = std::sqrt(std::max(0.f, 1.f - z * z));
                float phi = 2 * M_PI * y;
                return CLTest::CommonStruct::make_QuadLight_float3(r * std::cos(phi), r * std::sin(phi), z);
            };

            auto pdfHemisphere = []()->float {return 1.f / (2.f*M_PI); };
            std::random_device rd;  //Will be used to obtain a seed for the random number engine
            std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
            std::uniform_real_distribution<> dis(0.0, 1.0);

            constexpr int iteration = 1000000;
            constexpr int lmax = 9; // order 9

            
            Eigen::VectorXf MCResultCoeffVec((lmax + 1)*(lmax + 1));
            MCResultCoeffVec = Eigen::VectorXf::Zero((lmax + 1)*(lmax + 1));
            for (int i = 1; i <= iteration; ++i)
            {
                Eigen::VectorXf ylm((lmax + 1)*(lmax + 1));
                ylm = Eigen::VectorXf::Zero((lmax + 1)*(lmax + 1));

                CLTest::CommonStruct::QuadLight_float3 sample = uniformSamplingHemisphere(dis(gen), dis(gen));
                SHEvalFast(sample, lmax, ylm);
                TW_ASSERT(ylm.size() == (lmax + 1)*(lmax + 1));
                MCResultCoeffVec += ylm / pdfHemisphere();

                std::cout << "MC Progress: " << i << " / " << iteration << "     \r";
                std::cout.flush();
            }
            MCResultCoeffVec *= (1.0 / iteration);

            std::vector<float> MCResultCoeff;
            for (int i = 0; i < 100; ++i)
            {
                MCResultCoeff.push_back(MCResultCoeffVec(i));
            }
#endif

            std::vector<float3> v{ make_float3(0.f),A1,B1,C1,D1 };
            //std::vector<float3> v{ make_float3(0.f),A1,B1,D1 };
            std::vector<float> t1 = computeCoeff<4, 9>(make_float3(0.f), v, basisData, true, a);
            
            float ylmCoeff[100];
            computeCoeff<4, 9>(make_float3(0.f), v.data(), basisData.data(), rawA, ylmCoeff);
            if (!equal(t1.begin(), t1.end(), &ylmCoeff[0], [epsilon](const float& x, const float& y)
            {
                return std::abs(x - y) <= epsilon;
            }))
            {
                ++nfails;
                TW_ASSERT(t1.size() == 100);
                printf("{\n");
                for (int i = 0; i < t1.size(); ++i)
                {
                    printf("Test failed at Line:%d t1[%d]:%.7f ylmCoeff(GPU)[%d]:%.7f\n", __LINE__, i, t1[i], i, ylmCoeff[i]);
                }
                printf("}\n");
            }
            ++ntests;

#if VALIDATE_SH_ORDER9_MC
            if (!equal(MCResultCoeff.begin(), MCResultCoeff.end(), &ylmCoeff[0], [epsilon](const float& x, const float& y)
            {
                return std::abs(x - y) <= epsilon;
            }))
            {
                ++nfails;
                TW_ASSERT(MCResultCoeff.size() == 100);
                printf("{\n");
                for (int i = 0; i < MCResultCoeff.size(); ++i)
                {
                    printf("Test failed at Line:%d t1[%d]:%.7f ylmCoeff(GPU)[%d]:%.7f\n", __LINE__, i, MCResultCoeff[i], i, ylmCoeff[i]);
                }
                printf("}\n");
            }
            ++ntests;
            std::exit(0);
#endif

            std::vector<float> t2{ 0,-2.38419e-07,1.53499,-1.78814e-07,8.0361e-09,1.29553e-08,-1.49964e-09,2.37276e-08,-5.70635e-08,1.90735e-06,0,
1.19209e-07,-0.586183,-1.19209e-07,-1.19209e-07,4.91738e-07,2.90852e-07,-1.73211e-08,3.61446e-07,-5.52696e-07,2.85673e-07,
2.19718e-07,2.16946e-07,4.27483e-07,2.03351e-07,0,3.21865e-06,-3.93391e-06,-9.53674e-07,1.07288e-06,0.367402,
-9.53674e-07,3.8147e-06,-1.66893e-06,-7.45058e-07,4.05312e-06,1.59473e-07,8.44052e-07,-1.00783e-07,4.63194e-07,6.57873e-07,
-1.27605e-07,6.28974e-07,-9.65823e-07,-9.55999e-07,1.80002e-06,-1.09245e-06,-9.892e-07,-3.4858e-07,1.62125e-05,-1.14441e-05,
2.38419e-06,-2.86102e-06,-4.76837e-06,1.90735e-06,1.81198e-05,-0.26816,3.8147e-06,-3.33786e-06,6.67572e-06,7.62939e-06,
3.8147e-06,8.58307e-06,-7.62939e-06,-1.8975e-06,-5.77771e-06,-7.41833e-06,-2.07832e-06,-7.66758e-06,-6.26134e-07,3.82385e-06,
-1.88402e-06,-3.5203e-06,1.18708e-06,8.25938e-06,1.41067e-05,-4.0676e-06,-5.4201e-06,6.67927e-06,-4.89425e-06,-4.6691e-06,
1.5,0.00292969,0.265625,1.625,0.21875,1.375,0.484375,1.875,-0.625,-0.375,
-0.4375,1.375,0.65625,-0.00683594,1.25,-0.8125,0.0859375,0.75,0.1875, };
            TW_ASSERT(t2.size() == 100);
            if (!equal(t1.begin(), t1.end(), t2.begin(), [epsilon](const float& x, const float& y)
            {
                return std::abs(x - y) <= epsilon;
            }))
            {
                ++nfails;
                TW_ASSERT(t1.size() == t2.size());
                for (int i = 0; i < t1.size(); ++i)
                {
                    printf("Test failed at Line:%d t1[%d]:%f t2[%d]:%f\n", __LINE__, i, t1[i], i, t2[i]);
                }
            }
            ++ntests;
#if VALIDATE_SH_ORDER9_MC
            if (!equal(t1.begin(), t1.end(), MCResultCoeff.begin(), [epsilon](const float& x, const float& y)
            {
                return std::abs(x - y) <= epsilon;
            }))
            {
                ++nfails;
                TW_ASSERT(t1.size() == MCResultCoeff.size());
                for (int i = 0; i < t1.size(); ++i)
                {
                    printf("Test failed at Line:%d t1[%d]:%f MCResultCoeff[%d]:%f\n", __LINE__, i, t1[i], i, MCResultCoeff[i]);
                }
            }
            ++ntests;

            if (!equal(t2.begin(), t2.end(), MCResultCoeff.begin(), [epsilon](const float& x, const float& y)
            {
                return std::abs(x - y) <= epsilon;
            }))
            {
                ++nfails;
                TW_ASSERT(t2.size() == MCResultCoeff.size());
                for (int i = 0; i < t2.size(); ++i)
                {
                    printf("Test failed at Line:%d t2[%d]:%f MCResultCoeff[%d]:%f\n", __LINE__, i, t2[i], i, MCResultCoeff[i]);
                }
            }
            ++ntests;
#endif
        } 

        // Unit Test Polygon with edge = 5. UnitHemisphere case.
         {
         // Well projected on hemisphere
         auto A1 = make_float3(0.0f, 1.0f, 0.0f);
         auto B1 = make_float3(-1.0f, 0.0f, 0.0f);
         auto C1 = make_float3(0.0f, -1.0f, 0.0f);
         auto D1 = make_float3(1.0f, 0.0f, 0.0f);
         auto E1 = make_float3(sqrt(2) / 2.f, sqrt(2) / 2.f, 0.f);

         /* MC Validation of canonical cases: */
#if VALIDATE_SH_ORDER9_MC
         auto uniformSamplingHemisphere = [](float x, float y)->CLTest::CommonStruct::QuadLight_float3
         {
             float z = x;
             float r = std::sqrt(std::max(0.f, 1.f - z * z));
             float phi = 2 * M_PI * y;
             return CLTest::CommonStruct::make_QuadLight_float3(r * std::cos(phi), r * std::sin(phi), z);
         };

         auto pdfHemisphere = []()->float {return 1.f / (2.f*M_PI); };
         std::random_device rd;  //Will be used to obtain a seed for the random number engine
         std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
         std::uniform_real_distribution<> dis(0.0, 1.0);

         constexpr int iteration = 100000;
         constexpr int lmax = 9; // order 9


         Eigen::VectorXf MCResultCoeffVec((lmax + 1)*(lmax + 1));
         MCResultCoeffVec = Eigen::VectorXf::Zero((lmax + 1)*(lmax + 1));
         for (int i = 1; i <= iteration; ++i)
         {
             Eigen::VectorXf ylm((lmax + 1)*(lmax + 1));
             ylm = Eigen::VectorXf::Zero((lmax + 1)*(lmax + 1));

             CLTest::CommonStruct::QuadLight_float3 sample = uniformSamplingHemisphere(dis(gen), dis(gen));
             SHEvalFast(sample, lmax, ylm);
             TW_ASSERT(ylm.size() == (lmax + 1)*(lmax + 1));
             MCResultCoeffVec += ylm / pdfHemisphere();

             std::cout << "MC Progress: " << i << " / " << iteration << "     \r";
             std::cout.flush();
         }
         MCResultCoeffVec *= (1.0 / iteration);

         std::vector<float> MCResultCoeff;
         for (int i = 0; i < 100; ++i)
         {
             MCResultCoeff.push_back(MCResultCoeffVec(i));
         }
#endif

         std::vector<float3> v{ make_float3(0.f),A1,B1,C1,D1,E1 };
         //std::vector<float3> v{ make_float3(0.f),A1,B1,D1 };
         std::vector<float> t1 = computeCoeff<5, 9>(make_float3(0.f), v, basisData, true, a);

         float ylmCoeff[100];
         computeCoeff<5, 9>(make_float3(0.f), v.data(), basisData.data(), rawA, ylmCoeff);
         if (!equal(t1.begin(), t1.end(), &ylmCoeff[0], [epsilon](const float& x, const float& y)
         {
             return std::abs(x - y) <= epsilon;
         }))
         {
             ++nfails;
             TW_ASSERT(t1.size() == 100);
             printf("{\n");
             for (int i = 0; i < t1.size(); ++i)
             {
                 printf("Test failed at Line:%d t1[%d]:%.7f ylmCoeff(GPU)[%d]:%.7f\n", __LINE__, i, t1[i], i, ylmCoeff[i]);
             }
             printf("}\n");
         }
         ++ntests;

#if VALIDATE_SH_ORDER9_MC
         if (!equal(MCResultCoeff.begin(), MCResultCoeff.end(), &ylmCoeff[0], [epsilon](const float& x, const float& y)
         {
             return std::abs(x - y) <= epsilon;
         }))
         {
             ++nfails;
             TW_ASSERT(MCResultCoeff.size() == 100);
             printf("{\n");
             for (int i = 0; i < MCResultCoeff.size(); ++i)
             {
                 printf("Test failed at Line:%d MonteCarlo[%d]:%.7f ylmCoeff(GPU)[%d]:%.7f\n", __LINE__, i, MCResultCoeff[i], i, ylmCoeff[i]);
             }
             printf("}\n");
         }
         ++ntests;
         std::exit(0);
#endif

         std::vector<float> t2{ 0,-2.38419e-07,1.53499,-1.78814e-07,8.0361e-09,1.29553e-08,-1.49964e-09,2.37276e-08,-5.70635e-08,1.90735e-06,0,
1.19209e-07,-0.586183,-1.19209e-07,-1.19209e-07,4.91738e-07,2.90852e-07,-1.73211e-08,3.61446e-07,-5.52696e-07,2.85673e-07,
2.19718e-07,2.16946e-07,4.27483e-07,2.03351e-07,0,3.21865e-06,-3.93391e-06,-9.53674e-07,1.07288e-06,0.367402,
-9.53674e-07,3.8147e-06,-1.66893e-06,-7.45058e-07,4.05312e-06,1.59473e-07,8.44052e-07,-1.00783e-07,4.63194e-07,6.57873e-07,
-1.27605e-07,6.28974e-07,-9.65823e-07,-9.55999e-07,1.80002e-06,-1.09245e-06,-9.892e-07,-3.4858e-07,1.62125e-05,-1.14441e-05,
2.38419e-06,-2.86102e-06,-4.76837e-06,1.90735e-06,1.81198e-05,-0.26816,3.8147e-06,-3.33786e-06,6.67572e-06,7.62939e-06,
3.8147e-06,8.58307e-06,-7.62939e-06,-1.8975e-06,-5.77771e-06,-7.41833e-06,-2.07832e-06,-7.66758e-06,-6.26134e-07,3.82385e-06,
-1.88402e-06,-3.5203e-06,1.18708e-06,8.25938e-06,1.41067e-05,-4.0676e-06,-5.4201e-06,6.67927e-06,-4.89425e-06,-4.6691e-06,
1.5,0.00292969,0.265625,1.625,0.21875,1.375,0.484375,1.875,-0.625,-0.375,
-0.4375,1.375,0.65625,-0.00683594,1.25,-0.8125,0.0859375,0.75,0.1875, };
         TW_ASSERT(t2.size() == 100);
         if (!equal(t1.begin(), t1.end(), t2.begin(), [epsilon](const float& x, const float& y)
         {
             return std::abs(x - y) <= epsilon;
         }))
         {
             ++nfails;
             TW_ASSERT(t1.size() == t2.size());
             for (int i = 0; i < t1.size(); ++i)
             {
                 printf("Test failed at Line:%d t1[%d]:%f t2[%d]:%f\n", __LINE__, i, t1[i], i, t2[i]);
             }
         }
         ++ntests;
#if VALIDATE_SH_ORDER9_MC
         if (!equal(t1.begin(), t1.end(), MCResultCoeff.begin(), [epsilon](const float& x, const float& y)
         {
             return std::abs(x - y) <= epsilon;
         }))
         {
             ++nfails;
             TW_ASSERT(t1.size() == MCResultCoeff.size());
             for (int i = 0; i < t1.size(); ++i)
             {
                 printf("Test failed at Line:%d t1[%d]:%f MCResultCoeff[%d]:%f\n", __LINE__, i, t1[i], i, MCResultCoeff[i]);
             }
         }
         ++ntests;

         if (!equal(t2.begin(), t2.end(), MCResultCoeff.begin(), [epsilon](const float& x, const float& y)
         {
             return std::abs(x - y) <= epsilon;
         }))
         {
             ++nfails;
             TW_ASSERT(t2.size() == MCResultCoeff.size());
             for (int i = 0; i < t2.size(); ++i)
             {
                 printf("Test failed at Line:%d t2[%d]:%f MCResultCoeff[%d]:%f\n", __LINE__, i, t2[i], i, MCResultCoeff[i]);
             }
         }
         ++ntests;
#endif
        }

//        {
//            // Well projected on hemisphere
//            auto A1 = make_float3(0.0f, 1.0f, 0.0f);
//            auto B1 = make_float3(-1.0f, 0.0f, 0.0f);
//            auto C1 = make_float3(0.0f, -1.0f, 0.0f);
//            auto D1 = make_float3(1.0f, 0.0f, 0.0f);
//
//            std::vector<float3> v{ make_float3(0.f),A1,B1,C1,D1 };
//            //std::vector<float3> v{ make_float3(0.f),A1,B1,D1 };
//            std::vector<float> t1 = computeCoeff<4, 9>(make_float3(0.f), v, basisData, true, a);
//            std::vector<float> t2{ 0,-2.38419e-07,1.53499,-1.78814e-07,8.0361e-09,1.29553e-08,-1.49964e-09,2.37276e-08,-5.70635e-08,1.90735e-06,0,
//1.19209e-07,-0.586183,-1.19209e-07,-1.19209e-07,4.91738e-07,2.90852e-07,-1.73211e-08,3.61446e-07,-5.52696e-07,2.85673e-07,
//2.19718e-07,2.16946e-07,4.27483e-07,2.03351e-07,0,3.21865e-06,-3.93391e-06,-9.53674e-07,1.07288e-06,0.367402,
//-9.53674e-07,3.8147e-06,-1.66893e-06,-7.45058e-07,4.05312e-06,1.59473e-07,8.44052e-07,-1.00783e-07,4.63194e-07,6.57873e-07,
//-1.27605e-07,6.28974e-07,-9.65823e-07,-9.55999e-07,1.80002e-06,-1.09245e-06,-9.892e-07,-3.4858e-07,1.62125e-05,-1.14441e-05,
//2.38419e-06,-2.86102e-06,-4.76837e-06,1.90735e-06,1.81198e-05,-0.26816,3.8147e-06,-3.33786e-06,6.67572e-06,7.62939e-06,
//3.8147e-06,8.58307e-06,-7.62939e-06,-1.8975e-06,-5.77771e-06,-7.41833e-06,-2.07832e-06,-7.66758e-06,-6.26134e-07,3.82385e-06,
//-1.88402e-06,-3.5203e-06,1.18708e-06,8.25938e-06,1.41067e-05,-4.0676e-06,-5.4201e-06,6.67927e-06,-4.89425e-06,-4.6691e-06,
//1.5,0.00292969,0.265625,1.625,0.21875,1.375,0.484375,1.875,-0.625,-0.375,
//-0.4375,1.375,0.65625,-0.00683594,1.25,-0.8125,0.0859375,0.75,0.1875, };
//            if (!equal(t1.begin(), t1.end(), t2.begin(), [epsilon](const float& x, const float& y)
//            {
//                return std::abs(x - y) <= epsilon;
//            }))
//            {
//                ++nfails;
//                TW_ASSERT(t1.size() == t2.size());
//                for (int i = 0; i < t1.size(); ++i)
//                {
//                    printf("Test failed at Line:%d t1[%d]:%f AxialMoment[%d]:%f\n", __LINE__, i, t1[i], i, t2[i]);
//                }
//            }
//            ++ntests;
//
//            float ylmCoeff[100];
//            // computeCoeff<M,lmax>
//            computeCoeff<4, 9>(make_float3(0.f), v.data(), basisData.data(), rawA, ylmCoeff);
//            if (!equal(t1.begin(), t1.end(), &ylmCoeff[0], [epsilon](const float& x, const float& y)
//            {
//                return std::abs(x - y) <= epsilon;
//            }))
//            {
//                ++nfails;
//                TW_ASSERT(t1.size() == 100);
//                for (int i = 0; i < t1.size(); ++i)
//                {
//                    printf("Test failed at Line:%d t1[%d]:%f ylmCoeff(GPU)[%d]:%f\n", __LINE__, i, t1[i], i, ylmCoeff[i]);
//                }
//            }
//            ++ntests;
//        }
//
//        {
//            auto A1 = (sphToCartesian(M_PI / 2.f, M_PI / 2.f));
//            auto B1 = (sphToCartesian(M_PI / 4.f, M_PI / 2.f));
//            auto C1 = (sphToCartesian(M_PI / 4.f, 0.f));
//            auto D1 = (sphToCartesian(M_PI / 2.f, 0));
//
//            std::vector<float3> v{ make_float3(0.f),A1,D1,C1 };
//            std::vector<float> t1 = computeCoeff<3, 9>(make_float3(0.f), v, basisData, true, a);
//            //computeCoeff<4>(make_float3(0.f), v, basisData, true, a);
//            std::vector<float> t2{ 0.221557,
//-0.191874,0.112397,-0.271351,0.257516,-0.106667,-0.157696,-0.182091,0.0910458,-0.168561,0.180663,0.0979564,
//-0.172452,0.0951848,0.100333,0.0409613,0.0590044,-0.159732,-0.0669044,0.123451,0.0352618,0.178412,0.0315391,0,
//-0.0417227,-0.0322195,0.0864866,-0.0240152,-0.149799,-0.022234,0.126357,0.0235817,-0.0415949,-0.0594356,-0.0360227,
//-0.0113902,0.0575103,-0.0397269,0.0645599,0.0734258,-0.020161,-0.0867893,0.0190712,-0.0860057,-0.0548337,-0.0482535,
//0.00720787,0.0112695,0.0227746,-0.0603642,
//0.0413408,
//-0.0334876,
//0.0108118,
//0.0544548,
//0.0726051,
//-0.00468302,
//-0.058526,
//-0.0298831,
//-0.0204041,
//0.0195189,
//0.0394006,
//0.0309601,
//0.0143535,
//0.00383496,
//0.0343636,
//-0.0536409,
//-0.00150036,
//-0.0274113,
//-0.0316856,
//0.000189409,
//0.018694,
//0.0490761,
//-0.00329074,
//0.02873,
//0.011591,
//0.0509859,
//0.0273456,
//0.0109474,
//-0.0135141,
//-0.00925419,
//-0.0138894,
//1.02033,
//0.0463813,
//-0.0542679,
//0.390764,
//-0.0158205,
//-0.833401,
//0.181615,
//0.800297,
//-0.0891647,
//0.474453,
//0.0354776,
//0.875259,
//-0.0860701,
//-0.00929889,
//0.539316,
//0.0355549,
//-0.018822,
//0.604826,
//-0.874716 };
//            TW_ASSERT(t2.size() == 100);
//            if (!equal(t1.begin(), t1.end(), t2.begin(), [epsilon](const float& x, const float& y)
//            {
//                return std::abs(x - y) <= epsilon;
//            }))
//            {
//                ++nfails;
//                TW_ASSERT(t1.size() == t2.size());
//                printf("{\n");
//                for (int i = 0; i < t1.size(); ++i)
//                {
//                    printf("Test failed at Line:%d t1[%d]:%.7f t2[%d]:%.7f\n", __LINE__, i, t1[i], i, t2[i]);
//                }
//                printf("}\n");
//            }
//            ++ntests;
//
//            float ylmCoeff[100];
//            computeCoeff<3, 9>(make_float3(0.f), v.data(), basisData.data(), rawA, ylmCoeff);
//            if (!equal(t1.begin(), t1.end(), &ylmCoeff[0], [epsilon](const float& x, const float& y)
//            {
//                return std::abs(x - y) <= epsilon;
//            }))
//            {
//                ++nfails;
//                TW_ASSERT(t1.size() == 100);
//                printf("{\n");
//                for (int i = 0; i < t1.size(); ++i)
//                {
//                    printf("Test failed at Line:%d t1[%d]:%.7f ylmCoeff(GPU)[%d]:%.7f\n", __LINE__, i, t1[i], i, ylmCoeff[i]);
//                }
//                printf("}\n");
//            }
//            ++ntests;
//        }

        {
            auto A1 = (sphToCartesian(M_PI / 2.f, M_PI / 2.f));
            auto B1 = (sphToCartesian(M_PI / 4.f, M_PI / 2.f));
            auto C1 = (sphToCartesian(M_PI / 4.f, 0.f));
            auto D1 = (sphToCartesian(M_PI / 2.f, 0));

            std::vector<float3> v{ make_float3(0.f),A1,D1,C1,B1 };
            std::vector<float> t1 = computeCoeff<4, 9>(make_float3(0.f), v, basisData, true, a);
            //computeCoeff<4>(make_float3(0.f), v, basisData, true, a);
            std::vector<float> t2{ 0.347247,-0.339578,0.236043,-0.339578,0.343355,-0.278344,-0.148677,-0.278344,
                -4.47035e-08,-0.114247,0.313703,0.0257661,-0.261895,0.0257661,2.38419e-07,0.114247,
                0,-0.113373,0.04956,0.169006,-0.0637204,0.169007,-1.19209e-07,0.113373,-0.00327934,
                -0.0622197,4.76837e-07,-0.0384729,-0.0998672,0.0666416,0.0806866,0.0666404,2.38419e-07,
                0.0384715,-0.00838393,-0.0622196,0.102581,-0.0612729,-1.90735e-06,0.00736611,-0.04067,
                -0.024226,0.0278019,-0.0242314,-9.53674e-07,-0.00736642,-0.0146675,-0.0612733,2.13087e-06,
                -0.0432224,0.105511,-0.0192096,5.72205e-06,-0.00535965,0.0132933,0.0169692,-0.0217743,
                0.0169868,1.3113e-05,0.00536156,-0.019925,-0.019206,2.86102e-06,0.0432119,9.76771e-06,
                -0.0456655,0.0437557,0.00913024,-7.15256e-07,-0.0132378,-0.0420922,0.0431198,0.0413863,
                0.0431109,6.91414e-06,0.0132626,-0.0216702,0.00912857,1.19805e-05,0.0456738,-0.000314415,
                1.13593,0.00504541,0.0256577,-0.345093,-0.0441895,0.851807,0.102097,0.293915,0.961365,
                -0.125366,-0.185883,0.0160217,0.294434,-0.0154741,-0.355438,-0.372314,0.0489883,-0.120514,-0.193359 };
            TW_ASSERT(t2.size() == 100);
            if (!equal(t1.begin(), t1.end(), t2.begin(), [epsilon](const float& x, const float& y)
            {
                return std::abs(x - y) <= epsilon;
            }))
            {
                ++nfails;
                TW_ASSERT(t1.size() == t2.size());
                printf("{\n");
                for (int i = 0; i < t1.size(); ++i)
                {
                    printf("Test failed at Line:%d t1[%d]:%.7f t2[%d]:%.7f\n", __LINE__, i, t1[i], i, t2[i]);
                }
                printf("}\n");
            }
            ++ntests;

            float ylmCoeff[100];
            computeCoeff<4, 9>(make_float3(0.f), v.data(), basisData.data(), rawA, ylmCoeff);
            if (!equal(t1.begin(), t1.end(), &ylmCoeff[0], [epsilon](const float& x, const float& y)
            {
                return std::abs(x - y) <= epsilon;
            }))
            {
                ++nfails;
                TW_ASSERT(t1.size() == 100);
                printf("{\n");
                for (int i = 0; i < t1.size(); ++i)
                {
                    printf("Test failed at Line:%d t1[%d]:%.7f ylmCoeff(GPU)[%d]:%.7f\n", __LINE__, i, t1[i], i, ylmCoeff[i]);
                }
                printf("}\n");
            }
            ++ntests;
        }

    }
    printf("\nTest coverage:%f%%(%d/%d) passed!\n", 100.f*static_cast<float>(ntests - nfails) / ntests, ntests - nfails, ntests);
}

/* Code from PBRT-v2. */
namespace PBRT
{
    constexpr float OneMinusEpsilon = 0.9999999403953552f;
    inline float VanDerCorput(uint32_t n, uint32_t scramble) {
        // Reverse bits of _n_
        n = (n << 16) | (n >> 16);
        n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
        n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
        n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
        n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
        n ^= scramble;
        return std::min(((n >> 8) & 0xffffff) / float(1 << 24), OneMinusEpsilon);
    }


    inline float Sobol2(uint32_t n, uint32_t scramble) {
        for (uint32_t v = 1 << 31; n != 0; n >>= 1, v ^= v >> 1)
            if (n & 0x1) scramble ^= v;
        return std::min(((scramble >> 8) & 0xffffff) / float(1 << 24), OneMinusEpsilon);
    }

    
    inline void Sample02(uint32_t n, const uint32_t scramble[2],
        float sample[2]) {
        sample[0] = VanDerCorput(n, scramble[0]);
        sample[1] = Sobol2(n, scramble[1]);
    }
}


void QuadLight::TestBSDFProjectionMatrix()
{
    float epsilon = 1e-5f;
    int nfails = 0;
    int ntests = 0;
    constexpr int iteration = 1000;

    auto uniformSamplingHemisphere = [](float x, float y)->float3
    {
        float z = x;
        float r = std::sqrt(std::max(0.0, 1.0 - z * z));
        float phi = 2.0 * M_PI * y;
        return make_float3(r * std::cos(phi), r * std::sin(phi), z);
    };
    auto pdfHemiSphere = []()->float {return 1.f / (2.f * M_PIf); };
    auto UniformSampleSphere =[](float u1, float u2)->float3 {
        float z = 1.f - 2.f * u1;
        float r = sqrtf(fmax(0.f, 1.f - z * z));
        float phi = 2.f * M_PIf * u2;
        float x = r * cosf(phi);
        float y = r * sinf(phi);
        return make_float3(x, y, z);
    };
    auto pdfSphere = []()->float {return 1.f / (4.f * M_PIf); };
    auto sphToCartesian = [](const float theta, const float phi)->float3
    {
        return make_float3(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
    };
    auto Eval_f = [](float3 wo, float3 wi)->float
    {
        //TW_ASSERT(wi.z > 0.f);
        //if (wi.z * wo.z < 0.f)return 0.f;
        if (wo.z < 0.f || wi.z < 0.f)return 0.f; // Single Side BRDF
        //if (wi.z < 0.f)return 0.f; // This is consistent with PBRT's SH BSDF computation
        return fabsf(wi.z) / M_PIf;
    };

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_int_distribution<unsigned int> dis_uint(0);

#define WRITE_BSDFMATRIX 0
#if 1

#if WRITE_BSDFMATRIX
    //std::vector<std::vector<float>> bsdfMatrix;

    /*std::vector<float> factorPerRow(100, 0.f);
    for (int i = 0; i < iteration; ++i)
    {
        float3 dir = uniformSamplingHemisphere(dis(gen), dis(gen));
        float ylmVector[(lmax + 1)*(lmax + 1)];
        SHEvalFast9(dir, ylmVector);
        for (int j = 0; j < 100; ++j)
        {
            factorPerRow[j] += dir.z / M_PIf * 1.f / (iteration*pdfHemiSphere())*ylmVector[j];
        }
        std::cout << "I Progress: " << i << " / " << iteration << "     \r";
        std::cout.flush();
    }*/

    // See factorPerRow[0]
#ifdef SEEfactorPerRow0
    std::vector<float> factorPerRow(100, 0.f);
    for (int i = 0; i < iteration; ++i)
    {
        float3 dir = uniformSamplingHemisphere(dis(gen), dis(gen));
        float ylmVector[(lmax + 1)*(lmax + 1)];
        SHEvalFast9(dir, ylmVector);
        for (int j = 0; j < 1; ++j)
        {
            factorPerRow[j] += dir.z / M_PIf * 1.f / (iteration*pdfHemiSphere())*ylmVector[j];
            
        }
        //std::cout << "I Progress: " << i << " / " << iteration << "     \r";
        //std::cout.flush();
    }
    std::cout << factorPerRow[0] << std::endl;
#endif
    std::vector<float3> w;
    float pbrtMatrixCompute_ylmVector[100 * iteration];
    /*uint32_t scramble[2] = { dis_uint(gen),dis_uint(gen) };*/
    uint32_t scramble[2] = { 0,0 };
    for (int wsamples = 0; wsamples < iteration; ++wsamples)
    {
        float u[2];
        PBRT::Sample02(wsamples, scramble, u);
        float3 dir = UniformSampleSphere(u[0],u[1]);
        SHEvalFast9(dir, &pbrtMatrixCompute_ylmVector[wsamples * 100]);
        w.push_back(std::move(dir));
    }

    std::vector<std::vector<float>> pbrtBSDFMatrix;
    pbrtBSDFMatrix.resize(100);
    for (auto& row : pbrtBSDFMatrix)
    {
        row.resize(100,0.f);
    }

    for (int osamples = 0; osamples < iteration; ++osamples)
    {
        float3 &wo = w[osamples];
        for (int isamples = 0; isamples < iteration; ++isamples)
        {
            float3 &wi = w[isamples];
            if (Eval_f(wo, wi) != 0.f)
            {
                for (int y = 0; y < 100; ++y)
                {
                    float *ylmVector_wi = &pbrtMatrixCompute_ylmVector[isamples * 100];
                    float *ylmVector_wo = &pbrtMatrixCompute_ylmVector[osamples * 100];

                    for (int x = 0; x < 100; ++x)
                    {
                        float factorPerRow_x_ = Eval_f(wo, wi) * 1.f / (iteration*pdfSphere())*ylmVector_wi[y];
                        pbrtBSDFMatrix[y][x] += 1.f * factorPerRow_x_ / (iteration*pdfSphere())*ylmVector_wo[x];
                    }
                }
            }
            
        }
        //std::cout << "II Progress: " << osamples << " / " << 100 << "     \r";
        //std::cout.flush();
    }

#if 0
    for (int y = 0; y < 100; ++y)
    {
        std::vector<float> bsdfMatrixRow(100, 0.f);

        for (int osamples = 0; osamples < iteration; ++osamples)
        {
            float3 wo = UniformSampleSphere(dis(gen), dis(gen));

            std::vector<float> factorPerRow(100, 0.f); // For each column
            for (int isamples = 0; isamples < iteration; ++isamples)
            {
                //float3 wi = uniformSamplingHemisphere(dis(gen), dis(gen));
                float3 wi = UniformSampleSphere(dis(gen), dis(gen));
                float ylmVector_wi[(lmax + 1)*(lmax + 1)];
                SHEvalFast9(wi, ylmVector_wi);
                if (Eval_f(wo, wi) != 0.f)
                {
                    for (int j = 0; j < 100; ++j)
                    {
                        //factorPerRow[j] += Eval_f(wo, wi) * 1.f / (iteration*pdfHemiSphere())*ylmVector_wi[y];
                        factorPerRow[j] += Eval_f(wo, wi) * 1.f / (iteration*pdfSphere())*ylmVector_wi[y];
                    }
                }
                
            }

            float ylmVector_wo[(lmax + 1)*(lmax + 1)];
            SHEvalFast9(wo, ylmVector_wo);
            for (int j = 0; j < 100; ++j)
            {
                /*result[j] += 1.f * factorPerRow[y] / (iteration*pdfSphere())*ylmVector[j];*/
                bsdfMatrixRow[j] += 1.f * factorPerRow[j] / (iteration*pdfSphere())*ylmVector_wo[j];
            }
        }

        bsdfMatrix.push_back(std::move(bsdfMatrixRow));

        std::cout << "II Progress: " << y << " / " << 100 << "     \r";
        std::cout.flush();
    }
#endif 
    std::ofstream file("diffuseBSDFMatrix.dat", std::ios_base::trunc);
    std::cout << "Trying writing" << std::endl;

    for (const auto& row : pbrtBSDFMatrix)
    {
        for (const auto& col : row)
        {
            file << col << ",";
        }
        file << std::endl;
    }
    std::cout << "End writing" << std::endl;
#endif // 

    float ylmVector[(lmax + 1)*(lmax + 1)];
    float FlmVectorRandom[(lmax + 1)*(lmax + 1)];
    SHEvalFast9(uniformSamplingHemisphere(dis(gen),dis(gen)), ylmVector);
    /* Matrix Multiplication. */
    for (int j = 0; j < (lmax + 1)*(lmax + 1); ++j)
    {
        float result = 0.0f;
        for (int i = 0; i < (lmax + 1)*(lmax + 1); ++i)
        {
#if WRITE_BSDFMATRIX
            result += pbrtBSDFMatrix[j][i] * ylmVector[i];
#else
            result += BSDFMatrix_Rawdata[j][i] * ylmVector[i];
#endif
        }
        FlmVectorRandom[j] = result;
    }

    std::vector<float> Flm_data2{ 0.886227012, -0, 1.02332675, -0, 0, -0, 0.495415896, -0, 0, -0, 0, -0, 0, -0, 0, -0, -0, 0, -0, 0, -0.110778376, 0, -0, 0, -0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0.0499271452, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, -0, 0, -0, 0, -0, 0, -0, 0, -0.0285469331, 0, -0, 0, -0, 0, -0, 0, -0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, };
    std::for_each(Flm_data2.begin(), Flm_data2.end(), [](float& flm) {flm *= 1.f / M_PIf; });
    if (!equal(Flm_data2.begin(), Flm_data2.end(), &FlmVectorRandom[0], [epsilon](float x, float y)
    {
        return (abs(x - y) <= epsilon);
    }))
    {
        for (int i = 0; i < Flm_data2.size(); ++i)
        {
            printf("(Diffuse)BSDFMatrix Test failed at Line:%d Ref[%d]:%.7f Proj[%d]:%.7f\n", __LINE__, i, Flm_data2[i], i, FlmVectorRandom[i]);
        }
        ++nfails;
    }
#if WRITE_BSDFMATRIX
    std::exit(0);
#endif

    //printf("{\n");
    //for (int i = 0; i < 100; ++i)
    //{
    //    printf("First BSFRow Ref:[%d]:%.7f PBRT:%.7f\n", i, result[i], QuadLight::BSDFMatrix_Rawdata[0][i]);
    //}
    //printf("}\n");
    //
    //// should be 0.88 =f0,0
    //for (float theta = 0.0f; theta <= M_PIf / 2.f; theta += 0.5f*M_PIf / 180.f)
    //{
    //    float ylmVector[(lmax + 1)*(lmax + 1)];
    //    SHEvalFast9(sphToCartesian(theta, 0.0f), ylmVector);
    //    double ans = 0.0;
    //    for (int j = 0; j < (lmax + 1)*(lmax + 1); ++j)
    //    {
    //        ans += result[j] * ylmVector[j];
    //        
    //    }
    //    std::cout << "theta:" << theta << "\t" << ans << std::endl;
    //}
#endif // 0



#if 0
    std::cout << std::endl;
    for (float theta = 0.0f; theta <= M_PIf / 2.f; theta += 0.5f*M_PIf / 180.f)
    {
        float ylmVector[(lmax + 1)*(lmax + 1)];
        SHEvalFast9(sphToCartesian(theta,0.0f), ylmVector);
        double ans = 0.0;
        for (int j = 0; j < (lmax + 1)*(lmax + 1); ++j)
        {
            ans += BSDFMatrix_Rawdata[0][j] * ylmVector[j];
            
        }
        std::cout << ans << std::endl;
    }
    std::exit(0);


    float3 zrefVec = make_float3(0.f, 0.f, 1.f);

    float ylmVector[(lmax + 1)*(lmax + 1)];
    SHEvalFast9(zrefVec, ylmVector);
    float FlmVector[(lmax + 1)*(lmax + 1)];
    /* Matrix Multiplication. */
    for (int j = 0; j < (lmax + 1)*(lmax + 1); ++j)
    {
        float result = 0.0f;
        for (int i = 0; i < (lmax + 1)*(lmax + 1); ++i)
        {
            result += M_PIf * BSDFMatrix_Rawdata[j][i] * ylmVector[i];
        }
        FlmVector[j] = result;
    }
    
    
    /*for (float theta = 0.0f; theta <= M_PIf / 2.f; theta += 0.5f*M_PIf / 180.f)
    {
        SHEvalFast9(sphToCartesian(theta,0.0f), ylmVector);
        double ans = 0.0;
        for (int j = 0; j < (lmax + 1)*(lmax + 1); ++j)
        {
            ans += FlmVector[j] * ylmVector[j];
            std::cout << ans << std::endl;
        }
    }*/

    
    float rawA[100][19]{ 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0.684451,0.359206,-1.29053,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0.491764,0.430504,0.232871,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
-1.66836,1.27394,0.689107,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0.330376,0.163767,-0.973178,-1.58573,-1.06532,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
-0.154277,0.466127,-0.597988,0.289095,-0.324048,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0.651874,0.53183,0.467255,0.225622,0.243216,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
-0.675183,0.736365,0.558372,0.231532,0.497581,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
1.05513,-0.712438,-1.15021,-0.780808,0.626025,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
-3.59672,-4.00147,-1.96217,-1.38707,-4.62964,1.83853,-4.19686,0,0,0,0,0,0,0,0,0,0,0,0,
-1.02631,-1.14148,-0.328964,-0.581013,-2.09467,1.84828,-1.76873,0,0,0,0,0,0,0,0,0,0,0,0,
-0.276614,0.125182,-0.74475,0.647966,-0.943961,0.578956,-0.379926,0,0,0,0,0,0,0,0,0,0,0,0,
2.8152,2.99951,1.55905,0.530353,2.94505,-1.80575,2.25959,0,0,0,0,0,0,0,0,0,0,0,0,
0.933848,2.36693,1.06564,0.138454,2.52347,-1.63423,1.77388,0,0,0,0,0,0,0,0,0,0,0,0,
-1.53162,-2.26699,-1.63944,-1.27998,-1.49923,0.802996,-1.87734,0,0,0,0,0,0,0,0,0,0,0,0,
2.20369,2.65718,2.3773,1.47866,2.61718,-0.154934,1.01416,0,0,0,0,0,0,0,0,0,0,0,0,
0.386892,-0.234194,0.425573,-2.07542,-0.99878,0.087789,-1.49088,-2.10113,-2.21767,0,0,0,0,0,0,0,0,0,0,
0.670532,1.11525,0.0213808,0.213706,-0.775345,-0.870597,-0.243734,-1.07331,-0.921999,0,0,0,0,0,0,0,0,0,0,
3.34363,2.88333,1.5528,-0.690988,-1.21196,1.38443,0.248891,-3.33599,-2.18742,0,0,0,0,0,0,0,0,0,0,
-4.43576,-3.10679,-1.50732,1.14355,3.05976,-1.32176,-0.279068,6.36581,4.37729,0,0,0,0,0,0,0,0,0,0,
0.801925,-0.589618,-0.498002,-1.99003,-1.17152,0.0225333,-1.83288,-2.48201,-0.800297,0,0,0,0,0,0,0,0,0,0,
-0.0741623,0.463091,-0.490407,-1.33794,-1.19302,-0.0671063,-0.991982,-2.579,-1.40303,0,0,0,0,0,0,0,0,0,0,
2.30898,2.59505,-0.245139,-0.143466,-1.28047,0.0225859,0.402924,-3.23301,-2.68613,0,0,0,0,0,0,0,0,0,0,
2.45217,1.30712,0.624695,-1.20453,-0.992719,0.8808,-1.4293,-4.16149,-2.09886,0,0,0,0,0,0,0,0,0,0,
-3.29553,-2.75451,-0.707257,-3.38379,1.15948,-1.85432,-2.29433,0.282948,-0.104368,0,0,0,0,0,0,0,0,0,0,
0.901645,0.385552,0.881322,0.762582,0.0627355,0.500188,0.815467,0.0169501,0.29148,1.15498,0.629604,0,0,0,0,0,0,0,0,
1.48735,0.327641,1.03651,0.605442,1.299,1.28978,0.0118243,0.774944,-1.53547,1.58578,0.857565,0,0,0,0,0,0,0,0,
-0.139832,-0.767387,0.690406,-0.647648,-1.73758,-0.953175,-0.415786,0.357295,0.342909,-0.860505,-1.00317,0,0,0,0,0,0,0,0,
-1.30254,-0.0299509,-0.923929,-1.09153,-0.484701,-1.1409,-1.01218,0.732852,0.567873,-1.39764,-0.58814,0,0,0,0,0,0,0,0,
1.3415,0.479091,0.816822,0.0875707,0.305267,0.492711,-0.38267,-0.252676,-0.294921,1.50257,-0.944971,0,0,0,0,0,0,0,0,
-3.18545,-3.58245,-1.57054,-3.91511,-4.13576,-3.40871,-3.03685,1.47586,1.64368,-3.67964,-3.48077,0,0,0,0,0,0,0,0,
-1.49758,-0.795641,-0.477492,-0.9121,-0.961176,-0.978628,-0.587473,0.514521,-0.10312,-0.437121,-0.99984,0,0,0,0,0,0,0,0,
2.04199,1.20223,0.339812,1.55051,1.45889,0.618371,1.08261,-0.0765456,-1.4675,2.22075,0.53648,0,0,0,0,0,0,0,0,
0.0453747,-0.721773,0.127358,0.344248,0.0228017,-0.923741,-0.898898,0.594424,0.0211068,0.407756,-1.21018,0,0,0,0,0,0,0,0,
0.369802,-0.429225,0.962211,-0.428983,1.22736,0.0473707,0.177308,0.884197,-1.56972,1.2173,0.321895,0,0,0,0,0,0,0,0,
0.838529,1.88589,0.571345,1.3097,1.89246,2.36037,2.18634,-0.36959,-0.305823,0.624519,1.9482,0,0,0,0,0,0,0,0,
0.130954,-0.516826,-0.0390864,-0.0555911,-0.483086,0.549499,-0.425825,0.274285,-0.624874,0.704007,0.68713,-0.507504,0.16394,0,0,0,0,0,0,
-0.319237,0.457155,0.0503858,0.0374498,-0.0900405,-0.0600437,-0.0621607,1.7398,0.379183,-0.33379,-0.700205,-1.53056,0.827214,0,0,0,0,0,0,
-1.42509,0.341737,-2.29356,0.486814,3.32227,0.521771,0.22662,2.09383,3.62748,1.29747,0.113476,-4.24123,-1.57304,0,0,0,0,0,0,
0.110626,0.570583,0.681116,0.393754,0.0764495,0.47705,0.0317332,1.01107,0.132282,-0.207397,-0.607639,-0.912909,-0.0276892,0,0,0,0,0,0,
-0.871339,0.953107,-1.23588,0.951312,2.71071,-0.676999,0.417402,1.64249,2.11142,0.667482,-0.64461,-2.83809,-0.224166,0,0,0,0,0,0,
1.02862,-0.207223,1.93275,-0.537461,-2.93969,-1.08259,0.74633,-3.07593,-2.64397,-2.02553,0.324457,3.92295,1.43658,0,0,0,0,0,0,
-0.596174,0.139976,-1.07057,0.92516,2.08283,-0.639657,1.00171,0.956261,1.18423,-0.420206,-1.38542,-2.30077,-0.244787,0,0,0,0,0,0,
-2.21381,0.563199,-2.75559,-0.108763,3.37868,-0.351411,1.51645,0.93244,3.01257,1.26995,-0.682257,-2.69755,-2.34132,0,0,0,0,0,0,
-0.803653,0.486332,-2.71571,0.00486127,3.2686,-0.556502,1.78242,0.779195,3.27404,1.49038,-0.610542,-2.35778,-2.57451,0,0,0,0,0,0,
0.847329,-0.834072,2.37916,1.24146,-1.98473,-0.52079,-0.48859,-0.115073,-1.32164,-0.886867,-0.0804312,1.37765,1.79765,0,0,0,0,0,0,
-2.44787,0.39709,-3.7761,0.015054,6.59958,0.329229,0.798877,3.76058,5.89172,3.02555,-0.532234,-6.77026,-3.27629,0,0,0,0,0,0,
-0.658281,0.529111,-1.75975,-0.618552,1.90219,0.644707,0.912778,0.531902,1.53514,1.07795,0.382837,-0.831314,-1.17779,0,0,0,0,0,0,
0.90254,-0.00195845,0.416233,-0.534067,-1.33826,-0.406687,0.157748,-0.934546,-1.60637,-1.04187,0.646452,1.97913,0.0958256,0,0,0,0,0,0,
1.41138,-1.49318,-0.267813,0.118793,0.0667789,1.1114,1.21832,-0.364957,0.996974,-1.37857,-0.590952,-0.990955,1.44545,0.300644,0.521915,0,0,0,0,
-0.980688,0.364658,0.617238,0.970392,-0.845702,0.32811,-0.286463,0.866263,-0.592107,0.645209,-0.224906,0.547207,-0.936674,-0.788088,-0.536917,0,0,0,0,
-0.168275,0.53673,-0.365787,-1.25915,0.0107433,-0.413228,-0.0320747,-0.366879,-0.353566,0.0366365,0.302125,0.738732,0.35326,0.0523419,-0.221827,0,0,0,0,
-0.584207,0.430717,-0.130176,-0.274328,0.382646,-0.992711,0.0961735,-0.261535,0.0946536,0.772936,-0.148429,0.808034,-0.98955,0.367983,-0.497198,0,0,0,0,
-0.655831,0.734315,0.474604,-0.242935,-0.174109,0.226868,0.216102,0.234184,0.0758351,0.312709,0.304648,0.691801,0.132165,0.248373,-0.771094,0,0,0,0,
0.0795894,1.07338,-0.292855,-1.0238,0.581984,-0.873444,-0.632578,-0.599404,-0.774384,0.293745,0.164963,0.878368,0.574305,0.0938578,1.00816,0,0,0,0,
2.66202,-2.02176,0.195195,0.551417,0.618997,1.44304,2.92024,-0.450233,1.02399,-3.37295,-0.106694,-1.96011,1.64395,0.940143,0.462851,0,0,0,0,
-0.105968,-1.25284,0.864732,2.02985,-0.311623,1.51714,0.530248,-0.186852,-0.190595,-1.46531,-0.509711,-0.848307,-0.040913,0.517662,-1.19258,0,0,0,0,
-1.87156,1.57585,0.171384,-1.07235,0.0795015,-1.88109,-1.77911,-0.466125,-0.225306,2.0612,0.746487,1.15275,-0.836341,-1.0385,-0.0588058,0,0,0,0,
-0.231926,1.37785,-0.192581,-1.36978,-0.125444,-1.93895,-1.58626,-0.52281,-0.00773775,1.94619,1.14006,1.36407,-0.205571,-0.710586,0.220972,0,0,0,0,
0.33655,-0.574124,-0.732785,-0.764633,-0.384849,-0.0135144,0.504584,0.0967235,0.278052,-0.246882,0.53561,0.588689,1.36747,0.94626,0.718744,0,0,0,0,
-1.47714,1.41647,0.480085,-1.61308,0.495642,-1.87418,-1.98503,0.0255505,-1.03677,2.6324,-0.0743271,1.9304,-1.19671,-0.655958,0.10449,0,0,0,0,
0.804917,0.149715,-1.2958,-1.7613,1.1501,-0.56573,0.34409,-0.14935,0.177333,0.810151,0.991728,0.996871,0.634889,-0.423213,0.898464,0,0,0,0,
1.26666,-0.647383,-0.70616,-0.628073,0.550705,-0.287921,1.01286,0.604584,0.565855,-0.58263,0.00775118,0.532163,1.49201,0.565321,0.325189,0,0,0,0,
1.11794,-1.13155,-0.282903,0.617965,-0.0717177,1.57804,1.29605,0.671933,0.738547,-2.33639,-0.274473,-0.262591,1.11294,0.807418,0.257607,0,0,0,0,
-0.377914,-0.225818,-0.429096,0.987763,0.193171,0.714889,-0.666905,-0.929931,-0.588023,-0.0435213,0.465649,1.11136,0.810951,-0.312313,-0.0704491,-0.539556,0.159737,0,0,
-2.32767,1.34197,0.36476,0.48528,-2.29735,-0.107763,-0.157041,2.38077,-1.10103,0.9694,-0.118429,-1.97709,-0.678479,2.7164,1.27078,1.77615,-1.76046,0,0,
-0.506219,0.36112,-0.772923,0.637045,-1.9832,-1.24995,0.187018,1.29697,0.753882,-0.780084,-0.108084,-1.1623,0.228745,0.782582,0.190188,1.46219,-1.24104,0,0,
-0.261945,-0.134153,-0.861752,-0.32569,-1.08022,-0.635845,0.108112,0.980172,0.272034,-0.176725,-0.170833,-0.771681,-0.31043,0.87253,0.529705,1.48879,-0.608076,0,0,
-0.652701,0.343429,-0.860292,1.39669,-1.21608,-0.217333,0.624246,0.513427,-0.448237,0.419166,-0.201683,-0.834232,0.63071,0.541281,-0.198191,1.73257,-1.33826,0,0,
-0.143953,1.26514,0.252472,-0.406242,-0.671232,-0.463832,-0.187793,-0.0536602,0.755577,0.0418132,-0.613325,0.185727,-0.582403,0.168035,-0.114024,0.891265,-0.929824,0,0,
2.01231,-1.57626,-0.800351,0.856102,2.55656,1.95036,0.395023,-3.5701,0.742491,-0.329472,-0.0741527,2.63708,0.83174,-2.53329,-1.54782,-1.52773,1.88953,0,0,
-1.01344,0.222599,0.0148034,0.204784,-0.807036,0.182928,-0.523892,1.60103,-0.937233,0.743981,-0.674546,-0.0547825,-0.667966,1.43427,0.187707,0.861661,-0.698571,0,0,
-0.496894,0.258762,0.294853,0.568549,-0.587026,-0.761855,-0.250601,0.208739,0.283704,0.0268767,0.470202,-0.815505,-0.244517,-0.188146,0.19042,0.823236,-0.0702735,0,0,
-0.400609,-0.530642,-0.0301947,-0.01536,0.655302,-0.239775,0.572657,-0.241502,0.26003,-0.401339,0.12529,-0.0177895,0.198477,0.419563,-0.149376,0.522912,-0.248691,0,0,
3.02225,-1.04811,0.382163,-0.814561,2.24272,-0.140416,0.693969,-2.79055,1.04339,-0.215989,-0.0298695,1.39015,0.197856,-1.48015,-1.53468,-1.01782,1.39767,0,0,
3.50719,-1.32831,0.82969,-1.76717,3.12907,0.64418,0.485468,-4.61659,1.32673,0.264079,-0.585126,2.83722,0.276637,-3.21029,-2.21937,-3.05265,2.95631,0,0,
-0.687074,-0.364741,0.182821,0.36512,-0.775456,0.474574,-0.0408075,0.633208,-0.0875694,-0.0766544,-0.14942,-0.318291,0.280064,0.234616,0.977562,0.441624,-0.662151,0,0,
0.0898836,0.0633354,-1.49628,1.36927,-0.473625,0.208693,-0.458777,-0.25294,0.156376,-0.349746,0.342975,0.425743,-0.28819,-0.386056,-1.10283,0.639174,-1.61187,0,0,
0.683439,-0.256975,0.853269,-1.25306,0.689052,-0.205386,-0.250166,-0.0950969,0.375352,0.789996,-0.948669,-0.12304,-0.222474,0.474984,1.02151,-1.0293,1.25793,0,0,
-1.32926,0.386258,-0.413633,0.452075,-1.29237,0.123832,-0.775261,2.05353,-0.438136,0.371959,-0.196067,-1.72413,0.537271,1.33648,0.961259,0.902856,-0.412672,0,0,
-2.26639,1.17612,0.583651,0.185289,-1.79347,-0.720326,-0.414004,2.51146,-1.16678,-0.257522,-0.307256,-2.13279,-0.37188,1.88216,1.74421,1.33016,-1.07328,0,0,
18392,-38257.4,37806.7,-21284.7,32237,-62523.1,-41790.9,110133,92665.1,4731.58,92667,110134,-41792.6,-62522.7,32237,-21286.5,37806.1,-38255.3,18392.3,
-75.8607,157.288,-155.575,87.4328,-132.743,256.299,172.322,-453.639,-381.202,-18.6079,-381.246,-453.673,172.329,256.294,-132.735,87.4507,-155.557,157.267,-75.8604,
-2503.15,5205.17,-5145.71,2897.2,-4387.52,8508.29,5688.74,-14988.2,-12612.2,-644.042,-12610.3,-14988.5,5687.31,8508.34,-4386.49,2896.59,-5145.41,5206.82,-2502.99,
-12750.3,26524.9,-26210.7,14755.7,-22349.6,43346.7,28971.1,-76353.7,-64242,-3280.42,-64245.4,-76354.1,28975.1,43346,-22348.9,14758.9,-26210.4,26520.3,-12751.3,
3672.85,-7639.39,7547.34,-4249.92,6436.45,-12483.9,-8343.34,21989,18501.1,944.875,18501.9,21989.2,-8344.08,-12483.2,6436.8,-4250.16,7547.37,-7638.16,3672.5,
-14546,30256.1,-29899.7,16833.2,-25494.3,49446.8,33050.4,-87099,-73285,-3742.06,-73285.7,-87100.7,33052.3,49446.3,-25495.2,16834.4,-29898.9,30254.3,-14545.2,
2599.25,-5405.28,5340.93,-3007.28,4554.29,-8833.02,-5905.07,15560.5,13091.6,666.984,13092.1,15560.6,-5905.66,-8832.45,4554.75,-3007.63,5340.95,-5405.05,2598.8,
9713.83,-20206.3,19968.3,-11241.4,17025.7,-33021.3,-22071.7,58167.3,48941,2499.18,48942.9,58167.7,-22073.9,-33020.6,17025.1,-11242.7,19967.2,-20203.1,9713.36,
-15217.6,31652.3,-31280,17611,-26671,51729.2,34576,-91119.5,-76667,-3914.73,-76668.9,-91120.6,34578.1,51727.6,-26671.3,17610.6,-31278.9,31650.4,-15216.8,
22110.1,-45994.4,45450.8,-25586.7,38754.6,-75164.7,-50238.3,132400,111398,5688.3,111403,132400,-50244.2,-75162.2,38754.3,-25591.2,45448.8,-45987.3,22111.3,
6281,-13066.7,12911.9,-7269.6,11010.4,-21354.4,-14272,37613.1,31647.3,1616.11,31648.1,37613.6,-14272.9,-21353.7,11010.5,-7269.43,12911.4,-13066,6280.7,
9762.08,-20306.4,20065.7,-11296.4,17110.4,-33184.5,-22179.3,58452.4,49180.7,2511.16,49182.5,58452.7,-22181.6,-33183.8,17109.8,-11297.7,20064.6,-20303.1,9761.61,
-6209.93,12916.3,-12764.4,7185.95,-10883.3,21110,14108.8,-37183.1,-31285.4,-1598.15,-31286.7,-37183.3,14110.2,21108.6,-10884.4,7186.79,-12764.4,12915.8,-6208.84,
-70.5743,147.913,-146.15,82.1403,-125.439,243.072,161.074,-425.993,-358.776,-19.0758,-358.764,-426.003,161.08,243.066,-125.446,82.1452,-146.143,147.9,-70.5705,
9020.58,-18763.9,18542.3,-10439,15809.9,-30664.7,-20495.9,54014.2,45446.5,2320.56,45448.5,54014.7,-20497.7,-30663,15810.7,-10439.6,18542.3,-18760.9,9019.74,
12565.3,-26138.9,25829.3,-14540.3,22024.7,-42715.6,-28550.8,75243.1,63307.8,3232.64,63311.2,75243.5,-28554.8,-42715,22024,-14543.4,25829,-26134.3,12566.3,
-1062.07,2209.78,-2182.31,1229.07,-1862.27,3611.85,2412.59,-6359.96,-5351.25,-273.013,-5350.44,-6360.09,2411.99,3611.87,-1861.83,1228.81,-2182.18,2210.48,-1062.01,
7764.91,-16152.3,15962.1,-8985.76,13610.3,-26396.8,-17643.6,46496.1,39121.1,1997.65,39123.3,46497.6,-17644.3,-26395.7,13609.6,-8987.13,15960.5,-16150.2,7764.96,
-7382.98,15356.7,-15175.6,8543.61,-12939.8,25096.9,16775.5,-44208.7,-37195.9,-1899.55,-37196.7,-44209,16776.2,25096.8,-12939.8,8544.31,-15175.3,15355.8,-7383.1 };

    std::vector<std::vector<float>> a;
    for (int i = 0; i < 100; ++i)
    {
        std::vector<float> row;
        for (int j = 0; j < 19; ++j)
        {
            row.push_back(rawA[i][j]);
        }
        a.push_back(std::move(row));
    }
    std::vector<float3> basisData{ make_float3(0.320145, 0.000000, 0.947368),
                               make_float3(-0.397673, -0.364301, 0.842105),
                               make_float3(0.059105, 0.673476, 0.736842),
                               make_float3(0.471729, -0.615288, 0.631579),
                               make_float3(-0.837291, 0.148105, 0.526316),
                               make_float3(0.765317, 0.486832, 0.421053),
                               make_float3(-0.246321, -0.916299, 0.315789),
                               make_float3(-0.450576, 0.867560, 0.210526),
                               make_float3(0.934103, -0.341132, 0.105263),
                               make_float3(-0.924346, -0.381556, 0.000000),
                               make_float3(0.421492, 0.900702, -0.105263),
                               make_float3(0.292575, -0.932780, -0.210526),
                               make_float3(-0.820937, 0.475752, -0.315789),
                               make_float3(0.885880, 0.194759, -0.421053),
                               make_float3(-0.489028, -0.695588, -0.526316),
                               make_float3(-0.099635, 0.768883, -0.631579),
                               make_float3(0.516953, -0.435687, -0.736842),
                               make_float3(-0.538853, -0.022282, -0.842105),
                               make_float3(0.226929, 0.225824, -0.947368), };
    {
        // Well projected on hemisphere
        auto A1 = make_float3(0.0f, 1.0f, 0.0f);
        auto B1 = make_float3(-1.0f, 0.0f, 0.0f);
        auto C1 = make_float3(0.0f, -1.0f, 0.0f);
        auto D1 = make_float3(1.0f, 0.0f, 0.0f);

        
        std::vector<float3> v{ make_float3(0.f),A1,B1,C1,D1 };
        //std::vector<float3> v{ make_float3(0.f),A1,B1,D1 };
        std::vector<float> t1 = computeCoeff<4, 9>(make_float3(0.f), v, basisData, true, a);

        printf("{\n");
        for (int i = 0; i < t1.size(); ++i)
        {
            printf("First BSFRow Ref:t1[%d]:%.7f PBRT:%.7f\n", i, t1[i]*(1.f/(2.f*sqrt(M_PIf))), QuadLight::BSDFMatrix_Rawdata[0][i]);
        }
        printf("}\n");

        
    }



    std::vector<float> Flm_data{ 0.886227012, -0, 1.02332675, -0, 0, -0, 0.495415896, -0, 0, -0, 0, -0, 0, -0, 0, -0, -0, 0, -0, 0, -0.110778376, 0, -0, 0, -0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0.0499271452, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, -0, 0, -0, 0, -0, 0, -0, 0, -0.0285469331, 0, -0, 0, -0, 0, -0, 0, -0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, };
    if (!equal(Flm_data.begin(), Flm_data.end(), &FlmVector[0], [epsilon](float x, float y)
    {
        return (abs(x - y) <= epsilon);
    }))
    {
        for (int i = 0; i < Flm_data.size(); ++i)
        {
            printf("(Diffuse)BSDFMatrix Test failed at Line:%d Ref[%d]:%.7f Proj[%d]:%.7f\n", __LINE__, i, Flm_data[i], i, FlmVector[i]);
        }
        ++nfails;
    }

    printf("\nTest coverage:%f%%(%d/%d) passed!\n", 100.f*static_cast<float>(ntests - nfails) / ntests, ntests - nfails, ntests);
#endif // 
}

void QuadLight::TestZHRecurrence()
{
    //TestSolidAngle();
    TestBSDFProjectionMatrix();
    //TestYlmCoeff();
    auto sphToCartesian = [](const float theta, const float phi)->float3
    {
        return make_float3(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
    };

    std::vector<float3> basisData{ make_float3(0.6, 0, 0.8),
                                   make_float3(-0.67581, -0.619097, 0.4),
                                   make_float3(0.0874255, 0.996171, 0),
                                   make_float3(0.557643, -0.727347, -0.4),
                                   make_float3(-0.590828, 0.104509, -0.8), };

    std::vector<std::vector<float>> a{ {1},
                                           {0.04762, -0.0952401, -1.06303},
                                           {0.843045, 0.813911, 0.505827},
                                           {-0.542607, 1.08521, 0.674436},
                                                   {2.61289, -0.196102, 0.056974, -1.11255, -3.29064},
                                                   {-4.46838, 0.540528, 0.0802047, -0.152141, 4.77508},
                                                   {-3.36974, -6.50662, -1.43347, -6.50662, -3.36977},
                                                   {-2.15306, -2.18249, -0.913913, -2.24328, -1.34185},
                                                   {2.43791, 3.78023, -0.322086, 3.61812, 1.39367} };

    // Well projected on hemisphere
    /*auto A1 = make_float3(0.0f, 1.0f, 0.0f);
    auto B1 = make_float3(-1.0f, 0.0f, 0.0f);
    auto C1 = make_float3(0.0f, -1.0f, 0.0f);
    auto D1 = make_float3(1.0f, 0.0f, 0.0f);*/
    auto A1 = (sphToCartesian(M_PI / 2.f, M_PI / 2.f));
    auto B1 = (sphToCartesian(M_PI / 4.f, M_PI / 2.f));
    auto C1 = (sphToCartesian(M_PI / 4.f, 0.f));
    auto D1 = (sphToCartesian(M_PI / 2.f, 0));

    //std::vector<float3> v{ make_float3(0.f),A1,B1,C1,D1 };
    std::vector<float3> v{ make_float3(0.f),A1,C1,D1 };
    //std::vector<float3> v{ make_float3(0.f),A1,B1,D1 };
    computeCoeff<3,2>(make_float3(0.f), v, basisData, true, a);
    //computeCoeff<4>(make_float3(0.f), v, basisData, true, a);

#if 0
    auto uniformSamplingHemisphere = [](float x, float y)->float3
    {
        float z = x;
        float r = std::sqrt(std::max(0.0, 1.0 - z * z));
        float phi = 2.0 * M_PI * y;
        return make_float3(r * std::cos(phi), r * std::sin(phi), z);
    };

    auto pdfHemisphere = []()->float {return 1.f / (2.f*M_PI); };
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    constexpr int iteration = 10000000;

    //std::vector<float3> testData{ make_float3(0.f,1.f,1.f) };
    //testData[0] = TwUtil::safe_normalize(testData[0]);
    for (const auto& wo : basisData)
    {
        double result = 0.0;
        for (int i = 1; i <= iteration; ++i)
        {
            auto wi = uniformSamplingHemisphere(dis(gen), dis(gen));
            result += (sqrt(5.0 / (4.0*M_PI)) * 0.5 * (3.0*(dot(wo, wi)*dot(wo, wi)) - 1.0)) / pdfHemisphere();
            //result += (sqrt(3.0 / (4.0*M_PI))) * dot(wo,wi) / pdfHemisphere();
        }
        result = result / iteration;

        std::cout.precision(dbl::max_digits10);
        std::cout << result << std::endl;
    }
#endif // 0

}

bool QuadLight::TestZHIntegral(int order, const std::vector<optix::float3>& lobeDirections, int maximumIteration)
{
    TW_ASSERT(false);
    return false;
    TW_ASSERT(order == 2 || order == 3);
    #if  0
// Well projected on hemisphere
    auto A1 = make_float3(0.0f, 1.0f, 0.0f);
    auto B1 = make_float3(-1.0f, 0.0f, 0.0f);
    auto C1 = make_float3(0.0f, -1.0f, 0.0f);
    auto D1 = make_float3(1.0f, 0.0f, 0.0f);

    std::vector<float3> v{ make_float3(0.f),A1,B1,C1,D1 };
    computeCoeff<4>(make_float3(0.f), v, lobeDirections, true);


    auto sphToCartesian = [](const float theta, const float phi)->float3
    {
        return make_float3(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
    };

    auto uniformSamplingHemisphere = [](float x, float y)->float3
    {
        float z = x;
        float r = std::sqrt(std::max(0.0, 1.0 - z * z));
        float phi = 2.0 * M_PI * y;
        return make_float3(r * std::cos(phi), r * std::sin(phi), z);
    };

    auto pdfHemisphere = []()->float {return 1.f / (2.f*M_PI); };
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);

    //std::vector<float3> testData{ make_float3(0.f,1.f,1.f) };
    //testData[0] = TwUtil::safe_normalize(testData[0]);
    for (const auto& wo : lobeDirections)
    {
        double result = 0.0;
        for (int i = 1; i <= maximumIteration; ++i)
        {
            auto wi = uniformSamplingHemisphere(dis(gen), dis(gen));
            result += (sqrt(5.0 / (4.0*M_PI)) * 0.5 * (3.0*(dot(wo, wi)*dot(wo, wi)) - 1.0)) / pdfHemisphere();
            //result += (sqrt(3.0 / (4.0*M_PI))) * dot(wo,wi) / pdfHemisphere();
        }
        result = result / maximumIteration;

        std::cout.precision(dbl::max_digits10);
        std::cout << result << std::endl;
    }
#endif // 0
}

bool QuadLight::TestDiffuseFlmVector_Order3(const std::vector<float>& flmVector, int maximumIteration)
{
    /* Sampling functions. */
    auto uniformSamplingHemisphere = [](float x, float y)->CLTest::CommonStruct::QuadLight_float3
    {
        float z = x;
        float r = std::sqrt(std::max(0.f, 1.f - z * z));
        float phi = 2 * M_PI * y;
        return CLTest::CommonStruct::make_QuadLight_float3(r * std::cos(phi), r * std::sin(phi), z);
    };

    auto pdfHemisphere = []()->float {return 1.f / (2.f*M_PI); };

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);

    constexpr int order = 2;
    Eigen::VectorXf ylmRes(9);
    ylmRes << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    for (int i = 1; i <= maximumIteration; ++i)
    {
        Eigen::VectorXf ylm(9);
        ylm << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

        CLTest::CommonStruct::QuadLight_float3 sample = uniformSamplingHemisphere(dis(gen), dis(gen));
        SHEvalFast(sample, order, ylm);
        assert(ylm.size() == (order + 1)*(order + 1));
        ylmRes += ylm * sample.z / pdfHemisphere();
    }
    ylmRes *= (1.0 / maximumIteration);

    int numFailures = 0;
    for (int i = 0; i < ylmRes.size(); ++i)
    {
        if (std::abs(ylmRes[i] - flmVector[i]) >= 1e-5f)
        {
            ++numFailures;
            std::cout << "Failed Test ylmRes[i]:" << ylmRes[i] << "  flmVector[i]:" << flmVector[i] << std::endl;
        }
    }
    std::cout << "Num of failed tests:" << numFailures << std::endl;
    return numFailures == 0;
}

void QuadLight::TestLegendreP(float epsilon)
{
    int nfails = 0;
    int ntests = 0;
#if _HAS_CXX17
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i <= 10; ++i)
    {
        if (std::abs(std::legendref(i, 0.0f) - LegendreP(i, 0.0f) >= epsilon))
        {
            ++nfails;
            printf("Test failed at Line:i=%d std:%f myP:%f\n", __LINE__, i, std::legendref(i, 0.0f), LegendreP(i, 0.0f));
        }
        float x = dis(gen);
        if (std::abs(std::legendref(i, x) - LegendreP(i, x) >= epsilon))
        {
            ++nfails;
            printf("Test failed at Line:i=%d std:%f myP:%f\n", __LINE__, i, std::legendref(i, x), LegendreP(i, x));
        }
        ++ntests;
    }
    printf("\nLegender Polynomial Test coverage:%f%%(%d/%d) passed!\n", 100.f*static_cast<float>(ntests - nfails) / ntests, ntests - nfails, ntests);
#endif
}

void QuadLight::initializeAreaLight(optix::Context& context)
{
#define SHIntegration_Order2 0
#define SHIntegration_Order9 1

#if SHIntegration_Order9
    constexpr size_t lmax = 9; // N=10 order SH, lmax = 9, l goes from [0,9]
    constexpr size_t AProws = (lmax + 1)*(lmax + 1);

    TW_ASSERT(AProws == 100);

    /* Basis Vector. */
    optix::Buffer areaLightBasisVectorBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 2 * lmax + 1);
    std::vector<float3> basisData{ make_float3(0.320145, 0.000000, 0.947368),
                                   make_float3(-0.397673, -0.364301, 0.842105),
                                   make_float3(0.059105, 0.673476, 0.736842),
                                   make_float3(0.471729, -0.615288, 0.631579),
                                   make_float3(-0.837291, 0.148105, 0.526316),
                                   make_float3(0.765317, 0.486832, 0.421053),
                                   make_float3(-0.246321, -0.916299, 0.315789),
                                   make_float3(-0.450576, 0.867560, 0.210526),
                                   make_float3(0.934103, -0.341132, 0.105263),
                                   make_float3(-0.924346, -0.381556, 0.000000),
                                   make_float3(0.421492, 0.900702, -0.105263),
                                   make_float3(0.292575, -0.932780, -0.210526),
                                   make_float3(-0.820937, 0.475752, -0.315789),
                                   make_float3(0.885880, 0.194759, -0.421053),
                                   make_float3(-0.489028, -0.695588, -0.526316),
                                   make_float3(-0.099635, 0.768883, -0.631579),
                                   make_float3(0.516953, -0.435687, -0.736842),
                                   make_float3(-0.538853, -0.022282, -0.842105),
                                   make_float3(0.226929, 0.225824, -0.947368), };
    TW_ASSERT(basisData.size() == 2 * lmax + 1 && basisData.size() == 19);
    void *areaLightBasisVectorBufferData = areaLightBasisVectorBuffer->map();
    memcpy(areaLightBasisVectorBufferData, basisData.data(), sizeof(float3)*basisData.size());
    areaLightBasisVectorBuffer->unmap();
    context["areaLightBasisVector"]->setBuffer(areaLightBasisVectorBuffer);

    /* Flm diffuse Vector. */
#if 0
    // Randomly multiplied BSDFMatrix with Ylm.
    std::vector<float> Flm_data{ 0.268985,0.000214693,0.336971,-0.00179676,-0.00377481,0.000431436,0.178585,-0.00309102,0.00110261,0.000244611,-0.00270744,-0.000907655,0.00505085,-0.00282868,0.00296516,-0.00409108,-0.00139595,-0.000243531,0.000206071,-0.00431938,-0.0337186,-0.0016056,0.00408457,-0.00476357,0.00109925,-5.56032e-05,-0.00203488,-0.00167503,0.00396617,-0.00848528,0.00180117,0.00132287,0.00390038,-0.00464888,0.00149478,0.00307399,0.00216639,-0.00166132,-0.00221808,-0.0033795,0.00494139,-0.00662686,0.0164715,0.00305539,0.00112277,-0.00355957,0.000907865,0.00509134,0.00189163,0.000922727,0.00271026,-0.00314071,-0.0030905,-0.00420886,0.0039766,-0.00101252,0.000895643,0.00350988,-0.00303527,-0.00366246,-0.000299833,0.00525,0.00204088,0.0034147,-0.00212067,0.00185999,0.00212068,-0.00184453,-0.00372554,-0.00204841,0.00387355,0.00441764,-0.00851125,0.00168139,-0.00681009,-0.00414122,-0.00158837,0.00210927,0.00215199,0.00537809,0.00213542,-0.000394887,-0.00258227,0.00287402,0.000434673,0.00139832,-0.00323931,0.0033196,0.00479884,0.00683564,-0.00169632,
        -0.000511877,-0.00804228,-0.00350908,-0.00234572,3.40814e-05,0.000449437,0.002933,0.00289487,-0.000556304, };
#else
    std::vector<float> Flm_data{ 0.886227012, -0, 1.02332675, -0, 0, -0, 0.495415896, -0, 0, -0, 0, -0, 0, -0, 0, -0, -0, 0, -0, 0, -0.110778376, 0, -0, 0, -0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0.0499271452, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, -0, 0, -0, 0, -0, 0, -0, 0, -0.0285469331, 0, -0, 0, -0, 0, -0, 0, -0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, 0, -0, };
    // Add 1.f/M_PI
    std::for_each(Flm_data.begin(), Flm_data.end(), [](float& v) {v *= 1.f / M_PIf; });
#endif
    TW_ASSERT(Flm_data.size() == 100);
    //TW_ASSERT(TestDiffuseFlmVector_Order3(Flm_data, 100000));

    optix::Buffer areaLightFlmVectorBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, AProws);
    void *areaLightFlmVectorBufferData = areaLightFlmVectorBuffer->map();
    memcpy(areaLightFlmVectorBufferData, Flm_data.data(), sizeof(float)*Flm_data.size());
    areaLightFlmVectorBuffer->unmap();

    context["areaLightFlmVector"]->setBuffer(areaLightFlmVectorBuffer);

    std::vector<std::vector<float>> BSDFMatrix_data;
    for (int i = 0; i < (lmax + 1)*(lmax + 1); ++i)
    {
        std::vector<float> row;
        for (int j = 0; j < (lmax + 1)*(lmax + 1); ++j)
        {
            row.push_back(BSDFMatrix_Rawdata[i][j]);
        }
        BSDFMatrix_data.push_back(std::move(row));
    }

    TW_ASSERT(BSDFMatrix_data.size() == (lmax + 1)*(lmax + 1));
    optix::Buffer BSDFMatrixBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, (lmax + 1)*(lmax + 1), (lmax + 1)*(lmax + 1));
    float *BSDFMatrixBufferData = static_cast<float *>(BSDFMatrixBuffer->map());
    for (int i = 0; i < BSDFMatrix_data.size(); ++i)
    {
        memcpy(BSDFMatrixBufferData+i * (lmax + 1)*(lmax + 1), BSDFMatrix_data[i].data(), sizeof(float)*(lmax + 1)*(lmax + 1));
    }
    
    BSDFMatrixBuffer->unmap();
    context["BSDFMatrix"]->setBuffer(BSDFMatrixBuffer);


#if 0
    /* Alpha coeff. */
    constexpr size_t rawArow = (lmax+1)*(lmax+1);
    constexpr size_t rawAcol = (2*lmax+1);
    float rawA[100][19]{ 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0.684451,0.359206,-1.29053,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0.491764,0.430504,0.232871,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
-1.66836,1.27394,0.689107,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0.330376,0.163767,-0.973178,-1.58573,-1.06532,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
-0.154277,0.466127,-0.597988,0.289095,-0.324048,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0.651874,0.53183,0.467255,0.225622,0.243216,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
-0.675183,0.736365,0.558372,0.231532,0.497581,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
1.05513,-0.712438,-1.15021,-0.780808,0.626025,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
-3.59672,-4.00147,-1.96217,-1.38707,-4.62964,1.83853,-4.19686,0,0,0,0,0,0,0,0,0,0,0,0,
-1.02631,-1.14148,-0.328964,-0.581013,-2.09467,1.84828,-1.76873,0,0,0,0,0,0,0,0,0,0,0,0,
-0.276614,0.125182,-0.74475,0.647966,-0.943961,0.578956,-0.379926,0,0,0,0,0,0,0,0,0,0,0,0,
2.8152,2.99951,1.55905,0.530353,2.94505,-1.80575,2.25959,0,0,0,0,0,0,0,0,0,0,0,0,
0.933848,2.36693,1.06564,0.138454,2.52347,-1.63423,1.77388,0,0,0,0,0,0,0,0,0,0,0,0,
-1.53162,-2.26699,-1.63944,-1.27998,-1.49923,0.802996,-1.87734,0,0,0,0,0,0,0,0,0,0,0,0,
2.20369,2.65718,2.3773,1.47866,2.61718,-0.154934,1.01416,0,0,0,0,0,0,0,0,0,0,0,0,
0.386892,-0.234194,0.425573,-2.07542,-0.99878,0.087789,-1.49088,-2.10113,-2.21767,0,0,0,0,0,0,0,0,0,0,
0.670532,1.11525,0.0213808,0.213706,-0.775345,-0.870597,-0.243734,-1.07331,-0.921999,0,0,0,0,0,0,0,0,0,0,
3.34363,2.88333,1.5528,-0.690988,-1.21196,1.38443,0.248891,-3.33599,-2.18742,0,0,0,0,0,0,0,0,0,0,
-4.43576,-3.10679,-1.50732,1.14355,3.05976,-1.32176,-0.279068,6.36581,4.37729,0,0,0,0,0,0,0,0,0,0,
0.801925,-0.589618,-0.498002,-1.99003,-1.17152,0.0225333,-1.83288,-2.48201,-0.800297,0,0,0,0,0,0,0,0,0,0,
-0.0741623,0.463091,-0.490407,-1.33794,-1.19302,-0.0671063,-0.991982,-2.579,-1.40303,0,0,0,0,0,0,0,0,0,0,
2.30898,2.59505,-0.245139,-0.143466,-1.28047,0.0225859,0.402924,-3.23301,-2.68613,0,0,0,0,0,0,0,0,0,0,
2.45217,1.30712,0.624695,-1.20453,-0.992719,0.8808,-1.4293,-4.16149,-2.09886,0,0,0,0,0,0,0,0,0,0,
-3.29553,-2.75451,-0.707257,-3.38379,1.15948,-1.85432,-2.29433,0.282948,-0.104368,0,0,0,0,0,0,0,0,0,0,
0.901645,0.385552,0.881322,0.762582,0.0627355,0.500188,0.815467,0.0169501,0.29148,1.15498,0.629604,0,0,0,0,0,0,0,0,
1.48735,0.327641,1.03651,0.605442,1.299,1.28978,0.0118243,0.774944,-1.53547,1.58578,0.857565,0,0,0,0,0,0,0,0,
-0.139832,-0.767387,0.690406,-0.647648,-1.73758,-0.953175,-0.415786,0.357295,0.342909,-0.860505,-1.00317,0,0,0,0,0,0,0,0,
-1.30254,-0.0299509,-0.923929,-1.09153,-0.484701,-1.1409,-1.01218,0.732852,0.567873,-1.39764,-0.58814,0,0,0,0,0,0,0,0,
1.3415,0.479091,0.816822,0.0875707,0.305267,0.492711,-0.38267,-0.252676,-0.294921,1.50257,-0.944971,0,0,0,0,0,0,0,0,
-3.18545,-3.58245,-1.57054,-3.91511,-4.13576,-3.40871,-3.03685,1.47586,1.64368,-3.67964,-3.48077,0,0,0,0,0,0,0,0,
-1.49758,-0.795641,-0.477492,-0.9121,-0.961176,-0.978628,-0.587473,0.514521,-0.10312,-0.437121,-0.99984,0,0,0,0,0,0,0,0,
2.04199,1.20223,0.339812,1.55051,1.45889,0.618371,1.08261,-0.0765456,-1.4675,2.22075,0.53648,0,0,0,0,0,0,0,0,
0.0453747,-0.721773,0.127358,0.344248,0.0228017,-0.923741,-0.898898,0.594424,0.0211068,0.407756,-1.21018,0,0,0,0,0,0,0,0,
0.369802,-0.429225,0.962211,-0.428983,1.22736,0.0473707,0.177308,0.884197,-1.56972,1.2173,0.321895,0,0,0,0,0,0,0,0,
0.838529,1.88589,0.571345,1.3097,1.89246,2.36037,2.18634,-0.36959,-0.305823,0.624519,1.9482,0,0,0,0,0,0,0,0,
0.130954,-0.516826,-0.0390864,-0.0555911,-0.483086,0.549499,-0.425825,0.274285,-0.624874,0.704007,0.68713,-0.507504,0.16394,0,0,0,0,0,0,
-0.319237,0.457155,0.0503858,0.0374498,-0.0900405,-0.0600437,-0.0621607,1.7398,0.379183,-0.33379,-0.700205,-1.53056,0.827214,0,0,0,0,0,0,
-1.42509,0.341737,-2.29356,0.486814,3.32227,0.521771,0.22662,2.09383,3.62748,1.29747,0.113476,-4.24123,-1.57304,0,0,0,0,0,0,
0.110626,0.570583,0.681116,0.393754,0.0764495,0.47705,0.0317332,1.01107,0.132282,-0.207397,-0.607639,-0.912909,-0.0276892,0,0,0,0,0,0,
-0.871339,0.953107,-1.23588,0.951312,2.71071,-0.676999,0.417402,1.64249,2.11142,0.667482,-0.64461,-2.83809,-0.224166,0,0,0,0,0,0,
1.02862,-0.207223,1.93275,-0.537461,-2.93969,-1.08259,0.74633,-3.07593,-2.64397,-2.02553,0.324457,3.92295,1.43658,0,0,0,0,0,0,
-0.596174,0.139976,-1.07057,0.92516,2.08283,-0.639657,1.00171,0.956261,1.18423,-0.420206,-1.38542,-2.30077,-0.244787,0,0,0,0,0,0,
-2.21381,0.563199,-2.75559,-0.108763,3.37868,-0.351411,1.51645,0.93244,3.01257,1.26995,-0.682257,-2.69755,-2.34132,0,0,0,0,0,0,
-0.803653,0.486332,-2.71571,0.00486127,3.2686,-0.556502,1.78242,0.779195,3.27404,1.49038,-0.610542,-2.35778,-2.57451,0,0,0,0,0,0,
0.847329,-0.834072,2.37916,1.24146,-1.98473,-0.52079,-0.48859,-0.115073,-1.32164,-0.886867,-0.0804312,1.37765,1.79765,0,0,0,0,0,0,
-2.44787,0.39709,-3.7761,0.015054,6.59958,0.329229,0.798877,3.76058,5.89172,3.02555,-0.532234,-6.77026,-3.27629,0,0,0,0,0,0,
-0.658281,0.529111,-1.75975,-0.618552,1.90219,0.644707,0.912778,0.531902,1.53514,1.07795,0.382837,-0.831314,-1.17779,0,0,0,0,0,0,
0.90254,-0.00195845,0.416233,-0.534067,-1.33826,-0.406687,0.157748,-0.934546,-1.60637,-1.04187,0.646452,1.97913,0.0958256,0,0,0,0,0,0,
1.41138,-1.49318,-0.267813,0.118793,0.0667789,1.1114,1.21832,-0.364957,0.996974,-1.37857,-0.590952,-0.990955,1.44545,0.300644,0.521915,0,0,0,0,
-0.980688,0.364658,0.617238,0.970392,-0.845702,0.32811,-0.286463,0.866263,-0.592107,0.645209,-0.224906,0.547207,-0.936674,-0.788088,-0.536917,0,0,0,0,
-0.168275,0.53673,-0.365787,-1.25915,0.0107433,-0.413228,-0.0320747,-0.366879,-0.353566,0.0366365,0.302125,0.738732,0.35326,0.0523419,-0.221827,0,0,0,0,
-0.584207,0.430717,-0.130176,-0.274328,0.382646,-0.992711,0.0961735,-0.261535,0.0946536,0.772936,-0.148429,0.808034,-0.98955,0.367983,-0.497198,0,0,0,0,
-0.655831,0.734315,0.474604,-0.242935,-0.174109,0.226868,0.216102,0.234184,0.0758351,0.312709,0.304648,0.691801,0.132165,0.248373,-0.771094,0,0,0,0,
0.0795894,1.07338,-0.292855,-1.0238,0.581984,-0.873444,-0.632578,-0.599404,-0.774384,0.293745,0.164963,0.878368,0.574305,0.0938578,1.00816,0,0,0,0,
2.66202,-2.02176,0.195195,0.551417,0.618997,1.44304,2.92024,-0.450233,1.02399,-3.37295,-0.106694,-1.96011,1.64395,0.940143,0.462851,0,0,0,0,
-0.105968,-1.25284,0.864732,2.02985,-0.311623,1.51714,0.530248,-0.186852,-0.190595,-1.46531,-0.509711,-0.848307,-0.040913,0.517662,-1.19258,0,0,0,0,
-1.87156,1.57585,0.171384,-1.07235,0.0795015,-1.88109,-1.77911,-0.466125,-0.225306,2.0612,0.746487,1.15275,-0.836341,-1.0385,-0.0588058,0,0,0,0,
-0.231926,1.37785,-0.192581,-1.36978,-0.125444,-1.93895,-1.58626,-0.52281,-0.00773775,1.94619,1.14006,1.36407,-0.205571,-0.710586,0.220972,0,0,0,0,
0.33655,-0.574124,-0.732785,-0.764633,-0.384849,-0.0135144,0.504584,0.0967235,0.278052,-0.246882,0.53561,0.588689,1.36747,0.94626,0.718744,0,0,0,0,
-1.47714,1.41647,0.480085,-1.61308,0.495642,-1.87418,-1.98503,0.0255505,-1.03677,2.6324,-0.0743271,1.9304,-1.19671,-0.655958,0.10449,0,0,0,0,
0.804917,0.149715,-1.2958,-1.7613,1.1501,-0.56573,0.34409,-0.14935,0.177333,0.810151,0.991728,0.996871,0.634889,-0.423213,0.898464,0,0,0,0,
1.26666,-0.647383,-0.70616,-0.628073,0.550705,-0.287921,1.01286,0.604584,0.565855,-0.58263,0.00775118,0.532163,1.49201,0.565321,0.325189,0,0,0,0,
1.11794,-1.13155,-0.282903,0.617965,-0.0717177,1.57804,1.29605,0.671933,0.738547,-2.33639,-0.274473,-0.262591,1.11294,0.807418,0.257607,0,0,0,0,
-0.377914,-0.225818,-0.429096,0.987763,0.193171,0.714889,-0.666905,-0.929931,-0.588023,-0.0435213,0.465649,1.11136,0.810951,-0.312313,-0.0704491,-0.539556,0.159737,0,0,
-2.32767,1.34197,0.36476,0.48528,-2.29735,-0.107763,-0.157041,2.38077,-1.10103,0.9694,-0.118429,-1.97709,-0.678479,2.7164,1.27078,1.77615,-1.76046,0,0,
-0.506219,0.36112,-0.772923,0.637045,-1.9832,-1.24995,0.187018,1.29697,0.753882,-0.780084,-0.108084,-1.1623,0.228745,0.782582,0.190188,1.46219,-1.24104,0,0,
-0.261945,-0.134153,-0.861752,-0.32569,-1.08022,-0.635845,0.108112,0.980172,0.272034,-0.176725,-0.170833,-0.771681,-0.31043,0.87253,0.529705,1.48879,-0.608076,0,0,
-0.652701,0.343429,-0.860292,1.39669,-1.21608,-0.217333,0.624246,0.513427,-0.448237,0.419166,-0.201683,-0.834232,0.63071,0.541281,-0.198191,1.73257,-1.33826,0,0,
-0.143953,1.26514,0.252472,-0.406242,-0.671232,-0.463832,-0.187793,-0.0536602,0.755577,0.0418132,-0.613325,0.185727,-0.582403,0.168035,-0.114024,0.891265,-0.929824,0,0,
2.01231,-1.57626,-0.800351,0.856102,2.55656,1.95036,0.395023,-3.5701,0.742491,-0.329472,-0.0741527,2.63708,0.83174,-2.53329,-1.54782,-1.52773,1.88953,0,0,
-1.01344,0.222599,0.0148034,0.204784,-0.807036,0.182928,-0.523892,1.60103,-0.937233,0.743981,-0.674546,-0.0547825,-0.667966,1.43427,0.187707,0.861661,-0.698571,0,0,
-0.496894,0.258762,0.294853,0.568549,-0.587026,-0.761855,-0.250601,0.208739,0.283704,0.0268767,0.470202,-0.815505,-0.244517,-0.188146,0.19042,0.823236,-0.0702735,0,0,
-0.400609,-0.530642,-0.0301947,-0.01536,0.655302,-0.239775,0.572657,-0.241502,0.26003,-0.401339,0.12529,-0.0177895,0.198477,0.419563,-0.149376,0.522912,-0.248691,0,0,
3.02225,-1.04811,0.382163,-0.814561,2.24272,-0.140416,0.693969,-2.79055,1.04339,-0.215989,-0.0298695,1.39015,0.197856,-1.48015,-1.53468,-1.01782,1.39767,0,0,
3.50719,-1.32831,0.82969,-1.76717,3.12907,0.64418,0.485468,-4.61659,1.32673,0.264079,-0.585126,2.83722,0.276637,-3.21029,-2.21937,-3.05265,2.95631,0,0,
-0.687074,-0.364741,0.182821,0.36512,-0.775456,0.474574,-0.0408075,0.633208,-0.0875694,-0.0766544,-0.14942,-0.318291,0.280064,0.234616,0.977562,0.441624,-0.662151,0,0,
0.0898836,0.0633354,-1.49628,1.36927,-0.473625,0.208693,-0.458777,-0.25294,0.156376,-0.349746,0.342975,0.425743,-0.28819,-0.386056,-1.10283,0.639174,-1.61187,0,0,
0.683439,-0.256975,0.853269,-1.25306,0.689052,-0.205386,-0.250166,-0.0950969,0.375352,0.789996,-0.948669,-0.12304,-0.222474,0.474984,1.02151,-1.0293,1.25793,0,0,
-1.32926,0.386258,-0.413633,0.452075,-1.29237,0.123832,-0.775261,2.05353,-0.438136,0.371959,-0.196067,-1.72413,0.537271,1.33648,0.961259,0.902856,-0.412672,0,0,
-2.26639,1.17612,0.583651,0.185289,-1.79347,-0.720326,-0.414004,2.51146,-1.16678,-0.257522,-0.307256,-2.13279,-0.37188,1.88216,1.74421,1.33016,-1.07328,0,0,
18392,-38257.4,37806.7,-21284.7,32237,-62523.1,-41790.9,110133,92665.1,4731.58,92667,110134,-41792.6,-62522.7,32237,-21286.5,37806.1,-38255.3,18392.3,
-75.8607,157.288,-155.575,87.4328,-132.743,256.299,172.322,-453.639,-381.202,-18.6079,-381.246,-453.673,172.329,256.294,-132.735,87.4507,-155.557,157.267,-75.8604,
-2503.15,5205.17,-5145.71,2897.2,-4387.52,8508.29,5688.74,-14988.2,-12612.2,-644.042,-12610.3,-14988.5,5687.31,8508.34,-4386.49,2896.59,-5145.41,5206.82,-2502.99,
-12750.3,26524.9,-26210.7,14755.7,-22349.6,43346.7,28971.1,-76353.7,-64242,-3280.42,-64245.4,-76354.1,28975.1,43346,-22348.9,14758.9,-26210.4,26520.3,-12751.3,
3672.85,-7639.39,7547.34,-4249.92,6436.45,-12483.9,-8343.34,21989,18501.1,944.875,18501.9,21989.2,-8344.08,-12483.2,6436.8,-4250.16,7547.37,-7638.16,3672.5,
-14546,30256.1,-29899.7,16833.2,-25494.3,49446.8,33050.4,-87099,-73285,-3742.06,-73285.7,-87100.7,33052.3,49446.3,-25495.2,16834.4,-29898.9,30254.3,-14545.2,
2599.25,-5405.28,5340.93,-3007.28,4554.29,-8833.02,-5905.07,15560.5,13091.6,666.984,13092.1,15560.6,-5905.66,-8832.45,4554.75,-3007.63,5340.95,-5405.05,2598.8,
9713.83,-20206.3,19968.3,-11241.4,17025.7,-33021.3,-22071.7,58167.3,48941,2499.18,48942.9,58167.7,-22073.9,-33020.6,17025.1,-11242.7,19967.2,-20203.1,9713.36,
-15217.6,31652.3,-31280,17611,-26671,51729.2,34576,-91119.5,-76667,-3914.73,-76668.9,-91120.6,34578.1,51727.6,-26671.3,17610.6,-31278.9,31650.4,-15216.8,
22110.1,-45994.4,45450.8,-25586.7,38754.6,-75164.7,-50238.3,132400,111398,5688.3,111403,132400,-50244.2,-75162.2,38754.3,-25591.2,45448.8,-45987.3,22111.3,
6281,-13066.7,12911.9,-7269.6,11010.4,-21354.4,-14272,37613.1,31647.3,1616.11,31648.1,37613.6,-14272.9,-21353.7,11010.5,-7269.43,12911.4,-13066,6280.7,
9762.08,-20306.4,20065.7,-11296.4,17110.4,-33184.5,-22179.3,58452.4,49180.7,2511.16,49182.5,58452.7,-22181.6,-33183.8,17109.8,-11297.7,20064.6,-20303.1,9761.61,
-6209.93,12916.3,-12764.4,7185.95,-10883.3,21110,14108.8,-37183.1,-31285.4,-1598.15,-31286.7,-37183.3,14110.2,21108.6,-10884.4,7186.79,-12764.4,12915.8,-6208.84,
-70.5743,147.913,-146.15,82.1403,-125.439,243.072,161.074,-425.993,-358.776,-19.0758,-358.764,-426.003,161.08,243.066,-125.446,82.1452,-146.143,147.9,-70.5705,
9020.58,-18763.9,18542.3,-10439,15809.9,-30664.7,-20495.9,54014.2,45446.5,2320.56,45448.5,54014.7,-20497.7,-30663,15810.7,-10439.6,18542.3,-18760.9,9019.74,
12565.3,-26138.9,25829.3,-14540.3,22024.7,-42715.6,-28550.8,75243.1,63307.8,3232.64,63311.2,75243.5,-28554.8,-42715,22024,-14543.4,25829,-26134.3,12566.3,
-1062.07,2209.78,-2182.31,1229.07,-1862.27,3611.85,2412.59,-6359.96,-5351.25,-273.013,-5350.44,-6360.09,2411.99,3611.87,-1861.83,1228.81,-2182.18,2210.48,-1062.01,
7764.91,-16152.3,15962.1,-8985.76,13610.3,-26396.8,-17643.6,46496.1,39121.1,1997.65,39123.3,46497.6,-17644.3,-26395.7,13609.6,-8987.13,15960.5,-16150.2,7764.96,
-7382.98,15356.7,-15175.6,8543.61,-12939.8,25096.9,16775.5,-44208.7,-37195.9,-1899.55,-37196.7,-44209,16776.2,25096.8,-12939.8,8544.31,-15175.3,15355.8,-7383.1 };
    std::vector<std::vector<float>> aVec;
    for (int i = 0; i < 100; ++i)
    {
        std::vector<float> row;
        for (int j = 0; j < 19; ++j)
        {
            row.push_back(rawA[i][j]);
        }
        aVec.push_back(std::move(row));
    }

    optix::Buffer areaLightAlphaCoeffBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, rawAcol, rawArow);
    float *areaLightAlphaCoeffBufferData = static_cast<float *>(areaLightAlphaCoeffBuffer->map());
    for (int i = 0; i < aVec.size(); ++i)
    {
        memcpy(areaLightAlphaCoeffBufferData + i * rawAcol, aVec[i].data(), sizeof(float)*rawAcol);
    }


    areaLightAlphaCoeffBuffer->unmap();

    context["areaLightAlphaCoeff"]->setBuffer(areaLightAlphaCoeffBuffer);
#endif

#elif SHIntegration_Order2
    constexpr size_t l = 2; // N=3 order SH, lmax = 2, l goes from [0,2]
    constexpr size_t AProws = (l + 1)*(l + 1);
    constexpr size_t APcols = (l + 1)*(2 * l + 1);
    TW_ASSERT(AProws == 9 && APcols == 15);
    /*constexpr size_t AProws = (l + 1)*(l + 1);
    constexpr size_t APcols = (l + 1)*(2*l + 1);
    TW_ASSERT(AProws == 9 && APcols == 15);
    optix::Buffer areaLightAPMatrixBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, APcols, AProws);
    std::vector<float> APdata{ 0.282095, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0.0232673, 0, 0, -0.0465345, 0, 0, -0.519401, 0, 0, 0, 0, 0, 0, 0,
                               0, 0.411914, 0, 0, 0.397679, 0, 0, 0.247148, 0, 0, 0, 0, 0, 0, 0,
                               0, -0.265119, 0, 0, 0.530239, 0, 0, 0.329531, 0, 0, 0, 0, 0, 0, 0,
                               -0.824085, 0, 2.47225, 0.061849, 0, -0.185547, -0.0179691, 0, 0.0539073, 0.350888, 0, -1.05266, 1.03784, 0, -3.11352,
                               1.40929, 0, -4.22786, -0.170478, 0, 0.511434, -0.0252959, 0, 0.0758877, 0.047984, 0, -0.143952, -1.50602, 0, 4.51806,
                               1.06279, 0, -3.18837, 2.05213, 0, -6.1564, 0.452105, 0, -1.35631, 2.05213, 0, -6.1564, 1.0628, 0, -3.18839,
                               0.679056, 0, -2.03717, 0.688339, 0, -2.06502, 0.28824, 0, -0.864721, 0.707511, 0, -2.12253, 0.423207, 0, -1.26962,
                               -0.768895, 0, 2.30669, -1.19225, 0, 3.57675, 0.101583, 0, -0.304749, -1.14112, 0, 3.42337, -0.439553, 0, 1.31866 };
    void *APMatrixBufferData = areaLightAPMatrixBuffer->map();
    memcpy(APMatrixBufferData, APdata.data(), sizeof(float)*APdata.size());
    areaLightAPMatrixBuffer->unmap();

    context["areaLightAPMatrix"]->setBuffer(areaLightAPMatrixBuffer);*/

    /* Basis Vector. */
    optix::Buffer areaLightBasisVectorBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 2 * l + 1);
    std::vector<float3> basisData{ make_float3(0.6, 0, 0.8),
                                   make_float3(-0.67581, -0.619097, 0.4),
                                   make_float3(0.0874255, 0.996171, 0),
                                   make_float3(0.557643, -0.727347, -0.4),
                                   make_float3(-0.590828, 0.104509, -0.8), };
    void *areaLightBasisVectorBufferData = areaLightBasisVectorBuffer->map();
    memcpy(areaLightBasisVectorBufferData, basisData.data(), sizeof(float3)*basisData.size());
    areaLightBasisVectorBuffer->unmap();
    context["areaLightBasisVector"]->setBuffer(areaLightBasisVectorBuffer);

    /* Flm diffuse Vector. */
    std::vector<float> Flm_data{ 0.886227,-0,1.02333,-0,0,-0,0.495416,-0,0 };
    //TW_ASSERT(TestDiffuseFlmVector_Order3(Flm_data, 100000));
    TW_ASSERT(AProws == 9);
    optix::Buffer areaLightFlmVectorBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, AProws);
    void *areaLightFlmVectorBufferData = areaLightFlmVectorBuffer->map();
    memcpy(areaLightFlmVectorBufferData, Flm_data.data(), sizeof(float)*Flm_data.size());
    areaLightFlmVectorBuffer->unmap();

    context["areaLightFlmVector"]->setBuffer(areaLightFlmVectorBuffer);

    /* Alpha coeff. */
    constexpr size_t rawArow = 9;
    constexpr size_t rawAcol = 5;
    std::vector<std::vector<float>> rawA = { {1, 0, 0, 0, 0},
                                             {0.04762, -0.0952401, -1.06303, 0, 0},
                                             {0.843045, 0.813911, 0.505827, 0, 0 },
                                             {-0.542607, 1.08521, 0.674436, 0, 0},
                                             {2.61289, -0.196102, 0.056974, -1.11255, -3.29064},
                                             {-4.46838, 0.540528, 0.0802047, -0.152141, 4.77508},
                                             {-3.36974, -6.50662, -1.43347, -6.50662, -3.36977},
                                             {-2.15306, -2.18249, -0.913913, -2.24328, -1.34185},
                                             {2.43791, 3.78023, -0.322086, 3.61812, 1.39367}, };

    optix::Buffer areaLightAlphaCoeffBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, rawAcol, rawArow);
    float *areaLightAlphaCoeffBufferData = static_cast<float *>(areaLightAlphaCoeffBuffer->map());
    for (int i = 0; i < rawA.size(); ++i)
    {
        memcpy(areaLightAlphaCoeffBufferData+i*rawAcol, rawA[i].data(), sizeof(float)*rawAcol);
    }
    
    
    areaLightAlphaCoeffBuffer->unmap();

    context["areaLightAlphaCoeff"]->setBuffer(areaLightAlphaCoeffBuffer);
#endif // SHIntegration_Order2
}

#if 0
// Diffuse BSDF
const float QuadLight::BSDFMatrix_Rawdata[100][100]
{ 1.00373, -0.00175499, -0.00210972, 0.00259945, -0.00971962, 0.00545694, -0.0262888, 0.00398189, 0.0174917, -0.00976772, 0.00160802, 0.00798622, -0.00805144, 0.0114864, -0.0143986, -0.000467487, -0.0152528, -0.00554582, -0.0170132, 0.00339516, -0.000675059, -0.00491428, 0.0075166, 0.000529292, 0.00227628, 0.00406122, -0.00403348, 0.0210214, 0.0172439, -0.0176828, -0.0121924, -0.00420666, -0.00261027, -0.00283213, 0.00938655, 0.0124994, -0.00254663, -0.00445335, 0.000947172, -0.000645554, 0.004832, -0.00126256, 0.00858557, 0.0043435, 0.00110955, -0.0233145, -0.0179228, -0.00782895, -0.00628026, -0.00428087, 0.000952657, 0.0175735, -0.00935337, -0.0151181, -0.00865855, 0.00527394, -0.000979502, 0.015702, 0.0023208, 0.00296591, -0.0143029, 0.0163637, -0.0144374, 0.00849733, 0.00196864, -0.0034588, -0.00480379, -0.0132777, -0.00277268, 0.00174386, -0.00435926, -0.00198597, -0.00491458, 0.0102491, 0.0123841, -0.00212499, -0.0199415, 0.00377156, -0.0161174, -0.0186606, 0.00144725, -7.53577e-05, 0.00150828, 0.017526, 0.00346368, -0.00460015, 0.0141369, -0.0264999, -0.00227667, -0.00392816, 0.000603624, 0.0141739, -0.00546285, 0.00303398, -0.00289109, 0.0174657, 0.0144556, 0.00261184, 0.0145796, 0.00863278,
0.000884044, 5.87022e-07, 6.24446e-06, 1.16739e-06, -1.70308e-05, 9.07858e-06, -1.04075e-05, -1.1826e-05, 4.36396e-06, 1.21879e-05, -5.56878e-06, -3.92549e-06, 2.5556e-06, 1.11843e-05, 1.33437e-05, -3.57154e-06, 8.30894e-06, 1.46986e-05, 9.2656e-06, -7.62989e-06, 1.1509e-05, -1.55012e-05, 2.48098e-06, -4.31525e-06, 3.08714e-06, -1.27171e-06, 1.66283e-05, -1.77064e-06, -1.35876e-05, -5.99399e-07, -5.45967e-06, 1.23071e-06, -1.02945e-05, -1.08058e-05, 4.36015e-06, 1.21568e-05, -6.42265e-06, 7.78273e-06, 3.68927e-06, 1.65758e-05, -1.02277e-05, 5.08518e-06, -7.22789e-06, 3.37326e-06, 4.64682e-06, 5.47049e-06, 1.43802e-05, 5.70674e-06, 1.23662e-06, -1.33165e-05, -6.85996e-06, -5.00168e-06, 3.33121e-06, 3.98694e-06, -1.61139e-06, -2.3468e-06, -1.05004e-05, -6.86489e-06, 6.3935e-06, 8.65381e-06, -9.37268e-06, -4.17765e-06, 2.79945e-06, -8.35373e-06, -1.37831e-06, 1.67766e-06, -4.29662e-06, -1.55725e-06, -2.15817e-06, -1.39713e-05, -6.38844e-06, -3.87824e-06, 1.73247e-05, 1.18781e-05, -1.69804e-05, -1.00604e-05, 2.54407e-05, -3.56141e-06, -3.99582e-06, -2.14519e-06, -5.18208e-06, 1.50246e-06, -1.11628e-05, 3.70963e-06, 4.02822e-06, -2.43522e-06, -1.21452e-05, -1.1369e-06, 6.30415e-06, 2.55481e-06, 8.78514e-06, 5.35021e-06, -7.55345e-06, -7.35996e-06, 9.38144e-06, -4.95334e-06, 1.16852e-05, -2.57614e-05, -1.02158e-05, 7.35018e-06,
1.16538, -0.00287387, 0.00789419, -0.0113777, 0.00254261, -0.0201102, -0.00832948, -0.00479342, 0.0115501, -0.0101835, -0.0254328, -0.0210198, -0.00332665, -0.00568072, 0.000356201, 0.00676872, 0.00617093, 0.00354731, 0.0154256, -0.0101845, -0.0169777, -0.0014046, 0.00780405, 0.00118527, -0.01662, 0.00405372, -0.0180399, -0.00791852, -0.000970666, 0.00570926, 0.00476706, -0.00220175, -0.00294505, -0.00315558, 0.00442759, -0.00891815, 0.00105301, -0.0113973, -0.0101328, -0.0152768, 0.0131242, 0.000392025, 0.000817106, 0.0132376, -0.00305919, 0.00871584, -0.00789657, 0.0176626, 0.0278443, -0.00846773, -0.0079098, -0.00738216, -0.0123391, 0.0214131, 0.00436028, 0.00334537, -0.00266419, 0.00450766, 0.00444879, 0.00382488, -0.0161152, 0.0203659, 0.00638048, 0.0258666, 0.0203134, 0.0140363, -0.0165585, -0.00314368, 0.0126847, 0.00384823, 0.0133383, 0.00282874, -0.0154009, -0.00305473, 0.0106724, 0.00163296, -0.0161404, -0.00305034, 0.00397439, -0.00569381, -0.00305675, 0.00133893, -0.00383339, 0.0146823, 0.000550959, -0.0316554, -0.00234186, 0.0161236, -0.0204694, -0.00044098, 0.00615226, 0.0102554, 0.000255079, -0.0142988, 0.00976796, -0.0067039, 0.0081815, -0.0115325, 0.00882833, 0.00489179,
-0.00568415, 5.28988e-05, -1.35383e-05, -1.26814e-05, -0.000121684, 2.74327e-06, 9.62291e-06, 4.51654e-05, 1.64609e-05, 1.77659e-05, -7.8396e-05, 6.2304e-07, 4.84361e-05, 9.96043e-05, 1.53126e-05, -2.71128e-05, -3.59991e-05, -0.000110968, -5.73885e-05, -4.41153e-05, -4.60425e-05, -4.81346e-05, 5.23629e-06, -2.47629e-05, -4.2268e-05, 8.02652e-06, 7.27188e-05, 1.394e-05, -9.24465e-05, 2.6776e-05, -8.34053e-05, 3.00192e-05, 8.96577e-06, -9.37977e-05, -9.72588e-05, 5.90313e-05, 3.00405e-05, 8.74258e-05, -1.82854e-05, -3.34882e-05, -2.6994e-05, 9.46458e-05, 8.23092e-08, 2.52511e-06, -7.48022e-05, -9.26576e-06, 6.92788e-06, 4.19347e-05, -2.1264e-05, -1.9194e-05, -5.06752e-06, -6.05826e-05, -0.000111385, -5.86847e-05, -4.90153e-05, -2.8355e-05, -3.07933e-05, 4.95774e-05, 4.72121e-05, 1.4959e-05, 6.02171e-05, -6.11884e-05, 9.97669e-06, 1.07099e-05, 2.34909e-05, 3.06608e-05, -2.40361e-05, -4.18112e-06, 4.3475e-05, 4.01135e-05, 2.22564e-05, 1.87976e-06, 8.6778e-06, -0.000107985, -7.3165e-05, 4.23292e-06, 4.17549e-05, 0.000119162, 4.23791e-05, -4.0346e-05, 4.20288e-05, 3.4785e-05, -3.13366e-05, 0.000128661, -5.60832e-06, -9.1668e-05, -5.09375e-05, -1.76195e-05, 3.55339e-05, 2.00567e-05, -1.62743e-05, -3.27384e-05, 1.11178e-05, -2.09715e-05, -4.59653e-05, 6.3299e-05, 2.63251e-05, 5.35801e-05, 1.21334e-05, 8.22443e-05,
-0.0119327, -0.000320857, -5.06382e-05, -2.00128e-06, 3.19991e-05, -1.77404e-05, -8.79217e-05, 2.54679e-05, 0.000204008, 3.81484e-05, 1.43928e-05, 7.67101e-05, 2.11894e-05, -0.000139406, 5.44892e-05, -2.7546e-05, -0.000309446, 6.34059e-05, -2.57316e-05, -8.30661e-05, 0.000145831, 0.000115975, -3.63371e-05, -0.000211157, -4.59143e-05, -3.30603e-05, 2.28396e-05, 3.25956e-06, -0.000164541, 9.99366e-05, -0.000234186, -3.69889e-05, 3.05517e-05, 3.74353e-05, -4.41915e-05, -0.000105813, 0.00019489, -0.00010649, -2.34692e-05, -0.000428133, -5.11246e-05, -2.96679e-06, -4.14534e-05, -0.000172488, -6.84342e-05, 3.44089e-05, 7.29643e-05, -8.92267e-05, 2.36315e-05, 0.000204487, 0.000108833, 0.000115954, -5.71022e-05, -3.09326e-05, 0.000199889, 0.000150065, 1.79326e-06, -4.68834e-05, 0.000317838, 2.9737e-05, 6.42428e-05, -8.98496e-06, 4.97002e-05, 1.0615e-05, 8.74118e-06, 5.69571e-05, -8.78688e-05, 0.000164325, -3.36826e-05, -0.000103906, 0.000179384, -7.34955e-05, -7.2498e-05, 9.05764e-05, -1.81427e-05, 9.30795e-05, 0.000163511, -9.03329e-05, -0.000143451, -4.84763e-05, -0.000128451, 0.000318358, -0.00034909, 7.92187e-05, -8.29585e-05, 2.53474e-06, -0.000166318, 0.000154425, -0.000252576, -2.39504e-05, -0.000236823, 8.87493e-05, -7.54423e-05, -0.000109584, -0.000134666, 0.000102057, 2.67016e-06, -7.12018e-05, 4.72231e-05, -0.000137433,
0.00147733, 8.27469e-06, 2.89112e-06, 3.05016e-05, 5.80233e-06, 2.48658e-08, -3.87084e-06, -6.32973e-06, 1.49444e-05, -2.92907e-05, 1.76861e-05, -2.28916e-05, 6.6303e-06, 9.71186e-06, -2.96735e-05, 1.45783e-05, 7.34151e-07, 3.97823e-06, -1.23936e-05, 1.8512e-05, 2.65335e-05, -1.74414e-05, -9.43416e-07, 6.92793e-06, -6.20767e-06, -4.02495e-06, 3.70408e-06, 1.5325e-05, -3.27069e-05, -9.48466e-07, 1.55021e-05, 1.01301e-05, 8.66606e-06, -1.39486e-05, 1.13716e-05, -1.35557e-05, 7.32034e-06, 1.80714e-06, 2.10852e-06, -1.68265e-05, 1.59768e-06, 2.01024e-05, 2.53646e-05, -1.37143e-05, -2.14151e-06, -2.8735e-05, 5.06176e-06, -1.06229e-05, 2.24529e-06, -8.18822e-06, 8.73598e-06, 2.87921e-05, -2.70806e-06, 2.80582e-05, -1.14252e-05, -4.01294e-06, 9.95565e-06, -1.05949e-05, 2.27737e-06, -3.29876e-06, -1.85101e-05, -4.89264e-06, 2.04156e-05, 9.61785e-08, -9.10039e-06, 1.81665e-05, 4.20738e-07, 1.35395e-06, -1.80813e-05, 1.13059e-05, -1.29027e-05, -1.88806e-06, -2.22208e-05, 2.22251e-05, 4.27858e-06, -1.26732e-05, -1.89441e-05, 4.90013e-06, 1.58113e-05, 6.56251e-06, 3.80225e-06, 1.11335e-06, -3.14747e-05, 2.27181e-05, 5.58785e-06, -2.94378e-05, 2.36087e-06, 1.7035e-05, -2.56873e-05, -6.77974e-07, 2.98645e-06, -5.02944e-06, 2.39997e-05, -2.29372e-06, -8.286e-06, -1.50933e-06, -4.2084e-06, 2.08125e-05, -1.68761e-05, -3.07559e-06,
0.577581, 0.000199596, 0.00199514, -0.000853732, -0.00908794, 0.00142141, 0.000758418, -0.00129421, -0.00412942, -0.00394712, 0.00401844, -0.00102604, -0.00827826, 0.00820156, -0.00147421, -0.0097496, -0.0038253, -0.00213336, -0.00183971, -0.000631154, -0.0018121, 0.000446139, -0.00128489, 0.00155963, 0.00382454, 0.011478, -0.00881696, -0.00087195, 0.00452513, -0.00421619, 0.00189592, 0.00688047, -0.00301121, -0.00588822, 0.000534926, 0.00451252, -0.00994878, 0.00659832, 0.00195946, -0.000504788, -0.00940746, 0.00665869, -0.000281063, 0.00607842, -0.00260614, 0.00444979, -0.000571283, -0.00451444, -0.00282114, -0.00435903, -0.00780418, 0.000876651, 0.0148701, 0.000115739, 0.000321269, 0.0018198, -0.00506202, 0.00797093, -0.00478418, 0.00142743, -0.00569324, 0.00589537, -0.000909328, 0.00175569, -0.0100369, -0.00173586, 0.00462149, -0.00076637, -0.00150084, -0.00108436, 0.0123973, -0.00349058, -0.011146, -0.0133702, -0.00416853, 0.0134345, -0.00353736, -0.00117243, -0.00221684, -0.00118525, 0.000541511, -0.000237668, -0.00448963, 0.0103242, -0.000757939, 0.00252953, -0.00184884, 0.0076098, 0.000494471, 0.000471407, -0.00353257, -0.00655578, 0.0117591, 0.00177338, -0.000519598, -0.00718191, 0.00315774, -0.0112551, -0.0081073, 0.00351442,
-0.0101357, 0.000238815, -8.74358e-05, -0.000172408, -0.000166346, -2.03742e-05, -8.85733e-05, -1.2441e-06, -2.38115e-05, 0.000174866, -0.000127435, -0.0001285, -9.88042e-06, -0.000111602, -6.85007e-05, 8.22873e-05, 3.0492e-05, 7.23272e-05, -8.78097e-05, -2.28544e-05, 3.1884e-05, 3.16188e-05, 0.000101059, 7.73201e-05, -5.26469e-05, 7.26088e-05, -7.65729e-05, -5.38647e-05, 5.22632e-05, -4.96459e-05, 0.000124305, -0.000129176, -6.43139e-05, -7.44089e-05, -6.80304e-05, 5.73092e-05, 0.000102754, 2.07488e-05, -2.83157e-05, -0.000195232, -4.69215e-05, -0.000199141, -4.40828e-05, -3.27422e-06, 0.000137, -7.56512e-05, 7.446e-06, -3.32335e-05, -5.12044e-05, 3.55231e-05, -6.96535e-05, 3.30998e-05, 1.48542e-05, -6.88438e-05, 1.35532e-05, 5.26245e-05, -0.000126315, -0.000119352, 5.7274e-05, 8.22986e-05, -9.67812e-05, -6.4561e-05, 0.000110307, -0.000149853, 5.92908e-05, -5.82092e-05, -7.55941e-05, 3.40062e-05, -0.000164406, 0.000114104, 5.63837e-05, -0.000225667, -0.000104613, -4.6026e-05, 3.03694e-05, 5.39634e-05, -8.9816e-06, -6.62552e-05, -4.6115e-05, -8.17469e-05, 2.16523e-06, -3.294e-05, 5.01901e-05, -2.56449e-05, 1.50603e-05, 8.51868e-05, -6.13263e-05, 9.04644e-05, 0.00016822, -1.53589e-05, -1.11059e-05, 5.51705e-05, 2.80943e-05, 6.17881e-05, 4.06188e-05, -0.000137491, 0.000200882, 3.25349e-06, -0.00010708, -6.21599e-05,
0.00409274, 6.20333e-05, 0.000113178, 4.19703e-05, -1.67903e-05, -7.73215e-05, -1.45776e-05, 2.47783e-05, 4.41997e-05, 1.14988e-05, 2.95941e-05, 1.98794e-05, -1.8024e-06, -1.68947e-05, 1.6414e-05, 9.68488e-05, -4.01581e-05, -1.77235e-05, 5.07645e-05, 4.0222e-05, 4.77284e-05, -7.16567e-06, 2.2187e-05, -6.70908e-05, -5.50313e-05, -4.38334e-05, 5.74148e-06, -2.47849e-05, 4.10544e-05, -7.21377e-05, 1.28398e-05, 3.71653e-05, -6.11906e-06, 2.8873e-05, 9.62923e-06, -6.39039e-05, 2.73674e-05, 3.33531e-05, 4.8613e-05, -1.05002e-05, 2.73802e-05, -6.30756e-05, 1.94328e-05, 2.03015e-05, -0.000104318, 3.11251e-05, 3.57316e-05, -5.32454e-05, -1.92915e-05, 1.01266e-05, 4.48738e-05, -6.95371e-05, 6.84465e-05, -2.45654e-05, -4.16479e-05, -4.61293e-05, 2.00317e-05, 0.00010485, 2.83474e-05, 5.10661e-05, 1.46941e-05, -4.42428e-05, -7.18293e-05, 9.76144e-06, 2.10763e-05, -1.74313e-05, 2.47787e-05, -3.55556e-05, 4.37993e-06, -4.19246e-05, -2.68534e-05, 2.10538e-05, 1.76751e-05, 4.53742e-05, 9.14465e-06, 5.75488e-05, -5.29588e-05, 7.22122e-06, -1.87179e-05, -1.9249e-05, 3.77082e-06, 2.58518e-06, 3.8965e-05, -3.04412e-05, -9.17303e-06, -3.80943e-05, -5.24497e-05, -3.14187e-05, 3.08221e-05, 3.88116e-05, 6.37153e-05, 4.73147e-05, 3.28613e-05, -1.63666e-05, -4.62106e-05, 3.0997e-05, -4.25483e-05, 5.74084e-05, 9.72057e-06, 2.66573e-06,
0.00101003, -2.43335e-07, 3.16188e-06, -9.49269e-06, 1.31301e-05, 5.94825e-07, 2.67079e-06, -1.72532e-05, 4.91173e-06, -2.00023e-06, 9.74306e-06, 5.26395e-06, -2.96351e-07, -3.63826e-06, 5.1815e-06, 3.85011e-06, -3.68261e-06, -2.51453e-05, -8.20016e-06, -2.52758e-06, -4.59611e-07, -1.28646e-05, -3.51745e-06, 8.76899e-06, 5.70745e-06, 6.45266e-06, -9.67791e-06, -1.05617e-06, -2.41985e-05, -1.06782e-05, 2.21766e-06, -2.1209e-05, -1.5342e-06, -3.88649e-06, -1.91337e-06, 1.19094e-05, 5.99382e-06, 9.78884e-06, 7.9774e-06, -6.36562e-06, -2.97201e-06, -5.45277e-06, -2.35788e-05, -2.09667e-06, 6.47992e-06, 9.7213e-06, -8.24128e-06, 1.23465e-05, -1.45763e-05, -4.62286e-06, 9.30186e-06, -4.67522e-06, 8.37257e-07, -2.71745e-06, -1.24498e-05, 4.89169e-08, -1.96267e-06, -3.15496e-06, 2.75566e-06, 7.58572e-06, 6.87705e-06, -4.68738e-06, -2.05941e-05, -5.62981e-06, 6.32101e-06, 1.34105e-05, -1.44758e-06, -2.98726e-05, -1.40071e-05, -1.26227e-05, -6.32574e-06, 1.85303e-06, -1.99734e-05, 3.86089e-06, 2.61202e-05, 8.38732e-06, -5.69119e-06, -9.6409e-06, -1.46779e-05, -2.21918e-06, 5.66514e-06, 1.9953e-05, -3.01489e-06, -2.5814e-06, -1.77135e-05, 4.9534e-06, 1.13085e-05, -8.55003e-06, -1.4184e-05, -2.85381e-05, -3.65455e-06, 5.77311e-06, 6.18548e-06, -1.25834e-05, 1.32449e-05, 3.81331e-06, -1.05741e-05, 4.87037e-06, 3.4706e-06, 6.03933e-06,
-0.0115608, -2.33982e-05, 3.43492e-05, 6.76401e-06, -6.90323e-05, 5.01731e-05, -1.98781e-05, 1.20127e-05, -7.98481e-05, -8.26087e-05, 0.000138206, 5.46131e-05, 1.02751e-05, 8.23744e-05, -8.98761e-05, -6.92022e-05, 3.00712e-05, -0.000182338, -0.000162413, 0.00013743, -2.51783e-05, -5.06415e-05, -0.000164286, 1.56479e-05, 1.39302e-05, -1.8636e-05, -3.61932e-06, 4.16469e-05, 1.60033e-05, 7.43634e-05, -1.35577e-05, 0.000174743, -1.56954e-05, 5.56408e-05, 2.28255e-05, -3.85311e-05, 7.6341e-06, -9.58864e-05, 0.000158354, 0.000200139, 7.06768e-05, -7.60843e-05, -4.04078e-05, 0.000166214, 5.26988e-05, -7.3343e-05, -4.83955e-05, 8.02229e-05, -4.59088e-05, 8.97572e-05, 0.000117484, 6.72824e-05, -0.000105785, 0.000194659, 3.01144e-05, 8.05536e-05, -0.000136169, 8.52252e-05, -0.000196682, 0.000119597, 5.42947e-05, 9.33501e-05, 4.04212e-05, 3.18802e-06, 3.05738e-05, 3.54255e-05, 2.8015e-05, -4.21975e-06, 0.000213409, 0.000109354, 0.000108875, 0.000164476, 0.000188861, 1.76532e-05, -0.000154165, 0.000149694, -5.12791e-06, 7.59802e-05, 7.58218e-05, 0.000117366, 0.000174515, -5.34328e-05, -9.42533e-05, -9.65245e-06, -0.000108726, -4.75943e-05, -0.000114091, -4.6942e-05, -0.00021772, -0.000259183, 0.00025582, -3.80233e-06, -4.3793e-05, 8.84606e-05, -0.000176759, -8.11803e-05, -9.55631e-05, 3.45507e-05, 0.000104218, 0.000163478,
-0.00322466, 2.53407e-05, -2.03881e-05, -4.42661e-05, 6.69714e-05, 1.62419e-05, 1.19079e-05, -2.0147e-05, 6.90872e-06, 3.55094e-05, -2.78527e-05, 7.21608e-06, -2.44174e-08, -2.36445e-05, 8.09093e-06, -2.56833e-05, 4.84925e-05, 1.36199e-05, -9.68192e-06, 4.61128e-05, -5.66851e-06, 1.88249e-05, -2.64189e-05, 3.44008e-05, 4.6905e-05, 2.0768e-05, -6.55944e-05, 2.9511e-05, 7.97449e-05, -8.52634e-06, -3.6152e-05, 1.45688e-05, 6.54813e-05, 1.05971e-05, 5.29162e-06, 1.49189e-05, 1.28817e-05, -1.73878e-05, -1.49661e-05, 1.84234e-05, -2.00017e-05, -1.10427e-06, -3.49761e-05, 7.21804e-06, -3.60468e-05, -5.08812e-07, 2.04718e-05, 2.54455e-05, 3.34766e-05, 7.81782e-06, 4.20372e-05, -4.07417e-05, -6.01695e-05, -1.07039e-05, 7.94392e-08, 4.20538e-06, -3.80361e-05, 1.34775e-05, -2.51689e-05, -2.76488e-06, 4.01163e-05, 4.53116e-05, -9.80149e-06, 9.38449e-06, -6.21311e-06, 5.98596e-05, 6.917e-05, -2.60685e-05, -8.34734e-06, 5.69091e-06, 2.24526e-06, -3.26975e-05, -1.04064e-06, 1.07533e-05, 2.80734e-05, 1.24381e-05, 1.40691e-05, 3.55946e-05, -4.54785e-08, -4.20213e-05, -1.62832e-05, 1.47044e-05, 9.03797e-05, 2.65931e-05, -3.26982e-05, 4.03847e-05, 2.45316e-05, 3.65297e-05, 4.25363e-06, 1.69678e-05, 3.75783e-05, -1.88359e-05, 2.26083e-05, -4.75192e-06, 2.67611e-05, 7.03842e-06, 6.15764e-05, -1.03004e-06, -1.63175e-06, -2.5434e-05,
0.0202655, 8.709e-05, -7.29179e-05, 5.97384e-05, -3.71432e-05, -0.000245334, 1.29102e-05, 1.23117e-05, 0.000402594, -0.000185094, 0.000323818, -5.48592e-05, -0.000199102, -0.000445337, 0.000310249, 5.78841e-05, 0.00020315, 0.000227446, 6.16131e-05, -3.88705e-05, -0.000319698, -0.000182573, 2.48386e-05, -0.000113219, 2.02018e-05, -3.08728e-05, 1.99448e-06, -4.34489e-05, -9.90703e-05, -0.000306539, 3.62124e-05, -8.25308e-05, -5.59243e-05, 3.17564e-06, 8.11746e-06, 0.000205277, -3.7116e-05, 3.55441e-05, 0.00027053, -4.91312e-05, -0.000458666, -0.00014444, -2.33413e-05, 5.68591e-05, 0.000199089, -2.04029e-05, 7.62611e-06, -2.73183e-05, 4.76543e-05, -0.000421106, -0.00011732, 0.000117062, 3.10698e-05, -9.88873e-05, -8.02176e-05, -1.95502e-05, 0.00019052, 3.79605e-05, -0.000121012, 0.000367591, -0.000226554, -8.34076e-05, 1.75634e-05, -0.000385961, 0.000166417, 0.000171697, -0.000145411, -0.000195881, 3.80717e-05, -0.000692322, 0.000145896, -0.00035737, 6.21533e-06, -7.53471e-05, -6.03789e-05, -0.000107824, 0.00013943, -6.61039e-05, -4.00375e-05, 5.1355e-05, -0.000259154, 0.000180903, 8.8139e-05, -0.00031659, 0.000347614, 6.01261e-05, -0.000100555, 0.000235107, -1.42146e-05, -0.000379806, -0.000111669, -8.41853e-05, 0.000138212, -7.39133e-05, -0.000331961, 0.000136534, 0.00031207, -0.000204342, 0.000102644, -0.000233657,
-0.0108288, 2.77967e-05, -4.61413e-05, 0.000210105, -7.64941e-05, 6.928e-05, -3.62831e-05, 0.000114515, 6.68312e-05, 1.893e-05, -1.0119e-06, 7.40817e-05, -0.000122202, -7.41889e-05, 1.2874e-05, 5.11631e-05, -5.20107e-05, 0.000114661, 0.000133209, 1.53381e-05, 6.58337e-05, 2.3555e-05, 4.64978e-05, 1.76937e-05, -0.000161826, -4.70128e-05, 1.86634e-05, 0.000245442, 1.67504e-06, 4.32715e-05, 7.49405e-05, -8.56277e-05, -9.94529e-05, 0.0002729, 1.26129e-05, 4.42178e-05, -0.000252899, -5.657e-05, 1.33417e-05, 0.000108534, 7.35401e-05, -0.000187737, 5.87171e-05, -0.000113279, -0.000259114, 5.30722e-06, 0.000128404, 0.000231957, -0.000165332, -3.93065e-05, 0.000135382, -1.60258e-06, 0.000155828, 0.000152842, -5.1735e-06, -0.000100258, 3.85924e-05, -0.000151633, 6.66237e-05, 0.000127672, 5.72756e-05, -7.97575e-05, 4.9494e-05, 5.50567e-05, 7.03524e-05, -0.000214155, -2.13722e-05, 2.29451e-05, -0.000184452, 0.000211503, -3.17707e-05, 6.36345e-05, -3.55086e-05, 0.000158051, 8.25195e-05, 1.26459e-05, 3.36593e-06, -0.000133296, -1.61058e-05, -5.20611e-05, 2.31413e-05, 5.13499e-05, 0.000106457, 0.000115799, -5.05127e-06, -0.000175355, 2.96049e-05, -0.000180114, -0.000124047, 8.62997e-05, 0.000171845, 0.000159445, 2.17604e-05, -1.09303e-05, 6.4022e-05, -6.85188e-05, -0.00017133, 6.73281e-06, -0.000211396, 7.17838e-05,
0.0100369, 0.000109078, 0.000107799, 0.000121465, 6.47468e-05, 2.55648e-05, 4.97969e-05, 0.000164909, -0.000195048, 5.30974e-05, 3.23822e-05, 2.91679e-05, 8.59484e-06, -8.20675e-05, 9.21743e-05, -4.30902e-06, 9.50756e-05, 1.00326e-05, 4.68041e-06, -8.96843e-05, 0.00012329, 4.76601e-05, 0.000223082, 5.70365e-05, 1.70074e-05, 2.52986e-05, -1.98721e-05, -2.61562e-05, -3.29399e-05, 0.00010688, -2.70133e-05, 0.000220761, -7.22952e-05, 0.000109308, 0.000259111, 0.000205629, 9.80535e-05, 7.15275e-05, -3.83174e-05, 4.7344e-05, 0.000124205, 7.02512e-05, -1.9283e-05, 5.1874e-05, 9.10683e-05, -0.000129598, -0.000128641, -9.77955e-05, -4.43388e-05, 0.00013768, -2.64449e-05, -0.000102256, 9.27867e-05, -0.000180435, -1.27723e-05, -1.64432e-06, -0.000122313, 0.000121611, 7.8611e-05, -0.000144822, 0.000152644, -9.20648e-05, 2.41341e-05, 8.58576e-05, -6.92019e-05, 1.38744e-05, 1.20498e-06, 9.34075e-05, -0.00016323, -0.000121107, -3.48207e-05, -5.09477e-05, 9.28927e-05, 6.75176e-05, 0.000116267, -0.000127474, 5.49745e-05, 7.94096e-06, -8.38934e-05, 1.85024e-05, -0.000168938, -0.0001199, 3.92765e-05, -5.76325e-05, 3.75394e-05, -0.000142731, -6.30816e-05, -4.18406e-05, 4.60504e-05, -0.000138889, -0.000218058, 2.15123e-05, 8.47816e-05, 2.60443e-06, 3.53364e-05, -7.4499e-05, -2.998e-05, -3.87462e-05, -4.61243e-05, -5.27679e-05,
-0.0131941, 6.26479e-06, 0.000279606, -8.4314e-05, 0.000103702, -5.54646e-05, -2.02439e-05, 0.000101685, 0.00022411, 0.000228208, 0.000307024, 1.15493e-05, 8.58097e-05, 0.000182272, 8.75245e-05, 0.000222004, -2.81781e-05, 1.13424e-05, -0.000135914, -0.000372231, 5.86067e-05, -2.27621e-05, -9.40235e-05, -2.60226e-06, -0.000115836, 8.30124e-05, 9.01132e-05, 2.4913e-05, -0.000161708, -6.50526e-05, 6.55235e-06, 0.000201613, -0.000104681, -0.00025367, -2.58759e-05, -9.42227e-05, -0.000232094, -3.20871e-05, -3.96148e-05, 0.000101269, -4.83872e-05, -5.4402e-05, -6.16405e-05, -7.17752e-06, -3.82145e-05, -0.000120564, 2.0536e-05, 0.000104725, 4.74285e-05, -9.18586e-05, 5.70655e-05, 0.000170511, 6.87431e-05, -0.000457942, 3.90811e-05, 0.000272845, 0.000188972, 3.20505e-06, 0.000217688, 0.00016209, -5.64791e-05, -7.41438e-05, -0.000125733, 0.000294251, 0.000187563, 0.000132963, 0.000161793, 0.000120171, 0.000207793, -0.000252166, -0.000104658, -4.74637e-05, 0.0001119, 2.87783e-05, -1.07112e-05, -0.000102341, 0.000168053, -5.94723e-05, 3.02823e-05, -3.07661e-05, -2.04698e-05, -0.000166778, -0.000157801, 0.000121711, -2.69181e-05, 0.000119623, 0.000141289, 0.000191831, 1.72917e-07, -0.000122849, -9.47626e-05, 4.94141e-05, 0.000142933, 9.49755e-05, 5.75628e-05, 0.000122049, 1.44741e-05, -4.32157e-05, 0.000163004, 2.0093e-05,
-0.00488004, 2.64959e-05, 7.27052e-05, -4.86544e-05, -3.68249e-05, -2.52466e-05, -0.00011807, -1.97413e-05, 6.51929e-05, 3.80668e-06, 2.04515e-05, -3.47428e-05, -6.14339e-05, 8.72746e-07, -3.33348e-05, 1.73507e-05, 1.75464e-05, -3.37583e-06, -5.99943e-05, -1.02716e-05, 4.02473e-05, 1.00053e-05, -2.09862e-05, 2.24864e-05, -4.79447e-05, -8.29339e-06, -6.48615e-05, -2.34878e-05, 5.56645e-05, -4.53124e-05, 7.9912e-05, 2.37436e-06, 0.000102197, 3.72581e-05, 7.0719e-05, -1.15057e-05, -1.7714e-05, 6.61185e-06, -8.7678e-06, -6.2764e-05, -2.83867e-05, -5.04012e-05, -2.98374e-05, 7.11111e-05, 4.28108e-05, -7.9766e-06, 3.27283e-05, -2.79447e-05, -1.15053e-06, -3.52284e-05, 2.44667e-05, -2.90543e-05, 2.22953e-05, 4.40809e-05, 9.14095e-05, -2.43685e-05, -4.43581e-05, -1.17589e-05, -5.78926e-05, 1.34965e-05, -1.30862e-05, 2.303e-05, 6.40552e-05, 5.61717e-05, 9.159e-05, 7.99123e-07, 3.83746e-05, 6.66293e-05, -1.44742e-05, 1.6221e-05, -1.691e-06, -5.00638e-05, -2.03168e-06, 5.2734e-05, -5.67366e-06, 2.24085e-05, 3.20471e-05, 1.21041e-05, -4.88946e-05, 1.96341e-05, -3.5663e-05, -8.75208e-06, 2.51541e-05, 8.39249e-05, 0.000118063, -2.75409e-05, 1.03224e-05, 2.16708e-05, -5.25349e-05, -0.000107053, 3.3133e-07, -5.77131e-05, 6.12165e-06, 6.15804e-05, -7.91416e-06, -4.48635e-05, 2.27774e-06, -1.99436e-05, -9.3871e-05, -7.64817e-05,
-0.000839947, -1.0377e-05, 1.76989e-05, -4.06892e-06, -2.5776e-07, 7.6738e-06, 3.72609e-06, 1.14426e-05, -3.21258e-06, -1.59776e-06, 3.46989e-06, -2.58886e-06, -7.50084e-06, -5.85034e-06, -2.75962e-06, -6.73332e-06, 5.17717e-06, 6.24946e-07, 9.4854e-06, -1.72762e-05, -1.39968e-05, 1.61228e-05, -1.61171e-05, -1.57402e-06, 5.84436e-06, 1.09596e-05, 1.1922e-06, 5.57874e-06, -6.42769e-06, -1.13725e-05, 3.4861e-06, -2.10117e-06, -2.43798e-05, -4.34289e-06, 5.17515e-06, 1.3249e-05, 6.63411e-06, -8.72844e-06, -7.11476e-06, 1.05878e-05, 3.99361e-06, 6.02089e-06, 7.05234e-06, -5.74142e-06, -2.68978e-05, 1.22544e-05, 6.02066e-06, -6.20849e-06, 7.38913e-06, 1.13709e-06, 6.36325e-08, 9.89685e-06, -3.32297e-06, 7.14216e-06, -2.9064e-06, 2.29283e-06, 1.66917e-06, -1.70762e-05, -2.12082e-06, -4.87911e-06, -4.37099e-06, 5.94598e-06, -6.83897e-07, -2.67553e-06, 1.14549e-05, -1.97916e-06, 2.26547e-06, 1.00578e-06, -6.97956e-06, -2.20271e-06, 3.06379e-06, -1.50664e-05, 1.10141e-05, 6.05982e-06, -6.54619e-06, 4.84658e-06, -9.5537e-06, 3.90987e-06, 2.10908e-05, 1.32246e-05, -1.83971e-06, -1.58182e-05, -1.19187e-05, -1.37852e-06, -1.85871e-06, -5.34467e-06, -9.46575e-06, 1.35981e-06, 1.63494e-05, -1.12167e-05, -1.09446e-05, -3.87775e-07, 1.17477e-05, 7.8885e-06, -3.14459e-06, 1.32999e-05, 1.74066e-05, 5.62409e-06, 8.15359e-06, 6.31474e-06,
0.000570958, -7.96493e-06, 9.03883e-06, 1.69785e-06, 3.68513e-06, 2.72347e-07, -1.79968e-06, -5.87793e-06, -7.26532e-06, 3.54526e-06, 6.158e-06, -2.79318e-06, -6.07338e-06, 5.74327e-06, 5.82157e-06, 4.54185e-06, -5.02315e-07, 1.36807e-06, 1.12749e-05, -2.73354e-06, 2.77993e-06, 2.60566e-06, 1.82284e-06, -4.98162e-06, -2.57739e-06, 1.14305e-05, 5.35071e-06, 2.9216e-06, 6.04927e-06, -3.84088e-06, 9.44533e-06, 7.28308e-06, 2.00758e-06, -5.0529e-06, 4.55283e-07, -5.58827e-06, -7.98965e-06, 4.2304e-06, 1.2501e-06, 8.72598e-06, -2.83737e-07, 5.80023e-06, -6.96146e-06, -1.41686e-05, -8.44839e-06, -6.92306e-06, 2.01081e-06, 6.83186e-07, 1.28454e-06, -1.09299e-05, 2.94478e-06, -3.00977e-06, -4.86938e-07, -2.86972e-06, 5.72622e-06, 1.48814e-06, 5.56802e-06, -4.19776e-06, 4.69749e-06, 1.71025e-06, -3.40097e-06, -4.85771e-06, 4.84501e-06, -1.48035e-06, -1.48488e-07, 5.44697e-07, 2.12151e-06, -4.70923e-06, 6.1549e-06, -4.51122e-06, 2.08282e-06, -5.58851e-07, 6.9847e-06, -6.81359e-06, 4.0259e-06, -7.67608e-06, -2.21034e-07, -1.1836e-05, 2.58949e-06, -3.61649e-07, -3.67831e-07, 1.88819e-06, 1.05268e-05, 6.52438e-06, -5.9295e-06, 2.65467e-06, 5.09779e-06, -1.02348e-06, -5.44444e-06, -2.15564e-06, 2.65689e-06, -4.84205e-06, -6.47902e-06, -3.59683e-06, -3.64894e-06, -5.50084e-06, 6.15566e-08, 5.27898e-06, -4.9742e-06, 1.57527e-07,
-0.0154693, -0.000248182, -0.000106945, -3.5261e-06, -0.000210503, 0.000232087, 2.07725e-05, -8.97346e-05, 4.72382e-05, -4.52034e-05, -0.000173656, -7.37386e-05, -0.000230039, -0.000226184, -0.000367757, -5.24402e-05, 5.4258e-05, -0.000167749, 0.000209888, 0.000185359, -0.000125297, -0.000150043, 0.000161547, -3.14699e-05, 7.90228e-05, 0.000195289, -7.95535e-05, 1.81934e-05, -6.79202e-05, 0.000124472, -0.000179786, -0.000148087, 0.000151596, 6.31856e-05, -0.000211934, -6.27482e-05, -2.79778e-06, -0.000169306, -0.000392735, -0.000269104, 0.000364492, 0.000256656, -7.35182e-05, -1.77231e-05, 1.02867e-06, 0.000241707, -6.36589e-05, -0.000106341, -0.000218711, 6.97152e-05, -9.06315e-05, 2.1787e-05, -6.84949e-05, 0.000291623, 0.00010853, 0.000149515, 9.66014e-06, -0.00017627, -0.000182661, -5.49716e-05, -0.000239019, 0.000113827, 0.000294012, -0.000252168, -8.28582e-05, -5.06236e-05, -0.000144442, -7.71738e-05, 0.000167743, 0.000107157, -2.88346e-05, -0.000111765, 8.57985e-05, -5.58337e-05, 0.00020888, 0.000219817, -0.000122312, -6.55685e-05, 0.000278901, -0.000159722, 4.34329e-05, 9.62793e-05, -5.29103e-05, -1.22089e-05, 0.000121778, 0.000107393, 9.31856e-05, -4.51538e-05, -0.00010179, 0.000113483, -9.76544e-05, -0.000124585, 8.38539e-05, 6.7122e-05, 0.00019099, 0.000237649, 9.44624e-05, 9.05498e-05, -9.02945e-05, 0.000176997,
-0.111083, -0.000487983, 1.22919e-05, -0.00177811, -0.000255125, 0.000431673, -0.000995818, -0.000484071, 0.00131896, 0.00225378, -0.000380012, 0.00216106, 0.000844029, -0.00120877, 0.00186523, 0.000100935, -0.0011302, -0.000220666, 0.000850764, 0.00108709, -0.000715937, 0.00233308, 9.06115e-05, 0.00133438, -0.00031672, -0.000332774, 0.000971415, -0.00141928, 0.00112425, -0.00195665, 0.000213262, -5.41393e-05, -0.000625491, -0.000271684, 0.000238538, -0.00134782, -0.00118545, -0.00051717, -0.000179082, -0.000489945, 0.000930954, 0.00241404, -0.00172583, 0.00205817, -5.61244e-05, 0.00105533, 0.000201738, 0.000155079, 0.000474942, 0.000697573, -0.000891332, 0.000278731, -0.000621164, 0.000133052, 0.00120059, -0.00207881, 0.000776222, 0.000969374, 0.0013911, 0.000740903, 0.000751617, -0.000901764, -0.000205331, -0.000481937, 0.00104009, 0.000496687, -0.000863633, 6.0429e-06, -0.000706976, -0.00266222, -0.000447966, 0.000698544, 0.000455509, 0.00154527, -0.00110187, 0.0033004, 0.000941368, 0.00190918, -0.000500779, 0.000597417, -0.000893377, -0.001541, -0.00123416, -0.000611187, -0.00192712, 0.00062385, -0.00110473, -0.000174488, -0.000667588, 0.000321383, -0.00118863, 0.000547032, -0.00018576, -0.0012253, -0.000909773, -0.000399187, 0.00148559, 0.00115246, 0.000769775, 0.000184751,
-0.00538007, 4.74897e-05, 3.83273e-05, -0.000123273, 1.81522e-05, -5.82949e-05, 6.06361e-05, -4.8995e-05, 1.05876e-05, -2.96114e-05, 1.41369e-05, -5.98497e-05, -3.54837e-07, -5.08924e-08, 4.93121e-07, -1.62507e-05, -5.91073e-05, 5.86203e-05, -8.3797e-06, 5.61681e-05, -7.12938e-05, -6.81887e-05, -4.97238e-05, 6.6831e-05, 3.65078e-05, -7.54234e-05, -8.00143e-05, -7.8991e-05, -4.8051e-06, -2.68901e-05, 3.30475e-05, 1.32911e-05, 1.3993e-05, 6.25601e-05, 1.97844e-05, 3.92084e-06, 1.0001e-05, 4.31479e-05, -5.82905e-05, -5.51069e-05, 6.56846e-06, -3.37794e-05, -0.000108976, 6.78824e-05, -2.5743e-05, -3.08666e-05, -7.66449e-05, 7.77191e-05, -4.68053e-05, 6.15973e-05, 6.85959e-05, -4.64813e-05, 3.66815e-05, -5.06991e-05, 2.30758e-07, -3.6053e-05, -9.32748e-05, -8.00224e-06, 5.27987e-05, 8.01086e-05, -4.02762e-05, 2.79795e-05, 5.19722e-05, 9.43761e-05, 3.26179e-05, 4.74942e-05, 1.0748e-05, 1.06769e-05, 1.89309e-05, -7.98564e-05, 1.0638e-05, 4.25772e-05, -1.18005e-05, -4.01073e-05, -6.08262e-05, -8.83548e-05, 6.04823e-05, -2.13313e-05, -5.1601e-05, 9.365e-05, -5.90582e-05, -3.55045e-05, 6.35089e-05, 2.68776e-05, -6.29713e-05, -1.09764e-05, 8.95822e-06, 3.425e-05, -2.28361e-05, 0.000123029, -2.07354e-05, 6.10203e-06, -2.88917e-05, 2.58523e-05, -6.19677e-05, 8.85572e-06, -5.64014e-05, 8.99458e-05, 9.23968e-05, -5.50116e-06,
0.0148662, 0.000138236, -5.30798e-05, -0.00016852, 0.000302062, -0.000136453, -0.000112242, -0.000133256, -1.12076e-06, 2.98092e-05, 0.000233478, 6.54076e-05, 5.70197e-05, 0.000179924, -0.000202853, 7.74884e-05, 1.18406e-05, -8.4384e-05, 0.000291433, -0.0001898, -3.57494e-05, -0.000104083, 0.000214246, 9.12014e-05, 0.000259448, 0.000221546, -0.000297227, -9.11925e-05, 0.000312595, -0.000220931, 3.24422e-05, 4.84447e-05, 4.39697e-05, -4.51314e-05, -3.34076e-05, 0.000124598, -0.000265245, -0.000191736, -0.000148631, -0.00031781, 0.000126777, 4.92562e-07, -4.88127e-05, -0.000134424, 1.53254e-05, 0.000152509, 8.56073e-05, 6.33058e-05, -6.68599e-05, 0.000133114, 0.000127511, 8.52652e-05, 0.00014181, 0.000236482, 3.1582e-05, 7.25474e-05, 0.000149576, 7.92216e-05, 0.000111123, 2.70105e-05, 9.23953e-05, -0.000287861, 6.02269e-05, -0.000259804, -0.000116044, 2.52684e-05, -1.75951e-05, -8.21453e-05, -3.37924e-05, -0.000248405, 9.17234e-05, 2.02407e-05, 0.000148897, 6.01032e-05, -8.83003e-05, 0.000178497, 0.00024319, -0.000104421, -7.08102e-05, 3.42586e-05, -5.57448e-05, 0.000213729, 7.57189e-05, -0.000313184, -4.97161e-06, -0.000255848, -0.000235679, 0.000187766, -0.000157859, 2.67036e-05, -3.75994e-05, 8.25972e-05, 0.000231554, 0.000259048, -9.20078e-05, -8.66896e-05, 9.08328e-05, 0.000102753, -0.000220262, -4.48525e-05,
-0.0183455, -0.000106368, -2.26751e-05, 7.33812e-05, 1.74552e-06, 0.000114156, 3.94391e-06, 0.000151598, -4.27824e-06, -3.9212e-05, 0.000116041, -1.8642e-05, -0.000359934, 0.000167041, -8.93349e-05, 5.82213e-05, 0.000245363, 0.000177835, -6.49124e-05, 0.000150515, -0.000159731, -0.000371971, -3.17361e-05, -0.000374093, -0.00025257, 0.000351158, -4.72152e-05, 0.000336024, 4.57539e-05, 0.000239373, 0.000385681, -0.000256183, 0.000152831, -2.15445e-05, -0.000284026, 0.000120233, -8.01793e-05, 6.27592e-05, -0.000133762, -6.74247e-05, 0.000212486, -2.39328e-05, -0.000232524, 0.000239804, 1.0323e-05, 0.000130945, -0.000377307, 0.000182447, 0.000113859, -1.55662e-06, 0.000147946, -1.78088e-05, -3.58881e-05, -7.14875e-05, 0.000380911, -0.00015936, -0.000183053, -4.17801e-05, -1.81563e-05, 0.000143701, 5.22691e-05, 7.40382e-05, 3.10608e-05, -0.000197381, -6.82123e-05, -4.8471e-06, -0.000271617, -0.000119895, 6.87564e-05, -0.000214152, 0.000127788, 6.22424e-05, -2.89012e-05, -0.000288027, -0.000208129, -0.000137118, 0.000138002, -2.91815e-05, 0.000146094, -6.95727e-05, 0.000165773, 1.37397e-05, 5.56066e-05, -0.000275922, 0.000204862, -0.000283101, -0.000443484, 0.000237527, 5.57461e-05, -0.000161195, 0.000247127, 0.000135178, -1.81759e-05, -0.000165839, -5.17094e-05, 0.000160272, 0.000300995, 0.000136045, 0.000138002, 0.000225638,
0.00332664, -1.26906e-05, 3.13256e-05, -5.88093e-06, 1.90286e-06, 1.9494e-05, -1.76624e-07, -1.10588e-05, -4.45599e-05, 9.52769e-06, 6.76965e-05, 5.38869e-05, -8.20276e-06, -2.15103e-05, 2.28461e-05, -5.51493e-05, -1.56468e-05, 2.94245e-05, 2.79082e-05, 3.98857e-07, 2.13783e-05, 1.24124e-05, -3.46969e-06, -1.35542e-05, 3.65298e-05, -1.22871e-05, 3.34219e-05, -9.607e-06, -7.63723e-05, 9.67952e-06, -9.07414e-05, 3.93456e-05, 3.76415e-06, 3.15997e-05, -2.82911e-05, -4.69318e-06, -2.50726e-05, 3.97912e-05, 3.52887e-05, 4.68995e-05, 4.44264e-05, 6.94869e-06, 3.12379e-05, 2.00891e-05, -1.79787e-05, 2.56372e-05, 4.29612e-05, -1.37782e-05, 9.34457e-06, 1.31043e-05, -2.1607e-05, -5.23359e-05, -2.05837e-05, -1.20435e-05, -3.66579e-06, 3.58224e-05, 1.14902e-05, 2.41826e-05, -5.5757e-06, -4.96971e-05, -1.71153e-07, 1.23492e-05, -2.48688e-05, 9.79423e-06, 2.23157e-05, -6.36041e-05, -2.15897e-05, 2.19337e-05, 3.38022e-05, 4.38952e-05, 3.7483e-05, 6.86706e-05, -1.05784e-05, -4.7678e-05, -3.78137e-05, -4.55536e-05, -1.39303e-05, 7.12349e-06, -1.8838e-05, 4.11698e-05, -5.6062e-05, -3.25974e-05, 2.99971e-06, -6.20117e-05, -1.6149e-05, -1.9316e-05, -2.08292e-05, 4.58024e-05, 4.25306e-05, -3.96885e-05, -1.41223e-05, -3.83121e-05, -9.30159e-06, 5.03334e-07, 5.57279e-05, -3.77859e-05, -1.12657e-05, -4.22104e-05, -3.00818e-05, -4.1218e-05,
-0.000197571, -1.5502e-06, -1.17839e-06, 2.14232e-07, -8.98317e-07, 1.21595e-06, 3.08636e-07, 3.69745e-07, -1.41577e-06, -1.34261e-06, -2.16051e-08, 5.04552e-09, 2.8142e-06, -9.00584e-07, 9.12341e-07, -9.00087e-07, -3.97801e-07, 3.60008e-06, -5.49932e-06, 1.31303e-06, -1.75531e-07, -6.86018e-07, 8.53576e-07, 2.24705e-06, 2.0159e-06, -1.96984e-06, 1.80211e-06, -9.2389e-07, -1.0281e-06, -4.79631e-06, 3.76723e-06, 3.39933e-06, 1.56672e-07, -5.9391e-06, -2.7184e-06, 2.21608e-06, -3.98977e-06, 3.67729e-06, -4.36728e-07, 3.41596e-06, 3.23324e-06, -2.19939e-06, -7.85232e-07, 6.98627e-07, -2.15948e-07, -6.35803e-07, -9.77108e-07, 9.98744e-07, -1.76893e-06, 8.51341e-07, -1.32598e-07, 4.43222e-06, 1.1667e-06, 2.79434e-06, 1.59831e-08, -1.36003e-06, 1.82316e-06, 1.56217e-06, 1.22993e-08, 2.51113e-06, 1.94955e-06, 1.20784e-06, -1.4166e-06, 8.10105e-07, 5.20945e-06, -2.52563e-06, -9.48975e-07, 3.69436e-06, -2.69582e-07, 4.9777e-08, 3.23332e-06, -8.5454e-07, -2.83227e-07, -1.76139e-07, 2.2229e-06, 8.69216e-07, -5.1544e-08, 2.93938e-06, 9.87765e-07, 4.66175e-06, 1.7672e-06, -6.88958e-07, 1.5496e-06, 2.61103e-06, -9.34031e-07, 1.23752e-06, 2.60436e-06, 1.77676e-06, 2.37038e-06, 2.27544e-07, -3.33611e-06, -2.30568e-06, 2.86367e-07, -2.30463e-06, 2.1067e-06, 1.93751e-07, 4.9277e-08, 1.3746e-06, 1.3604e-06, 9.03697e-07,
-0.00641683, 9.16277e-05, 6.6816e-05, 9.78504e-05, -1.04166e-06, 1.20135e-05, -1.6951e-05, 3.64641e-05, -2.87921e-06, 9.91347e-05, -5.26805e-05, 5.31506e-05, 8.94092e-05, 0.000134326, 3.63388e-05, 1.95752e-05, 6.63911e-05, 2.29018e-05, -3.46526e-06, 1.96914e-05, -2.32657e-05, 0.000110081, -0.000106418, 2.5413e-05, 9.80783e-05, -6.61048e-05, 4.81116e-05, -0.000130906, -7.27028e-06, 6.60088e-06, 1.6311e-05, -8.72583e-05, 0.000121689, 4.56329e-05, -2.37767e-06, 6.34398e-06, -5.33783e-06, -3.33484e-05, -6.21908e-05, -0.000127467, 3.87048e-05, 5.79218e-05, -3.58032e-05, 8.17828e-05, 1.34648e-05, -8.68233e-07, 4.56564e-05, 1.05633e-05, -1.16092e-05, 3.91317e-05, 5.31627e-05, -1.5178e-05, 2.63242e-05, 5.3642e-06, 7.53044e-05, 2.52493e-05, 5.99034e-05, -8.32228e-05, -2.61025e-05, -5.30268e-06, -1.96645e-05, 9.44307e-05, 5.37941e-05, -1.58694e-05, 5.58666e-05, 8.18931e-06, 2.62587e-05, 7.84066e-05, 8.65106e-05, -6.47775e-05, -9.09649e-05, -5.21124e-06, 4.84583e-05, -4.78574e-06, 8.06302e-05, 5.03458e-05, 2.7316e-05, 9.92432e-05, 2.70619e-05, -9.018e-06, -3.96338e-05, 1.44593e-05, -2.81275e-06, 6.75518e-05, 1.52394e-05, 3.41579e-05, 2.25405e-05, -0.000109931, -4.20953e-05, 7.53873e-05, 7.22308e-05, 3.00321e-05, 3.64849e-05, 1.40008e-05, 0.000123253, 0.000131251, -2.54887e-05, 8.49304e-06, 4.0263e-05, 6.61768e-05,
-0.00648172, -6.46166e-05, -6.67727e-05, 2.59796e-05, 1.07628e-05, 4.56448e-05, -2.97908e-05, 6.92821e-05, 9.97262e-05, -0.000137496, 9.41745e-05, -7.84228e-05, -8.87596e-06, 7.10769e-05, 8.04437e-06, -3.71854e-05, -1.13455e-05, -0.000115109, 4.14904e-05, 7.01148e-05, -9.73344e-05, -7.3946e-05, -8.33711e-05, 8.32312e-05, -6.05974e-05, 5.81056e-05, -1.77885e-05, 4.18381e-05, 2.29977e-05, -0.000117385, -7.45849e-05, 8.52427e-06, -0.000111307, -4.42657e-05, -8.19735e-05, -2.9954e-05, -9.74324e-05, -0.000132235, 0.000103661, 2.16382e-07, 2.47312e-05, -6.70385e-06, 8.13643e-06, 0.000107794, 3.22522e-05, -3.71037e-06, 4.91771e-05, -7.66237e-05, -3.07403e-05, 6.52742e-05, 3.55005e-05, -3.14948e-05, 7.71312e-05, 4.16889e-05, -2.7456e-05, -3.01296e-05, 0.000123387, 6.34281e-06, -2.48229e-05, -7.89743e-06, 9.24564e-06, 3.09902e-05, -9.06566e-05, -6.64023e-05, 6.45016e-05, -6.35548e-05, 6.84062e-05, 3.48052e-05, 0.000128396, 2.75631e-05, -7.49759e-05, -6.23258e-05, -2.60734e-05, -7.89165e-05, -2.48439e-05, 8.03743e-05, 0.000113028, 9.07769e-05, 2.90231e-05, -4.08583e-05, 4.72664e-05, 2.25124e-05, 7.94104e-05, 8.39635e-05, 2.28223e-05, -0.000131396, 0.000155385, 7.85428e-06, -3.12623e-05, -9.86835e-05, 2.06526e-05, 3.6392e-05, -1.71858e-05, -2.34302e-06, 2.60072e-05, -6.69134e-05, -4.18233e-05, -0.000104677, -5.02985e-05, -7.87708e-05,
0.0133646, -0.00017031, -2.43593e-05, 2.92585e-05, -0.000152755, 0.000230371, -7.94001e-05, 0.000160503, 1.17246e-05, -2.49275e-05, 0.000159185, -8.67029e-05, 0.000352821, 0.000100806, 0.000158224, 9.72745e-05, 8.92498e-05, 4.3849e-06, 0.000136702, -0.000296644, -7.34866e-05, -1.38015e-05, -1.45579e-06, 0.000127571, -0.000202487, -0.000122153, -0.000132728, -0.000110552, -0.000104617, -0.000201717, 5.93116e-05, 7.99179e-05, -3.11281e-05, -1.8373e-05, -1.47378e-05, 8.06002e-05, -0.000108432, 4.96677e-05, 7.16214e-05, 0.000159202, 3.18052e-05, 4.40196e-05, 0.000139111, 0.000235349, 0.000153209, -1.09291e-05, 0.000180385, 1.49377e-05, -0.000111889, 8.03297e-05, 6.21514e-05, -0.000125651, 0.000184896, -2.69925e-05, -6.96199e-05, -0.000159404, -0.000189567, -5.65917e-05, 6.22981e-05, 5.6825e-05, 3.16168e-05, -0.000140029, 7.84605e-05, -4.44503e-05, -3.21059e-05, 3.42528e-05, 2.60493e-05, -8.47554e-05, -0.000157576, -4.77026e-05, -6.01427e-05, -8.87583e-05, -0.000193021, -0.000151741, -0.00014627, 0.000162673, -4.28067e-05, -4.40538e-05, -4.36697e-05, 0.000105222, 0.000223735, -0.000228722, 0.00020359, 7.01384e-05, 0.000126058, 9.44939e-05, 0.000126398, -0.00032363, 5.91532e-05, 4.80038e-05, -0.000156326, -5.81198e-05, 7.57263e-05, 0.000133433, -0.00019452, -5.42036e-05, -6.71236e-05, 9.82159e-05, 0.000163119, -0.000206614,
-0.0266169, -0.00028152, -0.000178215, -1.08474e-05, -0.00013429, -0.000159784, 0.000182376, 0.000409284, -0.00064297, -7.66231e-05, -0.000132575, -0.000118009, -0.000484349, -0.000145348, 0.000301715, 4.58372e-05, 9.63611e-05, 0.000393456, -5.05839e-05, 3.78375e-05, 0.000114196, -0.000129995, 0.000101784, 0.000358259, -0.000212977, 8.78991e-06, 0.000267662, -0.000207313, 0.000177679, 0.000105356, 0.000383587, -2.78203e-05, -0.000172078, 0.000263173, -0.000343484, -0.000226749, 0.000151495, -7.29105e-05, 2.31312e-05, -0.000265758, 9.03633e-05, -7.20092e-05, 0.000252028, 0.000282063, -0.000414592, 0.000583949, 0.000126181, 0.000326881, -0.000131828, -3.83969e-05, 0.000348789, 9.21557e-05, 0.000198054, 0.000303863, -0.000144787, -0.000381074, -2.91866e-05, -0.000202135, -0.000239865, -0.000590711, 0.000298374, -0.00015207, 0.000191609, 0.000298734, 0.0008538, -4.50759e-05, 0.000256385, 0.000230678, -4.69429e-05, -8.06201e-06, -0.000355467, -0.000244683, 6.58429e-05, -0.000309513, 0.00011786, 0.000127002, 0.000223635, 0.000306621, -0.000280358, 0.000177865, -0.00038905, -5.1792e-05, -0.000507087, -4.48549e-05, -0.000299844, 9.09436e-05, 0.000443045, 8.53144e-05, 0.000297445, 0.000175891, -7.93012e-05, 4.2924e-05, -0.00032568, 0.000218673, 0.00047383, 0.000153753, 0.000616012, 0.000273198, 0.000155913, 0.000212303,
0.00597301, -4.81272e-05, 0.000100718, -8.15108e-05, -2.53893e-06, 2.47345e-05, -7.85147e-05, -6.3893e-05, -6.04308e-05, 1.74348e-06, -9.30074e-05, -1.28895e-05, -2.59735e-05, 3.41724e-06, -2.54423e-05, 7.88225e-05, 5.06876e-05, 1.59601e-05, -4.56704e-05, 3.53232e-05, -8.09193e-05, 6.16515e-05, 9.48604e-07, 5.82114e-05, -3.04708e-05, -1.30692e-05, -2.41661e-05, -1.9582e-05, -0.000104181, 8.39274e-05, 1.88025e-05, -8.0651e-05, 6.19998e-05, 1.78317e-05, 5.21244e-05, 1.72115e-05, 4.66467e-05, -5.23583e-05, 1.32495e-05, -1.39118e-05, -6.30497e-05, -0.000101805, 5.97163e-05, -2.98181e-05, 7.00758e-05, 9.78024e-05, -0.000110293, 5.74923e-05, -1.35242e-05, -1.99436e-05, -2.73694e-05, 0.000130674, -6.74247e-05, 3.01462e-05, -6.63041e-05, -2.49005e-06, 6.71321e-05, -4.67152e-05, 1.14713e-05, -3.77606e-05, 1.87265e-05, -5.87112e-05, -0.000126142, -9.14818e-05, -5.69318e-05, -2.5378e-05, 7.18386e-07, 9.1525e-05, 9.99575e-05, -3.84592e-06, 1.96455e-05, 2.03712e-05, 8.36162e-07, -5.11474e-06, -4.34079e-06, -3.64805e-05, -8.40741e-05, -4.17371e-05, 3.52397e-05, 9.62201e-05, -1.68689e-05, 2.50131e-05, 2.92617e-05, 1.84137e-05, -2.70586e-05, -7.09771e-05, 7.06127e-05, 8.32232e-05, -3.31084e-05, -0.000108557, -2.91142e-05, -5.55838e-05, -0.000138671, -5.56909e-06, -3.88825e-05, -0.00010933, -3.85194e-05, 3.69765e-05, 5.58681e-05, 7.36175e-06,
0.00506311, -9.98648e-05, 3.02046e-06, -6.1215e-05, 2.96987e-05, -9.02427e-05, 1.0363e-05, -8.21168e-05, 3.54729e-05, -0.000101717, -3.35259e-05, 1.63008e-05, 3.33676e-05, -8.1509e-06, 2.54928e-05, 1.97215e-05, 2.94683e-05, 0.000127157, -1.6047e-05, 6.20539e-05, 8.55526e-05, -4.45045e-06, 1.9642e-05, 1.62709e-05, 8.44956e-05, 5.35852e-05, -0.000116439, -5.62463e-05, 4.81091e-06, 3.13292e-05, 6.43026e-05, 4.72632e-05, 1.57845e-05, -2.53928e-05, -5.85026e-05, 3.10358e-05, -9.33546e-05, -8.81167e-05, 1.37881e-05, 3.57621e-05, -8.22751e-05, -9.44703e-05, 3.32525e-05, 4.95182e-06, 3.98955e-06, 3.78191e-05, -2.41121e-06, -3.53744e-05, -2.70361e-05, -8.3924e-05, 6.67153e-05, -6.06942e-05, 0.00011783, -1.94654e-06, -2.92004e-05, -2.25854e-05, 4.61884e-05, 1.02189e-05, 4.13097e-05, 6.66769e-05, 2.382e-05, -0.000102552, 1.82325e-05, -6.44534e-05, -2.26783e-05, 3.8105e-05, 4.03559e-05, 5.58085e-05, -1.27456e-05, 3.64901e-05, -7.54555e-06, 3.33252e-05, -4.32848e-05, 5.5222e-05, -5.383e-05, -2.75596e-05, -7.83459e-05, 5.63845e-06, 6.67725e-06, 4.6939e-05, 6.32725e-05, -2.6596e-05, -4.34936e-05, -3.44254e-05, -3.64121e-06, -9.72787e-05, 2.67369e-05, -6.88493e-05, 6.98536e-05, 3.78033e-05, -0.000102543, -0.000117774, -7.14599e-05, -3.08881e-05, 3.10428e-05, 3.46317e-05, -3.63279e-05, -6.65619e-05, 2.86098e-05, -0.000108088,
0.0136227, -9.22404e-05, 4.65729e-05, -6.8946e-05, -0.000214424, 0.000140592, -0.000239692, -3.98172e-05, 0.000197502, -6.47918e-05, -0.000139892, 0.000272865, 0.000141201, -6.79287e-05, 1.8787e-06, 9.77588e-05, -1.08427e-05, 3.89092e-05, -2.93971e-05, 0.000277602, 0.000103413, 0.000175983, 1.93411e-05, 3.81385e-05, 0.000161402, 0.000232841, 3.81033e-05, 0.000103715, 9.52555e-05, -0.000162831, -1.98633e-05, 9.35671e-05, 5.29287e-05, -0.000178979, 2.28548e-05, -0.000164412, 0.000299244, 4.94289e-06, -6.51996e-05, 0.000242065, -0.000183416, -0.000188503, 2.24658e-05, -0.000137864, 0.000128551, 0.000129204, -2.8518e-05, -3.38982e-05, -8.78306e-05, 7.09561e-05, -8.23678e-05, 0.000122311, -0.00035324, -1.7432e-05, -0.000104815, -0.000137643, 6.79587e-05, 0.000166098, -0.0001288, -0.000146802, 0.000111498, -0.000216526, -9.76235e-05, 6.31895e-06, 0.000147005, 6.97849e-06, -4.45738e-05, -2.1257e-05, -9.63844e-05, -2.18649e-05, -0.000216894, 6.08847e-05, 3.61274e-05, -1.87517e-05, 0.000152649, 7.43781e-05, -2.28972e-05, 5.30497e-05, 0.000111896, 7.1284e-05, -0.000258452, 0.000193157, -6.31043e-05, -0.0001044, -0.00010721, -5.64283e-05, -5.50335e-05, 8.26726e-05, -1.53003e-05, -9.99827e-05, -0.000106209, -4.82505e-05, 2.67641e-05, 3.15413e-05, 0.000203715, -0.000169983, -6.3949e-05, 4.73203e-05, 5.40584e-05, 5.9514e-05,
-0.0156184, -0.000124985, -2.89862e-05, -0.000199374, 0.000225674, 0.000117749, 0.000163968, -1.60097e-05, -3.48966e-05, -0.000172222, 4.11374e-05, -0.000179768, -6.95566e-05, 8.15469e-05, -0.000380657, -7.65916e-05, 0.000340305, -6.80577e-05, 0.000155305, -7.40019e-05, -5.77501e-05, 3.13128e-05, 2.53073e-06, 0.000158581, -3.41819e-07, 2.7228e-05, -0.000416061, 1.32675e-05, -5.17933e-05, 8.50501e-05, -6.13805e-05, 9.19233e-05, 0.000289727, 5.94146e-05, -0.000100793, 3.76894e-05, 9.07288e-05, 0.000135774, -0.000109537, -8.18741e-05, -0.00015866, -0.000145878, -6.14647e-05, -0.000229249, -0.000170587, 0.000101228, -5.90227e-05, 7.42621e-05, 9.40537e-05, 8.20794e-05, -7.52224e-05, -7.19416e-05, 0.000206649, -0.000159984, -0.000118714, 0.000235703, -0.000112406, -0.000258759, -3.71286e-05, 0.000130252, -8.81891e-05, 0.000180504, 4.49827e-05, -1.47802e-05, -0.000129762, 0.000328436, 1.68192e-05, -8.2416e-05, -0.000172466, -0.000226016, -0.000190391, -6.38294e-05, -2.80799e-05, -0.000158711, -3.91616e-06, 0.000228499, 2.3063e-05, -6.12619e-05, -0.000256187, 0.00013055, -3.45897e-05, 0.000230222, -0.000127401, -6.76912e-05, -4.91249e-05, 0.000176353, -0.000165959, 0.000142286, -0.000162263, 0.000140805, -0.000147839, 7.61133e-07, 0.000137604, 0.000107498, -0.000116321, -4.79519e-05, 9.2404e-06, -6.80255e-05, 0.000374142, -0.000340955,
0.00470602, 4.67175e-06, -2.06683e-05, 2.03762e-05, 6.65378e-05, -9.80753e-06, 3.36866e-07, 7.14819e-05, -4.19771e-05, -1.71012e-05, 2.39493e-05, -5.01202e-06, 4.18195e-05, 5.05311e-05, -2.24929e-05, 8.88445e-05, -1.44285e-05, 2.22412e-05, -2.49613e-05, 9.65634e-05, 7.97106e-05, 7.33824e-05, -6.76248e-05, -1.81394e-05, -6.14114e-05, -8.13854e-05, -4.73297e-05, -2.71877e-05, 5.82258e-05, 1.77061e-05, 4.26022e-05, 4.72127e-05, 1.1302e-05, 2.9002e-05, -2.14014e-05, -4.83324e-05, -6.61581e-05, -1.25461e-05, 7.30891e-05, 5.79956e-05, 2.4823e-05, -0.000120161, -4.11008e-05, 3.22102e-05, 4.37354e-06, -2.68825e-05, -1.77249e-05, 2.02461e-05, 2.45499e-05, -7.88347e-05, -3.92206e-06, -7.32193e-05, 6.52897e-05, 1.41091e-05, -5.94323e-05, 1.06353e-05, 9.74586e-06, 2.97126e-05, -1.56616e-05, -6.37048e-05, 3.20318e-05, -1.03279e-05, 4.32099e-06, 3.29701e-05, -4.17548e-05, -4.28719e-05, -2.42702e-05, 0.000112733, 9.71184e-05, 1.41771e-05, 1.17234e-05, 0.00010989, -5.65861e-05, 4.1499e-05, 6.29161e-06, -1.83158e-05, 2.42128e-05, 6.94437e-05, 1.17047e-05, -1.67126e-07, 1.6387e-05, 3.6572e-05, -2.23028e-05, -1.83067e-05, 7.18546e-05, -2.29306e-05, -2.16938e-05, -3.62299e-05, -4.89268e-05, 7.89316e-05, -4.02366e-05, -4.83734e-06, -2.89323e-06, 3.15117e-05, -7.97411e-05, 1.75175e-06, -2.0188e-05, 6.99755e-05, -3.14052e-05, 1.5668e-05,
0.010729, -3.20571e-05, -1.62268e-05, 4.61457e-05, 4.09872e-05, -0.000159802, 5.66962e-05, -6.70597e-05, 0.000184707, 0.000151618, 0.000181181, 5.37844e-05, -9.43185e-05, -0.000115135, -3.92026e-05, -6.16321e-06, -3.3321e-05, 3.94491e-05, -5.21987e-05, 2.0993e-05, -0.000129089, 0.000157057, 3.64925e-05, 4.58622e-05, -0.000104943, -3.69101e-05, -1.12656e-05, 2.09056e-05, 0.00018915, -0.000131923, 0.000209178, -4.86791e-05, -0.000159768, -0.000213707, 0.000119164, -9.71145e-05, 8.65991e-05, 0.000109689, 4.55596e-05, -9.53541e-05, 1.48663e-05, -8.26943e-05, 3.59341e-05, 1.82809e-05, -3.38615e-05, -0.000231988, 1.31689e-05, 0.000153771, 8.33496e-05, 0.000183831, 8.02275e-05, 7.51212e-05, -4.14164e-05, -8.41488e-05, 4.34436e-05, 0.000117202, 0.000285226, -7.60905e-05, -4.20396e-05, 0.000115663, -6.21746e-05, -0.000106919, -0.000185821, -7.03426e-05, 9.00173e-05, -1.04448e-05, -0.00016667, -6.29298e-05, 7.80371e-05, -6.22031e-05, 0.000151588, 0.000250569, 9.20225e-05, 3.46939e-05, 0.000106541, 4.1845e-05, -2.48083e-05, -0.000170818, -2.82761e-07, -5.80874e-07, -6.87241e-05, 6.28255e-05, -3.05007e-05, 3.9221e-07, 9.60929e-05, 9.45832e-05, 0.000189056, 2.37208e-05, 3.41603e-05, 0.000105081, 4.42171e-05, -0.000208554, 0.000158031, -0.000105035, 6.49042e-05, -0.000184864, 6.12818e-06, 9.91313e-05, -6.28442e-05, 3.95457e-05,
0.00759028, -4.38412e-06, 2.081e-05, -0.000107379, -3.52131e-05, -2.94627e-05, -6.01207e-05, -1.8758e-05, -4.64635e-05, 1.35589e-05, 6.88977e-05, -9.34711e-06, 8.0249e-05, -7.60901e-05, -4.98323e-05, 2.20174e-05, -8.61378e-05, -0.000116005, -0.000143245, 0.000174259, -4.90502e-06, 1.49553e-05, -7.91497e-05, -0.000131115, 6.31522e-05, -7.51971e-05, -1.76465e-05, 5.13262e-05, -9.82652e-05, 5.02051e-05, -8.76319e-05, 3.05013e-05, 0.000109304, -6.92959e-05, -2.91754e-06, -4.66554e-07, 2.91971e-05, 0.000134557, -1.291e-05, -6.49005e-05, 8.32519e-05, 8.92224e-05, -4.33497e-05, 2.13243e-05, -6.90482e-06, 1.91068e-05, -1.28485e-05, -3.36761e-05, 3.24907e-05, -0.000122863, -2.81431e-05, 2.34929e-05, 2.80384e-05, 0.00012142, -7.47552e-05, -3.64635e-05, -2.05644e-05, 6.25356e-05, -1.21985e-05, -1.81781e-05, -9.5505e-05, -0.000165166, -0.000100923, 4.51416e-06, -5.09784e-05, 5.24238e-05, 0.000101872, 1.74927e-05, -2.19165e-05, 3.94027e-05, 2.59112e-05, -3.87224e-05, 0.000108496, 6.31934e-06, -2.90437e-05, -1.15748e-05, 3.29428e-05, -0.000115796, -5.57594e-05, 0.00013555, 1.9566e-05, 9.10258e-06, 8.19834e-05, -0.000109533, 4.52945e-06, -0.000101238, 3.96131e-07, 5.12842e-06, 2.67774e-05, -4.86652e-05, 5.2335e-05, 2.78725e-05, -7.05244e-05, -9.40428e-05, 0.000154593, 8.27587e-06, 0.000172306, -4.83709e-05, -8.33381e-05, -4.15456e-05,
-0.00505888, 4.9277e-05, -4.44999e-05, -3.12037e-05, -3.68835e-05, 5.4171e-05, -3.50113e-05, 3.36877e-05, -7.73632e-06, -1.66043e-05, -4.33097e-05, 6.06967e-05, 2.78237e-05, -8.37198e-05, -1.77976e-05, 1.47016e-05, 3.63583e-06, 2.12665e-05, -5.72886e-06, -3.45924e-05, 5.35894e-05, -4.83842e-05, 3.52396e-05, 7.82703e-05, 5.79847e-05, 9.1345e-05, 7.20075e-05, -6.45701e-05, -1.22249e-05, -7.62242e-06, 3.81754e-05, -2.42813e-05, -4.35808e-05, 1.48108e-05, 3.41548e-05, 5.23177e-05, -5.59744e-07, 2.37152e-05, 8.88498e-05, -6.47342e-05, 0.000138882, -6.37326e-05, 2.30054e-05, 3.07065e-05, 1.19292e-05, 5.40287e-05, 3.44039e-05, 2.67316e-05, -1.62972e-05, 6.90218e-07, 4.12233e-05, 3.00274e-05, -2.39536e-06, 5.05301e-05, -5.68841e-05, -4.31389e-05, -3.06224e-05, 3.94754e-05, -2.53653e-05, 1.76766e-05, -0.000107052, -0.000125694, 3.97375e-05, -2.84787e-05, 3.42759e-05, 2.34636e-05, 7.36249e-05, 8.43609e-05, -4.88688e-05, -0.000112853, 1.32057e-05, -2.47754e-07, 4.89116e-05, 8.4037e-05, 2.76722e-05, 4.21095e-05, -1.38408e-05, 1.07618e-05, 8.95925e-05, -3.32256e-05, -2.2141e-06, -5.33033e-05, 1.11897e-06, -1.4261e-05, -4.13652e-05, 2.44211e-05, -0.000114669, -4.14507e-06, 2.65983e-05, -4.03262e-06, -8.49477e-05, -3.04563e-05, -8.90161e-05, 5.73719e-05, 1.21195e-05, 1.70637e-05, -6.06093e-05, -7.04068e-06, 3.55226e-05, 6.71173e-05,
-0.00707002, -7.0993e-06, 8.21552e-05, -5.73311e-05, 7.6649e-05, 1.67931e-05, -0.000129986, -3.52193e-05, 4.28105e-05, 1.45405e-06, 2.23865e-05, -0.000105436, -3.95796e-05, -3.89111e-05, -0.000103843, 0.000133319, -1.72223e-05, 6.72361e-05, -3.87109e-05, 3.96325e-05, -9.73937e-06, -6.8229e-06, 8.70072e-05, 9.16323e-05, 2.43848e-05, -0.000168422, -4.00261e-05, 2.96413e-05, -2.77884e-05, -2.0791e-05, 3.0117e-05, 1.62948e-05, -0.000174668, 3.89497e-05, -0.000118566, 2.69873e-05, -3.27921e-05, -2.14345e-05, 3.32755e-05, -9.32898e-06, -6.74604e-05, -7.68847e-05, -8.90389e-05, 3.13942e-06, 8.81117e-06, 7.43786e-06, 1.87639e-05, -0.000142597, 6.15338e-06, 3.61973e-05, 0.000110998, 4.8425e-05, -3.85365e-05, -9.69342e-05, 9.10999e-06, 8.59296e-06, 4.15694e-05, 9.24128e-05, -8.425e-05, -1.59734e-05, 4.11278e-06, 6.78056e-05, -0.000144264, 8.51418e-05, -6.78859e-05, -8.5691e-06, -1.79007e-05, -0.000118131, -6.35228e-05, 1.79623e-05, -2.21957e-05, 2.1721e-05, -5.82419e-05, -3.42088e-05, -0.000214781, -6.30648e-05, -4.89206e-05, 3.70126e-06, 5.89587e-05, -0.000156173, 3.38352e-05, -0.000121533, -6.58578e-05, -6.31991e-07, 6.46859e-07, 4.61336e-05, 0.00019367, -0.000137956, 4.78579e-05, 5.5918e-06, -5.32428e-06, -1.42854e-06, -5.38975e-05, -5.00453e-05, 4.98989e-05, 3.04235e-05, -4.64534e-05, 1.16054e-05, -4.25942e-05, 0.000175731,
-0.0129646, -2.79924e-05, -4.89604e-05, 0.0001678, 9.90361e-05, -1.3466e-05, -8.3177e-06, 2.71142e-05, 0.000111586, -3.61204e-06, -0.000134051, 5.50851e-05, -5.45456e-05, 4.34727e-05, -5.19162e-05, -2.2456e-05, 0.000251201, 8.55861e-05, 4.52919e-06, -0.000171696, 9.97317e-05, -1.22395e-07, -0.000109253, -0.000172465, 0.000153316, -0.000252044, 7.31153e-05, 0.000149806, 0.0001208, 0.000104069, -0.000208344, 0.000135558, -9.9911e-05, 5.70374e-05, -0.000294062, 0.000136504, 0.000274222, 0.000155073, 6.17759e-05, -7.61325e-05, -2.14742e-05, 3.66601e-05, -0.000265617, 9.59725e-05, -5.43906e-05, -1.82127e-05, -8.79354e-05, -0.000283207, 1.83721e-05, 1.76133e-06, -0.000108121, -6.0343e-05, 0.000190634, -8.00525e-05, 0.00018414, -0.000138501, 6.35379e-05, -0.000185233, -0.000153711, 9.32064e-05, -0.000112597, -3.98897e-05, 0.00021364, 1.16404e-05, 0.000271237, 6.30358e-05, 6.28311e-05, 5.66905e-06, -0.000283799, -6.03683e-05, -5.29336e-05, 4.14602e-05, -8.41692e-05, 0.000118683, -0.000258295, 0.000102895, -0.000120036, -1.18531e-05, -0.000186815, -9.4887e-06, -6.32815e-05, -9.25979e-05, 6.77632e-06, 8.47358e-05, -0.000207921, 1.07601e-05, 6.05636e-05, -0.000205984, -4.16016e-05, 1.78461e-06, -0.000121977, -7.6643e-05, 7.96635e-05, -0.000186043, 0.000241291, 5.081e-06, -0.000101931, -1.54871e-05, 0.000178508, 7.59456e-07,
0.0171465, -0.00015528, 0.000293171, 1.76701e-05, 0.000171802, -7.7623e-05, 3.25512e-05, -0.000208176, 0.000298472, 0.00027189, -1.66807e-05, -1.59375e-05, -6.7057e-05, -0.000240276, 2.12128e-05, 1.02189e-05, -0.000260096, -2.44856e-05, -0.000136095, 0.000118011, -0.000106866, 3.72696e-05, -0.000163843, -0.00011977, 6.63784e-05, 0.000219565, 0.000225043, -0.000277278, 0.000137065, -0.000110479, -3.81571e-05, 3.08693e-05, 0.000129278, -0.000412469, 5.24035e-05, 4.13001e-06, -0.000116689, 0.000153192, -8.42901e-06, -6.07776e-05, -0.000376801, 0.000258786, -0.000256369, 2.46728e-05, 0.000344747, -9.68457e-05, -0.000109526, 0.000249753, 0.000207191, -8.29744e-05, -0.000187522, 4.07159e-05, -0.000245371, -5.71354e-05, 0.000145562, 9.31867e-05, -1.34086e-05, -0.000339042, -0.000124458, -0.000188752, -0.000224918, -2.1787e-05, 0.000358432, -6.36326e-05, 1.88408e-05, -8.96235e-05, 0.00010152, -4.48739e-05, 0.000193386, -9.6502e-05, -0.000146785, -8.16967e-05, 0.000151478, -0.000128773, -0.000174227, 5.22658e-05, -0.000137304, -0.00012747, -1.15014e-05, -5.10077e-05, 0.000159624, 0.00044777, 0.000141362, 0.000272511, -4.72777e-05, -0.000104493, -4.27676e-05, 0.000125086, 8.00028e-05, 9.26745e-05, -3.28421e-05, 9.51038e-05, 6.2514e-05, 0.000207256, 4.35339e-05, -3.06443e-05, -1.36653e-05, 0.000113104, -1.68248e-06, 0.000200682,
-0.023401, -7.53886e-05, 0.000102866, 6.14841e-05, -0.00025094, 0.000102781, -0.000260238, -6.99668e-05, -0.000329833, 0.000402261, -0.00026647, 0.000132937, -0.000227817, 0.000383488, 5.00038e-05, -0.000163428, -7.84583e-05, 8.68862e-05, -8.65722e-05, -0.000313099, 0.000137855, -3.97711e-05, 7.6245e-05, 0.000384857, 0.000373394, -0.000118004, -0.000264929, 0.000225192, -2.4384e-05, -0.000130738, 0.000502689, -0.000275561, -4.04816e-05, 1.37066e-05, -0.000169094, 0.000179951, 5.13695e-05, 5.82135e-05, -0.000156585, -0.000387008, -0.000227854, -0.000524164, -0.000309293, 0.000402516, 6.77394e-05, -0.000177278, 0.000125405, -0.000352789, 0.000293361, -0.00039022, 6.23003e-07, -0.00018939, -9.64428e-05, 0.000258159, -6.24322e-06, -7.09642e-05, -0.000166817, -7.72125e-05, -0.000226999, -3.35875e-05, -0.000243492, 7.86514e-05, -1.38302e-05, 0.000190333, 9.78576e-05, -0.000354809, -0.000393396, -0.000235056, 0.000590012, -0.000310228, 8.84777e-05, -0.000154182, -0.000628708, -0.000348088, -0.000159676, -0.000133557, 0.000172606, -8.09074e-05, 0.0001229, 0.000135733, -0.000115937, 0.000134793, -0.00042238, 0.000144811, 6.85825e-05, 0.00027988, 7.34649e-05, -0.000268837, -7.37325e-05, -0.00021418, 0.000226488, 0.000189424, -0.000213652, -0.000412437, 0.00020506, 7.79711e-05, -0.000177613, 4.81125e-05, -4.8651e-05, -0.000158118,
0.0590557, -0.000175287, -6.36384e-05, 0.00025588, -0.000173192, 0.000474336, 0.000206215, 0.000493466, 0.000759272, -0.0011493, 0.000545419, 0.00111286, 2.63983e-05, -0.000404598, 0.000588829, -0.00128354, -0.000700776, -6.5846e-05, 0.000363509, 0.000338211, 7.2729e-05, -0.000284371, -0.000264185, 0.00115348, -0.000284121, -0.000275087, 0.000875599, -0.000639874, -0.000594498, 0.000435624, -0.000404967, 0.000346098, -0.000200522, 0.000440626, 0.000543688, 0.00076108, 0.000843225, 0.000289699, 0.000396802, -0.00036158, -0.000607665, -0.000131156, -0.000714451, -0.00045486, 5.48883e-05, -0.000120561, 0.000201547, 0.000313484, 0.000391367, -0.000660843, -0.000867645, 0.000386688, 5.87877e-05, 0.00054334, -0.00164901, 0.000691969, -0.000663608, -0.0001405, 0.000152305, -0.000333208, -0.00042771, 0.000907497, 0.000607913, 0.000166875, 0.000445544, -0.00040736, 0.000506665, -0.000650028, 0.000461318, -0.00152023, -0.000754002, 8.67871e-06, 0.00114412, -0.00042582, 0.000189495, 0.000747592, 0.000269081, 0.000672641, -0.000538095, 0.000812075, -1.54405e-05, -0.000386245, 3.12357e-05, -0.000611805, 0.00125723, -0.000228971, 0.000581758, 0.00146346, -0.000445277, 9.79237e-05, -0.000519595, 0.000847179, 0.000166648, 0.000194653, -0.00087481, -0.000687864, 2.51148e-06, 0.000626845, 0.000633513, -0.000219125,
0.0140747, -9.41889e-05, -0.000166086, 0.000251832, -4.82692e-05, -3.43233e-05, 2.81834e-05, 4.42668e-05, -0.000227922, -0.000108991, 7.94601e-05, -0.000121826, 0.000156285, -9.1298e-05, 0.00018061, 0.000166239, 3.78141e-05, 3.24246e-05, 0.000103658, 0.0002042, 9.82648e-05, 5.62271e-05, -0.000175145, 0.000284546, 0.000229754, -1.6304e-06, 5.12196e-05, -5.96661e-05, -0.000209097, -6.26141e-05, -5.33292e-05, -0.000134222, -0.000160743, 0.000175666, -7.28739e-05, 4.99836e-05, -0.000140949, -6.37686e-05, -3.35617e-05, 0.000149736, 0.000139426, 4.20925e-05, 4.78859e-05, -0.000101599, -5.87372e-05, -3.4297e-05, 5.71562e-05, 0.00025327, -6.03158e-05, -0.000175268, 8.27304e-05, 9.69431e-05, -5.47119e-05, -0.00011623, 0.00010453, 7.11925e-05, 3.75123e-05, 0.000162731, 0.000165722, -2.16327e-05, 0.000179458, 7.62907e-06, -0.00012456, 6.62108e-05, -2.80862e-05, 6.28788e-05, -0.00016488, 0.000150289, -0.000122372, -0.000149005, -5.32443e-05, -0.000161253, -0.000176835, -6.91825e-06, 0.000108728, 0.000103244, 0.00020288, 0.000261265, 6.90572e-05, -3.1448e-05, 1.69582e-05, -0.000137034, -1.3727e-05, 8.23916e-05, -0.000124581, -3.96327e-05, -0.000383566, -4.55002e-05, 0.000135265, -9.37755e-05, 0.000160363, 2.33273e-05, -7.07074e-05, 0.000133176, 0.000236943, 9.40736e-05, -5.02904e-08, -0.000103144, 0.000103368, -0.000226539,
0.00407632, 3.07292e-06, -0.000110176, 4.1118e-06, -2.83434e-05, 4.09191e-05, 5.75071e-05, -1.21544e-05, -6.56699e-05, 7.25775e-05, -3.89827e-05, -2.85315e-05, 3.05477e-05, 8.34609e-06, -9.71918e-05, -7.11177e-05, -2.73847e-05, -2.08435e-06, 2.66736e-05, -8.13287e-05, -5.46036e-05, -9.47783e-05, -1.33818e-06, 6.39885e-06, 2.54249e-05, 3.52061e-05, -8.13815e-06, 8.61801e-05, 1.28434e-05, 1.16189e-05, 2.68409e-05, 3.174e-05, 2.63569e-05, 2.50013e-05, -1.82948e-05, -1.42123e-05, -2.04745e-05, 4.0153e-05, -3.42281e-05, 2.40245e-05, 5.79121e-05, -8.77239e-05, -5.10971e-05, -2.98007e-05, -7.50626e-06, 3.50458e-05, 1.26572e-05, 3.35537e-05, -1.94292e-05, -3.00804e-05, 7.22968e-07, 1.11407e-05, -6.93709e-05, -6.80446e-05, -3.50835e-07, 6.20265e-05, -5.8209e-06, 2.73622e-05, 1.9484e-05, 1.44371e-05, -8.30609e-06, 6.34198e-06, 1.96442e-05, -8.87614e-06, -3.12531e-05, -5.65149e-05, 3.77986e-05, 6.95659e-05, 1.89111e-05, 5.67198e-05, -6.57896e-06, -4.75388e-05, -6.85268e-05, -1.20302e-05, -5.1278e-05, -4.62559e-06, -6.69481e-05, 7.52861e-05, 5.20404e-05, -7.47484e-05, 1.76348e-05, -1.42576e-06, -2.87463e-05, 1.91781e-05, 1.87897e-05, 4.75274e-05, -3.06295e-05, -2.30126e-06, 3.97632e-06, -4.8816e-05, 4.46162e-06, 8.31156e-06, -6.90526e-07, -4.77039e-05, 1.20369e-05, 3.53523e-05, -1.14257e-05, -1.72746e-05, 9.12249e-06, -6.41878e-05,
-0.0122678, -2.96474e-05, -0.000194234, -0.000194272, 3.01917e-05, -4.54267e-05, 5.34952e-05, -0.000227721, -7.33173e-05, 2.6613e-05, -9.55459e-05, 9.43521e-05, -0.000178885, 9.58274e-05, 0.000108032, -0.000106496, 7.34014e-05, -5.57134e-05, -5.99125e-05, -7.20672e-05, -0.000186789, 0.000111576, -3.86417e-05, 0.000170334, 1.25984e-05, -0.000165341, -0.000120021, 0.000129465, 6.93782e-05, -6.569e-05, 0.000164076, -7.46761e-05, 3.9991e-05, 2.84274e-05, -8.31102e-05, -0.000112608, -2.50225e-05, -9.80328e-05, -7.94235e-05, -0.000150243, -3.57758e-05, -0.000265703, -0.000119046, 6.53283e-05, 0.000120803, -8.15781e-05, 0.000129909, -3.59928e-05, 0.000122691, 0.000135228, 1.36852e-06, 4.43862e-05, -1.69052e-06, 3.59156e-05, -9.28027e-05, 0.000160809, -0.000264427, -0.000199703, 3.67965e-05, 3.09302e-05, -5.90161e-05, 0.000246885, 0.000101108, 5.55999e-05, 1.73377e-06, 0.000153423, -0.000145426, -0.000137389, -1.78324e-05, 3.75448e-06, -0.000160996, 2.56947e-05, -2.64398e-05, 2.41497e-05, -2.49134e-05, -0.000223652, -0.000141484, -6.59629e-05, -0.000155189, 3.81245e-05, 0.00011564, 0.00014068, 0.000117221, 9.9477e-05, 2.37803e-05, 5.56982e-05, 0.000115897, -0.000102908, -0.000110533, 0.000103036, 3.49423e-05, -2.17395e-05, 0.000158669, -1.4803e-05, -4.117e-05, 0.000231569, -0.000164206, -7.64214e-05, 5.82125e-05, 2.86016e-05,
0.0030012, 1.48234e-05, -3.45024e-05, -8.52273e-06, 1.87906e-06, 1.92816e-05, 7.26066e-05, -3.11792e-05, -1.6668e-05, -2.96075e-05, 3.55445e-06, 6.63687e-05, -3.73976e-05, 1.91439e-06, -3.13855e-05, -1.18216e-05, -3.15068e-05, 2.96064e-05, -2.02123e-05, 1.2312e-05, -2.54624e-05, -3.94501e-06, 5.36874e-05, -1.35452e-05, -2.77459e-05, 1.5705e-06, -2.65927e-05, 1.92854e-06, 1.11581e-05, -4.7879e-05, 1.93947e-06, -6.12215e-05, 1.67795e-05, 1.96193e-05, 3.312e-05, 1.75601e-05, -1.61661e-05, -1.05314e-05, -3.50745e-05, 3.9074e-05, 2.58745e-05, -7.76495e-06, 1.40116e-05, 8.76713e-06, 3.70347e-05, -3.26828e-05, -7.33494e-05, -1.44412e-07, 2.0987e-05, 1.62971e-06, -1.08133e-05, -8.92409e-05, 2.59427e-05, -3.93146e-05, 1.44853e-05, -2.12829e-05, -2.53712e-05, 1.3984e-05, 8.6e-05, -1.68406e-05, -3.96385e-05, -6.80677e-06, 6.58371e-06, 1.22384e-05, 3.52446e-05, -4.05415e-06, 2.91128e-05, -8.22536e-05, 8.80825e-07, -1.21737e-05, -2.74759e-05, -1.63684e-05, 1.69784e-05, -1.02843e-05, -6.96638e-05, 9.50243e-07, -5.66952e-06, 5.43182e-05, 3.69031e-05, 3.49266e-05, -6.73325e-05, 1.9292e-05, -6.39504e-05, 6.76544e-05, 7.94735e-06, -7.45938e-06, -1.88935e-05, 1.69748e-05, 2.75102e-05, -9.49391e-06, -3.75457e-05, 2.08842e-05, -3.72225e-05, 3.12518e-05, -1.40847e-05, 2.64717e-06, -9.05268e-06, -1.70803e-06, -3.47324e-06, 9.0058e-05,
0.0172849, 0.000137723, 0.000195433, -9.9225e-05, -2.63941e-05, 0.000115114, -7.07816e-05, 6.00825e-05, -1.33186e-05, 0.000189723, 1.90942e-06, -1.21558e-05, -6.01984e-05, 0.000154054, 0.000388448, -7.96321e-05, -0.000224411, -1.53577e-05, -0.000412179, 1.99504e-05, 0.000185436, 0.000103807, 0.000288191, 8.62114e-06, -3.25771e-05, -0.000113113, -6.30261e-05, 0.000358153, 0.000186768, 0.000172796, 0.000251179, 5.9911e-05, -0.000124486, 0.000301758, -0.000191068, 0.000107442, -0.00023203, 9.41471e-05, -0.000149847, 0.000273322, -9.01591e-05, -1.71377e-05, 0.000220837, 0.000244104, -0.000281876, -0.000340712, -8.18626e-05, 0.000107123, -3.56118e-05, 5.17455e-07, 8.51542e-05, 0.000219588, -9.87657e-05, 0.000140971, 5.4868e-05, 5.2943e-05, 0.000237441, 0.000156849, 0.000153079, 0.000107955, -3.0326e-05, -0.000147937, -0.000262424, 0.000182262, 5.77469e-05, -0.000374086, 5.54542e-05, -0.000214752, -0.000129218, -0.000446174, 0.000293831, -2.75519e-06, -4.79103e-05, -0.000123406, -1.02924e-05, 0.000119444, -4.97074e-05, -0.000260739, 0.000196488, 0.000264832, -5.70794e-05, 0.000131057, 0.000273625, 7.61073e-06, 6.66199e-05, 5.30765e-05, 1.98357e-05, -0.000292071, -9.71627e-05, 0.000107555, 0.000133971, -3.343e-05, 9.15018e-07, -0.000183521, 4.67007e-05, -0.000334349, -8.36911e-05, 0.000126364, -8.79087e-05, -4.79688e-05,
0.00578778, -5.87314e-06, 3.42514e-05, 1.78559e-05, 5.90252e-05, 7.27381e-06, 0.00013527, 1.97246e-05, 8.23314e-05, 2.65783e-05, -7.54781e-06, 9.08444e-06, 4.42816e-05, 3.69972e-06, 3.57289e-05, -6.18712e-06, 7.32875e-05, 4.86123e-05, 2.18542e-06, 6.48892e-05, 8.39725e-05, 7.27211e-06, -7.16999e-05, -3.38777e-05, 1.4353e-05, -5.55851e-05, 8.72813e-05, 9.84383e-05, -3.02853e-05, 3.24064e-05, 1.47155e-05, 2.06515e-05, 3.97391e-05, -0.000113051, 5.80345e-05, 3.20978e-05, -1.84897e-05, -2.16839e-06, 2.33547e-05, -9.40672e-05, -1.60092e-05, 8.17609e-05, 0.000102356, 5.40125e-05, -8.88269e-05, 1.13563e-05, -5.15271e-05, 2.14484e-05, -9.02426e-06, -6.71439e-05, -1.55371e-06, 4.46237e-05, -8.25318e-05, -1.27018e-05, 1.51744e-05, 4.59613e-05, -2.41501e-05, 2.59972e-05, 3.08537e-05, 4.45413e-05, -2.20868e-05, -3.98687e-05, 5.62371e-05, 9.71181e-05, 2.11431e-05, 1.48326e-05, 0.000100554, 1.01266e-05, 4.88462e-05, 1.07104e-06, 8.11468e-05, -0.000121598, -3.91725e-05, 2.31807e-05, -4.99769e-06, 1.55936e-05, -2.27706e-06, -5.454e-06, 3.26318e-05, -9.9982e-05, 7.5289e-05, -5.41164e-05, 2.98501e-05, -4.4471e-05, 4.71421e-06, 3.56366e-06, 1.72034e-05, -2.83364e-06, -7.1758e-05, -6.15054e-05, 2.29887e-05, -3.01923e-05, 6.15869e-05, -4.19154e-05, -7.80869e-05, -8.98867e-05, -6.96564e-06, -5.51314e-05, 4.06548e-05, -1.62275e-05,
0.00287075, -6.91462e-06, 4.02191e-06, -1.42057e-05, -2.43458e-07, -2.85399e-05, -3.97124e-05, 2.30982e-05, 6.41966e-05, 1.98726e-05, 1.04583e-05, -3.04824e-06, -2.20805e-05, 3.56497e-05, 4.14654e-06, -4.15673e-07, 5.41508e-05, 4.10513e-06, 4.28954e-05, 4.71738e-07, -3.03641e-05, -5.64818e-06, 9.66502e-06, 2.94541e-05, -2.15608e-05, -9.82466e-07, 5.29488e-05, -2.10979e-05, 2.79752e-05, 3.2096e-05, 1.54964e-05, 2.14929e-05, 1.994e-06, -1.62368e-07, 2.18836e-06, -7.18086e-06, 6.18968e-05, -3.11503e-06, 2.57735e-05, 4.30619e-06, 3.2802e-05, 1.2878e-06, 1.20033e-05, 8.39204e-06, -1.04636e-05, -1.08868e-05, -3.59794e-06, 5.62377e-05, 4.05816e-05, -7.62451e-06, -1.33191e-05, -1.32416e-05, -2.69252e-05, -1.87218e-05, 3.55279e-05, 3.01701e-05, -2.34099e-05, -2.29466e-05, -7.18423e-06, 4.29571e-06, -2.63505e-05, 1.14303e-05, -3.27239e-05, 1.13323e-05, 6.42999e-05, -2.08436e-05, 3.09725e-05, 1.82973e-05, 2.74337e-05, -2.33277e-05, 3.95963e-06, 3.49225e-05, 2.34855e-05, 2.49438e-05, -4.46137e-05, -2.17977e-05, -4.45148e-05, -6.50044e-06, -3.19306e-05, 2.47845e-05, -8.55637e-06, -2.33783e-05, -2.46901e-05, -7.11076e-05, -3.90536e-05, -2.1409e-06, 5.58662e-06, -1.61084e-05, 3.17005e-05, 1.91181e-05, 1.29868e-05, -2.40676e-05, -3.74026e-06, -1.56419e-05, -2.89036e-05, 1.32516e-05, -4.38047e-05, -3.99447e-05, 6.72658e-06, -2.16801e-05,
0.0115143, 0.00024604, -9.74668e-07, -0.00014733, 3.2793e-05, 3.40506e-05, 9.23134e-05, -4.69995e-05, -6.7757e-05, -9.9981e-05, 7.35236e-06, -0.000156587, 7.71261e-05, 4.50955e-05, 4.90386e-05, -0.000384541, 0.000114487, 5.14549e-05, 0.000190718, -0.000114935, 2.19649e-05, -2.946e-05, -3.44786e-05, 0.000109631, -2.34144e-05, 3.27693e-05, -0.000101297, -7.50671e-05, -8.66306e-05, 0.000237673, -1.18239e-06, 0.000103796, -3.57106e-05, -4.4007e-05, 1.69185e-05, -2.59722e-05, -4.04822e-05, 6.9521e-05, -6.98225e-05, -3.4353e-05, -0.000100042, 0.000100451, 5.47573e-05, 0.000282422, 1.03158e-07, 0.000158505, -1.99418e-05, 8.97512e-05, -0.000101387, 0.000144333, 0.000170561, -0.000159172, 7.62996e-05, 9.84727e-05, -1.56147e-05, 7.46075e-05, 5.73833e-05, -9.60056e-06, 0.000250048, 0.000105388, 0.000193354, 3.7023e-05, 0.000219311, 0.000103351, 4.02583e-05, -2.31665e-05, 3.96521e-05, -0.000136103, -0.000147597, 0.000220651, -0.000150262, 0.000121167, -5.93347e-06, 0.000163544, 0.000210723, 0.000224595, 1.99375e-05, 7.48695e-05, 0.000108321, -0.00013867, 9.55556e-05, -0.000288075, -0.000203602, 0.000133736, 8.64081e-05, 6.53009e-05, -2.51961e-05, -0.000188693, 0.000192592, -0.000114363, -1.65279e-05, 5.80957e-05, 8.28964e-05, 0.000179917, 6.92716e-05, -5.75134e-05, 0.000116731, 6.16904e-05, 8.23635e-05, 7.4646e-05,
-0.00997741, 1.10792e-05, -0.00017227, -0.000174343, -0.000217143, -0.000138439, -6.16618e-05, 0.000116954, 0.000106947, -0.000136169, 4.20472e-05, 3.61039e-05, -7.20323e-05, -5.76884e-05, -2.90041e-05, 0.000185975, -9.55685e-06, -0.000129987, 8.45093e-05, 6.54536e-05, 4.43172e-05, -4.86732e-05, -8.89951e-05, -3.85739e-05, 4.68427e-05, -3.47642e-05, 2.81205e-05, 5.39602e-05, 1.72662e-05, 8.11372e-05, 0.000128635, -2.25614e-05, -0.000156052, 1.31225e-05, -3.07366e-05, -0.000133925, 1.44079e-05, 8.18362e-07, 3.32444e-05, -4.2883e-05, -0.000137933, -5.35507e-05, 8.93543e-05, -8.6586e-05, -8.51143e-05, 5.62072e-05, -4.04677e-05, -2.65785e-05, 9.91009e-05, 0.000182233, 2.1494e-05, -0.000157592, -3.59564e-05, 0.000156462, 0.000131728, 7.26594e-05, -4.0224e-05, -9.36035e-05, 5.74937e-05, -0.000171939, 3.4321e-05, -0.000111188, 0.000201556, -6.86891e-05, 1.97179e-05, -3.71073e-07, 0.000131039, -0.000235341, 3.23308e-05, 4.08726e-05, -6.51863e-05, 0.000115383, 0.00013324, -9.6061e-05, 0.000152424, 0.000190273, 5.60316e-05, -8.03954e-05, -7.16744e-05, 0.000170292, 2.64324e-05, -0.000307595, -4.8318e-05, -4.93508e-06, 0.00021756, -9.37775e-05, 2.99109e-05, 5.68611e-05, 3.88428e-05, 0.00026029, 1.33188e-05, -8.18097e-05, 5.68735e-05, -0.000131805, 8.55603e-05, 0.000117098, -3.2375e-06, 0.00014753, -1.09196e-05, 0.000141688,
-0.0101885, -0.000128936, -8.99639e-05, 2.29156e-05, 6.80954e-05, 1.56234e-05, 6.46556e-05, -0.000237843, -2.66563e-05, -6.12704e-05, 9.58966e-05, -0.000109882, -1.45847e-05, 7.24104e-05, -1.23549e-05, 0.000197353, -6.62098e-05, -7.45148e-05, -6.80458e-05, -1.23418e-05, -0.00011896, 2.6757e-05, -2.6603e-05, 8.64772e-06, 3.8663e-06, 0.000135781, -0.000122711, -0.000102119, 0.000144069, 0.00012296, 8.21182e-05, -0.000136273, -0.000137979, 2.35658e-05, -3.73866e-05, -0.000162426, 8.28748e-05, -6.67768e-05, 3.15945e-05, -8.48638e-05, 3.74323e-05, -7.89446e-05, -5.19082e-05, -0.000218941, 5.43258e-05, 0.000153606, -0.000146235, -8.16235e-05, -6.74976e-05, 9.305e-05, -8.49588e-05, -8.89793e-05, 3.04544e-05, -3.20283e-05, -0.000113792, -0.00015627, -0.000127342, -0.000204663, 8.25357e-05, 1.16438e-05, 6.30623e-06, 0.000193719, 7.16029e-06, -0.000112908, 3.69857e-06, 2.96172e-05, -2.63959e-05, -4.88204e-05, -7.2124e-05, 3.66007e-05, -0.000175402, 6.04806e-05, -1.50172e-05, -0.000176539, -4.28013e-05, -0.000187214, 8.76287e-05, -1.96037e-05, -0.000103394, -5.35017e-05, -3.32705e-05, -0.000335535, -5.35969e-05, 0.000180006, -8.94994e-05, 9.35903e-05, 6.57076e-05, 0.000144833, 4.97488e-05, 9.8477e-05, -0.000173161, 6.20066e-05, 1.62621e-05, 0.000101082, 0.000167563, 1.72316e-05, 0.000270144, 6.87924e-05, -5.17778e-05, 1.24828e-05,
-0.0143151, 0.000131824, 0.00022796, 0.000122503, 4.581e-05, -8.12446e-05, -1.22761e-05, 1.96691e-05, -0.000118358, -5.04886e-05, -0.000189057, -0.000114086, 0.000123788, -9.3268e-05, 0.000146342, -7.09014e-05, -0.000276324, -2.51972e-05, 8.52766e-05, -1.0148e-06, -9.83365e-05, -2.64532e-05, 3.22219e-05, 0.000171837, -0.000210435, -6.44457e-05, -0.000106355, 2.97875e-05, 6.50604e-05, -7.99859e-05, -1.98095e-05, -0.000224756, -0.000259289, 7.1224e-05, 0.000245684, 0.000266492, 0.000142202, -0.000106565, 2.3321e-05, -1.66672e-05, 0.000125513, 1.43308e-05, -3.22395e-05, -0.000421159, 3.287e-05, 0.000138568, 0.000106357, 6.99954e-05, -7.40098e-05, -6.15091e-06, 0.000101536, 3.6738e-05, -5.82622e-05, 4.40908e-05, -0.00016688, -5.02537e-05, -0.000221255, 0.000238508, -1.63034e-05, -9.03363e-05, 0.000103808, -0.000157045, -5.15322e-05, -2.07345e-05, 2.95185e-07, -6.72482e-05, -9.68639e-05, 2.32443e-05, -0.000106303, -0.0001652, 0.000257104, -3.83583e-05, 8.78804e-07, 0.000347101, 0.000142762, 0.000255787, -0.000239659, -0.000157389, 8.50095e-05, 8.64094e-05, 0.000197523, -6.54805e-05, 0.000156005, 0.000125108, 0.000212135, -0.000130541, -5.94662e-05, -7.6956e-05, 0.00023007, -9.01312e-05, 0.000124218, 0.000293952, 4.87048e-05, -5.3695e-05, -0.000100566, -0.000320226, -0.000269279, -9.67785e-05, -5.64425e-05, -0.000187976,
0.0141089, 6.88233e-05, 0.000123426, -0.000111003, 8.83523e-05, 7.71398e-05, 0.000165591, 8.66731e-05, -0.000267331, -4.95202e-05, -8.80515e-05, -7.15958e-05, 3.25474e-05, 0.000204542, 6.32746e-05, -0.000108882, 5.73478e-05, -1.04762e-05, 0.000181845, -9.59184e-05, 8.73891e-06, -6.17582e-05, -1.72559e-05, 0.000113397, 1.20541e-05, -1.83696e-05, 1.97309e-05, 5.83505e-05, 5.58987e-05, -5.53268e-05, 5.92417e-05, 5.4848e-05, -0.000160719, -0.000216721, -0.000125628, 0.000168478, 7.02117e-05, -5.56554e-05, 8.66232e-05, 0.000120216, -0.0003353, 0.000390618, -7.35737e-06, 0.000187574, -3.23359e-05, -0.000116724, -4.55641e-05, -0.000497453, -0.000223076, 0.000335997, 0.000268903, 0.000153186, -6.84382e-05, 1.42701e-05, 1.76332e-05, 0.000102624, 0.000119294, 0.000110975, -0.000318683, 0.000210512, 0.000184099, 0.000127706, 2.77155e-05, 0.000132665, -3.73543e-05, 0.000142863, 1.76511e-05, -3.86664e-05, -3.93974e-06, -2.49183e-05, 0.000156503, -6.27122e-05, 0.00013063, -6.72362e-05, -2.28929e-05, -0.000159164, -4.43742e-07, 0.000179704, -9.96358e-05, 8.61969e-06, -9.55364e-05, -0.000133042, -7.36177e-05, -0.000193638, 7.0064e-05, -0.00018198, 0.000159271, 5.40177e-05, 0.000106405, 8.82694e-05, -0.000132683, 7.22597e-06, -0.000179072, 9.30664e-05, 0.000114806, -0.00011723, -0.000138799, -0.000131085, -6.42225e-05, -0.000123607,
-0.00421674, -1.69326e-06, -3.97448e-05, 8.4422e-05, 5.1542e-05, -2.62139e-06, 2.62373e-05, 3.52993e-05, -8.0267e-05, 3.73699e-05, 8.62195e-05, 7.48689e-05, 8.85932e-05, 7.75517e-05, 6.31267e-05, -2.30955e-05, 1.57776e-05, 1.512e-05, 1.99944e-05, 7.78292e-05, 1.31513e-06, 1.2968e-05, -2.75518e-06, 3.06575e-05, -1.24654e-05, 5.14847e-05, -2.64345e-05, -4.07105e-05, -5.61569e-05, 5.76626e-05, 3.28381e-05, 9.14056e-06, 5.80385e-05, -2.26564e-05, -2.16342e-05, 1.01859e-05, 5.50612e-05, -2.40169e-06, 6.33221e-05, 3.56267e-05, -5.11026e-06, 2.83936e-05, 2.02099e-05, -4.18297e-05, -2.58537e-05, 4.04359e-06, -2.044e-05, 9.02129e-06, 1.95933e-05, -1.10242e-05, 2.42925e-05, 1.65642e-06, 1.18565e-05, 3.79925e-06, 3.57756e-06, 2.97222e-05, -9.32368e-06, -1.02793e-06, 6.91242e-05, -6.27626e-05, -6.44421e-06, -2.85152e-06, 3.89213e-05, -1.03049e-05, 7.97256e-05, -3.53605e-05, 2.11821e-05, 4.34789e-05, -3.64118e-05, -8.3118e-05, -5.62327e-08, -3.43431e-05, 1.64984e-05, 1.61187e-05, 4.83775e-05, 2.90762e-05, -3.04235e-06, 3.86683e-05, 2.2276e-05, -3.86314e-05, -2.78649e-05, -1.14476e-05, 2.39136e-05, -8.97426e-05, -1.57698e-07, -0.000125586, 6.67685e-05, -6.77408e-05, -3.14643e-05, -7.379e-05, 6.93087e-06, 1.18381e-05, 4.5096e-05, -1.00665e-05, -3.03706e-05, 1.40963e-05, -2.96107e-06, -5.45474e-05, -2.63409e-05, -4.3528e-05,
0.00277685, -1.31989e-05, -3.45768e-05, 1.71728e-05, 2.59024e-05, 2.75087e-05, 4.13044e-05, -2.40635e-05, -2.10821e-05, -2.58192e-05, -5.8041e-05, 4.13604e-05, -5.20614e-06, 2.15247e-05, -1.24346e-05, -3.0758e-05, 4.78188e-05, 2.49171e-05, -1.70731e-05, -1.15677e-05, -6.36119e-06, 3.95117e-05, -2.21327e-05, -5.31377e-06, 1.57124e-05, 7.38332e-06, -4.48755e-05, -7.8115e-06, -1.04973e-05, -6.62755e-06, -7.26017e-06, -1.46167e-05, -3.39305e-06, -6.60043e-06, -4.01126e-05, 1.11492e-05, -2.4283e-05, -1.52601e-05, -5.17417e-06, 4.05777e-08, -3.46231e-05, 5.21929e-06, -1.50563e-05, 1.70871e-05, -2.24823e-05, -2.00048e-05, 2.31658e-05, -8.27554e-06, 1.91366e-05, 8.55194e-07, -2.26101e-05, -2.75952e-05, 2.11966e-05, 1.77995e-05, 2.41089e-05, -3.88963e-05, -2.26082e-05, 3.55091e-05, 4.24417e-06, -8.92179e-05, -3.25172e-06, -1.54763e-05, -9.08917e-06, -2.09863e-05, -2.09006e-05, 2.07621e-05, -8.14411e-06, -1.00104e-06, 4.06459e-05, 1.05588e-05, 1.43034e-05, -3.83083e-07, 3.64297e-06, 6.51345e-06, 5.29145e-05, 1.50209e-05, -8.20453e-05, 2.3566e-06, -2.31791e-05, 2.92836e-05, -1.20098e-05, 2.62779e-05, 1.49367e-06, 1.33794e-05, -2.52239e-06, -3.27127e-05, -5.91039e-05, 2.42297e-05, -4.57473e-05, -8.53224e-06, -2.69022e-05, 5.4678e-06, -4.70387e-05, 3.49975e-05, -4.02133e-06, -1.02213e-05, 5.13827e-05, -2.3928e-05, 1.31101e-05, 6.62657e-05,
0.0142983, 0.000112342, -4.66339e-05, -4.13413e-05, 0.000309603, -0.000316948, 4.22536e-05, 0.000215781, 0.000258133, -0.00012338, 0.000198538, -0.00029328, 2.48391e-05, -0.000179177, -7.64569e-05, -6.17511e-05, 4.38795e-05, 0.000212542, 0.000242855, -0.000268719, 0.000199119, 8.78416e-05, 7.55024e-05, -1.59092e-05, 4.82038e-05, 9.45152e-05, -0.000131388, -0.000387755, -1.38443e-05, -0.000104306, -2.7031e-05, -0.000169889, 0.000138967, 0.00028052, -0.000113789, 2.96022e-05, -6.98275e-05, 0.000140872, 0.000223318, 0.000274732, 0.000150951, 0.00023939, -0.000119235, 0.000177274, 0.000170915, 0.000139813, 2.48528e-05, -1.50265e-05, 1.91295e-05, -3.6086e-05, -2.78991e-05, 8.00713e-05, -9.36489e-06, 9.80752e-06, -0.000198944, -6.30686e-05, -0.000157069, 0.000133999, -0.000120761, 0.000141277, 0.000218183, -8.08426e-05, -0.000205758, 9.53295e-05, -9.77383e-06, -1.16179e-05, 0.000241276, -0.000154287, -0.00020482, -0.000221587, -5.77857e-05, -3.44787e-05, 7.76193e-05, 0.000207615, -0.000210754, 0.000127659, 0.000112059, 2.15578e-06, -7.11047e-05, 0.000219113, 1.95823e-05, 3.50769e-05, 0.000210426, 9.96665e-05, 4.0588e-05, 0.000241474, -0.000103111, -0.000195419, -0.000231868, -0.00021536, 0.000183616, 0.000181196, 0.000223475, -1.35299e-05, -0.000149747, 7.42254e-05, -3.90989e-06, -7.81763e-05, 6.00542e-05, 0.000103554,
-0.0104521, -1.35201e-05, -6.44042e-05, -0.000115213, -0.000159477, 9.56275e-05, -0.000198975, 3.22498e-06, 0.000255636, -0.000110852, 5.71324e-06, 6.18297e-05, -8.81023e-05, 0.000203104, -5.3864e-05, 7.14396e-05, -0.000127688, 1.30878e-05, 1.62684e-08, 1.94774e-05, -6.19057e-05, -1.78991e-06, -0.000125799, -9.94888e-05, -8.8763e-05, 2.01866e-05, -0.000170965, -0.000175986, 1.34548e-05, -8.49406e-05, 0.000125902, 1.34807e-05, -0.00020807, 6.84615e-05, -0.000197852, 7.25726e-05, -4.77451e-05, -0.000175992, 0.000150987, -3.66021e-05, 4.72494e-06, 6.77744e-06, 2.2003e-05, 0.00021696, -9.44283e-05, 0.000114282, 0.000301344, 0.000173935, 0.000206564, 0.000187432, -0.000174304, 8.2958e-05, 3.33016e-05, -2.34449e-06, -0.00014588, 0.000149876, -7.92437e-06, 3.88841e-05, 0.000126421, -2.17681e-05, -0.000160817, 8.55032e-05, -9.48144e-05, -7.941e-05, -6.58782e-06, 7.49662e-05, 0.000146201, 1.64148e-05, 0.000100267, -7.98559e-05, -7.39386e-05, 2.42363e-05, -0.000172746, 0.00011475, -1.92182e-05, -5.73364e-05, -0.000162462, -4.55046e-05, -5.35867e-05, -4.67108e-06, -9.72802e-06, -0.000143449, -2.53612e-05, 9.72004e-05, -2.97867e-05, -1.62663e-05, 6.66964e-05, 1.1208e-05, 7.66252e-05, 5.49829e-06, 7.93682e-06, -0.000107323, 8.06985e-05, -6.70455e-05, -4.30244e-05, -0.000189037, 0.000168284, -9.72714e-05, 6.25913e-05, -2.86024e-05,
-0.0136906, 0.000186683, 3.99589e-05, -8.51733e-05, 7.38685e-05, -2.40505e-05, -6.27557e-05, -0.0002557, 7.31769e-06, -5.87067e-05, -1.42058e-05, 0.000121633, -0.000160032, 0.000283101, -0.000230965, -1.14919e-05, -7.30697e-05, -2.23898e-05, -0.000103393, 0.000145653, -5.53898e-05, 0.000312081, 0.000143954, 0.000128541, -0.000103988, -0.000107767, -3.84244e-05, -0.000110962, 9.10934e-07, -7.08488e-06, -0.000130675, 7.33392e-06, -0.00015033, 7.67584e-05, -0.000297338, 4.81614e-05, -0.000155257, -2.70693e-05, 3.49565e-05, -7.90606e-05, -0.000235569, 0.000145515, 2.76751e-05, -0.000100847, 0.000142387, -7.30949e-05, -0.00010537, 4.02595e-06, 3.87436e-05, -5.91303e-05, 0.000111846, -0.000109605, -6.03642e-05, -5.78865e-05, 0.000164872, 0.00014423, -0.00048342, -0.000208043, -9.11468e-05, -8.80639e-05, -0.000211469, -0.00013278, -0.000179958, 1.69935e-06, 0.000140067, 0.00014934, -4.86188e-05, 4.69468e-05, -7.86298e-05, 8.89884e-05, 0.000152901, 0.000145757, 4.21071e-05, -7.94471e-06, 0.000125126, -0.00013001, 0.000336021, 6.19468e-05, 0.000179138, -3.2503e-05, -0.000220857, -6.34478e-05, -4.13999e-05, -0.000184705, 8.83447e-05, 1.62322e-05, 2.27643e-05, 0.000107907, -0.000228112, 0.000174066, -0.000163579, 6.75677e-05, 0.000141539, 0.000185369, -0.000165437, 0.000160299, -4.07907e-06, -8.31167e-05, 4.5502e-05, -0.000143552,
-0.00121922, -1.24232e-05, 4.75447e-06, 1.01354e-05, 9.45497e-06, 3.00256e-05, -3.78734e-07, 6.25455e-06, -2.35128e-06, 1.24707e-05, 1.207e-05, 1.80186e-05, 1.69964e-05, -2.72471e-05, 1.54133e-06, 9.52962e-06, -1.39422e-05, 1.83122e-05, 1.06273e-05, -2.78255e-06, -1.98553e-06, -4.06806e-06, -5.49107e-06, -6.04802e-06, 4.03035e-07, -6.99282e-06, -1.81664e-06, 9.09336e-06, 5.54915e-06, 1.50624e-05, -3.70766e-06, 4.06782e-06, -3.382e-06, -5.37953e-06, 1.4661e-05, -4.63178e-06, -9.27535e-06, 7.98882e-06, 4.22249e-06, 4.65532e-06, -7.71813e-06, -5.34859e-06, 1.27757e-05, 1.93509e-05, -2.77596e-06, -3.33767e-07, 3.6977e-06, 1.0406e-05, -7.36046e-06, -4.13105e-05, 2.81049e-07, -1.70249e-06, -4.05117e-06, 1.37039e-05, 2.44645e-06, 3.36509e-05, -4.73946e-06, -9.11174e-06, -2.20039e-05, 8.15244e-06, 4.84443e-06, -4.04509e-06, 1.36429e-05, 2.42152e-05, 1.5674e-05, 7.76588e-06, -4.49153e-06, 1.28089e-05, 1.04144e-05, -1.31049e-05, 2.13778e-06, -1.38752e-05, -7.55661e-07, 6.08448e-06, 1.15194e-05, -1.0997e-05, -1.25218e-07, -7.07156e-06, -1.34621e-05, -2.2189e-06, -1.19576e-05, 1.45134e-05, -7.67472e-07, 1.1369e-05, 1.94028e-06, -1.18281e-05, 3.2618e-06, 6.48987e-06, -1.80958e-05, 1.14375e-05, -9.03821e-07, -1.17069e-05, 1.00832e-05, 3.02185e-06, 3.9791e-06, -1.27873e-05, -1.20499e-05, -2.1711e-05, 1.12185e-06, -8.54503e-06,
0.0154786, 6.8537e-05, -8.23094e-05, 7.6109e-05, 4.98291e-05, -6.8062e-05, 0.000152599, 0.000233028, -6.87975e-05, 0.000209242, 0.000129821, 6.79971e-05, 3.92891e-05, -0.000154275, 0.000102488, -9.07521e-06, -3.34974e-05, 5.03436e-05, 0.000329504, -4.83158e-05, 3.52554e-05, -7.97117e-05, 5.73639e-05, 1.52662e-05, 0.000183741, -0.000198625, -6.03625e-05, -0.000157018, 0.000325863, 6.3973e-05, 1.22553e-05, 0.000139454, 6.33481e-05, -8.98804e-05, -3.17035e-05, 3.93693e-05, 0.000110552, -0.000180009, 0.000288823, -4.29742e-05, 0.000267386, 6.24631e-05, -0.000128078, 1.61258e-05, -4.08024e-05, -0.000122919, -2.88076e-05, -1.07826e-05, -0.000115231, 0.000213781, 3.60139e-05, 1.64311e-08, -0.000205684, 2.22424e-05, -5.94689e-05, -7.29134e-05, -9.3974e-05, 0.00015473, -2.33043e-05, -0.000365083, -0.000194362, -7.98397e-05, -2.9421e-05, 0.000148374, 5.71272e-05, 0.000144757, 7.24536e-05, 8.17955e-05, -0.000168692, -0.000119885, 3.22486e-05, 6.01132e-05, -0.000108611, -0.000275697, -0.000118669, 0.000162807, 3.43368e-06, 7.33119e-05, 0.000132183, -3.3185e-06, -0.000575269, 5.95927e-05, 0.00021485, -7.95502e-05, 4.23043e-05, 0.000139947, 0.000280798, -1.26213e-05, 2.77217e-05, -0.0002293, 8.78732e-05, 0.0001229, 6.63258e-05, 9.02066e-05, -3.00444e-05, 2.57927e-05, 5.96901e-05, 0.000226792, -8.20947e-05, 0.000139532,
0.00878779, -6.77791e-06, -1.05494e-05, -7.7096e-06, -0.000170983, 9.17615e-05, -3.91895e-05, -0.000111811, -0.000117922, 1.77539e-06, 5.97666e-05, 0.000175983, 4.68907e-05, 0.000157806, -6.22251e-05, -4.10945e-05, -6.31827e-07, 5.21477e-05, -9.77267e-05, -1.62623e-05, 1.84547e-05, 1.65586e-05, 3.69318e-05, 0.000163607, 3.47902e-05, 8.44093e-05, 9.46193e-06, 4.31363e-05, 1.63619e-05, -5.15883e-06, 4.42586e-06, 0.000171781, 9.36357e-05, 3.77088e-05, 0.000127937, -9.55955e-05, -3.97155e-05, -3.30028e-05, -3.99689e-05, 8.20151e-05, 2.35646e-05, -8.72408e-05, 9.48185e-05, 9.76195e-05, -5.73176e-05, 0.000256995, -0.000103982, -9.7517e-05, 3.16043e-05, -5.94185e-05, -7.32809e-05, 6.69719e-05, 1.22677e-06, -0.000149913, -7.3944e-05, 3.10015e-05, -3.78878e-05, 8.83657e-05, 2.16757e-05, -6.02778e-05, 6.97432e-05, -2.64734e-05, -4.44121e-05, 1.82497e-05, 0.000137658, -0.000219206, 0.000166704, -0.000115663, -5.41671e-05, -2.20667e-05, -0.000117324, 8.61098e-05, 4.30663e-05, -9.42627e-06, 4.99182e-05, -1.55314e-05, 0.000223664, -7.20727e-06, 7.81428e-05, 6.53753e-05, -4.0061e-05, 6.94521e-05, -5.25068e-05, -2.9756e-05, 1.85248e-06, -6.41874e-06, -6.2419e-05, -7.27802e-05, 6.13354e-05, -5.27047e-05, 6.83425e-05, 0.000138395, -4.96156e-05, -6.0999e-05, 0.000188602, -5.5722e-05, 1.20046e-05, 5.45674e-05, -0.000111359, 1.85032e-05,
0.0116572, 0.000313971, -2.23927e-05, -0.000130032, 0.00013079, 0.000112077, 3.16872e-05, -1.70496e-05, -7.83445e-05, 0.000103645, 0.000167455, 0.000193714, 0.000100044, -0.000312724, 6.6739e-05, -0.000130754, 5.40016e-05, -2.25568e-05, 9.80857e-05, -0.000134078, -0.000225043, -0.000124674, 9.47695e-05, -2.89549e-05, 7.79797e-06, 2.37999e-05, 3.00204e-05, -9.17505e-06, -3.54757e-05, -3.50796e-06, -9.32635e-05, 1.96308e-05, 0.000118573, -8.73828e-05, -0.000289047, -6.86935e-05, -9.9624e-05, 6.20031e-05, 0.000339158, -3.70026e-05, -3.16236e-05, 5.22065e-05, -0.000245038, 0.000149534, 0.00022144, -6.12711e-06, 8.08791e-05, -0.000139536, -0.000281804, -0.000100108, 6.52171e-06, 0.000210351, 2.12827e-05, 0.000159321, 4.42548e-05, 3.97682e-05, -0.000272141, 7.15905e-05, -1.00794e-05, 9.36794e-05, 5.68021e-05, -0.000332242, 0.000162907, 6.63993e-05, 2.87898e-05, -0.000144939, -6.06135e-05, -0.000135081, -0.000197283, 1.59394e-05, 2.76102e-05, -7.60787e-05, 0.000127623, 2.57555e-06, -0.000143254, -6.80762e-05, -1.58889e-05, 0.000151886, 0.000186635, 9.48433e-05, -5.2202e-05, 0.000181816, 0.000148925, 4.52238e-05, 0.000271375, 5.26212e-05, -0.000256503, 6.99502e-05, 0.000193484, 7.83227e-05, 5.35203e-05, 0.000156654, -3.67716e-05, 6.74656e-05, 8.481e-06, -1.8457e-05, 0.000188364, -6.28844e-05, 0.000129571, 4.70602e-05,
-0.00732858, 3.06972e-05, 6.20607e-05, 9.71838e-06, 6.60415e-05, -5.47479e-05, 4.69094e-05, 3.03118e-06, 1.64002e-05, -9.72996e-05, -5.40418e-05, 3.1389e-05, -3.35968e-05, 1.08293e-05, -2.85057e-05, 3.87618e-05, -2.96582e-05, -2.81161e-05, -8.00102e-05, 7.82426e-05, 0.000206897, -8.72485e-05, 0.000181145, 4.79302e-05, 1.03157e-05, -2.25353e-05, -2.72143e-05, -1.74484e-05, -2.8777e-05, -2.90346e-05, -1.04455e-07, 3.73665e-05, -5.64283e-05, 7.06511e-05, -9.49746e-05, 2.73546e-05, -5.19702e-05, 0.000143358, 0.000201924, -5.77576e-06, -5.27031e-05, -0.000106946, 2.06328e-05, 5.70204e-05, -2.69399e-05, 4.53045e-05, -7.90251e-05, -9.97232e-05, -7.74448e-05, 0.000148707, -1.47598e-05, 1.92722e-05, 9.10381e-05, 0.000128719, -2.80777e-05, -2.52853e-05, -9.2565e-06, -3.69116e-05, -1.05199e-05, 5.99129e-05, 3.09733e-05, -6.40262e-05, 5.86049e-05, -1.80512e-05, 1.46579e-05, -0.000141828, 5.70704e-05, -5.11206e-05, 5.44277e-05, -1.25195e-05, -2.35489e-05, 0.000100344, -2.34399e-05, 0.000101243, 3.36971e-05, 7.12947e-05, 0.000152167, 6.80657e-05, -1.07537e-05, -0.000139389, -9.20886e-05, 5.80508e-05, 8.46335e-05, -4.15481e-05, 1.30224e-05, -9.14687e-05, -1.54641e-05, -2.5523e-05, -2.56967e-05, -4.59981e-05, -7.60972e-05, -0.000201515, 8.38667e-05, 0.000167888, -9.02601e-05, 5.22273e-05, -0.000129351, 2.10959e-05, -6.28044e-05, 9.84055e-05,
0.00734464, -6.57494e-05, -4.66785e-07, -9.8861e-05, -6.35581e-05, 6.86825e-05, -7.80842e-05, -1.50745e-05, 1.7824e-06, -1.07858e-05, -1.22316e-05, 6.42698e-06, -9.57756e-05, 6.8841e-05, -5.73955e-05, -7.565e-05, -3.81748e-05, 9.20916e-05, 1.37248e-05, 0.000117216, -4.10546e-05, -3.59845e-05, -9.79414e-05, 7.44557e-05, -3.17195e-05, 5.61577e-05, 3.5975e-05, -7.6579e-05, 4.96854e-05, -9.0475e-05, 5.87205e-05, -5.51961e-05, 1.08768e-05, 1.30647e-05, -0.000113171, -2.30102e-05, 9.5943e-07, 5.16381e-05, 4.62554e-06, 4.42367e-05, -0.000111706, 4.52206e-05, -1.31522e-06, 3.78715e-05, 2.36854e-05, -3.33476e-05, 2.83669e-05, 7.80533e-05, 4.85448e-05, -7.72437e-05, 1.74592e-05, -2.59052e-05, 1.03136e-05, 5.58485e-05, -1.58106e-05, 7.26708e-05, 6.32836e-05, -4.37903e-05, -9.97947e-05, 5.47248e-05, 9.94852e-05, 0.000125934, -1.75892e-05, -0.000126799, 0.000140755, 0.000176589, 5.12162e-05, -8.33018e-06, 3.38969e-06, -4.28868e-05, -8.90007e-05, -3.52892e-05, 6.26142e-05, -3.22501e-05, 0.000122996, 3.7841e-05, 3.32805e-05, 2.35211e-05, -3.17111e-05, 6.52081e-06, 4.48752e-05, -5.06616e-05, -7.69605e-05, -2.11714e-06, 1.93718e-05, 9.12512e-06, -6.23493e-05, -2.71302e-05, -3.25057e-05, 1.42455e-06, -6.18641e-05, 4.23484e-05, 2.14073e-05, -7.76121e-06, -0.000111982, -1.61798e-05, 5.51331e-05, 6.66166e-05, 1.45863e-05, -1.95395e-05,
0.00892595, 0.000116815, -4.64528e-05, -5.20656e-06, 0.000154622, 3.89269e-05, 1.48163e-06, 5.18231e-05, -0.000104226, 0.000143854, -8.04779e-05, -0.000108887, -0.000105375, -0.000158231, -5.69279e-05, -1.32282e-05, -0.000120644, 6.84594e-05, -0.000122617, -8.3487e-05, -7.10649e-05, -6.50481e-05, 1.03356e-05, 0.00019493, -5.5229e-05, -6.50008e-05, -0.000102743, 6.63394e-05, -6.43319e-05, -0.000176756, -8.99761e-05, 0.000113191, -2.17933e-06, 6.55917e-05, -0.000122393, 0.000166941, 0.000116442, 0.000101226, 1.52887e-05, 0.00010089, -1.70835e-05, 2.38287e-05, 3.31983e-06, -0.000210933, 4.92785e-05, -1.00383e-05, 8.16269e-06, 0.00020131, 2.08596e-05, 4.04549e-05, 9.74924e-06, -1.19184e-05, 3.11514e-05, -2.10452e-06, 0.000103475, -0.000106016, 5.18397e-05, 5.76011e-05, 0.000173018, 0.000138324, 7.54174e-05, 1.10651e-06, 0.000126755, 8.61237e-05, 0.000139014, 8.91306e-05, -6.77291e-05, -2.97899e-05, -8.93522e-05, -9.0604e-05, -2.1227e-05, 1.70385e-05, -0.000117658, 6.42157e-05, 1.9928e-05, 1.07419e-05, 8.50687e-05, 6.21637e-05, 9.56653e-05, -1.14243e-05, 0.000172545, -8.38682e-05, 7.90466e-06, 0.000122303, 6.23242e-05, 0.000120289, 8.34185e-06, -8.22106e-05, -0.000136936, 9.58754e-05, -6.05145e-05, -9.09286e-05, 8.63656e-05, -3.91557e-05, 4.43812e-05, -9.81379e-05, 2.59884e-05, 0.000209612, -8.23427e-05, -5.42007e-05,
-0.00702316, -2.67279e-05, -1.88425e-05, 0.000102991, -3.57232e-05, 0.000114993, 2.54271e-05, -4.63618e-05, -5.05901e-05, 9.8297e-06, -6.08287e-05, -2.52786e-05, -6.58845e-05, 9.71756e-05, -4.35831e-05, -6.13789e-05, -0.000100814, -0.00010397, 6.034e-05, -9.20263e-05, 4.43741e-05, 6.67462e-05, 5.06288e-06, 4.41629e-05, -0.000107211, -4.00196e-05, -1.08446e-06, 3.48982e-05, 0.000144819, 4.15069e-05, 2.17984e-05, -7.72152e-05, 0.000173294, 0.000108514, 3.98719e-05, -1.85131e-05, -3.02118e-05, 1.08877e-05, -5.72494e-05, -5.05758e-05, 0.000115718, 8.51324e-05, 8.78592e-06, 9.77748e-05, 7.02772e-06, -0.000106226, 3.10749e-05, 1.34008e-05, -2.4938e-05, -9.42454e-05, 7.8474e-05, -1.73408e-05, -2.37068e-05, 3.1436e-05, 4.94722e-05, 2.35396e-05, 1.97478e-05, 4.97524e-05, 6.69589e-07, 2.68161e-05, 0.000142787, 1.47785e-05, 2.47964e-05, -3.78599e-06, 1.78544e-05, 0.000114449, 0.000206279, 8.7343e-06, 3.14325e-05, 4.88179e-05, -2.57988e-05, 1.06821e-05, 5.7316e-06, -0.000126212, 9.28831e-05, 0.000126325, -7.29114e-05, 3.99838e-06, 2.70645e-06, -9.45027e-06, 3.93653e-05, -8.51608e-06, -4.78416e-05, -5.12881e-05, -2.99599e-05, 6.91778e-05, 1.89599e-05, 1.15345e-05, 6.85896e-05, 3.64738e-05, -3.45572e-06, 0.000115616, -5.16191e-05, 3.35019e-05, 2.13797e-05, -2.54439e-05, -0.000210526, -4.72139e-05, -5.97324e-05, 6.98427e-05,
-0.0136971, 0.000121997, 0.000101098, 5.57023e-05, 0.000135139, -7.791e-05, 3.31527e-05, -7.54265e-05, 0.000153535, -0.000203742, 0.000154982, -9.64332e-05, -6.45865e-05, 8.08166e-05, 1.32848e-05, 4.59973e-05, -3.545e-05, 1.62813e-05, 0.000147874, 3.48423e-05, -1.5609e-05, 1.98501e-05, 0.000203143, -0.000113849, -0.000120931, -0.000142897, -5.21486e-05, 2.63767e-05, -7.72324e-05, -0.000132869, 0.000358782, -0.000122431, 8.2605e-05, -0.000157428, 5.63199e-06, -3.24075e-05, 3.57023e-05, 7.83958e-05, 8.96161e-05, -1.09245e-05, 7.656e-05, -8.62365e-06, -1.60349e-05, 0.000137166, -1.55218e-05, -1.40588e-05, 3.48517e-05, 3.9406e-05, -0.000159297, -0.000195926, 0.000182021, -0.000163323, 0.000103657, 0.000207208, -0.000254991, 0.000297605, -2.88943e-05, 2.48502e-05, -0.000186145, -0.000197251, -2.28165e-06, -0.000130217, 2.67567e-06, -0.0001481, -0.000148539, 0.000143611, -8.24387e-05, 7.80336e-05, 1.51116e-05, 0.000170763, 0.000355168, 7.48942e-06, -0.000213966, -4.00791e-05, -0.000125595, 0.000116474, 0.000187789, -0.000107819, 5.71183e-06, 0.000189733, -0.000192831, -0.000113379, 0.000116859, 0.000100211, 0.000100499, -2.32367e-05, 7.7657e-05, -0.000244147, 5.13131e-05, 0.000181274, -0.000148963, -0.000162171, -0.000188704, 0.000121883, 6.07926e-05, -3.84707e-05, -8.77553e-05, -4.46981e-05, -7.10289e-05, 0.000211992,
-0.00635054, -2.31031e-05, 1.90678e-05, 2.19122e-05, -2.9163e-05, -7.33919e-06, 1.61174e-05, 7.80131e-05, 3.72193e-05, -0.00011139, 9.16738e-05, -1.09912e-05, -3.04536e-06, -4.24539e-06, -4.99848e-05, -2.22247e-05, 4.95724e-05, 2.52232e-05, -3.94515e-05, -3.26695e-05, -0.000112158, -3.24538e-06, -9.05438e-05, -6.6051e-05, 8.35675e-06, 4.71191e-05, 2.70693e-07, -0.000111579, -0.000105567, 7.55493e-05, 1.65233e-05, 1.55382e-05, -0.000104493, 4.44281e-05, 5.23245e-05, -4.65857e-05, 4.84115e-05, -8.10822e-05, -0.00011321, -3.32755e-05, 2.84274e-05, -1.70197e-05, 8.9493e-06, 2.1858e-05, -7.42509e-05, -1.93568e-05, -7.78843e-06, -5.70263e-05, -0.000115151, 5.50793e-06, -0.000100401, -4.51143e-05, -8.24458e-05, -5.3837e-05, -3.57652e-06, -3.53875e-05, -2.90877e-05, -2.94535e-05, 7.14551e-06, 6.31418e-05, 3.08754e-05, 4.93463e-05, 3.60203e-05, 1.32302e-05, -4.13165e-05, -0.000114099, -7.04553e-06, -8.10296e-05, 3.54264e-05, -6.59534e-05, 0.000118073, 2.21136e-05, 9.70773e-06, 4.36302e-06, -8.39074e-06, 2.89389e-05, -6.81578e-06, -0.000109132, 0.000112841, 2.96508e-05, 6.23656e-05, 1.72344e-05, 2.33582e-05, 4.16986e-05, -7.76735e-06, -6.8705e-05, 2.50705e-05, -6.25163e-05, 3.3334e-05, 5.84361e-05, -8.66108e-05, 8.00194e-05, 7.80382e-05, -3.91278e-05, -3.84229e-05, 1.08347e-05, 0.000108084, 2.36806e-05, 4.74574e-05, 2.08796e-05,
0.0127147, -0.000230186, 0.000143668, 0.000120965, -0.000197412, -0.000408921, 0.000138936, -4.74934e-05, 7.51949e-05, 7.57639e-06, 8.95233e-05, -3.17629e-05, -0.000158677, -1.18611e-05, 4.86395e-05, 5.98672e-05, -4.60637e-05, 1.37957e-05, -3.9275e-05, 5.6454e-06, 7.08811e-05, 6.74428e-05, 9.0926e-05, -4.40996e-05, -0.000115697, -0.000159, 8.82575e-05, 4.9623e-05, 3.53601e-05, 2.54351e-05, 4.48937e-05, 0.000119504, 0.000201295, -0.000248551, -9.33308e-07, 0.000242107, 6.47381e-05, 0.000101954, -5.43579e-05, 0.000255354, -4.02389e-05, -0.000119225, 3.51888e-05, 0.000305863, -0.000184595, 0.000108573, 3.88346e-05, -1.7177e-07, -1.3121e-05, -3.89292e-05, 1.59233e-05, -3.80713e-05, -2.41301e-05, -8.64601e-05, 0.00014707, 0.000189854, -3.33462e-06, 6.34273e-05, -9.6639e-06, 0.000101591, -0.000106815, -0.000154569, 6.94445e-05, -9.96253e-07, -0.000109127, 6.45577e-05, 1.45153e-05, -8.76464e-05, 6.33824e-05, 4.75547e-05, 0.000118955, -3.21028e-05, 6.83373e-05, 1.49818e-05, 0.000191684, -0.000105994, 2.18485e-05, 9.32237e-05, -1.75569e-05, 0.00019529, 0.000101137, 0.00012831, 0.00024742, 2.85728e-05, -1.84252e-05, -0.000146853, 7.63475e-05, 0.000224976, -8.12156e-05, 1.7472e-05, -5.54649e-05, 0.000238081, 0.000306083, 7.5258e-05, -0.000140819, -0.00018143, 0.000128897, 0.000210445, -2.81554e-05, -7.96663e-05,
0.0164543, 4.42774e-05, 2.64113e-05, -0.000133645, 0.000177772, 0.000127319, -0.000211679, -4.58959e-05, 0.000432157, -0.00020498, 8.62208e-05, -0.000119893, 2.1976e-05, 0.000302128, 0.000242611, 0.000536825, 3.59555e-05, -6.49481e-05, -0.000277619, -0.000127828, 0.000245528, 0.00012063, -5.52147e-05, -5.36393e-05, -0.000111127, 8.77251e-05, -3.07894e-05, -0.000323085, 0.000204298, 0.000208665, 0.000349831, 2.60003e-05, 0.00029817, 0.000165978, 7.73163e-05, 0.00020664, 9.52947e-05, -2.29101e-05, 6.59977e-05, 5.91022e-05, -0.00024416, -0.000291257, -4.56167e-06, -0.000218245, 1.8767e-05, 0.000121885, 4.02434e-05, -0.000314552, 0.000161454, -0.000140907, -0.000164354, 1.45473e-05, -0.000152852, 2.35257e-05, -0.000204275, -0.000157073, -3.27303e-05, -2.42718e-05, 0.000214145, -0.000239603, -4.58396e-05, 0.00019196, -1.78143e-05, 0.000181666, -0.000180799, -8.48918e-06, -0.000211845, -0.000151942, -7.87886e-05, 4.32539e-05, 8.76852e-05, -0.000139786, 0.000343508, 0.000214357, -0.000139602, -7.99259e-05, 0.000203764, -1.49891e-05, -2.28109e-05, -0.000115021, -1.42638e-06, 0.000173244, -0.000207189, 0.000221981, 0.0002775, -3.09847e-05, 9.91106e-06, 0.000219169, -0.000273402, 6.68118e-05, -2.5384e-05, -0.000193223, 0.000356527, 0.000105442, -0.000299766, 0.000248241, 5.20804e-05, 1.17178e-05, 0.000307446, -0.000309613,
-0.032033, -0.000116697, 0.000157522, 0.000275534, -0.000711362, -0.000555298, 6.50354e-05, 0.000259096, -0.000100342, 0.000558489, 4.02851e-05, 0.000617657, -0.000615463, 0.000781568, 0.000261092, 0.000144781, -0.00055849, -0.000613716, -0.000265412, -0.000130273, -0.000126875, -0.000315866, -0.000146277, 0.000242343, 4.37126e-05, -0.000505135, -2.82064e-05, -0.000156225, -8.47801e-05, -4.58238e-05, -1.27746e-05, -0.000245923, 0.000103113, 0.000290295, 7.08909e-05, -0.000207, 0.000538077, 0.000548665, -0.000241809, 0.000438799, 0.000102936, 1.90693e-06, -0.000144131, -4.50207e-05, -0.000344461, -0.000429306, 6.32523e-05, -0.000141831, 0.000215573, -0.000183225, -0.000686606, -0.000388952, -0.000265652, -0.000305664, 8.73129e-05, -0.000104938, 7.14226e-05, -0.000251316, 1.63192e-05, 0.000791592, -0.000723176, -0.000479606, -0.000443646, 0.000345731, 6.11922e-06, -2.97468e-05, -0.000328491, -2.36859e-05, -0.000137132, -0.000158254, -0.000150846, 2.02231e-06, -0.000205609, -0.000215285, 4.37667e-05, -0.000491003, -0.000528343, -0.00019029, -0.000191185, -7.18093e-05, -0.000730626, -0.000327437, 5.87109e-05, 0.000186993, 0.000242426, -0.000327813, 0.000330539, 0.000858784, 0.000389734, 7.55646e-05, -0.000438876, 0.000640907, -6.34128e-05, -0.000402806, 0.000478556, 6.12299e-05, 6.83966e-05, 6.19147e-05, -0.000623533, 0.000406526,
0.00581292, 8.48449e-05, 6.83837e-06, 1.12069e-05, -1.0538e-05, 7.69981e-05, -8.70168e-06, 9.4107e-05, 7.36721e-05, 3.33039e-05, -5.20568e-05, -4.92082e-05, 9.88547e-05, 4.45429e-06, 1.6125e-05, 3.27362e-05, 4.87562e-05, -8.22727e-06, -4.78591e-05, -3.96386e-05, 3.76467e-05, -0.0001172, 2.33706e-06, 2.88595e-05, -6.30914e-05, 3.79428e-05, 3.4647e-05, 5.39509e-05, -8.11516e-05, 5.95698e-05, -3.37441e-05, 0.000175537, 7.3122e-06, -6.89993e-06, -3.47977e-07, -3.62221e-05, 3.48216e-06, 1.15856e-05, 6.76311e-05, -5.00027e-05, 0.00011815, 1.1687e-05, 5.65309e-05, 0.000105543, 4.64024e-05, 5.08583e-05, 9.62271e-05, -3.52799e-05, -6.74401e-05, 6.50034e-05, 2.35873e-05, 6.50154e-05, 8.40915e-05, -4.99179e-05, -1.31541e-05, 6.23351e-05, 1.58216e-06, 1.85422e-06, -3.33868e-05, 9.52257e-05, -6.83453e-05, -0.000122572, -0.000109828, 3.23794e-05, -1.37157e-05, 6.17153e-05, -2.31071e-05, 2.66775e-06, -9.04202e-06, -6.58003e-05, 7.18259e-06, 5.86885e-05, -5.68619e-05, 3.4957e-05, -7.94687e-05, -1.49416e-05, -3.28062e-05, -0.000100141, -4.92806e-05, 1.3459e-06, 2.14301e-05, -3.75577e-05, -7.00549e-05, -2.82376e-05, -1.02346e-05, -9.06966e-05, -1.3106e-05, 4.82435e-05, -6.03385e-05, 7.2443e-05, 1.12195e-05, -3.04329e-05, -5.15649e-05, 3.02204e-05, 1.29994e-05, -4.18031e-05, 4.10088e-05, 7.96341e-05, 7.96583e-05, -6.94164e-05,
-0.0226806, -0.000362316, 7.22079e-06, -0.000190087, 6.78016e-05, -0.000400839, -0.000299071, 9.39397e-05, 0.000111414, -0.000100082, 7.03353e-05, 0.000331236, -0.000205109, -0.00021715, -1.98481e-05, 0.000345141, 0.000221475, -0.000439206, -0.000134763, -0.000604914, -0.000139596, -0.000125027, -0.000189822, 4.63786e-05, 0.000126838, 0.000187683, -0.000154061, 2.2863e-05, 0.000119175, 0.000144732, 0.000266797, 7.70356e-05, -0.000179868, 0.000107884, -7.14838e-05, 5.34379e-05, -0.000470122, 9.41573e-05, 0.000156833, 0.000583374, -8.59942e-05, -8.16085e-05, 0.000145138, 7.28124e-05, -0.000152215, -0.000254841, 6.45487e-05, 2.16533e-05, 0.000154496, 0.000161676, -0.000129203, 0.00016788, -0.000345175, -0.000112986, 0.00012849, -0.000245909, -0.00041307, -9.73657e-05, -0.00010768, -0.000183106, 0.000178722, 0.000227607, -0.000151065, -9.89974e-05, -2.65398e-05, 0.000679431, -9.72773e-05, -2.15371e-05, 0.000284393, -1.0273e-05, -0.000202743, -1.87194e-05, -0.000152883, 3.62269e-05, 4.86014e-05, -0.000227874, -4.8257e-05, 0.000169372, 4.03314e-05, -0.000493747, 0.000250498, 0.000208816, -0.000214311, -0.000246718, 0.000195738, -0.000412641, -0.000178468, -2.52616e-05, 9.8118e-05, -1.71627e-06, -0.000178952, -0.000210898, 9.41073e-05, 0.000320851, 0.000193478, -0.000251257, 0.000200688, 0.000264357, -0.000420725, 3.66284e-05,
-0.0160209, -6.63247e-05, 0.000156856, -0.000283418, -1.60001e-05, -4.58366e-05, -9.36348e-05, 0.000187084, -0.000457109, 3.63374e-06, -0.000171174, -1.20524e-05, 0.000194461, 0.000101802, 0.000137396, -8.83202e-05, 0.000137206, -4.85563e-05, -6.18385e-05, -0.000185147, -0.000274075, 0.000154539, -0.000159965, -7.04657e-05, -0.000424728, 4.0789e-05, 6.60641e-05, -0.000201986, -2.35366e-05, 0.000191192, -0.000108868, -0.000108205, 0.00011286, 0.000119628, -5.05563e-05, -0.000108301, -0.000128883, -9.3355e-05, -0.000115453, -6.10948e-05, 0.000488007, -6.55292e-05, 0.000169399, -3.53449e-05, -0.000273252, 0.000238548, 6.93622e-05, 0.000136115, -0.000172423, 5.5404e-05, 0.000146808, 0.000377188, 4.54951e-05, -1.13279e-06, -1.68767e-05, 7.83177e-05, 8.21981e-05, -1.4127e-05, -0.000236889, -5.87454e-05, 0.000156497, 2.89968e-05, 0.000245516, 0.000271714, -4.91764e-05, -7.78584e-05, 9.46326e-06, 6.76061e-07, -0.000219564, -6.14511e-05, 0.000344434, 1.20067e-05, 6.79674e-05, -0.000498927, -3.35337e-05, 7.19733e-05, -6.02648e-05, -0.00023497, -0.000104921, 0.000199282, 1.93053e-05, -0.00043184, -0.000142141, -6.6539e-05, 7.36886e-06, -1.90061e-05, 0.000142345, -0.000156195, 5.46965e-05, -0.000255488, -0.00018929, -0.000324793, -0.000141918, 3.24542e-05, -2.76458e-05, 1.74105e-05, -0.00015832, -5.79608e-05, 0.000189243, 0.000228086,
-0.00589875, -2.73991e-05, -2.7942e-05, -1.23538e-05, -6.90437e-07, 7.34791e-05, -3.77895e-06, 1.71785e-05, -4.08462e-05, 1.68253e-06, -5.09016e-05, 3.73739e-05, -5.40933e-05, -5.73594e-05, 8.14349e-05, 7.65688e-05, -9.45809e-06, 3.22871e-05, -4.42917e-05, 5.56231e-05, 1.42025e-05, 0.000113963, 8.89078e-05, 4.85382e-05, 1.03198e-05, -4.58647e-05, 5.50725e-05, -4.53172e-06, 1.60961e-05, -4.48096e-07, -3.48094e-05, -8.62294e-06, 8.35524e-05, -4.72972e-05, 8.43028e-05, -4.29959e-05, -3.33842e-05, 6.30835e-05, 2.45671e-05, -1.67495e-05, -5.32089e-05, 2.59177e-06, 6.22297e-05, -4.91854e-05, -0.000136369, -5.35987e-05, -4.82896e-05, 2.96119e-05, -3.12451e-05, 2.54273e-05, -3.04178e-05, 1.40524e-05, -4.60854e-05, 7.65579e-05, 1.4491e-05, -4.37295e-05, -2.55949e-05, 1.3123e-05, -0.000131221, 7.70958e-06, -0.00010197, -2.89209e-05, 6.20632e-06, 0.000102847, -5.49185e-05, 1.62119e-05, 3.95489e-05, 8.00513e-06, -4.27224e-06, 5.00069e-05, -0.00010544, -2.67746e-05, 4.65628e-05, -9.77638e-05, -3.87494e-05, 2.6071e-05, 5.45795e-05, 4.88702e-05, -1.55573e-05, 3.03147e-05, 2.36231e-05, -1.57097e-05, 2.06378e-05, 5.69597e-05, -2.59982e-05, -3.89391e-05, -3.95432e-05, 6.92238e-05, 1.66975e-05, 4.64109e-05, 5.13003e-06, 5.34547e-05, -2.18466e-06, -5.35857e-06, -4.49371e-05, 5.42766e-05, -1.62751e-05, 0.00010867, 3.82144e-05, -9.04626e-06,
0.00723543, 2.79905e-05, -9.58386e-05, 3.43413e-05, -2.11814e-05, -0.000100119, 3.16182e-07, -5.10169e-05, -0.000116262, 6.10414e-06, -9.45681e-05, 3.59383e-05, 0.000120251, 1.2748e-05, 2.25626e-05, -2.13135e-05, -2.21182e-05, -2.12416e-05, -7.89473e-05, -3.78303e-05, -1.09356e-05, 3.86156e-06, 6.11352e-05, -3.85302e-05, 8.47916e-06, -6.49536e-05, 0.000155351, -1.88539e-05, 4.36183e-05, -5.82663e-05, 1.66103e-05, 1.5749e-05, -4.62086e-05, 0.000167741, -1.78645e-05, 5.68615e-05, -2.42117e-05, 5.31334e-05, -2.02607e-05, 7.36554e-05, 6.17255e-05, 7.89555e-05, 0.000107876, -1.38455e-05, 5.81464e-05, 2.30382e-05, -3.8446e-06, -3.6304e-05, -1.54857e-05, -8.95495e-05, -2.4116e-05, 6.53094e-05, -2.60317e-05, -7.43905e-05, 7.05647e-05, 5.46393e-05, -5.18465e-05, 3.04952e-07, 0.0001316, -1.45633e-05, -9.64875e-05, -0.000112071, 7.38689e-06, 1.08136e-05, -9.84305e-05, 4.64754e-05, 5.61904e-05, -8.02921e-05, -1.58156e-05, 1.22194e-05, -3.42045e-05, -1.70856e-05, 6.20286e-06, 9.50042e-05, -0.00012388, 4.79536e-05, -0.000111028, -0.000106415, -1.40181e-05, -7.14523e-05, 8.93882e-05, 4.45542e-05, -2.06571e-05, -3.01331e-05, 3.11837e-06, 0.000182895, 6.58151e-05, -0.000159414, -8.24451e-06, -0.000104268, -5.61595e-06, -4.64146e-05, 3.0098e-05, -6.65291e-05, -1.29696e-05, -6.8701e-05, -6.50661e-06, -5.92288e-05, -1.34706e-06, 7.73405e-07,
0.00700827, 1.59895e-05, -9.34538e-05, 1.56684e-05, 4.7778e-05, -2.16529e-05, -8.29552e-05, 4.03245e-05, 0.000148827, -6.43051e-05, 7.14842e-05, 2.69842e-05, 1.41479e-05, 0.00016506, -4.2154e-05, -7.12373e-05, -5.08706e-05, 3.60265e-05, 3.70859e-05, -6.10797e-05, -5.56983e-07, 0.000147416, 9.96248e-05, 5.70901e-08, -0.000112916, -4.10993e-06, -9.32282e-05, -9.21648e-05, 6.40391e-05, -5.36729e-05, 2.2288e-05, 8.65523e-05, 0.000177553, 1.25321e-05, 1.85581e-05, 6.59795e-05, 0.00010584, 1.84374e-05, 9.23707e-05, -1.50839e-05, 2.17614e-05, -3.50466e-05, 7.07365e-05, 5.87098e-05, -8.36567e-06, -0.000155143, -5.53872e-05, -8.39947e-06, -0.000183078, 3.33062e-05, -2.17596e-05, -7.38538e-05, -1.19943e-05, -9.05216e-06, 1.4649e-05, 0.000144908, -1.10663e-05, -6.5234e-05, -6.03913e-05, -9.87236e-06, -0.000109314, 0.000132329, -5.4867e-06, 5.18124e-05, -2.39388e-05, -1.84113e-05, 9.25694e-07, 0.000191349, -0.000118211, 7.70672e-06, 4.45876e-05, 9.33877e-06, 3.10047e-05, -3.18559e-05, 3.67485e-05, 3.59235e-05, -2.79875e-05, 4.16866e-05, 0.000108801, -7.53879e-05, -6.33298e-05, 4.11381e-06, -5.03538e-06, 1.78763e-05, 0.000141358, 0.00015726, 1.0003e-06, 3.82224e-05, -0.00013337, 0.000100379, 2.16564e-05, -9.97851e-05, -3.43123e-05, -5.62384e-05, 1.24046e-05, 5.8598e-05, -1.67946e-05, 4.49906e-05, -0.000101734, 9.59784e-05,
0.0175294, -0.000106119, 0.000187143, -8.00116e-05, 1.70969e-05, -8.79147e-05, -0.000159431, 0.000131062, -0.00026596, 0.000230735, -6.99136e-05, 7.01139e-05, 8.09097e-05, -6.0425e-05, 2.06151e-05, 0.000172667, -0.00020064, 0.000169635, -2.0946e-05, 9.93375e-05, -2.78044e-05, 0.00031127, 4.78943e-06, 1.81676e-05, -6.92867e-05, -4.41987e-05, 3.40387e-05, -0.000231976, 0.000214239, -0.00016489, 0.000211523, 7.52889e-05, 9.67053e-05, 0.000243245, 2.81185e-05, 0.000188022, 4.84012e-05, 4.56024e-05, 1.23997e-05, 0.000118078, 0.0001925, 0.000170689, -7.58123e-05, -0.00012323, 0.000381852, -1.275e-06, 3.09232e-05, -2.11982e-06, 0.000375277, -6.05374e-05, -0.0001206, -0.00010643, 8.11542e-05, 0.000147741, -0.000239064, 2.34682e-05, -0.000105489, 0.000571532, -0.000346035, -2.59755e-05, 2.42203e-05, -4.8626e-05, -5.15574e-05, -1.52646e-05, -0.00011237, 0.000192229, -6.86383e-05, 5.74501e-05, -0.000294004, 0.000279046, 0.00023601, 3.39274e-05, -0.000208295, -0.000184583, 9.09717e-06, -5.07386e-05, -0.000127101, 0.000314517, -5.10576e-05, 5.94864e-05, 3.24414e-05, -2.56744e-05, -0.00034739, -7.91024e-05, -0.000221119, 0.000136516, -4.46122e-05, 0.000279093, -0.000204207, 0.000146801, 8.76395e-05, 0.000164673, -0.00040893, -1.7694e-05, 0.000212504, -0.000106073, 0.000172406, 9.13426e-05, 6.67963e-06, 0.000199183,
0.00699425, -1.49149e-05, 5.25142e-05, -6.99272e-05, 4.33303e-05, -7.0124e-05, -6.34102e-05, 1.51258e-05, -7.50018e-05, 3.49959e-05, 6.33141e-05, 9.26958e-05, -5.31233e-05, 4.94219e-06, 2.74878e-05, -3.433e-05, 3.26107e-05, 5.48429e-05, -8.86965e-05, -0.000127407, -0.000108993, 1.29425e-05, 0.000126666, -4.23471e-05, -6.32755e-05, -1.87984e-05, 1.2132e-05, 2.59478e-05, 1.60103e-05, 1.94307e-05, 3.94397e-05, 6.66686e-05, -0.000146012, -0.000110092, 6.32763e-06, -9.57549e-05, 0.000145787, 3.60656e-05, 1.05568e-05, 4.8098e-06, -1.1693e-05, 3.19584e-05, -7.96731e-05, -0.000117143, 0.000124481, -0.000181201, -2.60743e-05, 5.02173e-05, -6.8128e-05, -3.62157e-05, 3.62675e-05, -7.69465e-05, 3.30526e-05, 6.34056e-06, 8.19222e-05, -4.25479e-06, -3.37776e-05, -0.000133586, 1.68929e-05, 3.49051e-05, -1.81695e-05, 3.28987e-05, -1.95589e-05, 3.75633e-05, 5.55864e-05, -4.04628e-05, 0.000100546, -4.15712e-07, -1.38033e-05, 3.06474e-06, 1.02401e-05, -7.63643e-05, -9.05962e-05, -6.85851e-05, 2.61774e-05, -6.16681e-06, 3.91332e-05, -1.32687e-05, -8.3028e-07, 3.09931e-05, -4.88434e-05, -0.000120864, -1.77317e-05, -1.87181e-05, 9.96079e-05, 6.75822e-05, 0.000120436, -0.00011569, -4.88921e-05, 9.52357e-05, 2.95841e-05, 6.65118e-05, 3.05307e-05, 7.54268e-05, 4.31362e-05, -3.36919e-05, -8.00942e-05, -4.85535e-05, 3.64519e-06, -1.80254e-05,
-0.00133171, -1.02947e-05, -8.37288e-07, -4.47249e-06, -2.3912e-05, -9.97249e-06, -2.6823e-06, 1.25757e-05, 8.91091e-06, 3.23343e-05, -4.22123e-06, 3.14779e-06, 2.23124e-07, -1.57228e-06, -3.78206e-06, 7.91827e-06, -3.41763e-06, -1.62654e-05, -1.17496e-05, -6.41025e-06, -6.60057e-06, 1.72808e-05, -3.6095e-06, -1.83276e-05, -1.91992e-06, -4.57107e-06, -1.45525e-05, -1.94481e-05, -1.40781e-05, -1.79427e-05, 5.11135e-06, -1.16609e-05, 1.4522e-05, 8.92814e-06, 3.49326e-05, -3.74704e-05, 9.27877e-06, -5.1633e-06, 4.61073e-06, -7.03551e-06, -9.09541e-06, 4.29749e-06, -1.87048e-05, -1.64272e-05, -1.23987e-05, 9.16494e-06, 4.73387e-06, -1.81856e-05, 1.06627e-05, 1.42883e-05, 1.21018e-05, -1.26582e-05, -1.21286e-05, -9.5876e-07, 6.58514e-06, 1.24071e-05, -1.711e-05, -1.75602e-05, -2.4048e-05, 1.73306e-07, -1.44693e-05, 4.22647e-06, 1.02118e-05, -3.65971e-06, 1.7039e-05, 5.70508e-07, -1.6539e-05, 9.3851e-06, -1.06161e-05, 1.08065e-05, 1.08592e-05, 8.70656e-06, 1.76057e-05, 7.78233e-06, -2.49602e-06, -2.14232e-05, -1.21897e-05, -7.27402e-06, -4.97835e-06, 5.91143e-06, -7.99119e-07, -1.73852e-05, -1.61594e-05, 1.84201e-05, 5.48646e-06, -3.37544e-06, 1.46497e-05, -6.28958e-06, -4.15702e-06, 1.05592e-06, 1.58107e-05, 1.75301e-06, 1.11885e-05, 6.77632e-06, 8.66584e-07, -1.08363e-05, 1.88761e-05, -3.19397e-06, 1.20662e-05, 1.3809e-06,
-0.00895002, 4.65306e-06, -7.31104e-06, 5.36532e-05, -2.48767e-05, 2.70556e-05, 0.00012092, -5.09341e-06, 6.49894e-05, 0.000157362, 1.35711e-06, 1.00775e-05, -0.000127733, -4.8749e-06, -7.5614e-05, 0.00014151, -8.66708e-05, 0.000123195, -0.000142625, 0.00023438, -6.57936e-05, 4.34355e-06, -6.64064e-05, -0.000155031, 8.17332e-05, -0.000154406, 8.74868e-05, 5.13365e-05, -0.000172674, -3.07552e-05, 0.000103381, 0.000101259, -0.000126736, -2.00481e-05, -0.00012293, 0.000102047, -2.44789e-05, -0.000151125, -2.22779e-05, 5.75294e-05, -0.000145421, 3.11405e-05, 1.67679e-05, -9.60793e-05, -7.86524e-05, -6.46733e-05, -2.42608e-05, -1.14023e-05, -7.84117e-05, -0.000111885, 0.000153628, 4.69355e-05, -7.4225e-05, -0.000160753, -9.42622e-06, -0.000114045, 0.000111154, 1.93988e-06, -0.000113559, -1.08708e-05, -0.000111683, 3.63376e-05, 7.5684e-05, -4.80987e-05, -4.76871e-05, -2.57535e-05, 6.36908e-05, 2.06089e-05, 7.35328e-05, -7.36059e-06, 9.37279e-05, -7.88267e-05, 1.9136e-05, -7.1809e-05, 9.42465e-05, 7.01926e-05, -4.60888e-05, -9.59543e-05, 6.49494e-05, 0.000110096, 3.4422e-05, 4.86782e-05, -1.87907e-06, -6.66834e-06, 4.15437e-05, 6.03975e-05, 8.46834e-05, 7.37598e-05, -6.28104e-05, 5.30897e-05, -1.62812e-06, 0.00010853, 0.000109595, -6.3589e-05, -3.67925e-05, 2.14581e-05, -3.68867e-06, -7.3635e-05, 0.000149039, 0.000129826,
0.00981265, 0.000121466, 0.000118501, 1.18713e-05, -4.97673e-06, -8.33123e-05, -5.31029e-05, 0.000112878, -1.46171e-05, 0.000170198, 0.000214887, 9.5773e-05, 2.73998e-05, 0.000134714, -4.07198e-05, 0.00023005, -2.10921e-06, 7.58145e-05, -1.85862e-05, -2.80018e-05, -4.77551e-06, -0.000106375, -0.000116594, -0.000112991, 0.000108345, 2.45493e-05, -5.1209e-05, -0.000129993, -0.000132255, 0.000119475, 0.000102437, -8.8292e-06, 0.000141465, -3.03833e-05, -2.46154e-05, -4.87253e-05, -7.47924e-06, -7.9268e-06, -1.41572e-06, -0.000137266, 8.89768e-05, 0.000110458, 5.03824e-05, -1.66428e-05, 4.43283e-05, -5.13209e-05, -0.000110994, 7.27592e-06, 1.68355e-05, 0.000175039, 0.000102099, -6.33371e-05, 0.000136454, -0.000144252, 3.60391e-05, -7.32107e-05, -4.23007e-05, 9.31843e-05, 9.29066e-05, 7.98979e-06, 8.81296e-05, -3.2704e-05, 1.61455e-05, 0.000139407, 6.5029e-05, 6.88175e-05, -5.61275e-05, -0.000203889, -2.29595e-05, -2.70214e-05, 0.000117291, 2.00797e-06, -5.42772e-05, 8.25113e-05, -6.12932e-05, 6.3913e-05, -0.00026022, 5.99798e-05, -1.84828e-05, 5.82034e-05, -7.643e-05, -3.94639e-05, 3.2485e-05, 6.48147e-05, 8.59558e-05, 2.0939e-06, 0.000200702, 0.000101922, 4.21028e-05, 0.000308904, 0.000147349, 9.11544e-05, -4.93701e-05, -4.69656e-05, -0.000143127, 4.81905e-05, -3.25026e-05, 0.000147535, -7.54188e-05, -8.80036e-05,
0.00182198, -5.22646e-06, 9.67841e-06, -2.36565e-05, -1.97187e-05, 2.67621e-05, 1.40782e-05, -1.01912e-05, 6.7255e-06, -6.79334e-06, -9.42276e-06, 2.25078e-05, 2.39455e-05, -1.00941e-05, 1.20582e-05, -5.97357e-06, 1.87466e-06, -3.45736e-06, 2.20958e-06, 1.78445e-05, -5.45274e-06, 2.2159e-05, -8.04633e-06, 1.60685e-05, 4.90285e-06, -1.1702e-05, -3.26788e-05, 1.26461e-05, -2.29884e-05, 6.09888e-06, 2.75578e-05, -2.19719e-06, -1.20546e-05, -3.10979e-05, 2.47539e-06, 2.37342e-05, 3.93704e-05, 2.12222e-05, -2.5042e-05, 3.31331e-06, -2.30271e-05, 4.22757e-05, -4.98085e-06, -1.7921e-05, -8.8662e-06, -1.80206e-05, -2.69021e-05, -1.43797e-05, 1.37713e-05, -4.79531e-06, -1.47191e-05, -3.8543e-06, 4.13597e-06, -2.88799e-05, 6.67077e-06, -2.20324e-05, -2.49167e-06, 1.73841e-05, 3.49897e-05, -3.78517e-05, 1.60904e-05, -2.33079e-05, -4.73228e-05, 1.34463e-05, -4.69628e-06, -2.60719e-05, -1.06376e-05, -1.79249e-05, -2.65574e-05, -7.11348e-06, -3.66707e-06, -8.27301e-06, 4.60784e-05, 2.18271e-05, 1.23491e-05, 3.7943e-05, 3.38519e-05, 1.73858e-05, 4.3076e-05, 4.35568e-06, 1.72709e-05, 1.65561e-05, -9.54837e-06, -8.61547e-06, 8.34748e-06, 6.36154e-06, -3.11675e-06, 7.22381e-06, 1.96426e-05, 1.9006e-05, 5.65198e-06, 3.72618e-05, 1.11729e-06, 1.80185e-05, -9.82819e-06, -1.43106e-05, 1.1059e-06, -1.54006e-05, 1.16536e-06, -3.7571e-05,
0.00450763, -0.00010922, 1.46975e-05, 2.40335e-05, 2.80795e-05, 3.5525e-05, 2.19924e-05, 6.99828e-05, 4.27881e-05, 6.23731e-06, 9.68774e-05, -1.60022e-05, -2.44301e-05, -0.000130582, -7.58477e-06, 4.98089e-05, -5.85081e-06, 3.9203e-05, -8.41157e-06, 1.13263e-05, 6.11806e-05, -8.52414e-05, 9.1258e-07, -1.25485e-05, -2.74494e-05, -2.14741e-05, 2.16885e-05, 4.36163e-05, -2.34804e-06, 5.70667e-05, -1.62866e-05, -9.59666e-06, -1.40524e-06, 6.38915e-05, -3.1729e-06, 1.73379e-05, -3.58179e-05, -5.28818e-05, 2.30863e-05, 1.30894e-05, 1.11075e-05, 2.41069e-05, -4.06583e-05, 5.08114e-05, -5.55055e-05, 2.05874e-05, -3.61421e-05, -2.92026e-05, -3.79685e-06, -0.000114075, -6.28077e-06, -9.87901e-06, -4.20171e-05, 3.44909e-07, 6.02294e-05, 2.45311e-05, -2.53087e-05, 3.12519e-05, 4.10868e-05, -1.25571e-05, 2.69588e-05, -3.3479e-05, -1.91915e-05, 1.00182e-05, 5.58774e-05, -4.8052e-05, 3.13852e-05, 1.52489e-05, -9.22307e-08, 1.81532e-05, 2.93083e-05, -5.81699e-05, 2.58044e-05, -2.03132e-05, -6.28186e-05, -7.00588e-05, -5.41546e-05, 6.56589e-05, 2.83341e-05, -0.000104025, -3.28069e-05, 5.07586e-05, -1.66919e-06, -7.72854e-05, -5.54859e-05, -4.57137e-05, -1.73486e-05, 3.68959e-05, -6.39379e-06, -3.37209e-05, -9.17373e-05, -2.73981e-05, 1.79845e-05, -1.47589e-05, -7.6624e-06, 3.90954e-05, 5.95807e-05, 6.80549e-05, -3.26953e-05, 4.06156e-05,
-0.0113148, 0.000226502, -2.57667e-05, -1.29908e-05, -0.000206666, 1.92097e-05, -4.20276e-05, 0.00012767, 8.06919e-05, 2.21548e-05, -9.00133e-05, 0.000213296, -5.6217e-05, -9.73952e-05, -0.000104974, 7.23144e-05, 0.0001133, -9.36006e-05, 0.000133303, -0.00010484, 6.12612e-05, 0.000142837, -2.26813e-05, -0.000184334, 9.63695e-05, 0.000176352, -2.20612e-05, 5.15667e-05, 0.00019526, -1.66197e-05, 4.78446e-05, 1.72545e-05, 2.23777e-05, -7.34145e-05, 5.18056e-05, 3.29899e-05, -0.000188162, 8.95278e-05, 1.46423e-05, 3.59406e-05, 8.19957e-05, 0.000188125, 7.02971e-05, 6.33283e-05, -9.3578e-05, 0.000242434, -0.000165504, 8.60185e-06, -3.62022e-05, 0.000214183, -9.68536e-05, -5.93228e-05, -9.63041e-05, -1.77496e-05, -1.54481e-05, 0.000124058, 0.000108762, 0.000218515, -8.4379e-05, -0.000217612, 8.15462e-05, 7.47648e-05, 0.000210744, -4.2361e-05, 0.000102437, 1.74029e-05, 8.39782e-06, -0.000139317, -0.000120714, -4.21009e-05, -1.58064e-05, -6.40056e-05, 7.27371e-05, 0.000100818, 0.000175904, 0.000144224, 0.000130708, -0.000125546, -0.000135198, -5.76412e-05, -5.68468e-05, -9.6551e-05, 6.41005e-05, -5.41298e-05, -0.000173482, 0.00019817, -4.1663e-05, -1.84034e-05, 0.000149157, -0.00014917, -6.29835e-05, 0.000124229, 8.22971e-05, 0.000229533, -0.000138981, 4.68443e-05, -0.0001989, -4.98787e-05, -0.000204941, -6.22162e-05,
0.00958691, 0.000205509, 0.000110066, 1.26458e-05, 8.71488e-05, -8.78746e-05, 4.13374e-05, 0.000193503, -5.74684e-05, -0.000120387, -0.000165144, -6.51511e-05, -0.000213696, -0.00011898, -7.97163e-05, 0.000144143, 2.26256e-05, -8.25534e-05, 0.000113787, 5.39347e-05, -7.76769e-05, 0.000138778, 1.46904e-05, 0.000161607, 6.16706e-05, -0.000122158, 2.40779e-05, -6.1111e-05, -2.36837e-05, -2.16711e-05, -2.38067e-05, -0.000140413, 0.000132156, 5.06357e-06, -0.000111986, -5.8434e-05, -9.47193e-05, 0.000112926, -4.96509e-05, -5.83677e-05, -8.17258e-05, -0.000106192, -1.98028e-05, 5.86411e-05, 0.000179151, -0.000109964, 2.59056e-05, 0.0002469, 9.659e-05, -8.42077e-05, -9.14648e-05, 2.88911e-05, 0.000115353, 8.12478e-05, 0.000119614, -0.000219338, -3.42162e-05, 7.11627e-06, 3.93896e-06, -3.71658e-05, -0.000161904, -8.08313e-05, -9.3846e-05, 3.15161e-05, 1.53331e-05, 0.000128578, -7.58583e-05, 1.94806e-05, 0.000211932, 2.95149e-06, -6.32167e-05, -9.93508e-05, -0.000117874, -7.60036e-05, 0.00013198, -0.000138884, -6.90428e-05, -6.03992e-05, -3.14632e-05, -3.45979e-05, -7.11314e-05, -8.86874e-05, -3.44677e-06, 8.16278e-05, 3.8578e-05, -7.81794e-05, 0.000129285, -7.55299e-05, 4.38948e-05, -7.7413e-05, 8.34592e-05, -4.56155e-05, 3.5599e-06, 8.88303e-05, -0.00012742, 4.03484e-05, 1.47692e-05, -3.90101e-05, -3.39387e-05, -0.000122553,
0.0166148, 0.000123169, -4.51428e-05, 0.000144517, 0.000271372, 2.38541e-05, -1.15174e-05, -8.22861e-05, -0.000129797, -0.000175875, -0.000109406, -0.000192614, 0.0001155, 0.000123644, 5.5466e-05, 0.000187657, -3.95731e-05, 0.000130854, -1.22382e-05, -7.54818e-05, -0.000228904, -0.000234991, -7.99046e-05, -0.000148029, -7.04577e-05, 0.00038894, 0.000183774, 5.25323e-05, -3.48842e-05, -4.79871e-05, 0.000142224, 0.000233326, -4.13294e-05, 5.45105e-05, 0.000255139, -0.000212522, -0.000204601, -0.000173379, 5.13888e-05, 0.000227258, 5.49048e-05, 8.10862e-05, -6.71292e-05, 5.20796e-05, -3.56649e-05, 0.000162065, -4.64415e-05, -2.12452e-05, -0.000331055, -4.76669e-05, 1.6194e-06, -9.60756e-05, -3.48518e-05, 5.54804e-05, 0.000125376, -0.000211287, -1.57064e-05, 0.000135522, -0.00019337, -3.44112e-05, -7.98421e-05, -2.17796e-05, -9.30191e-05, -6.49322e-05, -0.000252606, 0.000267597, 0.000373659, 9.32094e-05, 0.000246807, 8.87442e-05, 6.45427e-05, 9.38269e-05, 8.84923e-05, 0.000181919, 2.50997e-05, -0.000155058, -4.57277e-06, -5.63455e-05, 2.32408e-05, -0.000217255, -9.33916e-05, 7.86601e-05, 3.20723e-05, -0.000118051, 0.000198599, -1.73905e-06, -0.000143121, 9.67013e-05, -4.6115e-05, -0.000131368, -0.000307504, -9.76656e-05, -8.13772e-05, 1.5359e-05, 2.30967e-06, 0.000297288, -0.000152888, 0.000186498, -8.60209e-05, -9.24432e-05,
0.0235764, 0.000252463, -6.70721e-05, 0.000161192, 0.000318973, 8.49084e-05, 0.00033832, 0.000162629, 0.000423415, 0.000202908, 0.000236222, -2.39023e-05, 0.00013738, 0.000392338, -0.000258993, 0.000110809, -0.000103156, -7.21325e-06, -0.000251628, 4.7052e-05, 6.6357e-05, 8.44195e-05, 6.85545e-05, 0.000320792, -0.000206572, -4.84896e-05, -0.000177769, 6.40494e-05, -6.32245e-06, 0.00024333, 1.13543e-06, 0.000353106, 0.00011499, 2.86309e-05, -0.000155152, -0.000159927, -8.31836e-05, 7.26874e-05, 3.27716e-05, -0.000154306, 0.00026952, -0.000169052, 0.000561719, 0.00011033, 0.000137752, -0.000505571, 6.96515e-06, 0.000212652, -0.000113335, -0.000290029, -0.00034234, -0.000129937, -0.000175125, 3.05401e-05, 9.13488e-05, 0.000632923, 1.19353e-05, 0.000296632, -0.000235682, 0.000333588, 0.000441306, -0.000437914, 4.38071e-05, -0.000376436, 3.47972e-05, -0.000619032, -0.000296893, -2.89518e-05, -0.000331876, 1.91478e-05, -0.000218315, -0.000184517, 0.000123684, 0.000108721, -0.000168073, 0.000323067, -0.000280244, -0.000117485, -7.69242e-05, -8.52265e-05, -0.00022058, 0.000159965, 0.000178404, 3.22996e-05, 0.000181391, -0.000310418, 0.000202188, -0.000387413, 0.000106077, -0.00013941, 0.000213339, 0.000162267, 0.000426735, -0.000232627, 0.000172891, -0.000178823, -9.39598e-07, -0.000331149, -0.000167652, 5.07098e-05,
-0.00606328, -1.77425e-05, -8.21672e-05, -8.81368e-06, -7.58417e-06, -1.67011e-05, -5.10538e-05, -1.92139e-05, 3.08842e-05, -8.47682e-05, -1.81663e-05, -1.24231e-05, 0.000131437, 5.62095e-05, 0.000108674, 5.27477e-05, 3.81663e-06, -3.7088e-05, 9.23224e-05, 5.22829e-05, -6.64068e-05, -6.37754e-05, 8.23752e-05, 0.000149553, 4.6259e-06, 0.000104491, -3.6845e-05, -7.96502e-05, 9.96607e-05, -3.51694e-05, -5.68009e-05, -7.43814e-05, 3.7208e-05, 8.18906e-05, -9.17948e-05, 5.37619e-06, -4.3166e-05, 2.19123e-05, 3.13615e-05, 3.47333e-05, -8.25358e-05, -5.74452e-05, 3.94801e-05, -3.76893e-05, 0.000108728, 3.58817e-06, -3.28632e-05, -1.39198e-05, -6.68708e-05, -4.08496e-05, -2.79113e-05, 1.54838e-05, 0.000113579, 4.0464e-05, -3.43576e-05, -0.000105602, 2.44254e-05, 2.82223e-05, 7.64462e-05, 5.05467e-05, -8.98492e-05, 3.93646e-05, -5.5861e-05, 6.67402e-05, 0.00011864, 3.56942e-05, -4.75472e-05, 2.22097e-05, 0.000115052, 3.88486e-06, -6.30524e-05, 0.000105559, 2.21838e-05, 0.000139624, -3.33665e-06, -1.52698e-05, -2.01858e-05, -1.60082e-05, -2.34824e-05, 5.87236e-05, 1.06749e-05, 2.05638e-05, -4.68926e-05, -4.99938e-05, 4.17252e-05, 0.000103709, -2.05216e-05, 4.35969e-05, 6.97774e-05, 7.61561e-05, 5.08544e-05, -2.65046e-05, -6.91211e-05, 5.37658e-06, -9.31642e-05, 8.00459e-05, 4.868e-05, -2.38523e-05, -7.21045e-05, -6.25166e-05,
-0.00222329, 1.6356e-05, 3.25559e-05, -2.7697e-06, 4.02147e-05, 3.03414e-05, 3.46496e-05, 7.70396e-06, 4.45047e-05, 3.2438e-05, 2.23721e-05, 1.5745e-05, 2.57403e-05, -2.43157e-05, -3.06839e-05, -5.71132e-06, 1.35417e-05, 4.15711e-05, -3.67533e-06, 1.4197e-05, 1.22681e-05, 7.2765e-06, -1.46858e-05, -3.0503e-05, 9.59888e-06, 1.67573e-05, 1.00834e-05, -1.32378e-05, 6.58942e-06, -2.45891e-05, 1.16236e-06, -3.30483e-06, 1.60438e-05, 6.46363e-06, 1.43472e-05, -2.66635e-05, 3.10306e-07, -7.32342e-06, -1.3528e-05, -3.26889e-05, -1.83076e-05, 3.55823e-06, -2.84891e-05, -1.59742e-05, 2.08683e-06, -3.40221e-05, 2.89957e-06, 9.70676e-06, 2.28604e-05, -3.38986e-05, 4.26836e-05, 1.88648e-05, 4.04802e-05, 2.35314e-05, 4.2716e-05, 5.94619e-06, -3.02987e-05, 1.77795e-06, -7.46536e-06, 3.78676e-05, -3.5218e-05, 1.68337e-05, -2.84202e-05, 7.82664e-06, -5.55432e-06, -1.72869e-05, -6.21474e-06, 2.17287e-05, -2.13454e-05, -1.48873e-05, 6.91621e-06, -3.05535e-05, 2.86192e-05, -1.74005e-06, -3.51863e-06, -7.31663e-06, -9.5059e-06, -3.04425e-05, -6.56574e-06, 4.66078e-05, -3.95634e-06, 3.56626e-05, -3.0636e-06, -1.97325e-05, 6.7861e-06, 5.07965e-06, 1.68461e-05, -6.12414e-06, 5.76853e-07, -4.06162e-05, 1.87011e-05, 9.43342e-06, -1.54683e-05, -4.86983e-05, 2.60783e-05, 3.93339e-06, 8.68655e-06, -6.23846e-06, 1.05229e-05, 1.09472e-05,
-0.0264733, -0.00023362, -8.98581e-05, 0.000204352, -0.000357118, 9.06493e-05, -0.000373209, 9.9699e-06, -9.13542e-05, 7.97122e-06, 4.6115e-05, -0.000233038, 8.9035e-05, -0.000237423, 0.000374107, 0.00023079, -0.00015985, 5.99534e-05, -0.000135568, -0.000301335, 0.00015641, 0.000422754, -0.000183777, -6.94715e-05, 0.000130129, -0.000448901, 0.000164941, -0.000393513, 7.74331e-05, 7.50988e-06, -5.7615e-05, 0.000346322, 0.000620978, -5.83726e-05, -9.95268e-05, -0.000583189, 5.14236e-06, -0.000155283, -0.000160949, -0.000169545, -0.000367563, 3.28467e-05, 0.000264523, -1.28711e-05, 0.000340524, 0.000210787, 0.000101552, 0.000194543, 0.000104514, 0.000342344, 0.000188224, 0.000158602, -0.000233744, 0.00058611, -0.000419084, -6.27097e-05, -0.000366128, -1.51919e-05, -1.10295e-05, -0.000137008, -0.000249681, 0.000157813, 0.000230054, -8.21872e-05, 0.000329465, 0.000256075, 0.000225744, 6.4167e-05, 0.000271611, -1.97126e-05, 0.000177572, -0.00029123, -0.000373823, -3.55254e-05, -0.000158978, -0.000224157, 0.000450493, -0.000123174, 0.0002351, 4.03463e-05, -0.00020096, -0.000178507, 7.97408e-05, -0.000336907, 8.58694e-05, -9.56922e-05, -6.19726e-05, -0.000281674, -0.000105988, -4.01477e-05, 7.47204e-05, -9.14543e-06, 0.000240088, -0.000142083, 0.000130227, 0.000350335, -2.27824e-05, 0.000244671, 0.000349422, 7.29987e-05,
-0.0134742, 6.59391e-06, 2.09902e-05, 0.000152491, -6.18385e-05, -9.42224e-05, 0.000102644, -1.3643e-05, -0.000186171, 2.48544e-05, 0.000130011, -0.00016931, -0.000299157, 8.22029e-05, 1.24339e-05, -0.000159324, -4.13988e-05, -0.000318714, -2.6568e-05, 0.000138056, 2.46595e-05, 7.60974e-05, 8.49131e-05, 0.000289502, 0.000251586, 5.42686e-05, 0.000114396, -0.000132045, -8.28309e-05, 5.05778e-05, -8.61827e-05, -0.000334638, 0.000171664, 0.000135132, 0.000186144, -0.000170633, 0.000137414, -0.000144876, -9.45315e-05, 6.81731e-06, -3.67862e-05, 6.06166e-05, -0.000303438, -6.94931e-06, 0.000120091, -0.000282807, 0.000137106, 1.03119e-05, 0.000153256, -0.000122836, 9.54594e-05, 0.000197349, 0.0001141, 2.76686e-05, 9.3574e-05, -0.000103744, -0.000136446, -0.000151749, 3.11137e-05, 8.60507e-05, 7.41346e-05, 0.000162802, 3.04701e-05, 0.000206072, 2.01113e-05, 0.00015393, -4.47258e-05, 0.000147307, 0.000111025, -7.49257e-05, 0.000138168, 8.48142e-05, -0.000410881, 1.48248e-05, 2.52488e-05, -0.000117985, 1.05905e-05, -9.73753e-05, -2.3306e-05, -8.85475e-05, 1.08721e-05, 2.81803e-05, -5.35855e-05, 0.000102058, 2.23869e-05, 5.02834e-05, -0.000181019, 0.000248798, 7.30842e-05, -0.000175347, -0.000111383, -5.39677e-05, -2.09627e-05, -0.000137031, 0.000129479, -0.000175078, -0.000149619, 0.00014379, 2.1397e-05, -2.45444e-05,
-0.00872986, 5.34741e-05, -0.000185342, -1.91432e-05, -9.20929e-05, -3.22374e-05, 2.17671e-05, 0.000114722, -0.00017115, 6.2423e-05, 2.24792e-05, 9.71377e-05, -0.000109716, -1.68607e-05, 6.93377e-05, -1.52556e-05, 8.33963e-05, -3.96133e-05, 2.64047e-05, 4.03999e-05, 0.000237259, -2.42826e-06, -1.13494e-05, 7.18636e-05, -4.37508e-05, 1.66358e-05, -9.22745e-06, 9.57802e-06, -0.000160766, -4.07189e-05, -2.50341e-05, -1.18673e-05, 5.57812e-05, -8.44544e-05, -4.33854e-05, -6.28028e-05, -8.3396e-06, -0.000247753, -3.34431e-05, 5.66241e-05, 5.68413e-05, -4.23219e-05, 0.000142786, -0.000142673, 8.473e-07, -0.000113566, 3.80069e-05, 0.000154338, 0.000120399, -7.7621e-06, -2.15125e-05, 8.57988e-05, -3.83632e-05, 9.62928e-05, 0.00017823, -5.19156e-05, 3.16501e-05, 6.43211e-06, -9.75055e-06, -0.000209252, 3.88127e-05, 6.5602e-06, 8.81763e-05, 6.9878e-05, 0.000115868, -2.96133e-05, -2.17841e-05, -5.78917e-05, 0.000116241, 5.89623e-05, -6.42817e-05, 6.46977e-05, 4.38734e-05, 4.85661e-05, -3.36344e-05, -0.000128628, -2.30209e-05, 9.41194e-06, 0.000120594, 4.78326e-05, -0.00013234, 3.41051e-05, -7.29044e-06, -0.00020066, 3.47981e-05, -0.00011668, 1.96525e-05, 3.52127e-05, 0.00011117, -4.51073e-05, 2.3387e-05, -1.29491e-05, 1.8147e-05, 8.53103e-05, 7.09413e-05, -7.77747e-05, 6.58434e-05, -0.000106566, 6.05357e-05, -0.000104257,
0.000153712, -3.22492e-06, -1.6796e-06, 1.29101e-06, -1.20776e-06, 2.39541e-06, 1.99047e-06, -2.14277e-06, 1.28273e-06, 1.54437e-06, 6.20579e-08, -1.08657e-06, -1.31319e-06, -9.79975e-07, -2.48492e-07, 7.57378e-07, 1.06104e-06, 1.98491e-07, -2.43834e-06, 7.23731e-09, 1.02316e-07, -1.26214e-06, -2.29165e-06, 4.24527e-07, 1.09458e-06, -1.12054e-07, -4.99034e-07, 8.5158e-07, -8.70765e-07, 9.29992e-07, 4.32668e-07, -5.61909e-07, 3.6648e-07, -7.34018e-07, 1.06852e-07, 1.15532e-06, 1.71277e-06, -5.23182e-07, 2.6658e-06, -1.16231e-06, -3.75723e-07, 4.54045e-07, 2.10819e-07, 1.34339e-06, 1.06601e-06, 2.19423e-06, 1.4368e-06, -1.71213e-06, 2.15305e-06, -1.61453e-06, -9.28683e-07, -3.00422e-07, -1.37545e-06, -1.42483e-06, -1.80356e-06, -4.3635e-07, -1.32361e-06, -1.84278e-06, 1.28465e-06, -1.03395e-06, 2.10395e-06, 1.03703e-07, -7.34753e-08, -1.83647e-09, -2.30005e-07, -1.13899e-06, -2.39024e-07, 2.08423e-07, -8.61767e-07, -6.97268e-07, 1.54498e-06, -6.23395e-08, -5.41411e-07, 1.94826e-06, -1.08275e-06, 5.87861e-07, 1.54799e-06, -4.99489e-07, 1.34016e-06, -1.04723e-06, -8.0603e-07, 1.85104e-06, -1.71532e-06, 9.2466e-07, -2.08962e-06, 3.58343e-06, 3.53684e-08, -1.81915e-06, -2.37038e-06, 6.65871e-08, 8.16867e-07, -1.51559e-07, -6.66973e-07, 1.10936e-06, 2.16112e-06, -2.47281e-06, 2.2629e-06, -8.38227e-07, 1.53043e-06, 2.38704e-06,
0.0017217, -1.00778e-05, -1.29031e-05, 1.18054e-05, 4.12947e-06, 2.61558e-05, -2.27879e-05, -2.31725e-05, -2.17061e-05, -1.09486e-05, 3.6091e-06, -3.63319e-06, 7.31452e-06, 1.14239e-05, 5.77818e-05, 9.45611e-06, -1.03681e-05, -1.80687e-05, -2.25638e-05, -1.96587e-05, 2.8409e-05, 3.89833e-06, -2.92094e-05, -2.34132e-05, -1.00743e-06, -9.54981e-06, -2.35281e-05, 1.28832e-05, -6.41269e-06, -2.35176e-05, 5.66763e-06, 4.14674e-05, -8.10263e-06, -1.47694e-05, 2.85066e-05, -1.46453e-05, -2.33927e-05, -2.54573e-05, 2.21931e-05, -7.69877e-07, -1.26318e-05, 7.84815e-06, -1.08189e-05, -4.17348e-05, 2.89052e-05, 1.65915e-06, -4.2921e-06, -1.05399e-05, -9.9512e-06, 1.71337e-05, 7.53312e-07, -1.59399e-05, -7.09059e-06, -3.79244e-06, -2.27645e-05, -1.8315e-05, 9.25301e-07, 7.50302e-06, -6.37583e-06, 1.6894e-05, 9.94267e-06, -2.18957e-06, 6.45164e-06, -2.67467e-05, 1.22486e-06, 2.34582e-06, 9.68361e-06, -5.49575e-06, -2.50574e-05, -7.34409e-06, 1.07219e-05, 1.72116e-05, 1.54982e-05, 2.16982e-05, 1.6529e-05, 4.0179e-06, -2.3448e-06, -3.09521e-05, 6.71602e-06, 9.96215e-06, 3.52727e-06, -2.2247e-05, -2.58078e-05, -6.49675e-06, 9.58428e-06, 2.57594e-05, -1.83081e-05, 4.97348e-05, -7.584e-06, -1.69909e-05, 2.13955e-05, 6.46888e-06, 8.61369e-06, -1.67658e-05, -1.73284e-05, 1.48714e-05, -1.12995e-05, 1.08481e-05, 5.57208e-06, -3.40473e-08,
0.0133092, -0.000238387, 0.000107174, -0.000147247, 2.30282e-06, -2.18399e-05, -0.000114154, -5.1374e-05, 2.57765e-05, -2.73062e-05, 0.000105127, -3.12226e-06, 0.000144714, -9.57409e-06, -3.13407e-05, 3.56772e-05, 3.33238e-05, 4.29063e-05, 0.000170708, -0.000390403, -0.000214212, -0.000247069, 7.55972e-05, -4.98208e-05, -5.08693e-05, -0.000133066, 3.44141e-05, -9.00708e-05, -0.000157301, -0.000279293, 0.000150097, -0.000217004, -0.000328736, -0.000195137, -6.91729e-05, -5.43274e-05, -6.985e-05, -0.000155516, 0.000202195, -1.80394e-05, 4.11952e-05, 7.11532e-05, 0.000175702, -8.24103e-05, -0.000131473, 0.000159575, 0.000128002, -6.1391e-05, 6.53396e-05, -7.87631e-05, 3.87638e-05, 3.6905e-05, -0.000135781, 1.21899e-05, -3.73846e-05, -0.000116171, 7.93498e-06, 0.000123551, 0.000101705, 1.85316e-06, -0.000133234, 4.66589e-05, 0.000123807, -7.71715e-05, -0.000256243, 7.93979e-06, 1.30941e-05, 0.000162161, 5.60254e-07, 0.000124997, -0.000150113, -5.93483e-05, 6.67954e-05, 0.000118882, -3.69502e-05, -8.08273e-05, 5.7693e-05, 0.000249847, 3.22934e-05, -0.000149776, -4.43677e-05, -0.00015773, 4.18569e-05, 6.9639e-05, -0.000241667, -6.19877e-05, 6.7741e-05, 1.11311e-05, 0.0002375, 3.64674e-05, 1.80206e-05, 9.48353e-05, -6.80063e-05, 0.000127491, 6.99257e-05, 8.28522e-05, 4.68875e-05, -7.51473e-05, 0.000134618, -0.00017281,
0.0100239, 4.39967e-05, -7.97095e-05, 0.000147311, -0.000112339, -0.00016601, 3.72719e-05, 0.000197976, -0.000127854, 7.07304e-05, 0.000102769, -3.6679e-05, 0.00024854, -0.000128685, -8.64067e-05, -3.35029e-05, 8.39235e-05, 7.37921e-05, -0.000300056, 5.3028e-06, 7.41312e-05, 8.34705e-05, -2.24117e-05, -8.52928e-05, -5.36841e-05, -6.9125e-05, 0.000154499, -6.65205e-05, 4.89454e-05, 3.61338e-05, -4.64449e-05, 0.000136097, 5.97964e-05, 0.00010508, 1.89185e-05, 7.66673e-05, -8.69052e-05, 3.89624e-05, 0.000155875, -8.80254e-05, 6.93722e-05, -1.22285e-05, -0.000131203, -7.44132e-05, 0.000164837, -9.99027e-06, -8.61124e-05, -0.000116237, -2.80269e-05, -0.000140107, 7.06352e-06, -5.55244e-05, 2.99934e-05, -5.45317e-05, 7.69934e-06, -1.60933e-05, -7.78913e-05, -4.55849e-05, 1.35497e-05, -5.32787e-05, -0.000115989, -6.8183e-05, -3.75347e-05, -0.000177302, -6.36895e-05, -1.08931e-05, -0.000107175, 0.000101995, 4.97027e-05, -9.62422e-05, -5.69846e-05, -0.000116169, 0.000118735, 2.46733e-05, -1.09784e-05, 1.99322e-05, -1.29313e-05, -0.000110021, 7.45877e-06, -0.000136084, -0.00012308, 0.000125539, -2.24002e-05, 4.85885e-05, 0.000144868, 3.72067e-05, 5.17895e-05, 0.00017317, 1.14458e-05, -8.77163e-05, 3.21534e-05, 0.000161223, -1.17232e-05, 0.000186827, 8.49292e-05, 0.000191941, -0.000160723, 0.000135812, -3.7933e-05, 1.98073e-05,
-0.00183152, 7.36763e-06, 3.13161e-06, -4.04476e-06, 1.69274e-05, 2.75621e-06, 1.22963e-05, -3.36622e-05, -1.0359e-05, 2.28897e-05, -2.96229e-05, 4.11792e-06, -2.89646e-06, -3.56714e-06, -4.53932e-05, 2.28877e-05, -1.26316e-06, 4.72875e-06, -1.44357e-05, -2.12716e-05, -9.25245e-08, -1.90047e-05, -1.56147e-05, -7.29204e-06, 1.158e-05, 3.95387e-05, 1.34721e-05, -8.55606e-06, -9.23258e-07, -1.8485e-05, -6.34934e-06, -1.31586e-05, -1.33622e-06, -8.42439e-06, 2.36902e-05, 1.01182e-05, 5.01659e-06, 6.66303e-06, -2.43089e-05, -2.45386e-05, -2.70919e-05, 2.99802e-07, 4.69443e-06, 2.45729e-05, -2.87025e-05, -8.92023e-06, 4.96004e-06, 1.28832e-05, -3.17218e-05, 1.01711e-05, 3.43043e-06, -6.28974e-06, 5.36474e-05, 1.16443e-05, -9.0582e-07, -3.27812e-05, -1.55255e-06, -3.69908e-05, -1.55678e-05, -2.62957e-06, 8.00324e-06, 2.9285e-05, -7.30451e-06, 1.24794e-05, -1.95129e-05, 6.8187e-06, 1.22711e-06, 4.22178e-07, 2.41968e-05, -2.7399e-05, -3.5167e-06, -1.15023e-05, 1.66898e-05, -3.77137e-06, -1.32095e-05, -1.42762e-05, -4.8783e-06, -8.64387e-06, 3.18888e-05, -9.88094e-06, 5.79247e-06, -1.05307e-05, -2.31014e-05, 2.49872e-05, -3.37327e-05, -5.17876e-06, -4.18913e-06, 3.79567e-05, -1.18025e-05, 1.47775e-05, -1.34749e-05, -8.29926e-06, -1.0394e-05, 1.31089e-05, -1.54863e-05, -2.14962e-06, -2.62527e-07, 9.62622e-06, -3.00422e-05, -4.55401e-05,
};
#endif
// Plastic BSDF FooTest
const float QuadLight::BSDFMatrix_Rawdata[100][100]
{ 0.447355, 0.00045284, 0.46206, 0.000113585, 0.00808554, 0.000356946, 0.125288, 0.000156355, -0.0045103, -0.00310479, 0.00324628, 0.000889256, -0.117025, 0.000121218, -0.00603861, 0.000592619, 0.000809988, -0.00471157, -0.00878434, 0.00272768, -0.0876337, 5.80625e-05, -0.00195204, 0.00131603, -0.0105474, -0.00295698, 0.000259735, -0.00410321, -0.0121817, 0.00398037, 0.0351062, 0.000117661, 0.00579605, 0.00167619, -0.00718666, 0.000826627, -0.0111672, -0.00421317, -0.000135928, -0.00195308, -0.00134785, 0.00235416, 0.0489427, 0.000601471, 0.0104675, 0.00127366, 0.00195546, 0.00129194, 0.00286512, -0.000366986, -0.0112559, -0.00324159, 0.001657, 8.60202e-05, 0.010184, -0.00122401, -0.0203422, 0.00173321, 0.00762633, 0.000559015, -0.00137517, 0.00106612, 0.00317828, 0.000315464, -0.0027358, -0.00063796, 0.000716685, -0.0014371, 0.00452728, 0.000849343, 0.00847204, -0.00357594, -0.0409896, 0.0031878, 0.000345524, 0.000485015, -0.0200934, 0.000296766, -0.000214761, 0.000404703, 0.0100522, 0.00186263, -0.00371752, -0.000658453, 0.0131963, -0.000727644, 0.00413888, 0.000378988, -0.00122246, -0.00366206, 0.00445089, 0.0040893, -0.00471193, 0.00127057, -0.031325, -0.000379556, -0.00506877, 0.000281098, 0.0133131, 0.000412844,
0.000224288, -0.118043, 0.000276484, -0.00630265, -0.00024369, -0.120147, 0.000140322, -0.00283978, -3.1835e-05, -0.00543602, -2.76001e-05, -0.0262966, -4.26799e-05, 0.00551144, -7.98578e-06, 0.00714639, -6.53513e-05, -0.00253538, 0.000267659, 0.0478353, -0.000102398, 0.00669832, 1.15903e-05, 0.00510273, 0.000426886, 0.00866885, -1.22929e-05, 0.00375917, 0.000108182, 0.0334639, 3.30668e-05, -0.00150723, -2.8969e-05, -0.00370673, 0.000114744, 0.00813322, 0.000574425, 0.0072029, 7.60648e-05, 0.00285076, -0.000307401, -0.0174473, 0.00021143, -0.00770171, -7.16232e-05, -0.00866979, -0.000478867, 0.00716121, 5.42701e-05, 0.00701348, 0.000249166, -0.00230277, 5.85222e-05, -0.0065952, -0.000262984, -0.0255338, 0.000170288, -0.00334761, -2.32609e-05, -0.00342679, -0.000430625, -0.0016773, 6.55352e-09, -0.00733177, 5.13079e-05, 0.00782532, -0.000443392, -0.0076157, -5.8447e-05, -0.0120148, 0.000275421, 0.00816716, -0.00010361, 0.00474947, 7.58929e-05, 0.00497354, 0.000308588, -0.00810275, -8.67264e-05, -0.00745082, -0.000663002, -0.00977181, 0.000118053, 0.00254155, -0.000433244, -0.000343578, -0.000107229, -0.00367987, 0.000462296, 0.0251762, -0.000260478, 0.00462255, 8.34319e-05, 0.00574631, 0.000633137, -0.0041895, -8.28375e-05, 0.000347116, -0.000135531, -0.0083792,
0.519302, 0.000544089, 0.559354, 9.77413e-05, 0.00278511, 0.000451575, 0.195881, 0.000203745, -0.00481964, -0.0039602, -0.000864159, 0.00113993, -0.0949992, 0.000205016, -0.00622812, 0.000824714, 0.000491877, -0.00587576, -0.00750867, 0.00338331, -0.103634, 4.84244e-05, -0.00150736, 0.00160356, -0.00603814, -0.0035471, 0.000257313, -0.00498116, -0.00867297, 0.00487073, 0.0105103, 2.40593e-05, 0.00709495, 0.00192092, -0.00414222, 0.00122795, -0.00645387, -0.00528096, 0.000555914, -0.00234907, -0.00158375, 0.00287802, 0.0394205, 0.000688998, 0.0123424, 0.00152916, -0.00145414, 0.00163882, 0.0022468, -0.00027715, -0.00640317, -0.00429239, 0.00267216, -1.6746e-05, 0.00636296, -0.00144447, -0.0139182, 0.00225252, 0.00941134, 0.000865048, -0.00948913, 0.00108475, 0.00218147, 0.000233512, -0.00316751, -0.000850372, 0.00132434, -0.00191266, 0.00508156, 0.00088298, 0.00738575, -0.0043371, -0.034666, 0.0040499, 0.00127963, 0.000740235, -0.0259964, 0.000194922, -0.0010723, 0.000520813, 0.00480302, 0.00191399, -0.00423713, -0.00122142, 0.0103614, -0.000625696, 0.00381908, 0.000530128, 0.00357384, -0.00460039, -0.000313329, 0.00497317, -0.00506513, 0.0014211, -0.0314161, -0.000280618, -0.00551229, 0.000605534, 0.00790858, 0.00027981,
3.83817e-05, -0.00630322, 0.000121081, -0.115167, -0.000164182, -0.00283975, 0.000133546, -0.117064, 5.67028e-05, -0.00787729, -1.37301e-05, 0.00551264, -2.96577e-05, -0.0263432, 1.38806e-05, 0.00822281, -9.32226e-06, -0.00522208, 0.000223112, 0.00669982, -0.000184423, 0.0438264, -5.57882e-05, 0.00689957, 0.000479899, 0.00733055, -9.3143e-05, 0.00423715, 0.00013951, -0.00150659, -9.545e-05, 0.0285714, -3.70544e-05, -0.000879373, 0.000192813, 0.0055211, 0.000481958, 0.0071613, -9.82911e-05, 0.00816098, -0.00020835, -0.00770235, 0.000100713, -0.0191111, 4.59848e-05, -0.00413602, -0.000356619, 0.00359626, 0.000158806, 0.00827879, 1.08693e-05, -0.00056893, 7.40415e-05, 0.00105348, -0.000274483, -0.00334939, 5.15281e-05, -0.0230807, 5.10873e-05, 0.00190674, -0.000275166, -0.00269589, 0.000199011, -0.0101037, -9.74649e-05, 0.00936326, -0.000785825, -0.00788885, 0.000242841, -0.00693496, 0.000123979, 0.0047465, -0.000221992, 0.0118456, -5.16529e-05, 0.00758118, 0.000386621, -0.00369653, 8.23599e-05, -0.0120439, -0.000512055, -0.00764534, -0.000123214, 0.00220733, -0.000645793, -0.00680856, 0.000148597, -0.00409451, 0.000393118, 0.00461826, -0.000273246, 0.0270895, -0.000114938, 0.00219929, 0.000555288, 0.00463602, -4.68169e-05, -0.00436181, -0.000204439, -0.006745,
0.00746274, 0.00017461, 0.00212512, 6.15539e-05, 0.116579, -2.33399e-05, -0.00897623, -1.1609e-05, 0.000382609, 0.00034116, 0.120312, -0.000253479, -0.00769997, -6.84689e-05, 6.17134e-05, -0.000312529, -0.00291967, 4.81979e-05, 0.0290778, -6.65003e-05, 0.00522559, 6.68365e-05, -0.00029198, -4.29827e-05, -0.0167833, -0.000430615, -0.00399792, -0.000374401, -0.0450554, 0.000283453, 0.0113148, 0.000222903, 0.000179654, 0.000396052, -0.0138687, -0.000576124, -0.0171387, -0.000117201, -0.00190144, -0.000186578, -0.0283756, 0.000173649, 0.00167717, 5.65956e-05, 0.000974472, 0.000277035, 0.00495397, -0.000203182, -0.0010064, -0.00068892, -0.0164469, 0.000413373, 0.00190968, 0.000410515, 0.0253871, -0.000297267, -0.00962995, -0.000331126, 0.000619315, -0.000326961, 0.0173355, 0.000589948, -0.000989351, 0.000566456, 0.00324754, -0.000399999, 0.00033198, 0.000290767, 0.00376449, 0.000437394, 0.028779, -0.000333672, -0.00630445, -0.000369647, -0.00104541, -0.000474956, 0.00785313, 0.000614435, -0.00037666, 8.89969e-05, 0.0170986, 0.000267489, 0.00365944, 0.000388573, 0.0127044, -0.000382353, 0.00213543, -0.000274979, -0.0151699, 0.000255412, 0.00604071, 0.00012623, -0.00214791, 0.0001733, -0.00969989, -0.000306452, -0.000226716, -0.000868426, 0.0186742, 0.000924865,
0.000975544, -0.120281, 0.00108717, -0.00284874, -0.000265148, -0.146158, 0.000435891, -0.000410048, -3.67626e-05, -0.00190383, -0.000114062, -0.0760803, -0.000142738, 0.00428408, -3.1651e-05, 0.00381493, -4.60338e-05, -0.000683804, 0.000183366, 0.00822897, -0.000193119, 0.00435606, -1.46596e-05, 0.00199754, 0.000393994, 0.00446378, -9.90094e-06, 0.000487936, 0.000150268, 0.0332374, 9.94916e-05, -0.00116007, -2.63975e-05, -0.00384631, 0.000162492, 0.00444567, 0.000513767, 0.0035611, 5.48711e-05, -0.003029, -0.000169712, 0.0085949, 0.000297854, -0.00538967, -4.45972e-05, -0.00707564, -0.000380176, 0.00373313, 3.25276e-05, 0.00424716, 0.000351272, -0.00101812, 4.8368e-05, -0.00959335, -0.000232013, -0.0101715, 0.000196421, -0.00324985, -1.1395e-05, -0.00360102, -0.000474951, -0.00146692, -7.45584e-06, -0.00403912, 3.55949e-05, 0.00560214, -0.000197693, -0.00233519, -3.58284e-05, -0.0107291, 0.00012478, -0.000232032, -6.3352e-05, 0.0014449, 5.82822e-05, 0.00252305, 5.6757e-05, -0.00536945, -8.00188e-05, -0.00373321, -0.000523162, -0.00565626, 8.92433e-05, 0.00380918, -0.000347518, 0.00334635, -8.90321e-05, -0.00187702, 0.000358572, 0.0138377, -0.000203329, 0.00191975, 6.8594e-05, 0.00467716, 0.00045392, -0.00345044, -9.89013e-05, 0.00121109, -0.000157516, -0.00470455,
0.252127, 0.000264611, 0.310542, -4.74101e-05, -0.00877286, 0.000270902, 0.184258, 0.000106149, -0.00110702, -0.0025992, -0.00742701, 0.000775361, 0.035304, 0.000213933, -0.00107855, 0.00071287, -0.000210397, -0.0036238, 0.000676576, 0.00207634, -0.0368205, 1.00198e-05, 0.000887483, 0.00099171, 0.00589409, -0.00184441, 0.000318783, -0.00282164, 0.00434961, 0.00283173, -0.0374657, -0.000192275, 0.00407932, 0.000900927, 0.00347535, 0.00114372, 0.00704477, -0.00316215, 0.00139677, -0.00126505, -0.000719501, 0.00164245, -0.0161996, 0.00027233, 0.00619574, 0.00076778, -0.00695482, 0.00114689, -0.000830953, 0.000207263, 0.00739622, -0.00300771, 0.00216641, -0.000194297, -0.00518759, -0.000766315, -0.00195663, 0.00147561, 0.00539388, 0.000744145, -0.0163659, 0.000308258, -0.00157866, -0.000159088, -0.00147802, -0.000494758, 0.00159039, -0.00145108, 0.00175753, 0.000227007, 7.03705e-05, -0.00244151, 0.000790779, 0.00259845, 0.00208655, 0.000702522, -0.015703, -0.000223096, -0.0019207, 0.000267374, -0.00842887, 0.000523189, -0.00165686, -0.00135263, -0.00261209, -2.34164e-05, -5.54292e-05, 0.000359748, 0.00966946, -0.0028415, -0.00519395, 0.00288107, -0.00157584, 0.000684778, -0.00678252, 4.84494e-05, -0.00202623, 0.000742021, -0.00836871, -0.00024873,
0.000111491, -0.00284942, 0.000188359, -0.117207, -0.000159849, -0.000410658, 0.000149781, -0.14333, 7.6357e-05, -0.00390529, -1.37726e-05, 0.00428467, -3.19111e-05, -0.0771503, 4.0638e-05, 0.00513844, 6.74501e-05, -0.00206385, 0.000230542, 0.00435839, -0.000197592, 0.00278603, -4.87142e-05, 0.00533946, 0.000493595, 0.00443967, 3.2751e-05, 0.00324912, 0.00018457, -0.00115775, -0.000167638, 0.0269659, -6.45319e-05, 0.00218844, 0.000337154, 0.00222895, 0.000495163, 0.0041, -9.96971e-06, 0.00518998, -0.000141035, -0.00539101, -2.7873e-05, 0.00590009, 9.26276e-06, 0.00107532, -0.000161147, 0.00131054, 0.000151291, 0.00508179, 0.000188933, -0.00128098, 6.04032e-05, 0.00104826, -0.000277358, -0.00325708, -3.44101e-05, -0.0079131, 4.7434e-05, 0.00367579, -0.000264153, -0.00040755, 0.000202169, -0.00654098, -1.08995e-07, 0.00604602, -0.000524805, -0.0068401, 0.000186434, -0.00324243, 2.02873e-06, 0.00143251, -0.000209642, 0.00463359, -2.44115e-05, 0.00429629, 0.000183201, 0.00196117, 0.000126603, -0.00856094, -0.000427516, -0.00391375, 4.27209e-06, 0.00183105, -0.000631198, -0.00718077, 0.000174062, -0.0013666, 0.000288094, 0.00190513, -0.000263891, 0.0179296, -0.000115155, -0.00185863, 0.000455337, 0.00891769, 3.25793e-05, -0.00470988, -0.000274411, -0.00441547,
-0.0041298, 3.0208e-05, -0.00395324, -4.89298e-05, 0.000379092, -6.72369e-06, 5.87706e-05, 1.17167e-05, 0.102461, -7.97603e-05, 6.1715e-05, -5.32793e-05, 0.00322182, 7.7902e-05, 0.111727, -0.000114565, 0.000301132, 6.61587e-05, -0.000280407, -3.13983e-05, 0.00248869, 1.30156e-05, 0.0357054, -3.68041e-05, 0.000454409, 5.29505e-05, -0.00017609, 0.000182157, 0.000201641, 2.61026e-05, -0.000861686, -0.000104956, -0.0369475, 9.29323e-05, -0.000451378, -3.21824e-05, -6.33188e-06, 0.000222651, -0.000845686, 4.26385e-06, 0.000992908, 3.0931e-05, -0.00304491, -7.74037e-05, -0.0363876, 5.75568e-05, -0.00195357, -0.000105212, 0.00132942, 0.000125495, -0.000939568, 0.000249172, -0.000540288, -0.000267598, 0.000619854, -5.92676e-06, -0.00212598, 8.4633e-05, 0.00846274, -0.00010157, -0.0021881, -0.000117865, 0.00370388, -0.000155946, 0.000349547, 0.000104088, -0.00199221, -3.42471e-05, 0.000845062, -0.000195526, -0.0010628, 9.49154e-06, 0.000218772, 0.000142503, 0.0278436, -0.000118882, -0.000409545, -2.71857e-05, 0.00531228, -0.000165747, -5.22397e-05, 0.000304723, -0.000474943, -4.80773e-05, -0.0014639, -0.000337895, 0.0017079, 0.000183274, -0.00217207, 8.54168e-05, 0.00132867, -3.0814e-06, 0.00628596, 6.32678e-05, 0.00173324, 4.76464e-05, 0.0033895, -1.75293e-05, 0.00102014, 0.000234653,
-0.00258173, -0.0054357, -0.00272022, -0.00787598, -0.000218604, -0.00188976, -0.000906246, -0.00389236, 0.000227347, -0.101469, 4.25091e-05, 0.00460784, 0.000388076, 0.00592842, 5.83455e-05, 0.00838783, 5.86857e-05, -0.111474, 0.000338076, 0.0028292, 0.000321094, 0.00715324, -0.000229098, 0.00645768, 0.000416468, 0.010634, -5.77298e-06, -0.0366812, 0.000107453, -0.00583767, -8.76699e-05, -0.00231672, -0.000190644, -0.00303924, 8.80114e-05, 0.00968963, 0.000493558, 0.0118502, -7.23645e-05, 0.0351916, -0.000360276, -0.00767507, -5.48371e-06, -0.00845905, 0.000105971, -0.00837612, -0.000478913, 0.00910371, 8.24908e-05, 0.00678706, 0.000210078, 0.00338751, -2.37274e-05, 0.032575, -0.000264034, 0.00258175, 0.00013928, -0.00190891, 0.000171562, -0.00283149, -0.000343342, -0.000739695, 0.000233214, -0.00854058, 0.000443044, 0.00612974, -0.000373533, -0.00508344, 5.02905e-05, -0.0133076, 0.000348956, 0.0118499, -0.000150079, 0.00770099, -8.81657e-05, 0.00554109, 0.000460696, -0.00851272, 0.000246305, -0.00890797, -0.000736577, -0.014373, 0.000426157, -0.00150504, -0.000344573, -0.00409382, -7.42775e-06, -0.0281752, 0.000478877, 0.00637884, -0.000392626, 0.00578466, -0.000192019, 0.00518239, 0.000725836, -0.00482639, 2.08676e-05, -0.000427097, -0.000334942, -0.00890572,
0.00262389, 0.000257997, -0.00134712, 0.000162646, 0.120496, 5.52582e-05, -0.00724104, 7.89632e-05, 6.85375e-05, 0.000420797, 0.150157, -0.000283594, -0.00458471, -6.80671e-05, 5.53019e-05, -0.000407594, -0.0031862, 0.000194747, 0.0837987, -0.000204637, 0.00497667, 1.2772e-05, 0.000295412, -0.000151326, -0.0110665, -0.000503616, -0.00421045, -0.000280241, -0.000180466, 0.000181856, 0.00901293, 0.000204139, 0.000859094, 0.000392575, -0.0084869, -0.000612078, -0.0112314, -0.000297581, -0.00187905, -0.000258574, -0.0239487, 0.000241631, 0.00209098, 9.74348e-05, 0.00100103, 0.000424155, 0.00515046, -0.000303859, -0.000684216, -0.000714792, -0.011418, 0.000262251, 0.00225233, 0.000261626, 0.00100895, -0.000130117, -0.00608345, -0.000306493, -0.000110716, -0.000153466, 0.0144114, 0.000514469, -0.0010219, 0.000599661, 0.00223766, -0.000572323, -0.00188126, 0.00036203, 0.00472667, 0.000446198, 0.0128453, -0.00028226, -0.00465321, -0.000453839, -0.00197884, -0.000484868, 0.00839096, 0.000724926, -0.00108658, 0.000226289, 0.0113646, 0.000265441, 0.00233115, 0.000140441, 0.00426932, -0.000144173, 0.0040475, -5.53714e-05, -0.00781054, 0.000127699, 0.00265705, -4.36895e-05, -0.00280683, -6.15504e-05, -0.00394974, -2.14897e-05, -0.0012842, -0.000729404, 0.0122263, 0.000829669,
0.00244345, -0.0264368, 0.0026009, 0.00550337, 7.91727e-06, -0.0762504, 0.000883167, 0.00427423, -1.4507e-05, 0.00458786, -0.000115377, -0.113029, -0.000395295, -0.00111026, -4.90654e-05, -0.00527906, 3.73722e-05, 0.00143108, -0.000175078, -0.0880431, -0.00031049, -0.00308347, -5.19904e-05, -0.00533899, -0.000142342, -0.00605501, 1.04998e-05, -0.00685744, -3.4983e-06, -0.0139669, 0.000254601, 0.00030813, 4.66944e-06, -0.00128741, -2.11062e-05, -0.00543254, -0.000165851, -0.00491379, -3.27625e-05, -0.0113137, 0.000199028, 0.0400778, 0.000326723, 0.00262018, 6.03996e-05, 0.00119802, 0.000119683, -0.00518544, -3.45014e-05, -0.00371807, 8.17981e-05, 0.00284279, -1.92839e-05, -0.00711939, 0.000107667, 0.0325917, 6.18304e-05, -0.000951748, 4.54291e-05, -0.00103934, -6.68776e-05, -8.04433e-05, -2.15476e-05, 0.0049647, -3.15737e-05, -0.00265427, 0.000389111, 0.00920286, 3.34811e-05, 7.4526e-05, -0.000163118, -0.00372277, 1.73715e-05, -0.00580208, -1.19398e-05, -0.00351853, -0.000426974, 0.0033714, -9.89113e-06, 0.00571857, 0.000297045, 0.00585693, -4.01449e-05, 0.0026915, 0.000240396, 0.00691505, 2.0658e-05, 0.00242205, -0.000137576, -0.0144463, 0.000109908, -0.00430364, -3.28957e-05, -0.000332008, -0.000350305, 0.000231846, -4.99666e-05, 0.00179008, 0.000100578, 0.00531607,
-0.00616124, -4.3787e-05, 0.0335494, -0.000159137, -0.00785184, 4.37169e-05, 0.0980998, -5.64034e-05, 0.00237951, -0.000785053, -0.00507538, 0.000299725, 0.1135, 9.94442e-05, 0.00339714, 0.000446514, 0.000121514, -0.000893226, 0.0041331, 0.000554292, 0.0562456, -9.02744e-06, 0.00250161, 0.000361087, 0.00548137, -4.96398e-06, 0.000644961, -0.000452095, 0.00742877, 0.000546108, -0.0180525, -0.000249282, 0.000674866, -4.85146e-06, 0.00237739, 0.000697063, 0.00844677, -0.000486297, 0.000825187, -0.000102257, 0.000456023, 0.000247632, -0.0439704, -0.000163968, -0.000271769, -7.80819e-05, -0.00569232, 0.000552135, -0.00204079, 0.000475022, 0.00887039, -0.000952484, -0.000154046, -0.000174105, -0.00621058, -0.000104419, -0.0198914, 0.000309938, 0.00032017, 0.000236325, -0.00773367, -0.000127335, -0.00272393, -0.000336679, 0.000249809, 8.10446e-05, 0.00104532, -0.000702191, -0.00183597, -0.000264773, -0.0023434, -0.000366417, 0.00472969, 0.000674704, 0.00138027, 0.00040148, 0.000763067, -0.000412285, -0.00137237, -9.4036e-05, -0.00915726, -0.000361908, 0.00100664, -0.000728959, -0.0063528, 0.000103237, -0.0026446, -1.72767e-05, 0.00659367, -0.00062839, -0.000364963, 0.000604688, 0.00149005, 0.000149573, 0.00930472, 1.98892e-05, 0.000695265, 0.00040445, -0.0113026, -0.000442615,
0.000202295, 0.00550307, 0.000178498, -0.0264891, 7.42924e-05, 0.00427261, 8.20381e-06, -0.0773274, 1.67834e-05, 0.00591742, 7.16245e-05, -0.00111186, -7.20375e-05, -0.115707, 3.1615e-05, -0.00396991, 0.000107381, 0.00488858, 3.64387e-05, -0.00308197, -7.0514e-05, -0.0918505, 1.55698e-05, -0.0010286, -4.01676e-05, -0.00447898, 0.000204115, -0.000973224, 5.45837e-05, 0.000311654, -0.000143691, -0.0178047, -3.29003e-05, 0.00629341, 0.00017936, -0.00440399, -8.42792e-05, -0.00506414, 0.000190117, -0.00367911, 8.79625e-05, 0.00261777, -0.000267801, 0.0374804, -5.93884e-05, 0.00933865, 0.000346723, -0.00236722, -1.1892e-05, -0.00467231, 0.000161084, -0.00205663, 5.93259e-05, 0.000775761, -7.12248e-06, -0.000967693, -0.000255675, 0.0325739, -2.31526e-05, 0.00340083, 0.000130297, 0.00509194, 1.62121e-05, 0.00510484, 0.000124413, -0.00497103, 0.00038019, 7.45712e-05, -2.75746e-05, 0.00614449, -0.000191337, -0.00583051, -0.000106315, -0.000539923, 2.20343e-05, -0.00536363, -0.000230335, 0.0108931, 9.41673e-05, 0.00484083, 0.000192337, 0.00553248, 0.000191041, -0.000840044, 9.09807e-05, -0.00227729, 5.32174e-05, 0.0044893, -0.000167385, -0.00433661, -6.31091e-05, -0.00945391, -1.72991e-05, -0.00767328, -0.000151874, 0.00953467, 0.000155858, -0.000970559, -2.7685e-05, 0.00324996,
-0.00494599, 2.03214e-05, -0.00442826, -6.21014e-05, 5.70349e-05, -1.55849e-05, 0.000728219, -6.52676e-06, 0.111905, -0.000142269, 4.54444e-05, -6.14553e-05, 0.00444832, 7.39407e-05, 0.14369, -0.000163939, -0.000128045, -4.2202e-05, 0.000305375, -4.24522e-05, 0.00321824, 3.33195e-05, 0.0837758, -0.000101081, -0.000336487, -3.83682e-05, -0.000595356, 0.000108901, 0.000893336, 2.90316e-05, -0.00109337, -8.29374e-05, -0.00211649, 7.46125e-05, -0.0015754, -7.43275e-05, -0.000638739, 7.89908e-05, -0.000770997, 2.8142e-05, 0.00103906, 6.99695e-05, -0.00408045, -8.38941e-05, -0.0352905, 0.000110801, -0.00288177, -0.000148941, 0.00252897, 2.08576e-05, -0.00174277, 0.000165278, 5.62322e-05, -0.000210074, -9.69285e-05, 5.55135e-05, -0.00354726, 6.59985e-05, -0.0103462, -3.46047e-05, -0.00266807, -0.000147793, 0.00556771, -0.000172775, -0.000256471, -1.30864e-05, -0.00225736, 1.11733e-05, 0.00143314, -0.000224352, -0.00199759, 5.0645e-05, -0.000902734, 0.000169956, 0.0188634, -0.000120519, -0.00050948, -5.86754e-05, 0.00659373, -0.000203668, 0.000623706, 0.000151507, -0.00120732, -0.00011734, -0.000962307, -0.00026085, 0.00180289, 7.54572e-05, -0.0028436, 9.17075e-05, 0.00129149, 8.78629e-05, 0.0154828, -3.39785e-06, 0.00185687, 9.41759e-06, 0.00292595, -7.64419e-05, 0.00169166, 0.000225849,
0.000856178, 0.0071465, 0.00075981, 0.00822407, 0.000178788, 0.00380334, 7.78248e-05, 0.00514264, 8.50908e-05, 0.00838403, -3.78622e-05, -0.00529231, -0.000121852, -0.00397013, -3.24335e-05, -0.104912, 0.000119597, 0.00645776, -0.000270447, -0.00737371, 0.000160529, -0.00574139, -0.000150879, -0.114771, -0.000331601, -0.00864285, -7.91515e-05, -0.00303279, -2.49112e-05, 0.000766734, 0.000154847, 0.00207002, -2.92168e-05, -0.0361489, -1.60671e-05, -0.0085506, -0.000678809, -0.00724958, -0.000249199, -0.00836812, 0.000429995, 0.00778342, -0.000241104, 0.00671983, 0.000195347, 0.0393189, 0.000413888, -0.00765729, 7.42272e-06, -0.00972124, -0.000356141, 0.00277366, -1.8816e-05, -0.00282733, 0.000306774, 0.00407135, -0.000291282, -0.000196313, 0.000164651, 0.0358874, 0.000202066, 0.000828456, 5.61524e-05, 0.0102416, -0.000304022, -0.00911778, 0.000553344, 0.00964206, 0.000366622, 0.0055445, -0.000395426, -0.00446395, 0.000220625, -0.00908407, -0.000117966, -0.01416, -0.000433766, 0.00636945, 7.11097e-05, 0.00873194, 0.000503199, 0.00840158, -0.000364236, 0.00141906, 0.000794539, 0.00425229, 0.000307229, 0.00519136, -0.000641379, -0.00535324, 0.000475593, -0.00577843, -0.000238066, -0.0312234, -0.000468851, 0.00191937, 1.64869e-05, -0.00343866, 0.000266588, 0.0112113,
0.00053527, 1.17883e-05, 0.000135634, -0.000232442, -0.00292009, -8.28807e-06, -0.000448223, -6.27947e-05, 0.000305001, -0.000119235, -0.00318922, -1.72655e-05, 5.14238e-05, 0.00023812, -0.000140481, -0.000285835, 0.087939, -5.22822e-06, -0.000379898, 1.46269e-05, 0.000836885, 0.00018763, -0.000675444, -7.15694e-05, -0.000539969, -4.68979e-05, 0.101296, 0.000135761, 0.0027937, 3.2917e-05, 0.000204471, -0.000149989, -0.000253191, 0.000280428, -0.000872862, 5.57706e-05, 0.000946435, -4.99387e-06, 0.0397452, 5.29494e-05, 0.00283877, -1.09571e-05, -0.00140266, -0.000246741, 0.000886942, 0.000215109, -0.000824961, -0.000108726, -0.00017383, -0.000186154, -0.000611181, 3.05207e-05, -0.0271504, -0.000139826, -0.000111434, -5.42636e-05, -0.00154036, 3.81802e-05, 0.00114889, -0.000175386, -0.000295314, -0.000224915, -0.000560721, -5.71753e-05, 0.00423752, -0.000174174, -0.00313694, -1.47521e-05, -0.0330183, -0.000117554, -0.00225873, -1.35844e-05, 0.000575495, 0.000228856, -0.000202794, -0.000252569, 0.000459212, -3.04433e-05, -0.000668178, -0.000195058, -0.000422546, -0.000309821, 0.00363155, 2.04098e-05, -0.00342461, -4.55273e-05, 0.00525143, 0.000102959, -0.00106532, 6.0565e-05, 0.00242368, 1.41775e-05, -0.00160081, 0.000117942, 0.000745984, 0.000282219, -0.000312221, -0.000182446, 0.000424853, -7.17786e-05,
-0.00443558, -0.00254397, -0.00464957, -0.00523608, -0.00037983, -0.000672587, -0.00147497, -0.00206247, 0.000292417, -0.111688, -9.73533e-05, 0.00146259, 0.000789526, 0.00490712, 0.000123447, 0.00645844, 9.87937e-05, -0.145711, 0.000392208, -0.00185668, 0.000604247, 0.00526127, -0.000261586, 0.00472637, 0.000593347, 0.00985557, 4.9185e-05, -0.0883144, 0.000317662, -0.00721195, -0.000265722, -0.00200408, -0.000337664, -0.00312467, 0.000262128, 0.00758819, 0.000655886, 0.0124253, -6.1772e-05, -0.00344084, -0.000231091, -0.00481785, -0.000190478, -0.00627744, -2.68646e-05, -0.00780335, -0.000491741, 0.00727864, 3.29191e-05, 0.00445201, 0.000480085, 0.0065224, -8.81678e-05, 0.029063, -0.000367043, 0.00625917, 0.000249097, -0.00106411, 0.000196715, -0.00365118, -0.000541619, -0.000430935, 0.000158963, -0.0065064, 0.000521836, 0.00351506, -0.000198634, -0.00115416, -3.46168e-05, 0.00528835, 0.000155045, 0.0133626, -9.23323e-07, 0.00583912, 6.59769e-05, 0.00341133, 0.000300382, -0.00690527, 0.000232124, -0.00662704, -0.000828095, -0.00970204, 0.000559657, -0.00163725, -0.00044055, -0.00268799, -3.03612e-05, -0.0182924, 0.000455101, 0.00595758, -0.000507958, 0.00362389, -8.32754e-05, 0.00419157, 0.00081867, -0.00456254, 0.000110729, 0.000402828, -0.000525121, -0.00724342,
-0.00894274, 7.48726e-05, -0.00726366, 0.0001177, 0.029272, 8.4468e-05, 0.00153282, 0.000130277, -0.00027469, 8.18582e-05, 0.0840217, -2.9396e-05, 0.00495557, 5.51423e-05, 0.000316969, -6.63692e-05, -0.000381097, 0.000204076, 0.124614, -0.000189227, 0.000851381, -1.2026e-05, 0.00122607, -8.405e-05, 0.0086738, -4.516e-05, -0.00034896, 0.000190373, 0.101343, -0.000185906, -0.00198241, -1.42399e-05, 0.00123708, 2.19541e-05, 0.00877155, 3.2798e-05, 0.00809908, -0.000212164, 0.000164956, -2.41133e-05, 0.0255507, 4.58102e-05, 0.00168815, 5.46165e-06, -4.88295e-05, 0.000215143, 0.00200473, -4.24955e-05, 0.000256265, 2.29696e-05, 0.00617782, -0.000246822, 0.00127727, -0.000179686, -0.0339493, 0.000252194, 0.00552702, -4.44377e-05, -0.0016411, 0.000271196, -0.00182742, -4.59402e-05, -0.00038349, -3.94267e-05, -0.0015674, -0.00020954, -0.00531134, 2.01945e-05, 0.00313556, -5.13802e-06, -0.0325825, 0.000158463, 0.00169132, -0.000187337, -0.00235129, 8.16224e-06, 0.00347562, 0.000195702, -0.00154902, 8.00114e-05, -0.00834914, -5.95949e-05, -0.00228808, -0.000418236, -0.0156354, 0.000305454, 0.00523596, 0.00031456, 0.000819572, -9.93688e-05, -0.00659579, -0.000280563, -0.00199657, -0.000374248, 0.0106079, 0.000475283, -0.00239118, 0.000112601, -0.00971209, -0.000220693,
0.00396548, 0.047939, 0.00414601, 0.0067093, 0.000308956, 0.00815058, 0.00128771, 0.00435606, 4.66713e-06, 0.00282156, 4.20159e-05, -0.0882631, -0.000726536, -0.00309518, -3.47477e-05, -0.00738146, 7.90623e-05, -0.00187771, -0.000379451, -0.135107, -0.00047195, -0.00506536, -3.99343e-05, -0.00661447, -0.000621964, -0.00718943, 2.58053e-05, -0.0093969, -0.000269699, -0.0806172, 0.000438994, 0.00068341, 4.89731e-05, 0.00044686, -0.000337967, -0.00728982, -0.000699988, -0.00491564, -7.35792e-05, -0.00974786, 0.000263222, 0.0106057, 0.000423121, 0.00461819, 0.000145592, 0.00482143, 0.000354572, -0.00678885, -5.15185e-05, -0.00548197, -0.000389317, 0.00482407, -6.25717e-05, -0.00101748, 0.000399417, 0.0468328, -7.41406e-05, -7.35115e-05, 0.000111895, 0.00128179, 0.000361342, 0.000416282, -2.19412e-05, 0.00679783, -6.78188e-05, -0.00560927, 0.00046688, 0.0104675, 5.24043e-05, 0.0060911, -7.45395e-05, 0.0186303, -2.44273e-05, -0.0067739, -3.07675e-05, -0.0037869, -0.000446088, 0.00539071, 2.6523e-05, 0.00728376, 0.000812081, 0.00835233, -0.000104068, -0.000321664, 0.000678189, 0.00366168, 7.18799e-05, 0.00230789, -0.000333702, -0.0118509, 0.000330225, -0.00489573, -0.000114642, -0.00139472, -0.000802094, 0.00124431, -6.3747e-06, 0.00110069, 0.000437229, 0.00735996,
-0.0664567, -7.29674e-05, -0.071596, -8.92228e-05, 0.00509577, -1.09388e-05, -0.00818559, -0.000140683, 0.00237281, 0.000300026, 0.00477689, 2.23571e-05, 0.070867, -0.000126411, 0.00315013, 6.44848e-05, 0.000813481, 0.000454562, 0.000670498, -0.000135328, 0.103739, -8.4573e-05, 0.00182255, 4.9222e-05, -0.00583966, 0.000436109, 0.000426719, 0.000408753, -0.00126731, -0.000317751, 0.0668264, -0.000114481, -0.000625497, -9.10971e-05, -0.00476676, 1.33519e-05, -0.00358945, 0.000647676, -0.00111799, 0.000227208, 0.00150564, -0.000236147, -0.00221919, -0.000256685, -0.0023972, -0.000258359, 0.00321, 0.000102058, -3.69734e-06, 0.000109006, -0.00359991, 0.000449499, -0.0026664, 1.36777e-05, 0.00412231, 8.66597e-05, -0.0422345, -0.000407874, -0.00234464, -0.000258849, 0.0110737, 0.000113011, 0.000180722, -3.13328e-05, 0.00072673, 0.000253564, -0.00047596, -1.03066e-05, -0.00298312, -0.000185247, 0.00138281, 0.000360534, -0.0313097, -0.000420495, -0.000550489, -5.85133e-05, 0.0113685, -6.2445e-05, 0.000421287, -0.000151921, 0.00318343, 6.06405e-06, 0.00143796, 0.000233137, 0.00146748, -0.000269638, -0.0019571, -0.000317779, -0.00429465, 0.000429876, -0.00220765, -0.000286326, 0.00181187, 0.000111931, 0.00461857, -0.000257327, 0.000513496, -0.000159366, 0.00168919, -5.91254e-06,
0.00031105, 0.00670963, 0.000202818, 0.0439426, 0.000322286, 0.00435411, -0.000131079, 0.00271333, -8.39555e-05, 0.00716618, 0.000201958, -0.00309861, -0.000187423, -0.0920744, -3.61624e-05, -0.00573203, 5.77294e-07, 0.00525624, -0.000124919, -0.00506546, -1.53939e-05, -0.136331, 6.18318e-05, -0.0027541, -0.00057922, -0.00722849, 0.000140406, -0.00244024, -0.000148185, 0.00068683, -4.69631e-05, -0.0809639, 5.83997e-05, 0.00532541, -0.000254649, -0.00352654, -0.00067899, -0.00804534, 0.000251496, -0.00522962, 0.000157368, 0.00461461, -0.000329816, 0.00932485, -4.42146e-05, 0.00770727, 0.000469503, -0.000122437, -0.000152184, -0.00744847, -0.00025087, -0.0023312, 0.000149633, 0.00141781, 0.000232617, -9.68098e-05, -0.000432384, 0.0452251, -8.4406e-05, -4.01518e-05, 0.000544105, 0.00743098, -0.000163284, 0.00882075, 4.98135e-05, -0.00878619, 0.000694225, 0.00310873, -5.7314e-05, 0.0082635, -0.000145, -0.00681775, -0.000187519, 0.0189521, -2.90509e-06, -0.00840279, -0.000143708, 0.0102971, -1.3433e-05, 0.0103044, 0.000606762, 0.00686016, 0.000127766, -0.00288493, 0.000745589, 0.00159804, -7.33923e-05, 0.00493444, -0.000378287, -0.0049477, -2.12435e-05, -0.00877742, 4.78225e-05, -0.00635116, -0.0004635, 0.00432599, 0.000138011, 0.00301186, 0.000342608, 0.00566909,
-0.00069853, -3.30832e-05, 5.88021e-05, -1.29265e-05, -0.000294394, -3.76425e-05, 0.00184506, -2.45798e-05, 0.0358939, -9.19692e-05, 0.000290177, -2.3359e-05, 0.00271681, -1.29702e-05, 0.0839888, -5.63895e-05, -0.000660483, -0.000186996, 0.00121685, -2.59766e-06, 0.00170218, 2.13462e-05, 0.110372, -8.28675e-05, -0.0014665, -0.000122608, -0.000590089, -0.000176612, 0.00126242, 3.82015e-05, -0.000561945, 3.54349e-05, 0.0820242, -3.06095e-05, -0.00231711, -7.52425e-05, -0.00102569, -0.000219496, 0.000273565, -4.62938e-05, -1.67967e-06, 0.000100249, -0.00275351, 1.07339e-05, 0.0179284, 7.36026e-05, -0.00222223, -9.02846e-05, 0.00213856, -0.000144655, -0.00127336, -0.0001784, 0.00117129, 4.19719e-05, -0.00160617, 0.000137253, -0.00368916, 9.21071e-06, -0.0250481, 0.000113222, -0.00140673, -7.42563e-05, 0.003292, -3.02488e-05, -0.000911075, -0.000199162, -0.000408589, -2.25417e-05, 0.00119686, -2.96829e-05, -0.00235074, 0.000109921, -0.00265511, 9.40417e-05, -0.0183147, 2.65365e-05, -0.000469169, -7.36362e-05, 0.00212473, -6.85098e-05, 0.000938012, -0.000190301, -0.00128387, -0.000189105, 0.000834493, 3.79291e-05, 0.000435989, -0.000127328, -0.0020228, 5.22935e-05, -5.35536e-06, 0.000204639, 0.00854141, -8.29315e-05, 0.000139269, -8.41929e-05, -0.00125102, -0.000104826, 0.00111806, -3.05647e-05,
0.00160563, 0.00511553, 0.00150931, 0.00689386, 0.000377079, 0.00199451, 0.000232657, 0.00533839, 0.000112472, 0.00645114, 0.000145585, -0.00535962, -0.000354129, -0.00102575, -2.16262e-06, -0.114997, 0.00021085, 0.00472221, -0.000276237, -0.00662123, 5.73551e-05, -0.00275726, -0.000153887, -0.148511, -0.000417145, -0.00604811, 5.79811e-05, -0.00311871, -0.000166837, 0.000679117, 0.000325027, 0.00148413, -6.5488e-05, -0.0868913, -0.000146666, -0.00637613, -0.000731028, -0.00427587, -0.000175527, -0.00779101, 0.00036889, 0.00720911, -8.34674e-05, 0.00277711, 0.000183616, 0.00176976, 0.000361503, -0.00624591, 6.5058e-05, -0.00666169, -0.000438221, 0.00369964, -5.83969e-05, -0.00364083, 0.000427836, 0.0049086, -0.000371529, -0.00354041, 0.000222741, 0.033487, 0.000300621, -0.000586548, 0.000125891, 0.00635745, -0.000172029, -0.00574056, 0.000543751, 0.00857953, 0.000323279, 0.00341935, -0.000250549, -0.00194197, 4.25144e-05, -0.00892252, -4.13934e-05, 0.00558335, -0.000307239, 0.0033693, 0.000111602, 0.00420508, 0.00058746, 0.00637207, -0.000240884, 0.00237345, 0.000969798, 0.00394511, 0.000396865, 0.00420459, -0.000708597, -0.00360768, 0.000470624, -0.00326521, -0.000249328, -0.020233, -0.000513363, 0.000315495, 2.65262e-05, -0.00491579, 0.000465911, 0.00646306,
-0.00875756, -0.000106587, -0.00375316, -0.000156846, -0.0167877, 7.73606e-05, 0.00737272, 0.000129953, 0.000439756, -0.000190169, -0.0110634, 0.000202161, 0.00588841, 0.000359352, -0.000360069, 0.000231783, -0.000536189, 7.9604e-05, 0.00867555, -8.88345e-05, -0.00599629, -1.6544e-05, -0.00146923, -1.45679e-05, 0.107315, 0.000402747, -0.00087287, 0.000354227, 0.0162262, -0.000407305, -0.00876541, -0.000512061, -0.00117571, -0.000302184, 0.119416, 0.000350191, 0.0188834, 0.000149388, -0.000830315, 4.8418e-05, 0.00133091, -0.000111772, 0.00377857, -0.000240268, 0.000616819, -9.2555e-05, 0.0395632, 3.85441e-05, -0.00259481, 0.000850059, 0.0179087, -0.000409847, -0.000299483, -0.00053627, -0.014556, 0.000520213, 0.0133415, 0.00052438, 0.00188618, 0.000342656, -0.040274, -0.000428401, -0.00312098, -0.000307898, 0.000948206, 0.000597055, -0.00195478, -0.000434042, 0.000461714, -0.000445776, -0.0088953, 0.000484551, 0.0038488, 0.000555548, 0.00101662, 0.000232277, -0.0383838, -0.000267058, -0.00105172, -9.18087e-05, -0.0221669, -0.000501872, 0.000895783, -0.000411835, -0.0184168, 0.000223444, 0.000752368, 0.000390921, 0.00921932, -0.000350852, -0.0132136, -0.000355234, -0.0010487, -0.000376384, 0.0149945, 0.000371342, 0.00162219, 0.000279136, -0.0205896, -0.000686658,
-0.00170764, 0.00866923, -0.00202392, 0.00732921, 0.000229251, 0.00446702, -0.000927019, 0.00444127, 0.000363259, 0.0106244, -8.59978e-05, -0.0060433, 0.00033158, -0.00447741, 0.000163368, -0.0086437, 0.000109401, 0.00986006, -0.000387577, -0.0071698, 0.000665188, -0.00722989, -0.000264449, -0.00603206, -6.71353e-05, -0.0943788, -4.60868e-06, -0.000237033, -4.6245e-05, 0.00322003, 2.49832e-05, -9.4715e-05, -0.000287003, 0.00421708, 0.000137497, -0.00942749, -0.000245208, -0.108774, -0.000143279, -0.00740945, 0.000504721, 0.00950326, -0.00060082, 0.00678988, 0.000104455, 0.00880549, 0.00030459, -0.0081445, -6.97553e-05, -0.0127957, -6.91833e-06, -0.0411937, -7.16223e-05, -0.00346675, 0.000300617, 0.00129626, -0.000345198, 0.0040409, 0.000272112, 0.00115294, 4.57019e-05, 0.0021433, 0.00010217, 0.00944299, 0.000115664, -0.0100096, 0.000330419, 0.0326042, 0.000100612, 0.00445225, -0.000494386, -0.0102793, 0.000429411, -0.00374915, -2.98902e-05, -0.00809123, -0.000412575, 0.00930681, 0.000220723, 0.0094435, 0.000760793, 0.00946508, 7.06642e-05, 0.00661212, 0.000188342, 0.0375626, 7.15351e-05, 0.00458731, -0.000614925, -0.00802416, 0.000573635, -0.00524859, -0.000255654, -0.0055376, -0.000355422, 0.00430846, 4.08087e-05, 5.81198e-05, 0.000605455, 0.00860293,
0.000364975, 4.48321e-05, 0.000334948, -0.000285327, -0.00399294, 1.95468e-05, 0.000281725, -8.84218e-05, -0.000157001, -0.000125829, -0.00421048, -2.80333e-05, 0.000563901, 0.000301653, -0.000596539, -0.00038973, 0.101559, -2.80924e-05, -0.000353358, -2.11708e-05, 0.000401562, 0.000298762, -0.000613164, -0.00017144, -0.000881354, -0.000100472, 0.137236, 0.000120944, 0.00387438, 2.03e-05, -0.000909454, -0.0001076, 0.000355008, 0.000317973, -0.0015661, 0.000112863, -0.000507385, -7.4253e-05, 0.0882744, 7.20359e-05, 0.00421127, 1.66685e-05, -0.00220204, -0.000328218, 0.00146023, 0.000394127, -0.001479, -2.4503e-05, -0.000470687, -0.000312742, -0.00263703, 6.37155e-06, 0.00654842, -0.000115657, 0.000887883, -2.92941e-05, -0.00138967, -8.31756e-05, 0.00110284, -2.76023e-05, -0.000459956, -0.000189937, -0.00087702, -3.92584e-05, 0.00276249, -0.000344899, -0.00492715, 3.11055e-05, -0.0310206, -0.000142143, -0.00191381, -3.4277e-05, 0.00145457, 0.000197618, -0.000768454, -0.000293493, 0.00067906, -6.68981e-05, -0.00100892, -0.000143415, 0.000321048, -0.000364171, 0.00112759, -7.68285e-05, -0.00477319, 2.66668e-05, -0.0110282, 3.86944e-05, -0.00146955, 1.71346e-05, 0.0033886, 9.22473e-05, -0.00197201, -5.54224e-05, 0.000841219, 0.000273254, -0.00092136, -0.000168587, 0.00103297, 4.03495e-06,
-0.00450571, 0.0037522, -0.00475064, 0.00422137, -0.000188843, 0.000490702, -0.00151322, 0.00323509, 8.32382e-05, -0.0369134, -0.000163242, -0.00683772, 0.000880862, -0.00096715, 7.5298e-05, -0.00303406, 4.20233e-05, -0.0885693, 8.55141e-05, -0.00937681, 0.000721877, -0.00241828, -8.29167e-05, -0.00312041, 0.000201647, -0.000240411, 6.93677e-05, -0.118822, 0.000310076, -0.00335944, -0.000323249, 0.000634914, -0.000278207, -0.00118316, 0.000175919, -0.00300071, 0.000207923, 0.00268359, 1.95902e-05, -0.0917177, 0.000232604, 0.00521509, -0.000408717, 0.00323794, -0.000270423, -0.000538586, -7.62903e-05, -0.0026884, -7.30086e-05, -0.00351439, 0.000392146, 0.0069261, -9.80827e-05, -0.0248655, -8.37221e-05, 0.00844569, 0.000179134, 0.00109099, 1.16758e-06, -0.00248953, -0.000307918, 0.000231643, -0.000100553, 0.00305568, 0.000129822, -0.00367829, 0.000321748, 0.00776109, -0.00016716, 0.0229551, -0.000261902, 0.00527727, 0.000247422, -0.00316362, 0.000277068, -0.00387066, -0.000223263, 0.00190233, -7.00313e-06, 0.00375844, -6.4927e-05, 0.00743114, 0.000224465, 0.000795685, -4.45185e-05, 0.00309113, -0.000105057, 0.0193661, -8.72277e-05, 0.000229601, -0.000283488, -0.00381334, 0.000278032, -0.00144496, 0.000166422, -0.000337532, 0.000134538, 0.00221865, -0.00017552, 0.0025549,
-0.012058, -0.000265446, -0.00811189, -0.000127535, -0.045172, -4.95315e-05, 0.00533375, 9.38671e-06, 0.000202634, -0.000378219, -8.1214e-05, 0.00025697, 0.00820409, 0.000201876, 0.000901005, 0.000465619, 0.00278381, -4.4187e-05, 0.101595, 7.62234e-05, -0.00116155, 0.000110469, 0.00127304, 0.0002366, 0.016247, 0.000530187, 0.00387351, 0.000491047, 0.150935, -0.000356738, -0.00619557, -0.000165697, 0.000361343, -0.000268503, 0.0153402, 0.0007319, 0.0141219, 0.000217263, 0.00264736, 0.000384763, 0.0912296, -0.000283363, 0.00163253, -0.000187104, -0.00140171, -0.000230308, 0.000565155, 0.0004528, 0.000425023, 0.000751818, 0.0115552, -0.000447302, 0.000974004, -0.00026082, -0.00865551, 0.000304636, 0.0096419, 9.06396e-05, -0.00242586, 0.000343856, -0.00862848, -0.000342257, -1.63194e-05, -0.00072571, -0.00324592, 0.000437621, -0.0064423, -0.000470294, 0.00181718, -0.000414009, -0.04697, 0.000529127, 0.00343404, 0.000156531, -0.0019582, 0.000468659, 0.000671896, -0.000420665, -0.00119337, -0.000376196, -0.0155996, -0.000357075, -0.0042776, -0.000506814, -0.0211422, 0.000240227, 0.00526389, 0.000262636, -0.0137862, 1.30861e-05, -0.0104876, -0.000213589, -0.00102438, -0.000267489, 0.0152407, 0.00047033, -0.00221437, 0.000550222, -0.0179076, -0.00104044,
0.00414771, 0.033696, 0.00434346, -0.0014873, 0.00022074, 0.0333636, 0.00134594, -0.00114442, -7.11172e-06, -0.00584067, 0.000140659, -0.0141512, -0.000804717, 0.000301265, -4.35645e-07, 0.000745954, 1.56043e-05, -0.00721228, -0.0001481, -0.0808947, -0.000569739, 0.000665619, 3.75791e-05, 0.000660299, -0.000374632, 0.00320441, 5.90183e-06, -0.00336001, -0.000286384, -0.110426, 0.000441977, -0.000727293, 0.000105308, 0.000143866, -0.000328458, 0.00157233, -0.000353677, 0.00422353, -1.84094e-05, 0.00198554, -7.27844e-05, -0.0733556, 0.000526657, -0.00198642, 0.000157566, 0.000392239, 3.7049e-05, 0.00134241, 1.42115e-05, 0.00092485, -0.000432561, 0.00197819, -2.47351e-05, 0.00384908, 0.000248322, -0.00285636, -1.7774e-05, -0.00126044, 0.000132675, 0.0018779, 0.000258028, -0.000472041, 1.00599e-05, -0.00133421, -2.65635e-05, -0.000284669, -0.000113115, -0.00218508, -7.81493e-06, 0.00118978, 0.000307046, 0.0388569, -0.000121582, 0.000707632, 2.50979e-05, 0.0033386, 4.81118e-05, -0.00196265, -4.4025e-06, -0.00161068, 0.000269919, -0.00133815, -3.83923e-05, -0.00274568, 0.000326974, -0.00527446, -1.50237e-05, -0.00311866, 0.000109718, 0.0281611, 0.000237378, 0.00166961, -8.92801e-05, 0.00320579, -0.000285091, -0.00168129, -2.79642e-05, -0.000468827, 0.000286334, -0.00131919,
-0.00290653, 0.000149202, -0.0337418, 8.55808e-05, 0.0113739, 0.000122151, -0.0592965, -9.42193e-05, -0.000563039, 0.000498622, 0.009174, -2.99631e-05, -0.0181581, -0.000316958, -0.000733238, -0.000306612, 0.000182372, 0.000475844, -0.0017161, -0.000103491, 0.071974, -0.000255635, -0.000497895, -0.000120896, -0.00862984, -0.000245646, -0.000939293, 0.00010414, -0.00592537, -2.08057e-05, 0.119731, -3.76666e-05, -0.000447653, 0.000221737, -0.00370567, -0.000525454, -0.0106124, 0.000118281, -0.00256179, -7.73155e-05, 0.00216844, 5.84199e-05, 0.073841, -8.85072e-05, -0.00115535, 0.000151368, 0.0107644, -0.000262705, 0.00228346, -0.000506645, -0.0113344, 0.000618968, -0.00295818, 8.57304e-05, 0.0093456, 2.54473e-05, -0.0120473, -0.000418576, -0.00207495, -0.00027407, 0.0184816, 0.000374727, 0.00309696, 0.000384465, 6.90596e-05, -0.000149615, -0.00217323, 0.000439697, -0.00160843, 0.000100505, 0.00276632, 4.86316e-05, -0.0504052, -0.00052708, -0.00176335, -0.000366857, 0.00956863, 0.000448587, 0.00165672, 0.000145521, 0.00998517, 0.00063836, -0.000181746, 0.000605401, 0.00585991, -0.000358625, -5.7106e-06, -0.000328405, -0.0103267, 0.000279825, -0.0251229, -0.000172845, 0.000309608, 0.000104983, -0.00519318, -0.000216618, -0.000619251, -0.000357257, 0.0106418, 0.000497616,
0.000606129, -0.00148623, 0.000564888, 0.0288134, 0.000219753, -0.0011454, 2.85659e-05, 0.0271096, -0.00010112, -0.00229718, 0.000161157, 0.000299142, -0.000299841, -0.0179776, -8.43767e-05, 0.00207674, -0.000118434, -0.00198869, -6.54783e-05, 0.000667454, -0.000198479, -0.0812429, 3.00431e-05, 0.00149279, -0.000410632, -9.85108e-05, -9.84407e-05, 0.000623366, -0.000159707, -0.000721506, -3.48909e-05, -0.109145, 0.000114291, -0.00168131, -0.00042126, 0.00429986, -0.000448813, -0.000723281, 5.46387e-05, 0.00278624, -1.69438e-05, -0.00198829, -0.000139354, -0.072759, 7.56864e-05, -0.00458867, -1.47261e-05, 0.00565846, -0.000109843, 0.000298889, -0.000452962, -0.00123281, 0.000181929, 0.0022271, 0.000142777, -0.00128429, -0.000373628, -0.0037994, -2.38492e-05, -0.00420256, 0.000390945, 0.00304801, -0.000153646, 2.10143e-05, -0.000116534, -0.000590197, 7.80901e-06, -0.000576257, 0.000132986, -3.53505e-06, 0.00010875, 0.000659773, -0.000428925, 0.0371999, -5.68364e-05, -0.00081338, 0.000357649, -0.00123217, -0.000122167, 0.00148021, 0.000203781, -0.00215799, -0.00012022, -0.00187461, 0.000459224, 0.00117864, -2.21583e-05, -0.00110995, -3.85864e-05, 0.0016109, -0.000297722, 0.0275066, -8.04147e-06, 0.00250105, 1.60909e-05, -0.00326468, -5.11708e-05, 0.00338354, 0.000277631, -3.42917e-05,
0.0062722, -8.00426e-05, 0.00701917, 5.73169e-05, 0.000183182, -6.3858e-05, 0.00294984, -8.8257e-06, -0.0370656, 4.93361e-05, 0.000863821, 1.66145e-05, -0.000562246, -9.13868e-05, -0.00203276, 0.000144904, -0.000264425, -0.000166154, 0.00123813, 7.20677e-05, -0.00093704, -3.42435e-05, 0.0822568, 6.87795e-05, -0.00121235, -4.27201e-05, 0.000359764, -0.000372197, 0.000357317, 8.60352e-05, -2.24109e-05, 0.000105917, 0.130713, -8.03805e-05, -0.00127654, 8.58154e-06, -0.000117634, -0.000261004, 0.0013051, -0.000228571, -0.00136777, 0.000113965, -0.000607834, 0.000134425, 0.0914569, -5.0366e-05, -0.000405709, 4.57212e-05, -0.000550403, -0.000109171, 0.000773302, -0.000399509, 0.00132146, 0.000125997, -0.00237922, 0.000157784, -0.00253388, 4.5106e-05, 0.0092177, 0.000136221, 7.63671e-05, 3.5083e-05, -0.00255088, 0.000140085, -0.000408014, -0.000182804, 0.00190134, -0.000214253, 0.000172087, 0.000204311, -0.00193012, 0.000143789, -0.00323146, 5.21692e-05, -0.0351088, 0.000181213, -0.000775036, -5.63076e-05, -0.00490028, 0.000137303, -3.49756e-05, -0.000215463, -5.64652e-05, -0.000191042, 0.00169733, 0.000102709, -0.00060716, -8.64411e-05, -0.00102008, 5.32859e-05, -0.00136641, 0.000237227, -0.0202383, -1.92476e-05, -0.00211127, -0.000132419, -0.0050391, -1.68483e-05, -0.000578776, -0.000266351,
0.00181761, -0.00369083, 0.00187491, -0.000884582, 0.000262427, -0.0038367, 0.000502885, 0.00218358, 2.3578e-05, -0.00304293, 0.000272752, -0.00129868, -0.000465138, 0.00629456, 2.70399e-05, -0.036398, 0.000124784, -0.00313049, 5.56416e-05, 0.000427394, -0.000285279, 0.00532908, -3.19588e-06, -0.087163, -8.34497e-05, 0.00419478, 0.000214578, -0.00118495, -0.000128873, 0.000141418, 0.000282418, -0.00168502, -2.85223e-05, -0.116418, -0.000152774, 0.00278381, -1.55512e-05, 0.00505213, 0.000173042, -0.000529212, -6.46728e-05, 0.000455722, 0.00031993, -0.00812077, 1.52682e-05, -0.0885984, -9.256e-05, 0.00144004, 0.000109967, 0.00456765, -1.67314e-05, 0.00232438, 4.57157e-05, -0.0024753, 0.000134353, 0.00310863, -9.48636e-05, -0.0074418, 0.000105898, -0.0213922, 7.55059e-05, -0.00286219, 0.000134935, -0.00595059, 0.000166761, 0.00538073, 0.000110358, -0.000291087, 2.60161e-05, -0.00385802, 0.000143776, 0.00528432, -0.000282895, -0.000889365, 0.000113083, 0.0264913, 0.000124952, -0.00552861, 7.81519e-05, -0.00724921, 8.41588e-05, -0.0027578, 0.000192587, 0.00222154, 0.000397708, 0.00100273, 0.000170431, -0.00143217, -0.0001592, 0.00302369, -7.14273e-06, 0.00484447, -3.83634e-05, 0.0224788, -7.13386e-05, -0.00399789, 2.27442e-05, -0.00274036, 0.000280797, -0.00746851,
-0.00741549, -0.000366858, -0.00424162, -0.000347435, -0.0138855, -0.000112621, 0.00371221, 2.53048e-05, -0.000437904, -0.000450333, -0.00848684, 0.000294215, 0.00269476, 0.000510214, -0.00157046, 0.000419275, -0.000875206, -0.000160581, 0.00879331, 0.000114121, -0.0046802, 0.00021614, -0.00232254, 0.000164454, 0.119666, 0.000594671, -0.00156319, 0.000360943, 0.0153483, -0.000403113, -0.00381723, -0.000503741, -0.00127634, -0.000324251, 0.156387, 0.000459456, 0.0153844, 0.000337713, -0.00148395, 0.000222808, 0.00224097, -0.00031389, 0.00752459, -0.000476468, 0.00116778, -0.000289701, 0.0928558, 0.000182179, -0.00266754, 0.000952163, 0.0140307, -0.000428036, -0.000466428, -0.000490217, -0.0122137, 0.000421547, 0.0127265, 0.000357083, 0.00261626, 0.000206332, -0.00142425, -0.000392483, -0.00337968, -0.000466264, 0.000686067, 0.000742199, -0.00317494, -0.000669429, 0.000681116, -0.000650303, -0.00846975, 0.000682075, 0.000268918, 0.000691311, 0.0013249, 0.000304391, -0.0354256, -0.000400154, -0.00142731, -0.000337247, -0.0156644, -0.000738673, 0.000224783, -0.000376101, -0.0173072, -1.0544e-05, 0.000853555, 0.000162887, 0.00626663, -6.97938e-05, -0.0164971, -8.67174e-05, -0.00148662, -0.00022975, -0.00428197, 0.000203586, 0.00150486, 0.000123546, -0.0113926, -0.000890249,
0.000712781, 0.00813106, 0.000620634, 0.0055238, 8.18113e-05, 0.00444116, 8.32995e-05, 0.00223035, -8.18785e-05, 0.00969479, -0.000149085, -0.00544034, -4.74009e-06, -0.00440664, -7.72036e-05, -0.00853923, -0.000233809, 0.00757468, -0.000263877, -0.00729882, 0.000220052, -0.00353084, -2.37168e-05, -0.00637127, -0.000206314, -0.00943034, -9.99562e-05, -0.00301698, 0.000115696, 0.00156737, 6.10946e-05, 0.00429819, 1.88194e-05, 0.00277128, 3.54616e-05, -0.0908228, -0.000465924, -0.00814438, 0.000170294, -0.00872406, 0.000522389, 0.00801663, -0.000365612, 0.00668609, 4.99076e-05, 0.00681375, 0.000301415, -0.10644, -8.8662e-05, -0.00997501, -0.000238611, 0.00214725, 0.000179074, -0.00228759, 0.000195061, 0.0025244, -0.000262989, -0.00165852, 7.12582e-05, 0.000558477, 7.44716e-05, -0.0434365, -4.30182e-05, 0.0140733, -0.000280335, -0.0107153, 0.000357003, 0.00930922, -7.2005e-05, 0.00681752, -0.000574677, -0.00664587, 0.000369523, -0.00944755, 2.60687e-05, -0.00647919, -0.000375639, 0.0279246, 0.000104408, 0.0133009, 0.000390919, 0.00707287, -0.000384448, -0.000804702, 0.000467179, 0.00430527, -0.000171005, 0.00566946, -0.000601441, -0.00580507, 0.000518715, -0.00462998, -8.62616e-05, -0.00398597, -0.000265336, 0.0357224, 0.000149772, -0.00256102, 0.000218919, 0.00907301,
-0.0101363, -0.000123948, -0.00503386, -0.000352938, -0.0171261, 0.000143856, 0.00813112, 6.32337e-06, -1.20629e-05, -0.00018277, -0.0112356, 0.000305666, 0.00889215, 0.00043218, -0.000641849, -0.000120439, 0.000942499, 0.000162749, 0.00807668, -9.76488e-05, -0.00363688, 7.91227e-05, -0.0010206, -0.000275846, 0.0188855, 0.000139852, -0.000523377, 0.00045984, 0.0141102, -0.000507094, -0.0107671, -0.00055981, -0.00010564, -0.000181594, 0.0153361, 1.48513e-05, 0.101556, -6.39405e-06, -0.00258054, 8.2423e-05, -0.00155652, -0.000128582, -0.00219013, -0.000315055, 0.00141575, 0.000245312, -0.00553907, -0.000222124, 0.00026187, 0.000290978, 0.115928, -0.000172271, -0.00217645, -0.000528762, -0.0154022, 0.000589074, 0.00911928, 0.000592787, 0.00143702, 0.000522628, -0.0180717, -0.000354023, 0.000596806, -0.000415686, -0.000675711, 7.60087e-05, 0.0391935, -3.68987e-05, 0.000708225, -0.000357531, -0.00621577, 0.000445797, 0.00659099, 0.000652209, -0.000594744, 0.0001412, -0.00591328, -6.26253e-06, 0.000313999, -0.000249296, -0.0177015, -0.000373869, -0.00133879, -0.000386309, -0.0431704, 0.000226703, 0.00256848, 0.000484591, 0.0130817, -0.000512619, -0.00545714, -0.000455459, -0.00223494, -0.00054822, 0.0137675, 0.000517674, -0.000741319, 0.000242083, -0.0194622, -0.000630402,
-0.00287248, 0.00720683, -0.00339029, 0.00715671, 0.00045372, 0.00356978, -0.00160968, 0.00409994, 0.000493943, 0.0118244, 6.77385e-05, -0.00489888, 0.000362853, -0.00505897, 0.000270174, -0.00726538, 9.55157e-05, 0.0124134, -0.000502194, -0.00489548, 0.000953881, -0.00804164, -0.000319262, -0.00427207, -0.00032674, -0.109057, -7.68113e-06, 0.00268704, -0.000279416, 0.00424231, 0.000190592, -0.00071932, -0.000465684, 0.0050801, -0.000106443, -0.00816864, -0.000304867, -0.146482, -0.000145521, -0.00604614, 0.000468014, 0.00807134, -0.000658311, 0.00707378, -1.05253e-05, 0.00850655, 0.000302695, -0.00671433, -0.000101138, -0.00876871, -8.38476e-05, -0.0912676, -0.000102805, -0.00467752, 0.000506736, -0.00142709, -0.000538495, 0.00545158, 0.000336933, 0.000816493, 0.00021504, 0.00285991, 4.40025e-05, 0.00827993, 0.000198306, -0.00369124, 0.000318373, -0.00150966, 4.95419e-05, 0.00199219, -0.000334811, -0.0120932, 0.000283152, -0.00215193, 0.00012617, -0.00750696, -0.000355655, 0.00951296, 0.00020292, 0.00832156, 0.000946784, 0.00734282, 0.000220933, 0.0119408, 0.000268692, 0.036379, 5.26348e-05, 0.00334426, -0.00071598, -0.00767133, 0.000635346, -0.00493342, -0.000172553, -0.00466967, -0.000528283, 0.00480164, 9.70933e-05, 0.000224356, 0.000854612, 0.00742467,
0.000640239, 5.25994e-05, 0.00134889, -6.49949e-05, -0.00189759, 4.35509e-05, 0.00159926, -9.81005e-06, -0.000833591, -5.5777e-06, -0.00187729, -1.53432e-05, 0.000618951, 0.000138434, -0.000762296, -0.000133219, 0.0400487, -3.57747e-05, 0.000158509, -5.08916e-05, -0.0012564, 0.000216023, 0.000264729, -0.000111965, -0.000840311, -7.06813e-05, 0.0885988, -3.98238e-05, 0.0026364, -1.7472e-05, -0.0024859, 9.45184e-05, 0.00128944, 0.000104648, -0.00149129, 8.51621e-05, -0.00257562, -0.000112418, 0.112152, -4.78617e-08, 0.00375494, 3.45407e-05, -0.00186716, -0.000138137, 0.00117585, 0.000331544, -0.00139877, 0.000130768, -0.000451801, -0.000214335, -0.00422578, -7.28523e-05, 0.0827661, 1.61257e-05, 0.00301831, 2.44064e-05, 0.000203248, -0.000243684, -2.27036e-05, 0.000294704, -0.000598271, 8.19526e-05, -0.000757484, 2.80867e-05, -0.00271608, -0.000294473, -0.00458228, 3.85861e-05, 0.0210797, -4.2271e-05, 0.00105009, -4.0385e-05, 0.002162, -0.000108676, -0.00099161, -1.79541e-05, 8.43437e-05, 1.4285e-05, -0.0011641, 6.3298e-05, 0.00097592, -7.10774e-05, -0.00479945, -0.00018056, -0.00388125, 0.000136331, -0.0205301, -0.000109849, -0.00119337, -7.00508e-05, 0.00269481, 0.000113721, -0.000603408, -0.000301776, 0.000114825, 8.03513e-05, -0.00163766, -7.87013e-06, 0.000670156, 7.51743e-05,
-0.0027667, 0.00285889, -0.0030207, 0.00817154, 0.000303242, -0.00302677, -0.00111284, 0.00518382, -0.000236336, 0.0353068, 6.14765e-05, -0.0113189, 0.000484776, -0.00369766, -0.000111274, -0.00836819, -8.37696e-05, -0.00355642, -0.000270086, -0.00974891, 0.000573444, -0.0052257, 9.51129e-05, -0.00779508, -0.00051766, -0.00738335, -1.43797e-05, -0.0919861, -5.53433e-05, 0.00198834, -0.000120401, 0.00281514, -1.99239e-05, -0.00053411, -0.000295156, -0.00874255, -0.000490598, -0.00603257, 6.79928e-05, -0.141448, 0.000464315, 0.0102751, -0.000406785, 0.00778717, -0.000313815, 0.00344688, 0.000243088, -0.0084623, -8.05621e-05, -0.00617524, -0.000153753, 0.00312651, -1.12379e-05, -0.097821, 0.000419634, 0.00620463, -7.97726e-05, 0.00107156, -0.000246438, -0.00130865, 0.00021553, -5.38043e-05, -0.000204019, 0.00811238, -0.000372616, -0.00463357, 0.000533246, 0.0102538, -0.000179992, -0.0104195, -0.000256371, -0.00244492, 0.000220145, -0.0085012, 0.000235649, -0.00737257, -0.000409606, 0.00627283, -0.000206426, 0.0094336, 0.000835884, 0.0144121, -0.000324457, 0.00481, 0.000555616, 0.00689195, -0.000195296, 0.0334423, -0.000559078, -0.00354754, 2.10739e-05, -0.00714445, 0.000513891, -0.00483679, -0.000559921, 0.00203514, -2.59913e-05, 0.00332529, 0.000561683, 0.0084289,
-0.00138624, -0.000314063, -0.00150734, -0.000263086, -0.0286669, -0.000168042, -0.000475768, -0.000147291, 0.000992, -0.000359594, -0.024124, 0.000209342, 0.000678506, 0.000150031, 0.00104017, 0.000506452, 0.00283243, -0.000228064, 0.0257411, 0.0002673, 0.00153175, 0.00019021, 2.46734e-06, 0.000406708, 0.00136442, 0.00046971, 0.00420558, 0.000233364, 0.0915402, -7.69134e-05, 0.00206741, -6.40709e-05, -0.00136436, -0.000105051, 0.00226617, 0.00056877, -0.001569, 0.000441879, 0.00375768, 0.00045607, 0.117084, -0.000298749, 0.00182698, -0.000203369, -0.00206119, -0.000368055, 0.00262545, 0.000535247, -0.000262052, 0.00050899, -0.00280469, -5.41704e-05, 0.002733, 0.000163377, 0.0773427, -4.39275e-05, 0.00017334, -1.47379e-05, -0.00183465, -7.83503e-05, 0.00283766, 1.3766e-05, -0.000239762, -0.000547408, -0.000542283, 0.000543533, -0.00340981, -0.000460379, 0.00282717, -0.000209426, 0.00866132, 0.000347779, -0.0023909, 0.000201549, -0.00124564, 0.000274827, 0.0031602, -0.000329071, -0.000260127, -0.000510822, -0.000579, -0.000181746, -0.000758588, 6.17525e-05, -0.0033316, -0.000282793, 0.00409875, -0.000130603, -0.02977, 0.000352801, -0.00401898, 0.000113698, -0.000965381, 0.00013534, 0.00305213, -5.53719e-05, -0.000509616, 7.59139e-05, -0.00112237, -0.000540072,
0.00218086, -0.0174678, 0.00235894, -0.00770513, -0.000228048, 0.00869225, 0.000840342, -0.0053825, -3.0367e-05, -0.00769274, -1.91814e-05, 0.0401773, -0.000389203, 0.00262949, 2.64054e-05, 0.00778579, -7.56585e-05, -0.00482162, 0.00025569, 0.0104617, -0.000412263, 0.00460577, 0.000122307, 0.0071978, 0.000432, 0.00950226, -3.53055e-05, 0.00522808, 9.01636e-05, -0.0736305, 0.000163023, -0.00201327, 0.00016394, 0.000437988, 0.00018771, 0.00801651, 0.000520986, 0.00806256, 4.89997e-05, 0.0102792, -0.000302499, -0.123703, 0.00041753, -0.00659334, 0.000136365, -0.00294897, -0.000322358, 0.00731156, 7.64123e-05, 0.00532396, 0.000160335, -0.00354774, 4.04763e-05, 0.00395302, -0.000213332, -0.0810129, 0.000200237, -0.0012159, 0.00010325, 0.00221314, -0.000268537, -0.00101343, 4.67697e-05, -0.00755401, 2.86067e-05, 0.00428971, -0.000542357, -0.0132824, -6.97488e-05, -0.00542125, 0.000387871, 0.00691152, -4.22135e-05, 0.00713811, 8.23939e-05, 0.00857558, 0.000373873, -0.00715796, -4.12868e-05, -0.00819801, -0.00073969, -0.00852879, 3.00166e-05, -0.00241205, -0.000507867, -0.0103637, -0.000133713, -0.00696897, 0.00065741, 0.049776, 2.10308e-05, 0.0065212, 1.46045e-05, 0.00627934, 0.000586439, -0.00346031, -8.25518e-05, -0.00149867, -0.000410821, -0.00782917,
0.0296828, 0.000267254, 0.0128476, 9.67744e-05, 0.00178415, 0.000227957, -0.0364849, -4.93423e-05, -0.00291243, 0.00017703, 0.00224855, 6.03826e-05, -0.0520893, -0.000302552, -0.0039635, -0.000329178, -0.0013932, 7.75144e-05, 0.00182923, 0.00010193, -0.00104708, -0.000362084, -0.00280901, -0.000204965, 0.00371891, -0.000613269, -0.00222069, -0.000167807, 0.00172523, 0.000312011, 0.076832, -0.000188217, -0.000871688, 0.000215348, 0.00759498, -0.000424561, -0.0021854, -0.000578758, -0.00195964, -0.000238327, 0.00186503, 0.000328788, 0.108748, -5.35985e-05, 2.60373e-05, 0.000406327, 0.00893878, -0.000364775, 0.00156772, -0.000504713, -0.00321861, -1.19456e-05, -0.000765525, -6.06854e-05, 0.000641296, 5.60518e-05, 0.06671, -0.000119907, -0.000411164, 0.000127397, 0.0056858, 9.9557e-05, 0.00204216, 0.000301527, -0.000749849, -0.000446837, -0.00260241, 0.000375203, 0.000433484, 4.31694e-05, -0.00236598, -0.000176719, -0.00474865, -0.000194656, -0.00109214, -0.000202553, -0.000238129, 0.00041605, 0.00112613, 0.000289809, 0.00171162, 0.000393625, -0.00160708, 0.000107474, -0.00137694, 0.00010316, 0.00079193, -0.00017499, -0.00478402, -0.000119509, -0.0411621, -4.21876e-05, -0.000983561, -8.22721e-05, -0.00410272, 0.000182231, 2.15198e-06, -5.92262e-05, 0.0033226, 0.000320107,
0.0014317, -0.00770323, 0.00159027, -0.0191505, -0.000233751, -0.00538127, 0.000586545, 0.00600085, 1.22915e-05, -0.00847568, -0.000142928, 0.00263272, -0.000363315, 0.037606, -2.62411e-05, 0.00671355, -5.47503e-05, -0.00627963, 8.53881e-05, 0.0046173, -0.000529717, 0.00918912, -2.9639e-05, 0.00278006, 0.000357718, 0.00677402, -0.000183278, 0.00324817, 7.2304e-05, -0.00199648, -0.000151024, -0.073049, 7.2862e-05, -0.00810946, 2.26476e-05, 0.00668794, 0.000493316, 0.00705961, -0.000208528, 0.00777839, -0.000161397, -0.00658649, 7.95435e-05, -0.122099, 0.000186921, -0.0129877, -0.000481012, 0.00470007, 6.20769e-05, 0.00711894, 6.63545e-05, 0.00134003, -1.36264e-05, 0.00113259, -0.000152368, -0.00123267, -0.000153238, -0.0804687, 0.000153376, -0.00491983, -0.000303806, -0.00435344, 1.58937e-05, -0.00805742, -0.0001004, 0.00769059, -0.000673295, -0.00308036, 0.00021008, -0.00718258, 0.00023301, 0.00709995, -0.000536656, 0.00499831, -6.04445e-06, 0.00680654, 0.000451294, -0.0105835, -0.000126201, -0.0080928, -0.000569439, -0.00822756, -0.000205194, 0.00111296, -0.000508787, -7.44532e-05, 0.000174636, -0.00504614, 0.000442386, 0.00647431, -0.00059959, 0.046768, -9.78245e-05, 0.00893259, 0.000708864, -0.00687377, -0.000201742, 0.000529712, -0.000289966, -0.00505281,
0.0103274, -5.59244e-05, 0.0110945, 5.59482e-05, 0.000984786, -5.56516e-05, 0.00370803, 1.56731e-05, -0.0366847, 8.36381e-05, 0.00101201, 1.46959e-05, -0.00238287, -6.85711e-05, -0.0354866, 0.000202663, 0.000863888, -2.32509e-05, -4.35579e-05, 0.000103813, -0.00279241, -7.00442e-05, 0.0180874, 0.000195841, 0.00057084, 7.85703e-05, 0.00143325, -0.000231615, -0.00140009, 0.000141696, -0.000440543, 4.16961e-05, 0.0917481, 2.10794e-05, 0.00116053, 0.000118863, 0.0014004, -1.82714e-05, 0.00116921, -0.000280054, -0.00205976, 0.000126198, 0.000388063, 0.00015583, 0.122464, -9.51508e-05, 0.00128602, 0.000155477, -0.00346084, 7.98156e-05, 0.00248007, -0.000228618, 0.000153284, -5.99564e-05, -0.00179943, 0.000115007, -0.000775656, 0.000176415, 0.0810941, -9.40291e-06, 0.000364132, 9.18886e-05, -0.00692765, 0.000160668, 0.000807997, 3.76607e-05, 0.00226658, -0.000313857, -0.00052985, 0.000184713, -0.00120324, 0.000127222, -0.00186432, 0.000152145, 0.00642594, 0.000134352, -0.00150112, -1.58239e-05, -0.00783392, 0.000218338, -0.00119998, 8.23894e-05, 0.00147157, -6.04077e-05, 0.000618505, -0.000130652, 0.000195151, 0.000159593, -0.000931571, 0.000115327, -0.00179501, 0.000194641, -0.0362199, 0.000112578, -0.00313027, -8.31871e-05, -0.00434779, 0.000137567, -0.0016335, -0.000188372,
0.00145173, -0.00867471, 0.00169997, -0.00412953, -0.000217101, -0.00707195, 0.000752002, 0.00107655, -0.000119658, -0.00837689, 6.34683e-05, 0.00120723, -0.000294517, 0.00933411, -1.48642e-05, 0.0394269, -0.000110912, -0.0078092, 0.000419614, 0.00481792, -0.000548454, 0.00770614, 0.000142033, 0.00163824, 0.000380941, 0.00880614, 0.000140653, -0.000544931, 0.00022722, 0.000376944, -1.6757e-05, -0.00458751, 9.6539e-05, -0.0888814, 7.92577e-05, 0.00683034, 0.000777273, 0.00848957, 0.000425453, 0.00344685, -0.000325962, -0.0029494, 0.000493909, -0.0129935, -8.16586e-05, -0.140231, -0.000434004, 0.00525365, 4.05956e-05, 0.0091614, 0.000611169, 0.000384429, 0.0003071, -0.00129831, -0.0003927, 0.00202804, 0.000318533, -0.00701913, -8.48098e-05, -0.0948708, -0.000363969, -0.00327268, 1.03804e-05, -0.0099278, 0.000232901, 0.00945535, -0.000129438, -0.00487459, -7.91963e-05, -0.00735894, 0.000163525, 0.00881041, -0.000266589, 0.00529171, 0.000107504, -0.00359837, 0.000221996, -0.00953096, -2.66877e-05, -0.00919108, -0.000523497, -0.00716742, 0.000390839, 0.000765511, -0.00027388, 0.000576522, -0.000149245, -0.00482514, 0.000383257, 0.00626708, -0.000400347, 0.00888705, 0.000136825, 0.041132, 0.000368976, -0.00686948, 1.33204e-05, 0.00231134, -0.000225972, -0.0113353,
-0.00197234, -0.000378021, -0.00563504, -0.0002176, 0.00493261, -0.000300192, -0.008354, -9.1867e-05, -0.00191197, -0.000334843, 0.00514561, 7.20878e-05, -0.00497548, 0.000240005, -0.00282816, 0.000232329, -0.000834525, -0.000325, 0.00203962, 0.000221575, 0.00385732, 0.000337317, -0.00220905, 0.000231253, 0.0398729, 0.000269146, -0.00148559, -3.28023e-05, 0.000614314, -2.62088e-05, 0.0105077, 2.33488e-05, -0.000458064, -2.14872e-05, 0.0931671, 0.000105847, -0.00546153, 0.000253105, -0.00139922, 0.000160601, 0.0026382, -0.000254254, 0.00852738, -0.000329025, 0.00120797, -0.000273926, 0.122674, 0.000158326, -0.000248674, 0.00011374, -0.00657973, -7.39712e-05, -0.000603448, -1.65063e-05, 0.00358933, -5.50614e-05, -0.000184914, -0.000258122, 0.00166638, -0.000244547, 0.0932924, 5.78504e-05, -0.000601203, -0.000216193, -0.000600081, 0.000145341, -0.00330598, -0.000431684, 8.58619e-05, -0.000334788, -0.000498305, 0.000372068, -0.00794153, 0.000160165, 0.000630002, 4.7831e-06, 0.0243251, -0.000146363, -0.000644651, -0.000374068, 0.0108703, -0.000318817, -0.00145049, -4.7218e-05, -0.00028691, -0.000453445, 0.000135199, -0.000331385, -0.00624816, 0.000456799, -0.00948925, 0.000379302, -0.00102879, 0.000143265, -0.0243521, -0.000235379, 0.000142443, -0.000308918, 0.0160645, -0.00029331,
0.00121225, 0.00716031, 0.00104496, 0.00359524, 0.000427091, 0.00372828, 4.11479e-05, 0.0013092, -0.000109229, 0.00913101, 0.0001249, -0.00519534, -0.000225316, -0.00236899, -0.000132997, -0.00764995, -0.000338539, 0.00728373, -0.000345278, -0.00679861, 0.000253567, -0.000124993, -7.14541e-05, -0.00624266, -0.00043028, -0.00817203, -0.000210335, -0.00271189, -0.000133569, 0.00133897, 0.000312244, 0.00565597, 2.41418e-05, 0.00143224, -0.000176286, -0.106716, -0.00068215, -0.00671634, 0.000178502, -0.00846715, 0.000511487, 0.00731284, -0.000282403, 0.00469631, 0.000100597, 0.00524152, 0.000319399, -0.145765, -0.000102225, -0.00938658, -0.000464975, 0.00286249, 0.00031854, -0.00279355, 0.000469947, 0.00242962, -0.000477719, -0.0049795, 0.000111943, 0.000273567, 0.000281053, -0.0949188, -4.75473e-05, 0.0116146, -0.000270561, -0.00963394, 0.000352688, 0.00951767, 6.09869e-05, 0.00584965, -0.000399486, -0.00589299, 0.000168534, -0.0112928, 2.08656e-05, -0.00584547, -0.000241103, -0.00742326, 0.000121683, 0.00939973, 0.000669651, 0.00683272, -0.000326677, 0.000316244, 0.000713527, 0.00480232, -0.000168735, 0.00520973, -0.000799713, -0.00517954, 0.000616178, -0.00411267, -0.00012676, -0.00395619, -0.000340206, 0.0333671, 0.000173571, -0.00548425, 0.00057825, 0.00751152,
0.00224809, -1.2643e-05, 0.00142136, 0.000127878, -0.00100468, 2.90641e-08, -0.00142269, 0.000114883, 0.00131118, 0.000285458, -0.000675848, 7.80244e-06, -0.00225128, -1.0435e-05, 0.00253027, 3.37674e-05, -0.000182767, 0.000180665, 0.000265506, -1.14192e-05, 3.59676e-05, -0.000103983, 0.00215835, 4.34979e-05, -0.00259882, 0.00022768, -0.000480189, -0.000134555, 0.000425444, -1.92835e-05, 0.00235055, -7.52863e-05, -0.000552171, 5.28298e-05, -0.00267262, -7.32355e-05, 0.000267963, 9.51343e-05, -0.000458165, -0.000226894, -0.000267401, 1.66233e-05, 0.00155766, 2.32855e-05, -0.00349371, 3.99767e-05, -0.000238079, -4.00813e-05, 0.0745337, 2.70973e-05, 0.000596794, -0.000147626, -3.8921e-05, 1.27898e-05, -0.000547815, 4.35359e-05, -0.00130084, 8.33438e-05, -0.00317971, -4.47426e-05, 0.00205231, 2.289e-05, 0.093903, 9.70472e-05, 0.00182331, 0.000174118, 0.000304275, -0.000140907, 0.000186845, 0.000202222, 0.000103213, -2.47053e-06, -0.00235867, 5.74318e-05, 0.00094356, -0.000138299, 0.00163427, 3.62874e-05, 0.0498933, 2.34066e-05, -0.000275565, 0.000522239, 0.00205505, 0.000154582, -0.00075162, 6.8649e-05, -7.88331e-05, 5.73713e-05, 0.000741674, -7.03256e-05, -0.000146529, -1.35775e-05, 0.0047081, -7.63662e-05, -0.000580329, 3.01711e-05, -0.0106297, -7.9919e-05, -0.00152108, -0.000116903,
-5.04351e-05, 0.00698915, -0.000268074, 0.00827468, 4.01221e-05, 0.0042439, -0.00034302, 0.00507794, 0.000179193, 0.00679414, -0.000256682, -0.00369566, 3.63353e-05, -0.00467322, 6.68185e-05, -0.00971252, -4.12298e-05, 0.00446, -0.000352823, -0.00547733, 0.000384136, -0.00744615, -0.000162477, -0.00663414, 0.000287681, -0.0127885, -0.000140101, -0.00351271, 0.000132919, 0.000899452, 8.59433e-05, 0.000303206, -0.000168222, 0.00458984, 0.000480295, -0.00998104, -5.75928e-05, -0.00875307, -0.000190449, -0.0061795, 0.000567875, 0.00531171, -0.000457073, 0.00712128, 6.44665e-05, 0.00914127, 0.000249372, -0.0093703, 0.000275624, -0.0836395, 7.1369e-05, 0.00755824, -4.62497e-05, 0.000888774, 0.000143572, 0.00120011, -0.000302162, 0.00345723, 0.000184592, 0.000659786, -0.000350864, 0.00123049, 0.00016354, 0.00797361, -0.000105455, -0.102307, 0.000227266, 0.0156313, 0.000216596, 0.0078909, -0.000642802, -0.00463745, 0.000443142, -0.0047813, 2.43715e-05, -0.00876583, -0.000649705, 0.00980143, -9.52737e-05, 0.00826444, 0.000392549, 0.00797307, -4.94897e-05, -0.0469274, 0.00010871, 0.0027991, 0.000269122, 0.00404995, -0.000500034, -0.00301102, 0.000577307, -0.00551754, -0.000140271, -0.00560237, -0.000146096, 0.00550337, -0.00014038, 0.000417634, 0.000204879, 0.0103889,
-0.00986914, -0.000396765, -0.00458279, -0.000838433, -0.0164298, 3.11834e-06, 0.00865689, -0.000303965, -0.000949239, -0.000420777, -0.011417, 0.00051158, 0.00929316, 0.000688009, -0.00175092, 0.000231283, -0.00059623, 1.0925e-05, 0.0061554, 0.000173404, -0.0036959, 0.000511442, -0.00126973, 2.42047e-05, 0.0179563, 0.000316096, -0.00263698, 0.000606401, 0.0115379, -0.000559838, -0.0114778, -0.000570623, 0.000790199, -0.00021307, 0.0140221, 0.000254742, 0.116227, 0.000166526, -0.00423536, 0.000358807, -0.0027979, -0.000432961, -0.00319484, -0.000738025, 0.00250389, 5.25156e-05, -0.00666847, -7.17667e-06, 0.00059045, 0.000466862, 0.151855, -0.000173569, -0.00291036, -0.000471358, -0.014557, 0.000467137, 0.00864238, 0.000376089, 0.00153801, 0.000552874, -0.0188385, -0.000370317, 0.00065842, -0.000642245, -0.00116959, 0.000249186, 0.0859791, -0.00018982, 0.000836056, -0.000571078, -0.00418959, 0.000665661, 0.00715498, 0.000948848, -0.0015183, 0.000425564, -0.00644543, -0.000154688, -0.000315014, -0.00050437, -0.0173313, -0.000834139, -0.00212497, -0.000400219, -0.0104031, 0.000113555, 0.0033958, 0.000306103, 0.015065, -0.000297727, -0.00430746, -7.60793e-05, -0.00314147, -0.000396715, 0.013483, 0.000534831, -0.00175973, 0.000193013, -0.0188079, -0.000942431,
-0.00314313, -0.00230191, -0.00353356, -0.00056751, 0.000287812, -0.00100939, -0.00152195, -0.00128085, 0.000211035, 0.00337709, 0.000192046, 0.00285617, 0.000291621, -0.00205577, 0.000171215, 0.00275374, -4.16189e-05, 0.00650793, -0.000164078, 0.00483204, 0.000647998, -0.00232626, -0.000109271, 0.00368253, -0.000380524, -0.0415266, -2.49919e-05, 0.00692017, -0.000326482, 0.00198108, 0.000151707, -0.001226, -0.000348621, 0.0023315, -0.00038925, 0.00210569, -4.55443e-05, -0.0916148, -3.61413e-06, 0.00313065, -6.79059e-05, -0.00354113, -0.000223576, 0.00134672, -0.000264273, 0.00040994, -7.64116e-05, 0.00282452, -4.75448e-05, 0.00753262, -6.46309e-05, -0.115337, -3.6446e-05, -0.00245806, 0.000269378, -0.0067444, -0.000264136, 0.00378635, 7.44338e-05, 0.000130144, 0.000159236, 0.00222892, -8.42031e-05, -0.00168759, 0.000142117, 0.0125434, -1.22346e-05, -0.0840774, -8.79652e-05, -0.00522576, 0.000235484, -0.00479434, -0.000152409, 0.00371747, 0.000322346, 0.00118259, 1.52272e-05, 0.00156404, -3.3809e-05, -0.00153071, 0.00026945, -0.00371615, 0.000267859, 0.0112144, 7.42397e-05, -0.0198259, -6.18213e-05, -0.00325326, -9.36451e-05, 0.00075806, 6.59279e-05, 0.000566418, 0.000275103, 0.0012246, -0.000288801, 0.00158985, 4.70238e-05, 0.000528144, 0.000347875, -0.00189464,
0.00245676, -2.10361e-06, 0.00330535, 0.00024689, 0.00190509, 1.32429e-05, 0.00196542, 0.000145163, -0.000562296, 0.000101289, 0.00225167, 1.66869e-05, -0.000729141, -7.02801e-05, 4.96071e-05, 0.000309344, -0.0272453, -2.38317e-05, 0.00127779, -1.08242e-05, -0.00288474, -6.45978e-06, 0.00118496, 0.000169655, -0.000307798, 3.49427e-05, 0.0067155, -0.000184554, 0.000967397, -3.18412e-05, -0.00275949, 0.000229338, 0.00132478, -8.72371e-05, -0.000468502, -3.7431e-05, -0.00220337, -5.5155e-05, 0.0830809, -0.000126294, 0.00272471, -1.13896e-05, -0.000602066, 0.000168481, 0.000139106, 3.28554e-05, -0.000594006, 0.000139932, -3.84248e-05, 0.000113839, -0.00292143, -0.000137053, 0.128279, 6.63974e-05, 0.00481857, 1.13578e-05, 0.00152907, -0.000217896, -0.000781864, 0.00039423, -0.000882756, 0.000314256, -0.000416055, 5.73739e-05, -0.00518512, 6.44592e-05, -0.00244653, -5.18813e-05, 0.0929532, 6.77131e-05, 0.00388547, -2.13739e-05, 0.00208736, -0.000379955, -3.62923e-05, 0.000338903, -0.00115049, 0.000237675, -0.00117938, 0.0001523, 8.41632e-05, 0.000284814, -0.00593261, -0.000112636, -0.00215683, 0.000138806, 0.0149957, -0.000159653, -0.000679543, -8.10794e-05, 0.00157268, -1.38206e-05, 0.00151194, -0.000259604, -0.000850359, 2.49032e-05, -0.00160394, 0.000105929, -0.00108009, -2.07515e-06,
-0.000659233, -0.00658616, -0.000848886, 0.00108248, 0.000493394, -0.00958985, -0.000529086, 0.0010632, -0.000293529, 0.0329119, 0.000313893, -0.00713095, -5.76113e-05, 0.00075432, -0.000235144, -0.00283031, -0.000114366, 0.0292748, -0.000220806, -0.00103439, 0.000196479, 0.00139155, 4.46684e-05, -0.00364516, -0.000699005, -0.00343163, -0.000101498, -0.0250568, -0.00032633, 0.00384565, 9.13422e-05, 0.00223626, 0.000147372, -0.00248048, -0.000613866, -0.0023303, -0.000567255, -0.00464444, 2.90479e-06, -0.0981492, 0.000176888, 0.00396405, -0.000173538, 0.00116248, -6.16966e-05, -0.00130259, 2.58046e-05, -0.00283024, 2.58179e-05, 0.000891386, -0.000498519, -0.0024554, 4.44916e-05, -0.128091, 0.00054885, 0.000474208, -0.000266749, -0.00185278, -0.000230594, -0.00171931, 0.000356895, -0.00134497, -3.42356e-05, 0.00214558, -0.000418756, 0.00357038, 0.000131983, 0.00158628, -6.21282e-05, -0.0872738, 0.000223762, -0.00214867, -0.000102966, -0.00365544, -3.84455e-05, -0.00266496, -3.93678e-05, 1.64554e-05, -0.000166432, 0.00306153, 0.000750391, 0.000597313, -0.000500799, 0.00678323, 0.000595474, 0.00381484, -0.000193226, -0.0141206, -0.000327804, -0.00129793, 5.21632e-05, -0.001722, 0.000305774, -0.00197538, -0.00048886, -0.000864742, -0.000241032, 0.00215779, 0.000775748, 0.00277934,
0.0101084, 0.000105853, 0.00599641, -9.82551e-06, 0.0253928, -3.14786e-05, -0.00583713, -8.93171e-05, 0.000620028, 0.000214745, 0.000869862, -0.000142981, -0.0067206, -0.000130446, -0.000102011, -0.000156385, -0.000101348, -1.14201e-05, -0.0340748, 6.46331e-05, 0.00405255, -1.61048e-05, -0.00161345, 4.4271e-05, -0.0145552, -0.000279647, 0.000889042, -0.000267085, -0.00849024, 0.000325065, 0.00951412, 0.000139978, -0.00238089, 0.000268093, -0.0121902, -0.000400116, -0.0154171, 9.5592e-06, 0.0030089, -2.4634e-05, 0.077655, 0.000166075, 0.000691701, 0.000106155, -0.00179725, 4.80563e-05, 0.00362759, -0.000100308, -0.000547186, -0.000493016, -0.0145733, 0.000398998, 0.0048149, 0.000474084, 0.13419, -0.000237853, -0.0100251, -6.80131e-05, -0.000930144, -0.000427796, 0.0135753, 0.000464294, -0.000303602, 0.00042924, 0.00283462, -0.000160821, 0.00182369, 0.000200127, 0.00450799, 0.000440721, 0.0939531, -0.000203344, -0.0073447, -3.81801e-05, -0.000904371, -0.000421188, 0.00474209, 0.000399393, 0.000612823, 0.000110261, 0.0148017, 0.000321986, 0.00378195, 0.000549206, 0.0160061, -0.000420717, 0.00229132, -0.00019535, 0.00259423, 0.000364325, 0.00436908, 0.000273855, -0.00139023, 0.000194929, -0.0103146, -0.000351602, 0.00142982, -0.00049487, 0.0166449, 0.000736968,
-0.00109449, -0.0258052, -0.00105561, -0.00337644, -0.000430756, -0.0102789, -0.000185753, -0.00326955, -9.28479e-06, 0.00256585, -0.00021093, 0.0328168, 0.000288811, -0.000946697, 5.30337e-05, 0.00410167, -6.70088e-05, 0.00624607, 0.000326227, 0.0470302, 6.92513e-05, -7.41026e-05, 0.000145262, 0.00492848, 0.000767858, 0.00130454, -4.03741e-05, 0.00844815, 0.000420216, -0.00302801, -0.000166161, -0.00129442, 0.000176345, 0.00309509, 0.000614815, 0.00252443, 0.000742053, -0.00142181, 3.05688e-05, 0.00621262, -5.88155e-05, -0.0813305, 3.65909e-05, -0.00125732, 0.00012913, 0.00200529, -0.000128913, 0.00243002, 4.0426e-05, 0.0011779, 0.000611179, -0.00675038, 2.95712e-05, 0.00046727, -0.000362466, -0.113759, 0.000316486, 0.00181728, 8.83189e-05, 0.00346874, -0.000416033, -3.01795e-06, 2.72498e-05, -0.0026755, 9.85157e-06, 0.00136178, -0.000227945, -0.00931041, -6.64941e-05, -0.00412353, 1.97248e-05, -0.0698026, 0.000239891, 0.00462185, 0.000104776, 0.00472289, 0.000166446, -0.0017497, -3.41521e-05, -0.00303671, -0.000852305, -0.0028365, -1.83264e-05, -2.02026e-05, -0.000762958, -0.00563057, -0.000130752, -0.00456068, 0.000568398, 0.00499449, 1.70778e-05, 0.00307257, 0.000102378, 0.00243204, 0.000695495, -0.000361897, -8.17056e-05, -0.00127784, -0.000760348, -0.00283712,
-0.00138253, 5.3044e-05, 0.00692665, -0.000142775, -0.00963357, 0.000120932, 0.00618086, -0.000135766, -0.00228853, -0.000279505, -0.00613193, 0.000196339, -0.022287, -0.000115334, -0.00374834, 0.000102774, -0.00152493, -0.000233816, 0.00540322, 0.000235004, -0.045254, -0.000251234, -0.0037411, -3.09742e-05, 0.0132389, 3.37627e-07, -0.00137575, -6.87148e-06, 0.00948781, 0.000211339, -0.0110775, -0.000412308, -0.00233308, -0.000136308, 0.0126772, 0.000254768, 0.00901299, -0.000310283, 0.000226333, 3.01047e-05, 0.000106145, 0.00017666, 0.0687669, -0.000325483, -0.000423993, 7.15381e-05, -0.000227079, 6.70082e-06, -0.00125668, 0.00026709, 0.00853648, -0.000536536, 0.00160832, -0.000200113, -0.00992983, 0.000168517, 0.115086, -5.03936e-05, 0.000758147, 0.000382531, -0.0108049, -0.000364224, -0.00155244, -0.000264258, -0.00074602, -2.93637e-05, -0.00103861, -0.000203432, 0.00150008, -0.000327688, -0.00697593, 0.000102427, 0.0758533, 4.50026e-05, 0.00035304, 0.000318958, -0.00740917, -0.000234255, -0.000239164, -7.52481e-05, -0.00849623, -0.000453405, -0.00105187, -0.000489963, -0.00904506, 0.000436843, 0.000605111, -0.000141036, 0.00395295, -8.01957e-05, -0.00575817, -0.000138392, -0.00121551, -6.672e-05, 0.00280301, 0.000328575, 0.00200045, 0.00022339, -0.00797237, -0.000387732,
0.00295394, -0.00337295, 0.00326117, -0.0233764, -0.000497227, -0.00326563, 0.00125736, -0.00803692, 0.000117924, -0.00196009, -0.000411808, -0.000935836, -0.000507577, 0.0328067, 8.57891e-05, -0.000202974, 0.000100232, -0.0011072, 2.45041e-05, -4.64663e-05, -0.000761184, 0.0454338, -3.91733e-06, -0.00354369, 0.000706253, 0.0040233, -2.86966e-05, 0.00109748, 0.00019394, -0.00125733, -0.000160326, -0.00398372, 3.17352e-05, -0.00743604, 0.000520495, -0.00165026, 0.000885368, 0.00542708, -0.000253599, 0.0011026, -5.56014e-05, -0.0012349, 0.000152638, -0.0808104, 0.00019959, -0.00701106, -0.000282736, -0.00497073, 0.000157359, 0.00346221, 0.000649993, 0.00377423, -0.000271819, -0.00185218, -0.000192516, 0.00181167, -8.55875e-05, -0.113121, 0.000287097, -0.00129198, -0.000676511, -0.00805323, 0.000155304, -0.00398205, 5.68583e-05, 0.0047688, -0.000371693, 0.00156474, -2.14642e-06, -0.00374561, 0.000124518, 0.00459947, -0.000430441, -0.0706045, 0.000156269, 0.0047086, -0.000126874, -0.00797172, 8.43193e-06, -0.00508238, -0.000616457, -0.00216434, -5.11429e-05, 0.00276557, -0.000838811, 0.00172149, 0.000255316, -0.00104432, 0.000470118, 0.00304767, -0.000540768, 0.00299314, -5.95547e-05, 0.00602094, 0.000605927, -0.00463514, -0.000109367, -0.00175666, -0.000537913, -0.00218648,
0.00800047, 3.28478e-05, 0.00899977, -3.65717e-05, 0.000630781, 5.34873e-06, 0.00373903, -6.60371e-06, 0.00844869, -8.05781e-05, -0.000103129, -1.63982e-05, -0.00129947, 1.87298e-05, -0.0105134, 9.69434e-06, 0.00115867, 2.44479e-05, -0.00164378, 4.22794e-05, -0.0026928, -3.00931e-05, -0.0251953, 0.000109859, 0.00187289, 1.60692e-05, 0.00108687, 0.00011431, -0.00242964, 0.000127989, -0.00151985, -6.45453e-05, 0.00937309, 0.000161435, 0.00261818, 0.000122458, 0.00142097, 0.000128605, -5.5234e-05, -1.22273e-05, -0.00183102, 0.000135554, -0.000113936, 4.77192e-05, 0.0814053, 4.14473e-05, 0.00170851, 0.00013206, -0.0031599, 0.000129183, 0.00151669, 0.000147361, -0.000784194, -0.000190178, -0.000926494, 8.49343e-05, 0.000489349, 0.000239452, 0.119723, -0.000126377, -0.000226667, 6.97907e-05, -0.00472743, 6.6507e-06, 0.00105284, 0.00017642, 0.000300086, -4.01382e-05, 0.000165808, -9.20274e-05, -0.000870948, 9.17744e-05, 4.92471e-05, 0.000297692, 0.0794894, -0.000108665, -0.00182607, 2.33997e-05, -0.00332465, 7.98995e-05, -0.000755254, 0.000213821, 0.00153155, 0.000137892, -0.000890064, -0.000201871, 0.00209839, 0.000211091, -0.00134713, 0.000168705, -0.0014036, 0.000188049, 0.000978225, 7.10217e-05, -0.00230772, -9.41138e-07, 4.72079e-05, 0.000171451, -0.000695172, 0.000141729,
0.00114129, -0.00345057, 0.0013892, 0.00191698, -0.000488252, -0.00361521, 0.000730889, 0.00368426, -0.000143817, -0.00284014, -0.000251682, -0.00102438, -8.55394e-05, 0.00339896, -7.7098e-05, 0.0362259, -0.000198252, -0.00366065, 0.00034705, 0.00130527, -0.000435606, -4.69618e-05, 0.000111318, 0.033695, 0.000361743, 0.0011743, -6.59123e-05, -0.00249551, 0.000458243, 0.00187828, -0.000201158, -0.00420455, 0.000172143, -0.0215931, 0.000218739, 0.00055765, 0.000668357, 0.000830462, 0.000272637, -0.00131156, -9.82483e-05, 0.00219407, 0.000233339, -0.00491753, 2.1306e-05, -0.0952077, -0.000264537, 0.000277454, -0.000120005, 0.000707463, 0.000684239, 0.000115671, 0.000404709, -0.0017189, -0.000532564, 0.00346239, 0.000380783, -0.00129206, -0.000129315, -0.123203, -0.000495094, -0.00141223, -0.000178727, 0.000782898, -3.91387e-05, 0.000284946, 0.000147373, 0.00113417, 0.000138588, -0.00265624, -0.000261095, 0.00405001, 0.000126906, 0.00291957, -8.64132e-05, -0.0796257, -0.000181885, -0.0038427, -0.000125959, 0.00324678, -0.000452647, -0.0012196, 0.000116011, -0.00120127, -0.000197478, 0.00392574, -0.000176064, -0.00196452, 0.000222329, 0.0017545, -0.000156959, 0.00397375, 2.56476e-05, -0.00668324, 0.000204609, -0.0050988, 9.30153e-06, 0.00632713, -0.000425259, 0.00173888,
-0.00617536, 9.51587e-05, -0.0137063, 0.000292383, 0.0173285, -0.000141174, -0.0160956, 4.70253e-05, -0.00215487, 0.000300291, 0.0144064, -0.000364569, -0.00511953, -0.00024598, -0.00259636, -0.000298257, -0.000305031, 8.58539e-06, -0.00181081, -0.000140525, 0.012206, 4.14406e-05, -0.00135657, -5.38263e-05, -0.0402998, -0.000383987, -0.000464765, -0.00042798, -0.00858682, 0.000264406, 0.0175773, 0.000509375, 3.34331e-05, 0.000285621, -0.00125318, -0.000438236, -0.0180453, -0.000193591, -0.000593177, -0.000303124, 0.00287075, 0.000212838, 0.00487475, 0.000260157, 0.000257172, 8.6881e-05, 0.0935319, -0.000140553, 0.00201792, -0.000870321, -0.0187614, 0.000218712, -0.000887302, 0.000239043, 0.0135581, -0.000204006, -0.0104638, -0.000512062, -0.000287424, -0.00045551, 0.149466, 0.000422819, 0.00236681, 0.000292213, -0.00137301, -0.000739238, -0.00265672, 0.000110179, -0.00115126, 0.00028174, 0.00447503, -0.000173809, -0.0114806, -0.000604423, -0.000439157, -0.000488931, 0.104067, 0.000370872, 0.000945362, 4.22892e-06, 0.0231918, 0.000503321, -0.00195405, 6.08542e-05, 0.0108706, -0.000549969, -0.000823441, -0.00036316, -0.0143364, 0.000473619, -0.00175609, 0.000277902, 5.07462e-05, 0.000135485, 0.00827236, -0.000313017, -0.000270007, -0.000497555, 0.0242965, 0.000628421,
0.00107771, -0.00167359, 0.000986082, -0.00270075, 0.00052631, -0.0014679, 5.61108e-05, -0.00041135, -6.32943e-05, -0.000712546, 0.000447118, -8.72745e-05, -0.00038279, 0.00509219, -0.000125824, 0.000828807, -0.000163858, -0.000406025, -5.03983e-05, 0.000412146, 7.98055e-06, 0.00743161, -0.000117466, -0.000586208, -0.00030395, 0.00210607, -0.000165973, 0.000225324, -0.000289909, -0.000469071, 0.000406517, 0.00304518, -1.20326e-05, -0.00286015, -0.000301534, -0.0437633, -0.000311972, 0.00282124, 3.91348e-05, -8.27452e-05, 3.71214e-05, -0.00100968, 0.000157982, -0.00435752, 0.000101702, -0.00327087, -2.53791e-06, -0.0952623, -3.00307e-05, 0.00119782, -0.000311326, 0.0022252, 0.000276021, -0.00135456, 0.000408416, -4.3813e-06, -0.000337618, -0.00805524, 0.000112515, -0.00141609, 0.000270088, -0.119442, 4.24919e-07, -0.00462111, 1.77453e-05, 0.00225036, 5.29965e-05, 0.001567, 0.00029028, -0.00204342, 0.000212725, 0.00134635, -0.000336257, -0.00543099, 1.36583e-05, 0.000163553, 0.000222979, -0.0880127, 6.02479e-05, -0.00739231, 0.000445483, -0.00031076, 8.19993e-05, 0.00257065, 0.00047737, 0.00159746, 7.23964e-05, -0.000974176, -0.000359686, 0.00122438, 0.000135873, 0.000359613, -0.000108199, -2.84161e-05, -1.64299e-05, -0.02301, 6.81661e-05, -0.00530701, 0.000591876, -0.00273955,
0.00235731, -5.17401e-05, 0.001158, 0.000136596, -0.000995466, -3.29507e-05, -0.00220019, 0.000152694, 0.00368129, 0.000418584, -0.00101772, 1.05167e-05, -0.00285753, 4.16647e-05, 0.00556546, 3.08438e-05, -0.000565434, 0.000308719, -0.000367686, 7.33005e-06, 0.000265929, -9.19688e-05, 0.00331438, 7.64202e-05, -0.00311533, 0.000311541, -0.000883201, -0.000134595, -6.42666e-06, -1.96108e-05, 0.00314158, -0.000130136, -0.00254628, 0.000108244, -0.00337795, 2.32889e-05, 0.000601115, 0.000198793, -0.000763047, -0.000334314, -0.000245997, -2.80237e-06, 0.00200285, -4.7611e-05, -0.00695985, 5.42321e-05, -0.000594712, 4.2235e-05, 0.0942106, 4.2256e-05, 0.000664838, -0.000129037, -0.000418354, -6.78085e-05, -0.000313931, 3.17363e-05, -0.00159258, 7.77797e-05, -0.00475656, -0.000100204, 0.00239544, 2.12685e-05, 0.138565, 2.7613e-05, 0.00182616, 0.000163236, -0.000323304, -0.00023839, -0.000200101, 0.000221673, 0.000370977, 1.03557e-06, -0.00280982, 0.000134888, 0.00293947, -0.000209466, 0.00248685, -6.68553e-06, 0.10478, -4.42748e-05, -0.00134094, 0.000476704, 0.00195287, 0.000140763, -0.00177705, -4.71609e-05, 7.44415e-05, 0.00011509, 0.00115087, -6.78816e-05, -3.06319e-05, 7.35325e-05, 0.00832026, -8.51788e-05, 0.00021161, 3.39045e-05, 0.0237371, -9.40529e-05, -0.00317605, -6.53617e-05,
1.18739e-05, -0.00732783, 0.000170035, -0.0100742, -0.000208855, -0.00404082, 0.000231334, -0.00653183, -9.61654e-05, -0.00853105, 5.70814e-05, 0.00496074, -4.52724e-05, 0.00508078, -0.000106356, 0.0102454, -0.000257736, -0.00648155, 0.000306591, 0.00680156, -0.000278016, 0.00880574, -2.98859e-05, 0.00635338, 0.000204212, 0.00944804, -0.000156935, 0.00307563, -1.20079e-05, -0.00132839, -2.79373e-05, 4.0658e-05, 6.47105e-05, -0.00595809, -7.38103e-05, 0.0140758, 1.0711e-05, 0.00825827, 0.000114841, 0.0080944, -0.000491364, -0.00755871, 0.0003488, -0.00803841, 9.94991e-05, -0.00992433, -0.000383371, 0.0116312, 3.81748e-05, 0.00796714, -0.00023418, -0.00171629, 0.00017995, 0.002101, -0.000246077, -0.00268302, 0.000168189, -0.00399665, 4.65332e-05, 0.000794507, -0.000153512, -0.00462953, 2.20285e-05, -0.0840478, -8.8436e-05, 0.00826461, -0.000377519, -0.00835752, -2.37327e-05, -0.006295, 0.000541773, 0.00606311, -0.000385429, 0.00554358, -5.8569e-05, 0.0113738, 0.000365875, -0.0150186, -6.21736e-05, -0.10214, -0.00032132, -0.00905875, -0.000293983, 0.000427256, -4.35991e-05, -0.0030906, -0.000163293, -0.00545442, 0.000633319, 0.00563537, -0.000395873, 0.00644093, -0.000125852, 0.00623473, 0.000340269, -0.00471096, -0.000120423, -0.0457996, -8.50496e-05, -0.00729259,
-0.00150458, 8.14393e-05, -0.0014772, -0.000212398, 0.00323888, 6.0568e-05, -0.000200234, -6.32016e-05, 0.000332827, 0.000202476, 0.00222122, -3.33453e-05, 0.00075471, 0.000200722, -0.000288275, -0.00035771, 0.0042285, 0.000293313, -0.00157956, -7.82256e-05, 0.000659729, 0.000151736, -0.000922887, -0.000250684, 0.000951506, -7.7462e-05, 0.00276327, 0.00015149, -0.00324452, -1.54644e-05, -0.000100389, -0.000139403, -0.000366339, 0.000159131, 0.00068158, -0.000433014, -0.000681089, 5.84677e-05, -0.00271742, -0.000152883, -0.000536643, 6.00677e-05, -0.00074385, -0.000206769, 0.000871133, 0.000333628, -0.000629226, -0.000317963, 0.00183185, -9.61014e-05, -0.00118954, 0.000205722, -0.00518856, -0.00030968, 0.00283269, 2.93083e-05, -0.000635004, 4.41214e-05, 0.00106355, 4.53078e-05, -0.00142858, 0.000128713, 0.00182597, -0.000405856, 0.0723073, -4.68711e-05, -0.000940504, 0.00010469, -0.000560377, -9.83121e-05, 0.00232013, -6.84023e-05, 0.000262611, 0.000182763, -0.000138385, -0.000268031, -0.000436647, 0.000333346, -7.53364e-05, -0.000278048, -0.000670252, 3.8568e-05, 0.0906414, 0.000192833, -8.88406e-05, -0.000143393, 0.00567312, 0.000268972, -0.00122918, -7.33507e-05, 0.000971168, -1.90225e-05, -0.00108616, -0.000162427, 0.00138996, 7.08748e-05, -0.00188526, 0.00013226, -0.000809075, -6.88913e-05,
-9.07113e-05, 0.00779073, -0.000513049, 0.0093616, 0.000311395, 0.00559238, -0.000773172, 0.00604339, 0.000173766, 0.00613155, -0.000118518, -0.00262536, -0.000217933, -0.00497351, 4.04858e-05, -0.00913864, 3.908e-05, 0.00351705, -0.000565573, -0.00559131, 0.000551347, -0.00878557, -0.000223469, -0.00573333, 1.20288e-05, -0.0100133, -0.000128436, -0.00367997, -0.000161441, -0.000304215, 0.000417153, -0.000584424, -0.000243845, 0.00541658, 0.000253878, -0.0107421, -0.000344429, -0.00368566, -0.000299008, -0.00463923, 0.000598855, 0.00427111, -0.00041731, 0.00769739, 3.44788e-05, 0.00946319, 0.000291357, -0.00963625, 0.000301845, -0.102633, -0.000197572, 0.0125673, -0.000147768, 0.00356655, 0.000456402, 0.00137573, -0.000594284, 0.00476899, 0.000240814, 0.000241283, -0.000199831, 0.00228698, 0.000213886, 0.00829044, 0.000102995, -0.144543, 0.000229897, 0.0170037, 0.000265415, 0.009587, -0.000483225, -0.00370126, 0.00023902, -0.00424258, 0.000119875, -0.00967342, -0.000718621, 0.0114562, -7.05527e-05, 0.00852153, 0.000718953, 0.00945165, 0.000162964, -0.0981707, 0.000311366, -0.00131029, 0.000420306, 0.00304785, -0.000677161, -0.00270041, 0.000735319, -0.00606494, -0.000104975, -0.0062117, -0.000433614, 0.00616013, -0.000181317, 0.000441216, 0.000568277, 0.0109498,
0.00146742, -0.000379735, 0.00207123, -0.000799336, 0.000341834, -0.000177007, 0.00173978, -0.000530855, -0.00200172, -0.000312314, -0.00187333, 0.000327386, 0.000801456, 0.000385892, -0.0022694, 0.000576131, -0.00311959, -0.000176998, -0.00531381, 0.000412845, -0.000623275, 0.000699177, -0.000412665, 0.000535679, -0.0018933, 0.000238746, -0.00491115, 0.00024558, -0.00645138, -7.23043e-05, -0.00208753, 1.39594e-05, 0.00191064, 6.15329e-05, -0.00311687, 0.000365158, 0.0395798, 0.000245287, -0.00458548, 0.000424584, -0.00341433, -0.000453931, -0.00250201, -0.000652793, 0.00228272, -0.000166702, -0.00331556, 0.000343648, 0.000311019, 0.000280285, 0.0863723, 1.80278e-05, -0.0024607, 0.000109802, 0.00182954, -0.000202883, -0.00106344, -0.000352373, 0.000309802, 0.000162448, -0.00272687, 3.12741e-05, -0.000320627, -0.000346815, -0.000928963, 0.000264865, 0.109297, -0.0002039, 0.000260971, -0.000291176, 0.00574253, 0.000287557, 0.00141073, 0.000455021, -0.00190942, 0.000492049, -0.00183329, -0.000116397, -0.00157923, -0.000387092, 0.00071563, -0.000798762, -0.00186269, -5.47834e-05, 0.0840345, -0.000197229, 0.00238631, -0.000287593, 0.00645069, 0.000290727, 0.00281965, 0.000615431, -0.00230057, 0.000193605, -0.000765675, 0.000171621, -0.00222467, -5.25346e-05, 0.00113228, -0.000487207,
-0.00272546, -0.00762448, -0.0027206, -0.00787785, -0.000290135, -0.00233364, -0.000708108, -0.00683687, -0.000269278, -0.00504977, 6.47618e-06, 0.00921364, 0.000431708, 6.73402e-05, -0.000114461, 0.00964501, -0.000136209, -0.00113949, 0.000348721, 0.0104658, 5.86912e-05, 0.00310711, 0.000132434, 0.00856597, 5.90619e-05, 0.0326923, -2.32936e-05, 0.00774784, 6.03623e-05, -0.00220277, -0.000306537, -0.00056689, -8.72973e-06, -0.000312085, -0.000222962, 0.00930284, 0.000313218, -0.00169268, 0.000146134, 0.0102499, -0.000497813, -0.013291, 0.000126035, -0.00307484, -0.000368864, -0.00487214, -0.000507089, 0.00949107, 4.70005e-05, 0.015624, 0.000104663, -0.0844082, 8.99075e-05, 0.00160236, -0.000348686, -0.00929752, 0.000383186, 0.00155907, -0.000290823, 0.00116137, -0.000326699, 0.00153609, -0.000112898, -0.00836715, -3.93663e-05, 0.0169927, -0.000306205, -0.134928, -0.000128922, -0.00864305, 0.000453546, 0.00323859, -0.000181899, 0.00748081, 0.000310837, 0.00836856, 0.000150121, -0.00458825, -0.000248114, -0.00807323, -0.000701229, -0.00993242, 9.0469e-05, 0.00184214, -0.000294092, -0.0968994, -0.000144643, -0.00888345, 0.00069736, 0.00826492, -0.000574796, 0.00516653, 0.00069126, 0.00496437, 0.000203932, -0.00114736, -0.000159228, 0.000738442, -0.000619101, -0.00803807,
0.00439202, -7.02218e-05, 0.00446099, 0.000299481, 0.00376094, -5.07825e-05, 0.000652258, 0.000220874, 0.000811124, 3.64676e-05, 0.0047266, 2.6028e-05, -0.00273834, -6.19625e-05, 0.00140687, 0.000438103, -0.0333987, -4.08686e-05, 0.0031423, 5.15849e-05, -0.00314301, -0.000114782, 0.00120656, 0.000388948, 0.000453701, 5.99334e-05, -0.0312551, -0.000159255, 0.00182406, -1.54455e-05, -0.001303, 0.000126435, 0.000194911, 1.65226e-05, 0.000688437, -0.000108812, 0.000677672, 3.66939e-06, 0.0212949, -0.000168541, 0.00282587, -8.66667e-05, 0.000584823, 0.000253668, -0.000538474, -0.000138273, 0.000116947, 3.60752e-06, 0.000179647, 0.000359615, 0.000805417, -0.000107163, 0.0933168, -5.13816e-05, 0.00449964, -7.00808e-05, 0.00135038, 1.99276e-05, 0.000128727, 0.000130509, -0.00111486, 0.000264222, -0.000202349, 4.80561e-06, -0.000586936, 0.000415058, 0.000258517, -0.000114652, 0.123592, 1.92967e-05, 0.00362684, -5.68769e-06, 0.00134954, -0.000260028, 0.00177243, 0.000342824, -0.00184961, 0.000385982, -0.000586906, 1.98743e-05, -0.00150138, 0.000256455, 0.00128076, 0.000118215, -0.000707202, 3.85435e-05, 0.0852051, -7.65769e-05, -0.000307101, -5.07846e-06, 0.0014358, -0.0001569, 0.00257337, 3.90175e-05, -0.00142067, 0.00020927, -4.48608e-05, 8.01615e-05, -0.00243234, -9.46181e-05,
0.00038149, -0.0120279, 0.000379041, -0.00693047, 5.52613e-07, -0.0107341, 4.31448e-05, -0.00322868, 1.93211e-05, -0.0132674, 0.000160928, 7.82154e-05, -0.000229002, 0.00615166, -9.98956e-05, 0.0055357, 1.92466e-05, 0.0054783, 0.000218044, 0.00608463, -0.000220021, 0.00824554, -0.000158636, 0.00341381, 4.31746e-05, 0.0044474, -5.41536e-05, 0.0230926, -1.12488e-05, 0.00117583, -5.62517e-05, -2.73962e-05, 6.47133e-06, -0.00385793, -0.000213998, 0.00680969, 0.000196025, 0.00200403, -0.000110296, -0.0106023, -0.000206219, -0.00541867, -9.315e-06, -0.0071748, 0.000191907, -0.00736013, -0.000437961, 0.00582052, 8.09938e-05, 0.00788874, -6.49664e-05, -0.00520733, -5.86058e-05, -0.0876006, 4.06188e-05, -0.00410382, -0.000164714, -0.00371449, 9.25137e-05, -0.00265997, -0.000204484, -0.00207053, 0.000127926, -0.00628524, 6.32599e-05, 0.00958575, -0.000377115, -0.00864436, 1.89888e-06, -0.132108, 0.000468006, 0.00151326, -0.000309051, 0.00443913, -0.000166682, 0.00346166, 0.000275898, -0.00800612, 5.77151e-06, -0.00659462, -0.000350024, -0.015434, -7.34006e-05, 0.00375562, -0.000169598, -0.00339476, -7.70923e-05, -0.0914294, 0.000390539, 0.00192128, -0.000181994, 0.00520139, -0.000147525, 0.00240827, 0.000291242, -0.00444542, -0.00023453, -0.00073602, -4.27094e-05, -0.0064357,
0.00903507, 0.00047064, 0.00752265, 0.000343186, 0.0291041, 0.000219485, -0.000745253, 0.000131502, -0.0010671, 0.000584108, 0.0129848, -0.00030946, -0.00331255, -0.000308222, -0.00201037, -0.000724735, -0.00224365, 0.000317194, -0.0328219, -0.000245677, 0.00116015, -0.000303347, -0.00236482, -0.000493058, -0.0089454, -0.000751828, -0.0019044, -0.000374507, -0.0471915, 0.000369275, 0.00312864, 0.000159537, -0.00193102, 0.000256487, -0.0084882, -0.000936366, -0.00621524, -0.000556677, 0.00104187, -0.000484912, 0.00885097, 0.000596229, -0.0021762, 0.000403816, -0.00119015, 0.000427237, -0.000448362, -0.000717715, 0.000101914, -0.000893661, -0.00419838, 0.00029856, 0.0038739, 0.000205993, 0.0943094, 7.01623e-05, -0.00717058, 0.00012724, -0.000862306, -0.000253385, 0.00453334, 0.0002771, 0.000373321, 0.000956248, 0.00231918, -0.000719482, 0.00573095, 0.00071657, 0.00363473, 0.000701523, 0.128036, -0.000390561, -0.00358456, -0.000129177, -0.00110378, -0.00073912, -0.000550161, 0.000701347, 0.00115674, 0.000782814, 0.00845517, 0.000447905, 0.00337344, 0.000341587, 0.0145429, 0.000131171, 0.000784668, 0.000242929, 0.0812046, -3.24469e-05, 0.00459487, 0.000127318, -0.00126455, -0.000227112, -0.00839124, -6.37221e-05, 0.00202586, -0.000248826, 0.0103658, 0.0011135,
-0.00396331, 0.00807903, -0.00421054, 0.00473527, 7.1122e-06, -0.000362255, -0.00139285, 0.00141673, 7.22646e-05, 0.0118736, -5.46237e-05, -0.00372774, 0.000748009, -0.00583249, 0.000104691, -0.00445137, 3.90897e-05, 0.0133736, -8.73987e-06, 0.0188205, 0.00067798, -0.00679676, 0.00010525, -0.00192146, 6.16301e-05, -0.0102709, 1.04923e-05, 0.00526903, 0.000216549, 0.038994, -0.000238661, 0.000681525, 0.000105496, 0.00529363, 0.00027848, -0.00664592, -0.000116902, -0.0120757, -4.7676e-05, -0.00244908, 0.000342384, 0.00672829, -0.000376297, 0.00709443, 0.000119389, 0.00879671, 0.000417669, -0.00589238, -6.19443e-05, -0.00465529, 0.00014598, -0.0047844, -6.48192e-05, -0.00214387, 0.000106775, -0.0701156, 0.000175797, 0.00457981, 0.000134596, 0.0040334, 0.000232973, 0.0013487, -5.17125e-05, 0.00605642, -5.00585e-05, -0.00371901, 0.00034467, 0.00322551, -3.48065e-05, 0.00150503, -0.000208982, -0.115475, 0.000464062, -0.00221831, 0.000141847, -0.00299302, -8.64736e-05, 0.00683847, -6.05246e-06, 0.00606511, 0.000244216, 0.00666854, -9.56747e-05, 0.00148976, -1.2115e-05, 0.00327314, -2.06595e-05, -0.0001864, -6.77171e-05, -0.0755679, 0.000243634, -0.00347882, 0.000136935, -0.00380953, -8.43619e-05, 0.00400928, 1.1151e-06, -0.000351006, -5.5096e-05, 0.00598305,
-0.0240841, -0.000240583, -0.0137911, -0.000322999, -0.00638041, -0.000100867, 0.0131204, -0.000248194, 7.54463e-05, -0.000484927, -0.00476058, 0.000217921, 0.00698164, -1.03367e-05, -0.00105749, 0.000477554, 0.0005775, -0.000380522, 0.00159121, 0.000275079, -0.0333383, -8.40694e-05, -0.00265531, 0.000276747, 0.00382808, 0.00066907, 0.00146594, 7.9753e-05, 0.00335169, 4.44328e-05, -0.0513075, -0.000451339, -0.00301664, -0.000272734, 0.000221635, 0.000673855, 0.00653681, 0.00041217, 0.00221027, 0.000231864, -0.0024402, -5.15122e-05, -0.00371303, -0.000596045, -0.00154477, -0.000373627, -0.00812831, 0.000461564, -0.00234265, 0.000791344, 0.00710994, -0.000390926, 0.00219529, -0.000141457, -0.00730925, 0.000179098, 0.0769047, -0.000309896, 0.000290417, 0.000171379, -0.0119782, -0.000335366, -0.00280814, -0.000546258, 0.000215326, 0.000567675, 0.00146323, -0.000623722, 0.00148517, -0.000457678, -0.00342678, 0.000387621, 0.113046, -3.6099e-05, 0.000586229, 0.000605849, -0.00640249, -0.000672557, -0.000628771, -0.000470004, -0.00539649, -0.000696886, 0.000733201, -0.000385744, -0.00395655, 0.000136906, 0.000987641, -0.000228727, 0.00461682, 0.000243192, 0.0715532, -0.000184338, -0.000773169, 0.000329335, 0.00177308, -4.96583e-05, 0.00257365, 0.000100521, -0.00634348, -0.00062687,
0.00473022, 0.00474079, 0.00501714, 0.0117704, -0.000141675, 0.00142336, 0.00167242, 0.00449334, 6.04435e-05, 0.00766901, -0.000269247, -0.00581494, -0.000835215, -0.000573025, 0.000107284, -0.00907491, 7.90331e-05, 0.00579034, -0.000276192, -0.00675342, -0.00077993, 0.0191362, 0.000122992, -0.00892009, 2.49609e-05, -0.0037582, 8.63437e-05, -0.00318829, -9.59992e-05, 0.000740032, 0.00015063, 0.0373515, 0.000129388, -0.000895064, 0.000278125, -0.00943497, -2.96225e-05, -0.00217629, -5.15228e-05, -0.00847667, 0.00013947, 0.00713444, 0.000195085, 0.00480828, 0.00017719, 0.00528836, 0.000339093, -0.0112802, 6.96546e-05, -0.00477154, 0.000263214, 0.0036966, -0.000235371, -0.00361572, 0.000197079, 0.0045854, -0.000311557, -0.0709483, 0.000256687, 0.00292089, -0.000105842, -0.0054303, 0.000134281, 0.00556577, 5.51677e-05, -0.00423634, 0.000472913, 0.00748547, -0.000216699, 0.00443834, 7.31579e-05, -0.00222764, -0.000382672, -0.114923, 0.000262726, -0.00228185, -0.000589471, 0.000338753, 0.000172856, 0.00502571, 0.000286402, 0.00717882, 5.43126e-05, 0.00161811, 0.000117563, 0.00456433, 7.56641e-05, 0.00512888, 6.35367e-05, -0.00348444, -0.000179125, -0.0747845, 0.000114123, -0.000682082, -0.000343458, -0.00118839, 0.000148388, -0.000980692, 0.000128434, 0.00370945,
0.00142872, 9.06861e-05, 0.00242266, -0.000112331, -0.00104222, 6.58769e-05, 0.00245418, -7.01052e-05, 0.0281462, -0.000204217, -0.00197986, -2.63484e-05, 0.0011752, 3.56167e-05, 0.0189805, -0.000200937, -0.000164353, -3.1623e-05, -0.00236056, -5.22967e-05, -0.00071766, 2.83554e-05, -0.018554, -0.000109146, 0.00102981, -0.000138567, -0.00074395, 0.000305554, -0.00196888, 2.26554e-05, -0.00168135, -7.10439e-05, -0.0353127, 0.000132577, 0.00133624, -1.94263e-05, -0.000604703, 2.58158e-05, -0.00101161, 0.000342959, -0.00124656, 9.40462e-05, -0.000985916, -5.95178e-05, 0.00662755, 0.000176096, 0.000622289, -2.72504e-05, 0.000924635, -1.79537e-05, -0.00152657, 0.00033527, -6.66356e-05, -3.08471e-06, -0.000894941, 9.27123e-05, 0.000327107, 0.000124485, 0.0798445, -7.0539e-05, -0.000470843, -6.26841e-06, 0.0029294, -0.000157143, -8.74001e-05, 8.68223e-05, -0.0019062, 0.0004126, 0.00178649, -0.00025383, -0.001095, 9.29316e-05, 0.000536069, 0.000281131, 0.114142, -0.000279828, -0.000854232, 2.8928e-05, 0.00457617, -0.000160293, 0.000849961, 8.93264e-06, -6.28532e-05, 0.000222208, -0.00125155, 0.000156895, 0.0030045, -5.44553e-05, -0.00123215, 0.000172987, -0.000799775, 0.00023784, 0.0746844, -0.000174124, -0.000350811, 1.56784e-05, 0.00360847, -2.50185e-05, 0.0010442, 0.000327987,
0.00145888, 0.00496564, 0.00148862, 0.00758006, -6.10793e-05, 0.00250787, 0.000446239, 0.00430196, 1.16577e-05, 0.00552216, -0.000185289, -0.00352412, -0.000196839, -0.00535522, -4.57745e-05, -0.0141231, 3.47353e-05, 0.00339536, -0.000176991, -0.00376771, -9.10786e-05, -0.00840285, -5.05873e-05, 0.0057755, -0.000237192, -0.00809038, -8.5506e-05, -0.00387418, 6.53555e-05, 0.00336408, 6.33217e-05, -0.000820696, 7.25338e-05, 0.026634, -6.15758e-05, -0.00649233, -0.000315902, -0.00749608, -0.000132199, -0.00737221, 0.000242889, 0.008572, -6.36819e-05, 0.00680449, 0.000163782, -0.00378285, 0.000157404, -0.00585031, -0.000179191, -0.0087578, -6.23912e-05, 0.00119254, 6.41205e-05, -0.00266711, -7.11004e-06, 0.00469262, -2.28812e-05, 0.00471341, 3.0181e-05, -0.0799577, -5.57224e-05, 0.000170429, -0.000220102, 0.0113744, -0.000207092, -0.00964253, 0.000455723, 0.00835738, 0.000264924, 0.00346192, -0.000460411, -0.00300499, 0.000355433, -0.00228017, -0.000214059, -0.126307, -0.000489452, 0.00353204, -0.000122259, 0.013658, 0.000286327, 0.00605628, -0.00024254, -0.00221321, 0.000502323, 0.00580428, 0.000114223, 0.00241521, -0.000453646, -0.00453682, 0.00049225, -0.00262889, -0.000265455, -0.0880594, -0.000437641, -0.000293108, 4.68141e-06, 0.00541116, 3.73946e-05, 0.0140286,
-0.0218087, 0.000534288, -0.0253727, 0.000581249, 0.00785866, 0.000217306, -0.0108807, 0.00029003, -0.00039653, 0.00076415, 0.00838672, -0.000504878, 0.00583536, -0.000356209, -0.000454544, -0.000535906, 0.000451121, 0.000529909, 0.00346062, -0.000594423, 0.0125731, -0.000307989, -0.000400345, -0.000368053, -0.0386449, -0.000600726, 0.000684936, -0.000304676, 0.000663656, 8.79587e-05, 0.00783177, 0.000410136, -0.000758719, 0.000195935, -0.0356436, -0.000484056, -0.00594706, -0.000503397, 0.000111765, -0.000625932, 0.00316999, 0.000559344, -0.00125463, 0.000663941, -0.00154219, 0.000342323, 0.0243683, -0.000312589, 0.0015961, -0.000836058, -0.00644699, 6.53303e-05, -0.00112842, -5.58885e-05, 0.00475952, 0.00020772, -0.00658933, -3.37683e-05, -0.00184709, -0.000171192, 0.104241, 0.000295251, 0.00244033, 0.000441786, -0.000376291, -0.00089794, -0.00177842, 0.000296851, -0.00186798, 0.000471609, -0.000536234, -0.000293774, -0.00559213, -0.000706373, -0.000809716, -0.000598683, 0.134914, 0.000561144, 0.00217916, 0.0003549, 0.00257233, 0.000750639, 1.63525e-05, -0.000249265, 0.00163354, -0.000182095, -0.00140754, 0.000143835, -0.00840985, -8.04483e-05, -0.00219726, -0.000368204, 0.00109452, -0.000298664, 0.0886185, 0.000114601, 0.00154617, -0.000188885, -0.00358187, 0.000818288,
0.000321428, -0.00809831, 0.000448443, -0.00369985, -1.72271e-05, -0.00536744, 0.000235862, 0.001959, -2.86403e-06, -0.00852537, 0.000272886, 0.00337013, -0.000179156, 0.0108941, -6.94346e-05, 0.00636888, 0.000187867, -0.00689674, 0.000455433, 0.00539237, -0.000319715, 0.010298, -0.000120981, 0.00336955, 0.000218516, 0.00930812, 9.79968e-05, 0.00192407, 0.000132096, -0.001957, 5.4409e-05, -0.00123588, -7.3335e-05, -0.00552457, -4.2819e-05, 0.0279962, 0.000440955, 0.00949067, -4.57727e-05, 0.00626578, -0.000312369, -0.00715474, 0.000432519, -0.0105872, 3.02406e-05, -0.00952448, -0.000360029, -0.00761668, 7.88327e-05, 0.00979725, 0.000298575, 0.00152827, 7.01203e-05, -2.14085e-05, -0.0001669, -0.00175144, 0.000174581, -0.00796805, 7.49289e-05, -0.00384093, -0.000144232, -0.0883408, 8.33271e-05, -0.0149814, 0.00026268, 0.0114323, -0.000108602, -0.00459207, 0.000350639, -0.00801833, 0.000381843, 0.0068361, -0.000444004, 0.000348609, 2.54861e-05, 0.0035277, 0.00042205, -0.137109, 6.21348e-06, -0.0155527, -0.000266326, -0.00613242, 0.000348064, 0.00344137, -4.08486e-05, -0.00114005, 0.000356829, -0.00561451, 0.000333302, 0.00616346, -0.00047783, 0.00269663, -6.61112e-05, 0.00366086, 0.000503294, -0.100018, -5.26858e-05, 0.000906781, -4.67906e-05, -0.00928833,
-0.000688727, -6.92423e-05, -0.00145281, 3.98404e-05, -0.00038553, -6.91422e-05, -0.00181449, 0.000108365, 0.0053098, 0.000243378, -0.00109125, -1.75261e-05, -0.00104362, 0.000128283, 0.00659447, 1.76576e-06, -0.000658492, 0.000255238, -0.0015403, 1.48935e-05, 0.000549695, 2.40563e-05, 0.00213124, 7.4319e-05, -0.00104051, 0.000140644, -0.00100275, 3.2082e-05, -0.00117972, -2.22748e-06, 0.00154717, -0.000132426, -0.00489453, 0.000127, -0.00142184, 0.00016533, 0.000321709, 0.000177014, -0.0011658, -0.000197274, -0.000259014, -2.91748e-05, 0.0010341, -0.000165456, -0.00783799, 5.36804e-05, -0.000653723, 0.000150173, 0.0502444, 3.61821e-05, -0.000305791, 3.36841e-05, -0.00118444, -0.000201511, 0.000600364, -2.85967e-05, -0.000197066, -9.13015e-06, -0.00333706, -0.000103213, 0.000937526, 9.04158e-06, 0.105144, -0.000134607, -6.09867e-05, 7.42429e-06, -0.00157434, -0.000185224, -0.000592239, -2.5783e-05, 0.00114935, -1.41367e-05, -0.000571122, 0.000186562, 0.00456898, -0.000172395, 0.00219521, -6.93192e-05, 0.126155, -0.000155261, -0.00198044, -7.62802e-05, -0.000429873, -1.77598e-05, -0.00223789, -0.000264191, 0.000958755, 0.000102604, 0.00120562, -5.03299e-06, 0.00025717, 0.000209816, 0.0080621, -4.92431e-05, 0.00218233, 1.06458e-05, 0.0852538, -5.43923e-05, -0.00337294, 0.000113219,
1.24619e-05, -0.00744046, 0.000285779, -0.0120151, -0.000757982, -0.00373182, 0.000464833, -0.00854941, -6.40153e-05, -0.00891989, -0.000359662, 0.00570828, 9.34738e-05, 0.00482017, -0.000114639, 0.00874205, -0.000336659, -0.00661738, 0.000463614, 0.00727883, -0.000377133, 0.010288, -9.56859e-05, 0.00420511, 0.000395426, 0.00947451, -0.000236434, 0.00378609, 0.000397059, -0.00160498, -0.000208731, 0.00149358, 2.10568e-05, -0.00726215, 4.37763e-05, 0.0132851, 0.000184413, 0.00832354, 0.000104072, 0.00943492, -0.000457407, -0.00819656, 0.000339466, -0.00807591, 0.000153741, -0.00919699, -0.000517195, 0.00940022, 1.77455e-05, 0.00827986, -9.53862e-05, -0.0015647, 0.000225477, 0.00302273, -0.000617617, -0.00304307, 0.000317577, -0.00509218, 0.000150513, 0.00325713, -0.000403619, -0.00739368, -6.26327e-06, -0.102457, -5.67817e-05, 0.00851566, -0.000430366, -0.0080882, 1.02102e-05, -0.00661847, 0.000346431, 0.00606674, -0.00032724, 0.0050056, -3.77273e-05, 0.0136659, 0.000292733, -0.0155698, -8.29411e-05, -0.143485, -0.000658715, -0.00941882, -0.000247596, 0.000449375, -0.000184996, -0.00284589, -0.000159232, -0.00669278, 0.000928139, 0.00576879, -0.000519085, 0.00717794, -0.000243718, 0.00624658, 0.000496701, -0.00109657, -0.000127782, -0.0965616, -0.000419181, -0.00725454,
0.0104544, 0.000120275, 0.00529007, 0.000133585, 0.0170791, -6.95094e-05, -0.0081745, -0.000104859, -5.79866e-05, 5.91459e-05, 0.0113349, -0.000188016, -0.00917327, -0.000256649, 0.000619186, -8.56094e-05, -0.000423788, -0.000230557, -0.00836395, 0.000115417, 0.00308025, 8.97257e-05, 0.000939455, 0.000183452, -0.0221715, 0.000241232, 0.000323264, -0.000348738, -0.0155787, 0.000390012, 0.00997055, 0.000413984, -2.98266e-05, 0.00033688, -0.0156663, -8.38044e-05, -0.017709, 0.000500791, 0.00097534, 0.00012059, -0.000541393, 1.7921e-05, 0.00178088, 3.93899e-05, -0.00119536, -2.14754e-05, 0.0108865, 0.000181071, -0.00027804, -9.37677e-05, -0.0172925, 0.000368163, 7.9853e-05, 0.000619067, 0.0148047, -0.000578819, -0.00845893, -0.000546911, -0.000750729, -0.000437673, 0.023201, 0.00044438, -0.00135155, 7.70928e-05, -0.000665061, 0.000193728, 0.000777486, -0.000245144, -0.00149592, 0.000265576, 0.00840849, -0.000368795, -0.00543273, -0.000287607, 0.000854994, -0.000189124, 0.00252504, 0.00018475, -0.00198219, -8.04261e-05, 0.0876225, 0.000443089, -0.000809076, 0.000486477, 0.0160477, -0.000686836, -0.00132115, -0.00062966, -0.00983799, 0.000552737, 0.00555646, 0.000605414, 0.00123649, 0.00044867, -0.0242705, -0.000443151, -0.00087329, -0.000364917, 0.108472, 0.000425453,
0.000783929, -0.0097245, 0.000775414, -0.00762805, -0.000266798, -0.00564218, 0.000115004, -0.00389346, 0.000436714, -0.0143603, -2.9416e-05, 0.00581214, -0.000327494, 0.00553286, 0.000264009, 0.0083952, -3.0744e-05, -0.00968815, 0.000261821, 0.00831401, -0.000134161, 0.00683286, -0.000236218, 0.00636684, 0.000257309, 0.00947563, -0.000178524, 0.00744974, 7.2841e-05, -0.00132212, 0.000302165, -0.00218536, -0.000367898, -0.00275858, -0.000128498, 0.00706853, 0.000149819, 0.00735494, -0.000191948, 0.0144246, -0.000295253, -0.00850035, 0.000335665, -0.00821784, 2.96015e-05, -0.00716954, -0.000512956, 0.00681047, 0.000743531, 0.007983, -0.000164404, -0.00370787, 5.46676e-05, 0.000578923, -0.000149581, -0.00284815, -0.000114247, -0.00212801, 0.000336494, -0.00122858, -0.000131152, -0.000335252, 0.000617256, -0.00906887, 0.000162644, 0.00952551, -0.000614568, -0.00992783, 0.000251159, -0.0154631, 0.000342103, 0.00664409, -0.000458282, 0.00717922, 0.000121685, 0.00604749, 0.000607011, -0.00612759, -0.000158363, -0.00940429, -0.000173314, -0.0767808, 4.62863e-05, 0.0027704, -0.000438305, -0.0023535, 5.93969e-05, -0.0105681, 0.000309015, 0.00577564, -0.000192324, 0.00585368, -0.000211797, 0.00400555, 0.000510995, -0.00333928, -0.000632985, -0.000258759, 0.000187796, -0.00593454,
-0.00290362, 0.000150743, -0.002926, -0.000248142, 0.00366199, 0.000115952, -0.000384283, -6.36109e-05, -0.000460184, 0.000117277, 0.00232258, -4.62752e-05, 0.00171076, 0.00027277, -0.00121602, -0.000489919, 0.00361585, 0.000288846, -0.00230364, -0.000124794, 0.00145865, 0.000233487, -0.00130683, -0.000362779, 0.000906162, -8.71436e-05, 0.00112371, 0.0002828, -0.00428266, -3.36575e-05, -0.000425337, -0.000148343, -4.15165e-05, 0.000217408, 0.000222021, -0.00044027, -0.00132001, 0.000109283, -0.00479448, -4.04208e-05, -0.000750709, 6.86305e-05, -0.00166487, -0.000314812, 0.00152917, 0.000539044, -0.0014859, -0.000318077, 0.00206585, -0.000207031, -0.00212284, 0.000329867, -0.00592588, -0.000383765, 0.00378777, 1.6282e-05, -0.000909611, -5.48768e-05, 0.00155986, 0.000188894, -0.00201061, 0.000141991, 0.00195502, -0.00052779, 0.0910372, -0.000136684, -0.00187763, 0.000214163, 0.00130282, -0.000283424, 0.00337464, -0.000108358, 0.000821205, 0.000190614, -0.000105319, -0.000350738, -2.77753e-05, 0.000344258, -0.000442438, -0.000425519, -0.000816456, -0.000213634, 0.132083, 0.00023872, -0.000880637, -0.000133243, 0.00895646, 0.000214387, -0.00134281, -8.21133e-05, 0.00130468, 6.4662e-05, -0.00151431, -0.00038286, 0.0026173, 6.86463e-05, -0.00269776, 9.55556e-05, -0.00122887, 1.03331e-05,
-0.000247606, 0.00252773, -0.000597979, 0.00221451, 0.000372994, 0.00380151, -0.000785327, 0.00183355, -3.14253e-05, -0.00151327, 0.000146475, 0.00270237, -0.000416021, -0.00084589, -0.000104417, 0.00137458, 0.000114119, -0.00164718, -0.000379167, -0.000302409, 0.00027207, -0.00288891, -0.000185029, 0.00234345, -0.000443858, 0.00657819, -1.26968e-05, 0.000789107, -0.000468549, -0.00273737, 0.00052768, -0.00186826, -0.000183202, 0.00224339, -0.00039502, -0.000841761, -0.000479904, 0.0119117, -0.000219536, 0.00480635, 5.99769e-05, -0.00241867, 7.40647e-05, 0.00112442, -4.72336e-05, 0.000809147, -2.55442e-05, 0.00028343, 3.13334e-05, -0.0473449, -0.000476469, 0.0112161, -0.000199954, 0.00677939, 0.000507013, -2.91092e-05, -0.000450979, 0.00277256, 0.000138571, -0.00119447, 9.20508e-05, 0.00257747, 6.98291e-05, 0.00045581, 0.000419047, -0.0985956, -3.79369e-05, 0.00186032, 8.53691e-05, 0.00375193, 0.000290905, 0.00148791, -0.000336023, 0.00162014, 0.000204905, -0.00224745, -0.000263203, 0.00347602, 3.4051e-05, 0.000483589, 0.000533442, 0.00271062, 0.000465973, -0.118637, 0.00031581, -0.0112587, 0.000301343, -0.00280467, -0.000197401, 0.00049656, 0.000215929, -0.000383026, 8.6675e-05, -0.00129735, -0.000543742, 0.00157039, -8.68079e-05, 0.000349881, 0.000606782, 0.000959222,
0.0130984, 0.00016048, 0.00964954, 0.000101121, 0.0127027, -4.96969e-05, -0.00399924, -0.000202225, -0.00147012, 0.000232561, 0.00427758, -0.000193003, -0.00752828, -0.000378521, -0.000976316, 0.000262966, -0.00343434, -5.1808e-05, -0.0156213, 0.000148498, 0.00124504, 8.29691e-05, 0.000820535, 0.00052363, -0.0184347, -0.000161208, -0.00477049, -0.000333923, -0.021137, 0.000483451, 0.0062617, 0.000594667, 0.00169347, 0.000519099, -0.0172649, 2.87253e-05, -0.0432059, -1.92997e-06, -0.00387461, -8.38695e-06, -0.00333654, 9.98848e-05, -0.00116831, 0.0002343, 0.000620704, 0.00018342, -0.000194793, 0.000297231, -0.000733744, -0.000220305, -0.0101683, 0.000209514, -0.00216814, 0.000552322, 0.0160034, -0.000598144, -0.00924426, -0.000601198, -0.000895814, -8.50064e-05, 0.0109087, 0.000512191, -0.00176206, 0.000343935, -0.000115066, -6.79271e-05, 0.0843915, 8.33002e-05, -0.000724659, 0.000376372, 0.014543, -0.000509076, -0.00414336, -0.000541901, -0.00126589, 8.81606e-05, 0.00157783, 0.000344621, -0.00223453, 0.000220257, 0.0160446, 6.50254e-05, -0.000884573, 0.000230722, 0.151042, -0.000291785, 0.000932602, -0.000519233, -0.000439306, 0.00037078, 0.00835125, 0.000555604, -0.00112346, 0.000387151, -0.0137264, 4.7422e-07, -0.00143586, -0.000111554, 0.0175916, 0.000497956,
-0.00211592, -0.000352046, -0.00188637, -0.00680352, -0.000594328, 0.00334357, -6.9194e-05, -0.00717755, -0.00041973, -0.00404833, -0.000267243, 0.0069194, 0.000657429, -0.00228052, -0.000309832, 0.00427892, -2.29546e-05, -0.00265005, 0.000441518, 0.00365772, -0.000154486, 0.00159367, 8.28197e-05, 0.00395781, 0.00053578, 0.0379729, 6.30432e-05, 0.00309049, 0.000448222, -0.00529379, -0.000726293, 0.00118107, 0.000165561, 0.00098047, 0.000247125, 0.00435073, 0.000300659, 0.0366374, 0.000170098, 0.00687439, -0.000303316, -0.0103802, -5.14487e-05, -6.97399e-05, -0.000162686, 0.000547294, -0.000511301, 0.00483011, 6.04818e-05, 0.0028405, 0.000161998, -0.0200464, 0.000145994, 0.00381582, -0.000651805, -0.00562432, 0.000719514, 0.00171501, -0.000312914, 0.00392898, -0.000769409, 0.00157224, -3.46388e-05, -0.0031287, -0.000137625, -0.00127049, -0.000264091, -0.097281, 1.61497e-06, -0.00337956, 6.47366e-06, 0.00329605, 0.000350655, 0.00454612, 9.74149e-05, 0.00583009, -0.000190459, -0.0011813, -0.000239175, -0.00287832, -0.000865878, -0.00235479, -0.000106244, -0.0112606, -0.000437224, -0.12827, -4.92216e-05, -0.00775484, 0.000718538, 0.00664096, -0.000480361, 0.00358467, 0.000611543, 0.00216422, 0.000368682, 0.00026077, -0.000352752, 0.00032956, -0.00100839, -0.00283078,
0.00305964, -6.01898e-05, 0.00225966, 1.81231e-05, 0.00214544, -7.14217e-05, -0.00134838, 0.000113215, 0.00171296, -0.000132249, 0.00405476, -2.77762e-05, -0.00323932, 0.00015216, 0.00179128, 4.94461e-05, 0.00517722, -9.49818e-05, 0.00523985, 1.06813e-05, -0.00192844, 3.3957e-05, 0.000418613, 0.00022809, 0.000745674, -6.3028e-05, -0.0112649, -1.67907e-05, 0.00527257, -2.22745e-05, 0.000195062, -7.72582e-05, -0.000604021, 0.000296263, 0.000860955, -4.53111e-06, 0.0025786, -3.68223e-05, -0.0206879, -6.83138e-05, 0.00410736, -9.61869e-05, 0.00081477, 2.44822e-05, 0.000204584, 9.14533e-05, 0.000167836, -4.549e-05, -8.43685e-05, 0.00019868, 0.00338689, 1.15919e-05, 0.0151961, -0.000187245, 0.00228409, -0.000106204, 0.000482812, 0.000211642, 0.00207392, -0.000140435, -0.000773596, 2.44445e-05, 7.30167e-05, -5.64723e-05, 0.00562907, 0.000318057, 0.00236347, 1.16836e-05, 0.0855551, -0.000168605, 0.000761369, -2.5644e-05, 0.000922562, 0.000169314, 0.00295791, -7.12435e-05, -0.00135542, 0.000225906, 0.00096276, -0.000107077, -0.00133723, -0.000156038, 0.0089305, 0.000262896, 0.000922259, 1.62706e-05, 0.125638, -2.74066e-05, -4.35375e-05, 3.38289e-05, 0.00225648, -8.28807e-05, 0.00205674, 0.000141363, -0.00197318, 0.000324472, 0.00235918, 3.68065e-05, -0.00121391, 1.00355e-05,
0.000275378, -0.00370277, 0.000452663, -0.00412657, -0.000585277, -0.00189121, 0.000374727, -0.00137804, 0.000317563, -0.0285057, -0.000249004, 0.00243092, -1.49676e-05, 0.00451733, 0.000159708, 0.00518793, 0.000135293, -0.0184307, 0.000486579, 0.00231858, -0.000422014, 0.00495705, -0.000207684, 0.00420186, 0.000821946, 0.00455, 5.81154e-05, 0.0196092, 0.000548994, -0.00312642, -0.000445447, -0.00113214, -0.000227693, -0.00143175, 0.00051076, 0.00571251, 0.000803049, 0.00332044, -0.000138066, 0.0336649, -0.000148525, -0.0069812, -0.000136902, -0.00507993, 0.00014983, -0.00482422, -0.000452536, 0.0052351, -1.01927e-05, 0.00404764, 0.000580582, -0.00323861, -0.000213106, -0.0143136, -0.000505814, -0.00455257, 2.5194e-05, -0.00103424, 0.000344527, -0.001967, -0.000751442, -0.00100117, 4.0505e-05, -0.0055064, 0.000480669, 0.00304601, -0.000383354, -0.0088581, -9.54851e-05, -0.0917907, 6.09435e-05, -0.000164889, -0.000167113, 0.00516845, 3.3592e-05, 0.00241177, 3.97368e-05, -0.00565526, 8.66172e-05, -0.00673055, -0.0010838, -0.0105907, 0.000458861, -0.00280303, -0.000893963, -0.0077559, 2.82545e-06, -0.124487, 0.000670318, -0.000223446, -0.000304026, 0.00426714, -0.000315977, 0.00151515, 0.000720583, -0.00191412, 4.03643e-05, -0.00296977, -0.000973487, -0.00640755,
0.000194981, 0.000188858, 0.00468534, 0.000235386, -0.0150427, 0.000195961, 0.00928677, 0.000182334, -0.00218721, 0.000102245, -0.00762595, 2.92292e-05, 0.00554867, -6.47027e-05, -0.00286592, -0.000333474, -0.00106661, 0.000156167, 0.00083882, -8.08487e-05, -0.00467822, -0.000185011, -0.00203606, -0.000449884, 0.00916653, -0.000179905, -0.0014664, 1.80035e-05, -0.0140138, 8.55298e-05, -0.00994223, -2.71353e-05, -0.00101133, -0.000260167, 0.00622898, -0.000178918, 0.013081, -0.000341725, -0.00118996, -0.000234783, -0.0299371, 0.000401897, -0.00447356, 0.000229415, -0.00090661, 4.66202e-05, -0.00622798, -0.000382318, 0.000750803, 1.55142e-05, 0.0150562, -0.000179312, -0.000679435, -0.000241067, 0.00280304, 0.000462214, 0.00378885, 0.000320887, -0.00132799, 0.000101758, -0.0142972, -0.000389159, 0.00115296, 0.000177362, -0.00125031, -0.000157781, 0.00644395, 0.000313245, -0.000300025, 0.000154268, 0.081565, 0.000112696, 0.0043856, 0.000193839, -0.00123461, -0.000199716, -0.00842919, -9.14274e-05, 0.00119818, 0.000438869, -0.00980416, -9.98484e-05, -0.00135465, -0.000173069, -0.000427408, 0.000578241, -2.29118e-05, 0.000509209, 0.131949, -0.000217962, -0.0011851, 4.82048e-05, -0.00035645, -0.000463565, 0.00310734, 0.000202611, 0.000785023, 0.000470435, -0.0100047, -2.79735e-05,
-0.00554942, 0.0254189, -0.00606364, 0.00464468, 0.000555729, 0.013905, -0.00228231, 0.00191127, 0.000140431, 0.00643366, 0.000320658, -0.0146688, 0.000853943, -0.00436418, 0.000155466, -0.00538637, 9.44034e-05, 0.00600896, -0.000251733, -0.0119897, 0.00106903, -0.00496508, 6.51811e-05, -0.00362612, -0.00079937, -0.00801039, 4.98576e-05, 0.000239443, -0.000249349, 0.0283459, -0.000127876, 0.00163719, 7.29336e-06, 0.00304143, -0.000468257, -0.00581565, -0.000919557, -0.00764269, -7.60166e-05, -0.00356116, 0.000370602, 0.0499659, -0.000553499, 0.0065085, 7.39406e-05, 0.00628919, 0.000535962, -0.00519025, -0.000102259, -0.00299521, -0.000673333, 0.000784898, -0.000126479, -0.00129886, 0.000648238, 0.0048222, -6.12603e-06, 0.00303701, 0.000185085, 0.00174424, 0.000894507, 0.00122173, -9.5119e-05, 0.00562392, -2.36989e-05, -0.00268887, 0.000348997, 0.0082693, -4.8421e-05, 0.00193098, 0.000129936, -0.0758953, 0.000453034, -0.00352461, 0.000213849, -0.00456242, 0.000124132, 0.00616598, 9.15678e-06, 0.0057558, 0.00114124, 0.00575712, -2.08918e-05, 0.000496858, 0.000768254, 0.00662493, 2.76244e-05, -0.000229244, -0.000361771, -0.110735, 0.000364204, -0.00391961, 0.000170447, -0.0046365, -0.000645863, 0.00357967, 8.62243e-05, 0.000556673, 0.000906924, 0.00579878,
-0.00280671, -0.000197963, -0.00898078, -0.000121738, 0.00602264, -0.000175599, -0.00981577, -0.000171631, 0.00138634, -0.00018747, 0.00265861, 7.24366e-06, -0.000756178, -0.000152556, 0.00134765, 0.000216848, 0.00242206, -0.000277267, -0.00653851, 0.000138127, -0.00134725, -0.000165066, -8.88981e-06, 0.000259642, -0.0131978, 0.000291853, 0.00337918, -0.000228396, -0.0103978, 0.000126263, -0.0251461, -0.000303632, -0.00144567, 4.95302e-05, -0.0164725, 0.000184187, -0.00542073, 0.000400117, 0.00267334, -0.000100115, -0.00397913, 7.73826e-05, -0.0417371, -0.00049768, -0.00191899, -0.000207446, -0.00942016, 0.000306723, -0.000152966, 0.000187758, -0.00428026, 0.000181232, 0.00153493, -2.42637e-05, 0.00430721, 0.000117919, -0.00580794, -0.000544285, -0.00151295, -0.000142361, -0.00156745, 0.000159462, -3.51304e-05, -4.49692e-05, 0.000998201, 0.000324579, 0.00279978, -0.000198434, 0.00138554, -6.88617e-05, 0.00448352, 0.000252329, 0.0719455, -0.000368439, -0.00083296, 0.000284826, -0.00186772, -0.000213047, 0.00027805, -0.000178337, 0.00559009, 5.17912e-05, 0.00135122, 0.000197923, 0.00827937, -0.000274873, 0.00222278, -0.000144927, -0.00125404, 0.000366385, 0.118429, -0.000182736, -0.000270665, 0.00061095, -0.00640729, -0.000372695, 0.000481214, -0.000138574, 0.00467579, 7.29822e-05,
0.00589168, 0.00465185, 0.00612028, 0.0273672, 0.000433909, 0.0019203, 0.00183752, 0.0180056, -9.382e-05, 0.0057941, 0.000162398, -0.00434204, -0.00112952, -0.00970777, 6.81553e-06, -0.00576317, -0.000108238, 0.00360978, -0.000422667, -0.00491259, -0.000719445, -0.00893371, 0.000214756, -0.00325414, -0.000837004, -0.00526339, -6.72298e-06, -0.00385471, -0.000473809, 0.00170863, 0.000573778, 0.0277197, 0.000312859, 0.00483998, -0.000451742, -0.00462035, -0.0011209, -0.00495803, 0.000151643, -0.00716902, 0.00013006, 0.00656103, 0.000389188, 0.0469866, 0.000240053, 0.00887511, 0.000555021, -0.00410258, -7.34566e-05, -0.0055142, -0.000717703, 0.000544928, 0.00011599, -0.00169474, 0.000549536, 0.00305246, -0.000532625, 0.00279946, 0.000164986, 0.00396876, 0.000733702, 0.00036109, 1.01419e-05, 0.00643031, -0.000126139, -0.0060634, 0.000663237, 0.00516249, -8.54472e-05, 0.00524614, 0.00025426, -0.00352465, -0.00056458, -0.0751512, 0.000204533, -0.00262676, -0.000265501, 0.00268952, 0.000192259, 0.00717714, 0.000940529, 0.00591925, -3.11513e-05, -0.000387782, 0.00118384, 0.00359818, -0.000132313, 0.00426475, -0.000162573, -0.00391405, 1.49873e-05, -0.108849, 0.000233237, -0.00145351, -0.000989896, -0.000773298, 0.000271948, 0.00117399, 0.000895063, 0.00358047,
-0.00425104, 3.66089e-05, -0.00400431, -7.09625e-05, -0.00215578, 4.60148e-05, -0.000154666, -0.000104688, 0.00641502, -1.37533e-05, -0.00281325, -5.74889e-07, 0.00249359, -7.2286e-05, 0.0156742, -0.000135754, -0.00159247, 3.30193e-05, -0.00199912, -7.71624e-05, 0.0019345, -7.55565e-06, 0.00857599, -0.000182787, -0.0010432, -6.98367e-05, -0.00194467, 0.000193575, -0.00102819, -0.00010149, -3.18801e-05, 5.18802e-06, -0.0204455, -9.00926e-05, -0.00148565, -0.000158163, -0.0022291, -2.50401e-05, -0.000581761, 0.000354704, -0.000970488, -3.39364e-05, -0.00111426, -4.89314e-05, -0.0363753, 4.12472e-05, -0.00105565, -0.000182979, 0.00468724, -9.46341e-05, -0.00313058, 0.000220913, 0.00149765, 0.000287898, -0.00138795, 6.74484e-05, -0.00103632, -6.12855e-05, 0.00118084, 1.63645e-05, -5.78853e-06, -0.000101079, 0.00829506, -0.000152965, -0.00105501, -4.57764e-05, -0.00229109, 0.000520954, 0.0025434, -1.88894e-05, -0.00125099, 0.000133143, -0.000642033, 5.8647e-05, 0.0750339, -0.00019098, 0.00104523, -3.25733e-05, 0.00805447, -0.000269743, 0.0012364, -0.000132392, -0.00146824, 9.97719e-05, -0.0011187, 0.000552951, 0.00206038, -0.000236575, -0.000339552, 0.000173961, -0.000333661, 0.000218864, 0.11412, -0.000350278, 0.00155965, -5.74919e-05, 0.00351461, -0.000297972, 0.00115239, 0.000106934,
0.0022023, 0.00577135, 0.00209423, 0.00219144, 0.000568388, 0.00468212, 0.000340765, -0.0018599, 0.000160013, 0.0051684, 0.000198564, -0.000358573, -0.000525847, -0.00766288, 5.9155e-05, -0.0315585, 0.000280218, 0.00417499, -0.000560741, -0.00140976, 2.18843e-05, -0.0063416, -0.000128774, -0.0203699, -0.000647998, -0.0055766, 7.2439e-05, -0.00145275, -0.000598395, 0.00323142, 0.000506224, 0.00249746, -9.62266e-05, 0.022731, -0.000423833, -0.00398389, -0.000961174, -0.00470026, -0.000365991, -0.00483544, 0.000162997, 0.00630954, 7.57454e-05, 0.0089229, 0.000129037, 0.0413609, 0.000275818, -0.00395694, -4.39853e-06, -0.0056568, -0.000811281, 0.00123491, -0.000442741, -0.00197364, 0.000556206, 0.00241991, -0.000356447, 0.00601863, 0.00017599, -0.0068786, 0.000430731, -3.00424e-05, 9.96617e-06, 0.0062496, -4.53321e-05, -0.00625309, 0.000186228, 0.00499297, -3.14186e-05, 0.00240641, -4.479e-05, -0.00384572, 8.96976e-05, -0.000677958, -0.000103962, -0.0884251, -0.000236264, 0.00366165, 2.01845e-05, 0.00625766, 0.000694707, 0.00400433, -0.000257863, -0.00127803, 0.00073333, 0.00216331, 0.000261707, 0.00151393, -0.000701519, -0.00464016, 0.000744553, -0.00145369, -0.000376785, -0.123978, -0.000731956, 0.00333472, 9.50836e-06, -0.000369471, 0.000530412, 0.0072473,
-0.0300046, 0.000236336, -0.0271102, 0.000120587, -0.00971016, 0.000255653, 0.000118714, 0.000216161, 0.00174934, 0.000316124, -0.00396198, -2.39039e-05, 0.0146903, 0.000135096, 0.00189301, -4.06674e-05, 0.000734589, 0.000476128, 0.0105917, -0.000355102, 0.00548151, -7.48595e-05, 0.000186743, -0.000193367, 0.0150053, 1.62815e-05, 0.000850063, 0.000284674, 0.0152145, -0.000303881, -0.00703368, -5.6789e-05, -0.0020696, -0.000218602, -0.00451857, 0.000158562, 0.0137757, -0.000153044, 0.00015604, -0.00016317, 0.00302859, 0.000174567, -0.00494571, 0.000313661, -0.00309845, 7.36143e-06, -0.0246904, 3.27223e-05, -0.000597171, 0.000241132, 0.0134749, -0.000312964, -0.000808312, -0.000393281, -0.0103004, 0.000510016, 0.00374733, 0.00052616, -0.00227205, 0.000166884, 0.00821444, -9.64035e-05, 0.000176003, -5.13232e-05, 0.00143277, -1.08657e-05, -0.000775579, -0.000188787, -0.00142084, -8.26673e-05, -0.00832301, 0.000197409, 0.00256822, 7.69082e-05, -0.000311002, -8.67365e-05, 0.088819, 0.000100693, 0.00214967, 0.000119083, -0.0242017, -4.78493e-05, 0.00263549, -0.000488098, -0.0136861, 0.000103868, -0.00201671, 0.000402591, 0.00319444, -0.000402819, -0.00675282, -0.000608293, 0.00156906, -0.000479389, 0.137414, 0.000424318, 0.0037102, 0.000220942, -0.0335692, -3.27459e-05,
-0.000390385, -0.00419245, -0.000165244, 0.00463766, -0.000649373, -0.0034509, 0.00030209, 0.00891972, -9.56469e-06, -0.00486954, -0.000255262, 0.000236014, 0.000177435, 0.00953538, -3.866e-05, 0.00191778, 0.000352085, -0.00458895, 0.000616099, 0.00124849, -0.000359047, 0.00432443, -7.8896e-05, 0.000316097, 0.000508671, 0.00435867, 0.000351668, -0.000314538, 0.000729206, -0.00168174, -0.000410357, -0.00326881, -9.35091e-05, -0.00399628, 0.000280177, 0.0361052, 0.000753367, 0.00483689, 9.00497e-05, 0.00206994, -9.83452e-05, -0.00346342, 0.000182097, -0.00687758, -5.60254e-05, -0.00686888, -0.000350767, 0.0335984, 0.000135123, 0.00554639, 0.000758179, 0.0015697, -3.9452e-05, -0.000873753, -0.000653881, -0.000362178, 0.000528334, -0.00463386, 1.89255e-06, -0.00509706, -0.000502604, -0.0232375, 0.000140763, -0.00471765, 0.000158641, 0.00619461, 0.000163966, -0.00119003, 0.000160643, -0.0044856, -0.000173308, 0.00401235, 7.29215e-05, -0.00118065, 1.02315e-05, -0.000289107, 9.9299e-05, -0.100393, 2.79945e-05, -0.00110302, -0.000735238, -0.00337185, 0.000143898, 0.00155574, -0.000181167, 0.000258255, 0.000363731, -0.00192114, 0.000471423, 0.00358078, -0.000487961, -0.000765544, -5.93591e-05, 0.00333523, 0.000623881, -0.132032, -6.38767e-05, 0.0102283, -0.000760408, -0.00320448,
-0.00499635, -3.11635e-05, -0.0049576, -1.355e-05, -0.000226536, -6.56305e-05, -0.000943477, 6.90392e-05, 0.00341493, -0.00011125, -0.00129011, -6.87625e-05, 0.0016035, 0.000158808, 0.00293941, 2.99468e-06, -0.000302506, 2.56838e-05, -0.00239813, -2.92785e-05, 0.000675985, 0.00010592, -0.00125993, 5.99279e-05, 0.00162921, -0.000143873, -0.00091237, 0.000193254, -0.00221197, -1.69945e-06, -0.000929299, -7.36247e-05, -0.00504263, 0.000101725, 0.00150251, 0.000109691, -0.000727297, -1.45121e-05, -0.00163323, 7.69087e-05, -0.00050103, -3.3757e-05, -0.000147805, -0.00017956, -0.00432893, 6.19226e-05, 0.000115344, 8.90115e-05, -0.010716, 1.04005e-06, -0.00174542, 0.000126365, -0.00160396, -0.000245043, 0.00143138, -7.93602e-05, 0.00215628, -7.00244e-05, 6.53283e-05, -2.27655e-05, -0.000311645, -1.91027e-06, 0.0239403, -0.000171288, -0.00187169, -0.000115253, -0.00221683, -2.96772e-05, -5.05548e-05, -0.000331818, 0.00202168, -5.31607e-05, 0.00270744, 0.000152946, 0.00360343, -6.22758e-05, 0.0015228, -4.35703e-05, 0.0856005, -0.00014992, -0.000888089, -0.000497101, -0.0026851, -0.000155979, -0.00143323, -0.000354209, 0.0023513, -9.48721e-06, 0.000791605, 3.66081e-05, 0.00042186, 0.000249149, 0.00350101, -1.89103e-05, 0.00371846, -7.07912e-06, 0.115562, 1.08237e-05, -0.00118103, 0.000169969,
-0.000217152, 0.000350575, -5.99518e-05, -0.00435742, -0.000971824, 0.00121273, 0.000275551, -0.00470454, 5.10652e-05, -0.000461642, -0.000793962, 0.00178347, 0.000267105, -0.000969169, -3.86836e-05, -0.0034338, -0.000133406, 0.000382071, 0.000167454, 0.00108736, -8.14101e-05, 0.00300742, -0.000154881, -0.00491455, 0.000247862, 8.99213e-05, -0.000163359, 0.0022386, 0.000645146, -0.00047787, -0.00027206, 0.00337734, -9.94374e-05, -0.00274883, 0.00011621, -0.00255169, 0.000250826, 0.00025851, -7.14206e-05, 0.00335999, 8.4151e-05, -0.00149684, -6.99109e-05, 0.000527674, 0.000117988, 0.00229861, -0.000258427, -0.00548148, -5.21143e-06, 0.000447343, 0.000189448, 0.000529552, 5.25144e-05, 0.0021633, -0.000570076, -0.00127153, 0.000162279, -0.00175586, 0.000229642, 0.00632322, -0.000418208, -0.00531051, -1.90237e-05, -0.0461969, 3.96566e-05, 0.00047171, -8.02056e-05, 0.000711132, 0.000105453, -0.000762656, -0.000294382, -0.000346617, 6.17438e-05, -0.00098642, 4.14753e-05, 0.00541796, -0.000140228, 0.000905209, -4.19469e-05, -0.0969675, -0.000611579, -0.00029252, 5.19589e-05, 0.000346552, -0.000150174, 0.00031898, 9.60233e-05, -0.00298993, 0.000504591, 0.000564185, -0.000148394, 0.00115677, -0.000284064, -0.000367561, 0.000217865, 0.0102268, -4.86193e-05, -0.117889, -0.000607796, 0.000121003,
0.0129292, 0.000749643, 0.00739564, 0.000423648, 0.0186787, 0.000350466, -0.00875575, 4.65311e-05, 0.00101984, 0.000543912, 0.0122142, -0.000456508, -0.0114882, -0.000455688, 0.00169287, -0.000323356, 0.00041579, 0.000144571, -0.00973255, -0.000358198, 0.00165781, -0.000160452, 0.0011191, 4.97739e-05, -0.0205915, 7.20211e-05, 0.00102913, -0.000470189, -0.0178995, 0.000414424, 0.0106809, 0.00047739, -0.000582855, 0.000516167, -0.0113892, -0.000373975, -0.0195135, 0.000397315, 0.000671949, -0.000214205, -0.00108137, 0.000434605, 0.00335724, 0.000299213, -0.00164362, 0.000268408, 0.0160915, -4.56119e-06, -0.00151279, -0.000348661, -0.0188092, 0.000464808, -0.00108106, 0.00062872, 0.0166631, -0.000457791, -0.00798162, -0.000486796, -0.000703397, -0.000409237, 0.0243295, 0.000607643, -0.00317637, 0.000525181, -0.000807896, -1.40158e-05, 0.00120236, -0.00011687, -0.00243467, 0.000625211, 0.0103207, -0.000734273, -0.0063803, -0.000469584, 0.00104557, -0.000443049, -0.00358901, 0.000482715, -0.00337504, 0.000345869, 0.108757, 0.000864752, -0.00122128, 0.000571121, 0.0176325, -0.000802164, -0.00121044, -0.000477897, -0.0100603, 0.000274913, 0.00465725, 0.000543533, 0.00115981, 0.000274933, -0.0335984, -0.000429312, -0.00117014, -0.00037332, 0.153119, 0.000831758,
0.000421658, -0.00836175, 0.000531654, -0.00669999, 8.93399e-05, -0.00467855, 0.000216504, -0.00440439, 3.38745e-05, -0.00890173, 0.000282289, 0.00532294, -0.00023371, 0.00321129, 8.26177e-05, 0.0112116, -0.000176884, -0.00724783, 0.000182095, 0.00733176, -0.000293708, 0.00565039, 5.4781e-05, 0.00646857, -0.000128708, 0.0085973, -0.000159869, 0.00254995, -0.000316454, -0.00134898, 0.00013707, -4.58902e-07, -7.99095e-05, -0.00747558, -0.000380088, 0.0090796, -5.50101e-05, 0.00739684, 2.68637e-05, 0.00843243, -0.000549205, -0.00780924, 0.000415735, -0.00502661, -0.00016202, -0.0113496, -0.00037873, 0.00750946, 2.37703e-05, 0.0103976, -0.000386873, -0.00192287, 0.000151816, 0.00277873, 3.2896e-05, -0.00278232, 2.53181e-05, -0.00221229, -2.15008e-05, 0.00175668, 9.72814e-05, -0.00275189, -6.94723e-05, -0.00729179, -0.000530207, 0.010925, -0.000516981, -0.00803154, 6.33484e-05, -0.00645014, 0.000764289, 0.00599242, -0.000509873, 0.00368192, 0.000215501, 0.0140681, 0.000562319, -0.00929226, -8.11531e-05, -0.00728963, -0.000105187, -0.0059415, -0.000356944, 0.000913093, -1.79182e-05, -0.00279674, -7.11643e-05, -0.00641247, 0.000482577, 0.00573338, -0.000283834, 0.0036054, 0.000206438, 0.00723294, 0.000316137, -0.00318894, 6.43999e-05, 8.53686e-05, 0.000121601, -0.075927,
};