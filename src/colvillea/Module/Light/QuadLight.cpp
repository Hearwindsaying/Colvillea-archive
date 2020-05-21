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

static __device__ __inline__ void SHEvalFast9(const optix::float3 &w, float *pOut) {
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
        if (wo.z < 0.f || wi.z < 0.f)return 0.f; // Single Side BRDF
        //if (wi.z < 0.f)return 0.f; // This is consistent with PBRT's SH BSDF computation
        return fabsf(wi.z) / M_PIf;
    };

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);

#define WRITE_BSDFMATRIX 1
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
    for (int wsamples = 0; wsamples < iteration; ++wsamples)
    {
        float3 dir = UniformSampleSphere(dis(gen), dis(gen));
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
        std::cout << "II Progress: " << osamples << " / " << 100 << "     \r";
        std::cout.flush();
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
