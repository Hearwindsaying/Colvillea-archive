// Include Eigen before STL and after OptiX (for macro support like __host__)
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix.h>

#include "colvillea/Device/SH/Test/SH.hpp"

// STL include
#include <random>
#include <utility>
#include <cstdio>
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
    //constexpr int lmax = 2;
    std::vector<float> ylmCoeff; ylmCoeff.resize((lmax+1)*(lmax+1));

    auto P1 = [](float z)->float {return z; };

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
    float S0 = computeSolidAngle<M>(we);

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
        // B1
        float S2 = 0.5f*Bl_1;

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
    for (int j = 0; j <= lmax; ++j)
    {
        if(lmax == 2)
            TW_ASSERT(2 * j + 1 == a[j*j+0].size());
        for (int i = 0; i < 2 * j + 1; ++i)
        {
            float coeff = 0.0f;
            for (int k = 0; k < 2 * j + 1; ++k)
            {
                coeff += a[j*j+i][k] * Lw[j][k];
            }
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
template<int M, int wSize>
void computeCoeff(float3 x, float3 v[], const float3 w[], const float a[][5], float ylmCoeff[9])
{
    constexpr int lmax = 2;

    auto P1 = [](float z)->float {return z; };
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
    float S0 = computeSolidAngle<M>(we);

    float Lw[lmax+1][wSize];

    for (int i = 0; i < wSize; ++i)
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
        float S1 = 0;
        for (int e = 1; e <= M; ++e)
        {
            ae[e] = dot(wi, we[e]); be[e] = dot(wi, lambdae[e]); ce[e] = dot(wi, ue[e]);
            S1 = S1 + 0.5*ce[e] * gammae[e];

            B0e[e] = gammae[e];
            B1e[e] = ae[e] * sinf(gammae[e]) - be[e] * cosf(gammae[e]) + be[e];
            D0e[e] = 0; D1e[e] = gammae[e]; D2e[e] = 3 * B1e[e];
        }

        //for l=2 to n
        float C1e[M + 1];
        float B2e[M + 1];

        float B2 = 0;
        for (int e = 1; e <= M; ++e)
        {
            C1e[e] = 1.f / 2.f * ((ae[e] * sin(gammae[e]) - be[e] * cosf(gammae[e]))*P1
            (ae[e] * cosf(gammae[e]) + be[e] * sinf(gammae[e])) +
                be[e] * P1(ae[e]) + (ae[e] * ae[e] + be[e] * be[e] - 1.f)*D1e[e] +
                (1.f)*B0e[e]);
            B2e[e] = 1.5f*(C1e[e]) - 1.f*B0e[e];
            B2 = B2 + ce[e] * B2e[e];
            D2e[e] = 3.f * B1e[e] + D0e[e];
        }

        // my code for B1
        float B1 = 0.f;
        for (int e = 1; e <= M; ++e)
        {
            B1 += ce[e] * B1e[e];
        }
        // B1
        float S2 = 0.5f*B1;

        Lw[0][i] = sqrtf(1.f / (4.f*M_PIf))*S0;
        Lw[1][i] = sqrtf(3.f / (4.f*M_PIf))*S1;
        Lw[2][i] = sqrtf(5.f / (4.f*M_PIf))*S2;
    }

    //TW_ASSERT(9 == a.size());
    for (int j = 0; j <= 2; ++j)
    {
        //TW_ASSERT(2 * j + 1 == 2*lmax+1); // redundant storage
        for (int i = 0; i < 2 * j + 1; ++i)
        {
            float coeff = 0.0f;
            for (int k = 0; k < 2 * j + 1; ++k)
            {
                coeff += a[j*j + i][k] * Lw[j][k];
            }
            ylmCoeff[j*j + i] = coeff;
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
            computeCoeff<3, 5>(make_float3(0.f), v.data(), basisData.data(), rawA, ylmCoeff);
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
            computeCoeff<4, 5>(make_float3(0.f), v.data(), basisData.data(), rawA, ylmCoeff);
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
            computeCoeff<4, 5>(make_float3(0.f), v.data(), basisData.data(), rawA, ylmCoeff);
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
#if VALIDATE_SH_ORDER9_MC

        {
            // Well projected on hemisphere
            auto A1 = make_float3(0.0f, 1.0f, 0.0f);
            auto B1 = make_float3(-1.0f, 0.0f, 0.0f);
            auto C1 = make_float3(0.0f, -1.0f, 0.0f);
            auto D1 = make_float3(1.0f, 0.0f, 0.0f);

            /* MC Validation of canonical cases: */
            /*auto uniformSamplingHemisphere = [](float x, float y)->float3
            {
                float z = x;
                float r = std::sqrt(std::max(0.0, 1.0 - z * z));
                float phi = 2.0 * M_PI * y;
                return make_float3(r * std::cos(phi), r * std::sin(phi), z);
            };*/
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

            constexpr int iteration = 10000000;
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
            }
            MCResultCoeffVec *= (1.0 / iteration);

            std::vector<float> MCResultCoeff;
            for (int i = 0; i < 100; ++i)
            {
                MCResultCoeff.push_back(MCResultCoeffVec(i));
            }


            std::vector<float3> v{ make_float3(0.f),A1,B1,C1,D1 };
            //std::vector<float3> v{ make_float3(0.f),A1,B1,D1 };
            std::vector<float> t1 = computeCoeff<4, 9>(make_float3(0.f), v, basisData, true, a);
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
        }
#endif 

        {
            // Well projected on hemisphere
            auto A1 = make_float3(0.0f, 1.0f, 0.0f);
            auto B1 = make_float3(-1.0f, 0.0f, 0.0f);
            auto C1 = make_float3(0.0f, -1.0f, 0.0f);
            auto D1 = make_float3(1.0f, 0.0f, 0.0f);

            std::vector<float3> v{ make_float3(0.f),A1,B1,C1,D1 };
            //std::vector<float3> v{ make_float3(0.f),A1,B1,D1 };
            std::vector<float> t1 = computeCoeff<4, 9>(make_float3(0.f), v, basisData, true, a);
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
            if (!equal(t1.begin(), t1.end(), t2.begin(), [epsilon](const float& x, const float& y)
            {
                return std::abs(x - y) <= epsilon;
            }))
            {
                ++nfails;
                TW_ASSERT(t1.size() == t2.size());
                for (int i = 0; i < t1.size(); ++i)
                {
                    printf("Test failed at Line:%d t1[%d]:%f AxialMoment[%d]:%f\n", __LINE__, i, t1[i], i, t2[i]);
                }
            }
            ++ntests;

            /*float ylmCoeff[9];
            computeCoeff<4, 5>(make_float3(0.f), v.data(), basisData.data(), rawA, ylmCoeff);
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
            ++ntests;*/
        }

        {
            auto A1 = (sphToCartesian(M_PI / 2.f, M_PI / 2.f));
            auto B1 = (sphToCartesian(M_PI / 4.f, M_PI / 2.f));
            auto C1 = (sphToCartesian(M_PI / 4.f, 0.f));
            auto D1 = (sphToCartesian(M_PI / 2.f, 0));

            std::vector<float3> v{ make_float3(0.f),A1,D1,C1 };
            std::vector<float> t1 = computeCoeff<3, 9>(make_float3(0.f), v, basisData, true, a);
            //computeCoeff<4>(make_float3(0.f), v, basisData, true, a);
            std::vector<float> t2{ 0.221557,
-0.191874,0.112397,-0.271351,0.257516,-0.106667,-0.157696,-0.182091,0.0910458,-0.168561,0.180663,0.0979564,
-0.172452,0.0951848,0.100333,0.0409613,0.0590044,-0.159732,-0.0669044,0.123451,0.0352618,0.178412,0.0315391,0,
-0.0417227,-0.0322195,0.0864866,-0.0240152,-0.149799,-0.022234,0.126357,0.0235817,-0.0415949,-0.0594356,-0.0360227,
-0.0113902,0.0575103,-0.0397269,0.0645599,0.0734258,-0.020161,-0.0867893,0.0190712,-0.0860057,-0.0548337,-0.0482535,
0.00720787,0.0112695,0.0227746,-0.0603642,
0.0413408,
-0.0334876,
0.0108118,
0.0544548,
0.0726051,
-0.00468302,
-0.058526,
-0.0298831,
-0.0204041,
0.0195189,
0.0394006,
0.0309601,
0.0143535,
0.00383496,
0.0343636,
-0.0536409,
-0.00150036,
-0.0274113,
-0.0316856,
0.000189409,
0.018694,
0.0490761,
-0.00329074,
0.02873,
0.011591,
0.0509859,
0.0273456,
0.0109474,
-0.0135141,
-0.00925419,
-0.0138894,
1.02033,
0.0463813,
-0.0542679,
0.390764,
-0.0158205,
-0.833401,
0.181615,
0.800297,
-0.0891647,
0.474453,
0.0354776,
0.875259,
-0.0860701,
-0.00929889,
0.539316,
0.0355549,
-0.018822,
0.604826,
-0.874716 };
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

            /*float ylmCoeff[9];
            computeCoeff<3, 9>(make_float3(0.f), v.data(), basisData.data(), rawA, ylmCoeff);
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
            }*/
            ++ntests;
        }

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

            /*float ylmCoeff[9];
            computeCoeff<3, 9>(make_float3(0.f), v.data(), basisData.data(), rawA, ylmCoeff);
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
            }*/
            ++ntests;
        }

    }
    printf("\nTest coverage:%f%%(%d/%d) passed!\n", 100.f*static_cast<float>(ntests - nfails) / ntests, ntests - nfails, ntests);
}

void QuadLight::TestZHRecurrence()
{
    TestSolidAngle();
    TestYlmCoeff();
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
}