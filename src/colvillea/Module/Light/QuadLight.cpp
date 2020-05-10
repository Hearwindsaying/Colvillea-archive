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
template<int M>
std::vector<float> computeCoeff(float3 x, std::vector<float3> & v, /*int n, std::vector<std::vector<float>> const& a,*/ std::vector<float3> const& w, bool vIsProjected, std::vector<std::vector<float>> const& a)
{
    constexpr int lmax = 2;
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

        //for l=2 to n
        std::vector<float> C1e; C1e.resize(v.size());
        std::vector<float> B2e; B2e.resize(v.size());

        float B2 = 0;
        for (int e = 1; e <= M; ++e)
        {
            C1e[e] = 1.f / 2.f * ((ae[e]*sin(gammae[e])-be[e]*cos(gammae[e]))*P1
                                  (ae[e]*cos(gammae[e])+be[e]*sin(gammae[e]))+
                                  be[e]*P1(ae[e])+(ae[e]*ae[e]+be[e]*be[e]-1.f)*D1e[e]+
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

        Lw[0][i] = sqrt(1.f / (4.f*M_PIf))*S0;
        Lw[1][i] = sqrt(3.f / (4.f*M_PIf))*S1;
        Lw[2][i] = sqrt(5.f / (4.f*M_PIf))*S2;
    }

    for (const auto& l0wi : Lw[0])
    {
        std::cout << "l0wi:   " << l0wi << std::endl;
    }
    std::cout << "--------------end l1wi" << std::endl;

    for (const auto& l1wi : Lw[1])
    {
        std::cout << "l1wi:   " << l1wi << std::endl;
    }
    std::cout << "--------------end l1wi" << std::endl;

    for (const auto& l2wi : Lw[2])
    {
        std::cout << "l2wi:   " << l2wi << std::endl;
    }
    std::cout << "--------------end l2wi" << std::endl;

    //TW_ASSERT(9 == a.size());
    for (int j = 0; j <= 2; ++j)
    {
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

    for (const auto& ylmC : ylmCoeff)
    {
        printf("%f\n", ylmC);
    }
    
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
        if (!(t1 == t2 == t3 == 0.5f*M_PIf))
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
        if (!(t1 == t2 == t3 == 0.5f*M_PIf))
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
        if (!(t1 == t2 == t3))
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
        if (!(t1 == t2 == t3 == 2.f*M_PIf))
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
        if (!(t1 == t2 == 2.f*M_PIf))
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
        if (!(t1 == t2 == t3 == M_PIf / sqrt(8)))
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
        if (!(t1 == t2 == t3))
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
    float epsilon = 1e-5f;
    int nfails = 0;
    int ntests = 0;

    auto sphToCartesian = [](const float theta, const float phi)->float3
    {
        return make_float3(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
    };

    printf("\n");

    float rawA[9][5] = { 1, 0, 0, 0, 0,
0.04762, -0.0952401, -1.06303, 0, 0,
0.843045, 0.813911, 0.505827, 0, 0,
-0.542607, 1.08521, 0.674436, 0, 0,
2.61289, -0.196102, 0.056974, -1.11255, -3.29064,
-4.46838, 0.540528, 0.0802047, -0.152141, 4.77508,
-3.36974, -6.50662, -1.43347, -6.50662, -3.36977,
-2.15306, -2.18249, -0.913913, -2.24328, -1.34185,
2.43791, 3.78023, -0.322086, 3.61812, 1.39367, };
    // Test statistics against computeCoeff(vector version), computeCoeff(gpu version), AxialMoments.
    {
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
        std::vector<float3> v{ make_float3(0.f),A1,D1,C1 };
        //std::vector<float3> v{ make_float3(0.f),A1,B1,D1 };
        std::vector<float> t1 = computeCoeff<3>(make_float3(0.f), v, basisData, true, a);
        //computeCoeff<4>(make_float3(0.f), v, basisData, true, a);
        std::vector<float> t2{ 0.221557, -0.191874,0.112397,-0.271351,0.257516,-0.106667,-0.157696,-0.182091,0.0910456 };
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
        computeCoeff<3, 5>(make_float3(0.f), v.data(), basisData.data(), rawA, ylmCoeff);
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

        std::vector<float3> v{ make_float3(0.f),A1,B1,C1,D1 };
        //std::vector<float3> v{ make_float3(0.f),A1,B1,D1 };
        std::vector<float> t1 = computeCoeff<4>(make_float3(0.f), v, basisData, true, a);
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
        auto A1 = make_float3(0.0f, 1.0f, 0.0f);
        auto B1 = make_float3(-1.0f, 0.0f, 0.0f);
        auto C1 = make_float3(0.0f, -1.0f, 0.0f);
        auto D1 = make_float3(1.0f, 0.0f, 0.0f);

        std::vector<float3> v{ make_float3(0.f),A1,B1,C1,D1 };
        //std::vector<float3> v{ make_float3(0.f),A1,B1,D1 };
        std::vector<float> t1 = computeCoeff<4>(make_float3(0.f), v, basisData, true, a);
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
    computeCoeff<3>(make_float3(0.f), v, basisData, true, a);
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