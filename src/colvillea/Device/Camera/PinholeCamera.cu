#include <optix_world.h>
#include <optix_device.h>

#include "colvillea/Device/Toolkit/NvRandom.h"
#include "colvillea/Device/Toolkit/Utility.h"
#include "colvillea/Device/Toolkit/CommonStructs.h"
#include "colvillea/Device/Sampler/Sampler.h"
#include "colvillea/Device/Filter/Filter.h"

#define EIGEN_DONT_VECTORIZE
#include "colvillea/Device/SH/SphericalIntegration.hpp"
#include "colvillea/Device/SH/DirectionsSampling.hpp"
#include "colvillea/Device/SH/Test/SH.hpp"

#define xstr(s) str(s)
#define str(s) #s
#define err_msg(x) #x " is " xstr(x)

//static_assert(false, err_msg(CUDA_VERSION));


#define GLM_FORCE_CUDA
#include "glm/glm/glm.hpp"

using namespace optix;
#pragma region SH_TEST_HIDDEN

//////////////////////////////////////////////////////////////////////////
//Forward declarations:
//system variables related:->Context
rtBuffer<float4, 2>	        sysOutputBuffer;         /*final result buffer. Necessary to keep Alpha=1*/
rtBuffer<float4, 2>         sysHDRBuffer;            /* Final result buffer for exporting to OpenEXR. */
rtBuffer<float, 2>          sysSampleWeightedSum;
rtBuffer<float4, 2>         sysCurrResultBuffer;     /*the sum of weighted radiance(f(dx,dy)*Li) with 
												       respect to the current iteration;
													   A is useless, note that for RenderView display,
													   it's necessary to keep as 1.f to prevent from 
													   alpha cutout by Dear Imgui*/
rtBuffer<float, 2>          sysCurrWeightedSumBuffer;/*weightedSum buffer with respect to the current
												       iteration*/

//rtDeclareVariable(float,    sysFilterGaussianAlpha, ,) = 0.25f;//Gaussian filter alpha paramter
//rtDeclareVariable(float,    sysFilterWidth, ,) = 1.f;//Gaussian filter width>=1.f

rtDeclareVariable(float,    sysSceneEpsilon, , );
rtDeclareVariable(rtObject, sysTopObject, , );


#ifndef TWRT_DECLARE_SYSLAUNCH
#define TWRT_DECLARE_SYSLAUNCH
rtDeclareVariable(uint2,    sysLaunch_Dim,       rtLaunchDim, );
rtDeclareVariable(uint2,    sysLaunch_index,     rtLaunchIndex, );
#endif

//camera variables related:->Program
rtDeclareVariable(Matrix4x4, RasterToCamera, , );
rtDeclareVariable(Matrix4x4, CameraToWorld, , );
rtDeclareVariable(float,     focalDistance, ,)=0.0f;
rtDeclareVariable(float,     lensRadius, ,)=0.0f;



namespace TwUtil
{
	/*
	 * @brief degamma function converts linear color to sRGB color
	 * for display
	 * @param src input value, alpha channel is not affected
	 * @return corresponding sRGB encoded color in float4. Alpha channel
	 * is left unchanged.
	 * @ref https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_texture_sRGB_decode.txt
	 * @see convertsRGBToLinear
	 **/
	static __device__ __inline__ float4 convertFromLinearTosRGB(const float4 &src)
	{
		float4 dst = src;
		dst.x = (dst.x < 0.0031308f) ? dst.x*12.92f : (1.055f * powf(dst.x, 0.41666f) - 0.055f);
		dst.y = (dst.y < 0.0031308f) ? dst.y*12.92f : (1.055f * powf(dst.y, 0.41666f) - 0.055f);
		dst.z = (dst.z < 0.0031308f) ? dst.z*12.92f : (1.055f * powf(dst.z, 0.41666f) - 0.055f);
		return dst;
	}

	/*
	 * @brief converts one of the sRGB color channel to linear
	 * @param src input value
	 * @return corresponding linear space color channel
	 * @ref https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_texture_sRGB_decode.txt
	 * @see convertFromLinearTosRGB
	 **/
	static __device__ __inline__ float convertsRGBToLinear(const float &src)
	{
		if (src <= 0.f)
			return 0;
		if (src >= 1.f)
			return 1.f;
		if (src <= 0.04045f)
			return src / 12.92f;
		return pow((src + 0.055f) / 1.055f, 2.4f);
	};

	/*
	 * @brief converts sRGB color to linear color
	 * @param src input value, alpha channel is not affected
	 * @return corresponding linear space color in float4. Alpha channel
	 * is left unchanged.
	 * @ref https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_texture_sRGB_decode.txt
	 * @see convertFromLinearTosRGB
	 **/
	static __device__ __inline__ float4 convertsRGBToLinear(const float4 &src)
	{
		return make_float4(convertsRGBToLinear(src.x), convertsRGBToLinear(src.y), convertsRGBToLinear(src.z), 1.f);
	}
};
#pragma endregion SH_TEST_HIDDEN



struct Vector : public glm::vec3 {

    __device__ Vector() : glm::vec3() {}
    __device__ Vector(float x, float y, float z) : glm::vec3(x, y, z) {}
    __device__ Vector(const glm::vec3& w) : glm::vec3(w) {}

    static inline __device__  float Dot(const glm::vec3& a, const glm::vec3& b) {
        return glm::dot(a, b);
    }

    static inline __device__ glm::vec3 Cross(const glm::vec3& a, const glm::vec3& b) {
        return glm::cross(a, b);
    }

    static inline __device__ glm::vec3 Normalize(const glm::vec3& a) {
        return glm::normalize(a);
    }

    static inline __device__ float Length(const glm::vec3& a) {
        return glm::length(a);
    }

    #if 0
Vector(){}
    Vector(float x, float y, float z) 
    { 
        this->vec.x = x; 
        this->vec.y = y; 
        this->vec.z = z; 
    }
    Vector(const Vector& w) {
        this->vec.x = w.vec.x; 
        this->vec.y = w.vec.y;
        this->vec.z = w.vec.z;
    }

    static inline float Dot(const Vector &a, const Vector &b) {
        return dot(a.vec, b.vec);
    }

    static inline float3 Cross(const Vector &a, const Vector &b) {
        return cross(a.vec, b.vec);
    }

    static inline float3 Normalize(const Vector &a) {
        return safe_normalize(a.vec);
    }

    static inline float Length(const Vector &a) {
        return optix::length(a.vec);
    }

    float& operator[](int idx) {
        if (idx == 0)return vec.x;
        if (idx == 1)return vec.y;
        if (idx == 2)return vec.z;
    }
    const float& operator[](int idx) const
    {
        if (idx == 0)return vec.x;
        if (idx == 1)return vec.y;
        if (idx == 2)return vec.z;
    }
    float& x()

    float3 vec;
#endif // 0
};

struct Edge {
    __device__ Edge() {}
    __device__ Edge(const Vector& a, const Vector& b) : A(a), B(b) {}
    Vector A, B;
};

struct Triangle{
    __device__ Triangle() {}
    __device__ Triangle(const Vector& A, const Vector& B, const Vector& C) :e0(Edge(A, B)), e1(Edge(B, C)), e2(Edge(C, A)) {

    }

    __device__ Edge& operator[](int idx) {
        if (idx == 0)return e0;
        if (idx == 1)return e1;
        if (idx == 2)return e2;
    }
    __device__ const Edge& operator[](int idx) const
    {
        if (idx == 0)return e0;
        if (idx == 1)return e1;
        if (idx == 2)return e2;
    }
    __device__ int size() const
    {
        return 3;
    }

    Edge e0;
    Edge e1;
    Edge e2;
};

struct SH {

    // Inline for FastBasis
    static inline __device__ Eigen::VectorXf FastBasis(const Vector& w, int lmax) {
        const auto size = Terms(lmax);
        Eigen::VectorXf res(size);
        SHEvalFast<Vector>(w, lmax, res);
        return res;
    }
    static inline __device__ void FastBasis(const Vector& w, int lmax, Eigen::VectorXf& clm) {
        assert(clm.size() == Terms(lmax));
        SHEvalFast<Vector>(w, lmax, clm);
    }

    // Inline for Terms
    static inline __device__ int Terms(int band) {
        return SHTerms(band);
    }

    // Inline for Index
    static inline __device__ int Index(int l, int m) {
        return SHIndex(l, m);
    }
};

template<typename T>
__device__ inline bool closeTo(const T& a, const T&b, const T& Percentage = T(0.01)) {
    if (a == T(0.0) || b == T(0.0)) {
        return abs(a - b) < Percentage;
    }
    else {
        const T c = max(max(a, b), Percentage);
        return (abs(a - b) < Percentage * c);
    }
}

__device__ bool CheckNormalization(const Eigen::VectorXf& clm, const Vector * basis, int basisSize) {

    Triangle tri;

    float shI = 0.0f;

    // Upper hemisphere
    tri = Triangle(Vector(0, 0, 1), Vector(0, 1, 0), Vector(1, 0, 0));
    shI += clm.dot(AxialMoments<Triangle, Vector>(tri, basis, basisSize));

    tri = Triangle(Vector(0, 0, 1), Vector(1, 0, 0), Vector(0, -1, 0));
    shI += clm.dot(AxialMoments<Triangle, Vector>(tri, basis, basisSize));

    tri = Triangle(Vector(0, 0, 1), Vector(0, -1, 0), Vector(-1, 0, 0));
    shI += clm.dot(AxialMoments<Triangle, Vector>(tri, basis, basisSize));

    tri = Triangle(Vector(0, 0, 1), Vector(-1, 0, 0), Vector(0, 1, 0));
    shI += clm.dot(AxialMoments<Triangle, Vector>(tri, basis, basisSize));

    // Lower hemisphere
    tri = Triangle(Vector(0, 0, -1), Vector(1, 0, 0), Vector(0, 1, 0));
    shI += clm.dot(AxialMoments<Triangle, Vector>(tri, basis, basisSize));

    tri = Triangle(Vector(0, 0, -1), Vector(0, 1, 0), Vector(-1, 0, 0));
    shI += clm.dot(AxialMoments<Triangle, Vector>(tri, basis, basisSize));

    tri = Triangle(Vector(0, 0, -1), Vector(-1, 0, 0), Vector(0, -1, 0));
    shI += clm.dot(AxialMoments<Triangle, Vector>(tri, basis, basisSize));

    tri = Triangle(Vector(0, 0, -1), Vector(0, -1, 0), Vector(1, 0, 0));
    shI += clm.dot(AxialMoments<Triangle, Vector>(tri, basis, basisSize));

    bool check = closeTo(shI, 0.0f);
    if (!check) { rtPrintf("Error, lost in precision: I=%f\n",shI); }

    return check;
}

__device__ __inline__ void UnitIntegral()
{
    int nb_fails = 0;
    constexpr int maxorder = 8;
    constexpr int maxsize = (maxorder + 1)*(maxorder + 1);
    // Precompute the set of ZH directions
    constexpr int basisSize = 2 * maxorder + 1;
    Vector basis[2 * maxorder + 1];
    SamplingFibonacci<Vector, 2 * maxorder + 1>(basisSize, basis);
    //const auto basis = SamplingBlueNoise<Vector>(2*maxorder+1);
    rtPrintf("Done sampling enough directions\n");

    // Get the Zonal weights matrix and the Zlm -> Ylm conversion matrix
    // and compute the product of the two: `Prod = A x Zw`.
    const auto ZW = ZonalWeights<Vector>(basis, basisSize);
    const auto Y = ZonalExpansion<SH, Vector>(basis, basisSize);
    const auto A = computeInverse(Y);
    const auto Prod = A * ZW;
    rtPrintf("Done with precomputing the matrix\n");

    for (int order = 1; order <= maxorder; ++order) {

        // Loop for each SH coeff on this band
        int size = (order + 1)*(order + 1);
        for (int i = order * order; i < size; ++i) {
            // Set the ith coeffcient to one
            Eigen::VectorXf clm = Eigen::VectorXf::Zero(maxsize);
            clm[i] = 1.0f;

            // Perform integral and check the result
            bool check = CheckNormalization(clm.transpose()*Prod, basis, basisSize);
            if (!check) {
                //std::cout << " for i=" << i << " at order=" << order << std::endl;
                rtPrintf("error in check!");
                ++nb_fails;
            }
        }
    }

    {
        rtPrintf("nb_fails %d",nb_fails);
    }
}

__device__ __inline__ int CheckEquals(const glm::vec3& wA, const Triangle& triA,
    const glm::vec3& wB, const Triangle& triB,
    int nMin, int nMax, float Epsilon = 1.0E-3f) {

    // Track the number of failed tests
    int nb_fails = 0;

    auto momentsA = AxialMoment<Triangle, Vector>(triA, wA, nMax);
    auto momentsB = AxialMoment<Triangle, Vector>(triB, wB, nMax);

    // Test the difference between analytical code and MC
    if (!momentsA.isApprox(momentsB)) {
        ++nb_fails;
        for (unsigned int n = 0; n < nMax; ++n) {
            rtPrintf("Error for n=%d: AXIAL\n", n);
            /*std::cout << "Error for n=" << n << " : "
                << momentsA[n] << " != " << momentsB[n] << std::endl;*/
        }
    }

    if (nb_fails == 0) {
        rtPrintf("Test pass\n");
    }
    return nb_fails;
}

__device__ __inline__ int testing()
{
    glm::vec3 A, B, C;
    Triangle  triA, triB, triC, triD;
    glm::vec3 wA, wB, wC, wD;

    // Track the number of failed tests
    int nMin = 0, nMax = 10;
    int nb_fails = 0;

    // Configuration A
    A = glm::vec3(0.0, 0.0, 1.0);
    B = glm::vec3(0.1, 0.0, 1.0);
    C = glm::vec3(0.0, 0.1, 1.0);
    triA = Triangle(A, B, C);
    wA = glm::normalize(glm::vec3(0, 0, 1));

    // Configuration B
    A = glm::vec3(0.0, 0.0, 1.0);
    B = glm::vec3(0.0, 0.1, 1.0);
    C = glm::vec3(-0.1, 0.0, 1.0);
    triB = Triangle(A, B, C);
    wB = glm::normalize(glm::vec3(0, 0, 1));

    // Configuration C
    A = glm::vec3(0.0, 0.0, 1.0);
    B = glm::vec3(-0.1, 0.0, 1.0);
    C = glm::vec3(0.0, -0.1, 1.0);
    triC = Triangle(A, B, C);
    wC = glm::normalize(glm::vec3(0, 0, 1));

    // Configuration C
    A = glm::vec3(0.0, 0.0, 1.0);
    B = glm::vec3(0.0, -0.1, 1.0);
    C = glm::vec3(0.1, 0.0, 1.0);
    triD = Triangle(A, B, C);
    wD = glm::normalize(glm::vec3(0, 0, 1));

    /* Check for the case where A == B */
    rtPrintf("# Check for ABC == ABC'\n");

    nb_fails += CheckEquals(wA, triA, wA, triA, nMin, nMax);


    /* Keep w = (0,0,1) and rotate the spherical triangle
       by 90, 180, and 270 degress. */
    rtPrintf("# Check for w = z and rotate the spherical triangle around z\n");

    // Check for the case where A and B are symmetric
    nb_fails += CheckEquals(wA, triA, wB, triB, nMin, nMax);

    // Check for the case where A and C are symmetric
    nb_fails += CheckEquals(wA, triA, wC, triC, nMin, nMax);

    // Check for the case where A and B are symmetric
    nb_fails += CheckEquals(wA, triA, wD, triD, nMin, nMax);

    // Check for the case where B and C are symmetric
    nb_fails += CheckEquals(wB, triB, wC, triC, nMin, nMax);

    // Check for the case where B and C are symmetric
    nb_fails += CheckEquals(wB, triB, wD, triD, nMin, nMax);

    // Check for the case where A and C are symmetric
    nb_fails += CheckEquals(wC, triC, wD, triD, nMin, nMax);

    /* Make w the first axis of the triangle projected on
       the ground plane. */
    rtPrintf("# Check for w = AB and rotate the spherical triangle around z\n");

    wA = glm::vec3(1, 0, 0);
    wB = glm::vec3(0, 1, 0);
    wC = glm::vec3(-1, 0, 0);
    wD = glm::vec3(0, -1, 0);

    // Check for the case where A and B are symmetric
    nb_fails += CheckEquals(wA, triA, wB, triB, nMin, nMax);

    // Check for the case where A and C are symmetric
    nb_fails += CheckEquals(wA, triA, wC, triC, nMin, nMax);

    // Check for the case where A and B are symmetric
    nb_fails += CheckEquals(wA, triA, wD, triD, nMin, nMax);

    // Check for the case where B and C are symmetric
    nb_fails += CheckEquals(wB, triB, wC, triC, nMin, nMax);

    // Check for the case where B and C are symmetric
    nb_fails += CheckEquals(wB, triB, wD, triD, nMin, nMax);

    // Check for the case where A and C are symmetric
    nb_fails += CheckEquals(wC, triC, wD, triD, nMin, nMax);

#ifdef CHECK_ORIENTATION
    // Configuration B is similar to A but with swap vertices
    A = triA[0].A;
    B = triA[2].A;
    C = triA[1].A;
    triB = Triangle(A, B, C);
    wA = glm::normalize(glm::vec3(0, 0, 1));

    nb_fails += CheckEquals(wA, triA, wA, triB, nMin, nMax);
#endif

    if (nb_fails == 0)
        return 0;
    else
        return 1;
}

/* AP Matrix for SH Integration. */
rtBuffer<float, 2> areaLightAPMatrix;

/* Flm Diffuse Matrix. */
rtBuffer<float> areaLightFlmVector;

static __device__ __inline__ void Test_AP_Flm_MatrixOrder3()
{
    /* Test if AP Matrix is loaded correctly. */
    rtPrintf("%f == 0.282095f, %f == 0.0232673, %f == 1.31866 \n", areaLightAPMatrix[make_uint2(0, 0)], areaLightAPMatrix[make_uint2(1, 1)], areaLightAPMatrix[make_uint2(14, 8)]);
    rtPrintf("%f == -1.05266f \n", areaLightAPMatrix[make_uint2(11, 4)]);

    rtPrintf("\n");

    auto sizexy = areaLightAPMatrix.size();
    for (int j = 0; j < sizexy.y; ++j)
    {
        for (int i=0; i<sizexy.x;++i)
            rtPrintf("%f ", areaLightAPMatrix[make_uint2(i, j)]);
        rtPrintf("\n");
    }

    /* Test Flm Vector. */
    rtPrintf("Flm Vector:\n");
    for (int i = 0; i < areaLightFlmVector.size(); ++i)
    {
        rtPrintf("%f ", areaLightFlmVector[i]);
    }
    rtPrintf("------------Done printing------------\n");
}



//////////////////////////////////////////////////////////////////////////
//Program definitions:
RT_PROGRAM void RayGeneration_PinholeCamera()
{
    if (sysLaunch_index == make_uint2(0, 0))
    {
        Test_AP_Flm_MatrixOrder3();
        //testing();
        return;
    }
    else { return; }

    /* Make sampler and preprocess. */
    GPUSampler localSampler;  /* GPUSampler is a union type, use out parameter instead of return value to avoid copying construct on union, which could lead to problems. */
    makeSampler(RayTracingPipelinePhase::RayGeneration, localSampler);

    /* Fetch camera samples lying on [0,1]^2. */
    float2 qmcSamples = Get2D(&localSampler);

	/* Calculate filmSamples in continuous coordinates. */
	float2 pFilm = qmcSamples + make_float2(static_cast<float>(sysLaunch_index.x), static_cast<float>(sysLaunch_index.y));

	/* Generate ray from camera. */
	float3 rayOrg = make_float3(0.f);
	float3 rayDir = rayOrg;
    if (lensRadius > 0.f)
    {
        float2 lensSamples = Get2D(&localSampler);
        TwUtil::GenerateRay(pFilm, rayOrg, rayDir, RasterToCamera, CameraToWorld, lensRadius, focalDistance, &lensSamples);
    }
    else
    {
        TwUtil::GenerateRay(pFilm, rayOrg, rayDir, RasterToCamera, CameraToWorld, lensRadius, focalDistance, nullptr);
    }

	/* Make ray and trace, goint to next raytracing pipeline phase. */
	Ray ray = make_Ray(rayOrg, rayDir, toUnderlyingValue(RayType::Radiance), sysSceneEpsilon, RT_DEFAULT_MAX);

    CommonStructs::PerRayData_radiance prdRadiance;
	prdRadiance.radiance = make_float4(0.f);

	rtTrace<CommonStructs::PerRayData_radiance>(sysTopObject, ray, prdRadiance, RT_VISIBILITY_ALL, RT_RAY_FLAG_DISABLE_ANYHIT);

	/*--------------------------------------------------------------------------------*/
	/*----Perform filtering and reconstruction so as to write to the output buffer----*/
	/*--------------------------------------------------------------------------------*/

    /* If filter width <= 0.5f, one sample could only contribute to one pixel
       -- and there is no chance that two samples not in the same pixel will
       -- contribute to the same pixel. So atomic operation could be saved for
       -- efficenciy. */
    float filterWidth = GetFilterWidth();
    if(filterWidth <= 0.5f)
    {
        float currentWeight = EvaluateFilter(qmcSamples.x - 0.5f, qmcSamples.y - 0.5f);

        float4 &currLi = prdRadiance.radiance;
        /*ignore alpha channel*/
        sysCurrResultBuffer[sysLaunch_index].x    += currLi.x * currentWeight;
        sysCurrResultBuffer[sysLaunch_index].y    += currLi.y * currentWeight;
        sysCurrResultBuffer[sysLaunch_index].z    += currLi.z * currentWeight;
        sysCurrWeightedSumBuffer[sysLaunch_index] += currentWeight;
    }
    else
    {
        /* Compute pFilm's raster extent
         * --1.get film sample's discrete coordinates. */
        float2 dCoordsSample = pFilm - 0.5f;

        /*--2.search around the filterWidth for raster pixel boundary*/
        int2 pMin = TwUtil::ceilf2i(dCoordsSample - filterWidth);
        int2 pMax = TwUtil::floorf2i(dCoordsSample + filterWidth);

        /*--3.check for film extent*/
        pMin.x = max(pMin.x, 0);                       pMin.y = max(pMin.y, 0);
        pMax.x = min(pMax.x, sysLaunch_Dim.x - 1);     pMax.y = min(pMax.y, sysLaunch_Dim.y - 1);

        if ((pMax.x - pMin.x) < 0 || (pMax.y - pMin.y) < 0)
        {
            rtPrintf("invalid samples:%f %f\n", dCoordsSample.x, dCoordsSample.y);
        }

        /* Loop over raster pixel and add sample with filtering operation. */
        for (int y = pMin.y; y <= pMax.y; ++y)
        {
            for (int x = pMin.x; x <= pMax.x; ++x)
            {
                /*not necessary to distinguish first iteration, because one sample
                 *could possibly contribute to multiple pixels so the 0th iteration doesn't
                 *have a specialized meaning. Instead, we use the modified version of progressive
                 *weighted average formula.*/

                 /*Pass 1:accumulate sysCurrResultBuffer with f(dx,dy)*Li and sysCurrWeightedSumBuffer with f(dx,dy)*/
                uint2 pixelIndex = make_uint2(x, y);
                float currentWeight = EvaluateFilter(x - dCoordsSample.x, y - dCoordsSample.y);

                float4 &currLi = prdRadiance.radiance;
                /*ignore alpha channel*/
                atomicAdd(&sysCurrResultBuffer[pixelIndex].x, currLi.x * currentWeight);
                atomicAdd(&sysCurrResultBuffer[pixelIndex].y, currLi.y * currentWeight);
                atomicAdd(&sysCurrResultBuffer[pixelIndex].z, currLi.z * currentWeight);
                atomicAdd(&sysCurrWeightedSumBuffer[pixelIndex], currentWeight);
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
//Initialize outputbuffer and sampleWeightedSum buffer, much more efficient than using serilized "for" on host 
RT_PROGRAM void RayGeneration_InitializeFilter()
{
	sysCurrResultBuffer[sysLaunch_index] = sysOutputBuffer[sysLaunch_index] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
	sysCurrWeightedSumBuffer[sysLaunch_index] = sysSampleWeightedSum[sysLaunch_index] = 0.0f;
}

//////////////////////////////////////////////////////////////////////////
//Perform filtering
RT_PROGRAM void RayGeneration_Filter()
{
	/* Perform gamma correction to resolve correct linear color for computation. */
	sysOutputBuffer[sysLaunch_index] = convertsRGBToLinear(sysOutputBuffer[sysLaunch_index]);
	sysOutputBuffer[sysLaunch_index] = (sysOutputBuffer[sysLaunch_index] * sysSampleWeightedSum[sysLaunch_index] + sysCurrResultBuffer[sysLaunch_index]) / 
                                       (sysSampleWeightedSum[sysLaunch_index] + sysCurrWeightedSumBuffer[sysLaunch_index]);
    sysHDRBuffer[sysLaunch_index]    = (sysHDRBuffer[sysLaunch_index] * sysSampleWeightedSum[sysLaunch_index] + sysCurrResultBuffer[sysLaunch_index]) / 
                                       (sysSampleWeightedSum[sysLaunch_index] + sysCurrWeightedSumBuffer[sysLaunch_index]);

	sysSampleWeightedSum[sysLaunch_index] += sysCurrWeightedSumBuffer[sysLaunch_index];

    /* Ensure w component of output buffer is 1.0f in case of being transparent
     * -- in RenderView. */
    sysOutputBuffer[sysLaunch_index].w = 1.f;
    sysHDRBuffer[sysLaunch_index].w    = 1.f;

    /* Prevent from precision error. Note that some filters (such as Gaussian)
     * -- could have zero sample weight to a pixel, which may cause that for some
     * -- pixels, they couldn't have any sample weight contribution at all at first
     * -- launch (almost only one sample distributes sparsely per pixel in the situation).
     * But after a few iterations, all pixels are able to have sample weight contribution.
     * For this reason, we skip the NaN values in output buffer to give a chance for
     * later accumulation. */
    if (isnan(sysOutputBuffer[sysLaunch_index].x) || isnan(sysOutputBuffer[sysLaunch_index].y) || isnan(sysOutputBuffer[sysLaunch_index].z) || 
        isnan(sysHDRBuffer[sysLaunch_index].x)    || isnan(sysHDRBuffer[sysLaunch_index].y)    || isnan(sysHDRBuffer[sysLaunch_index].z))
    {
        sysOutputBuffer[sysLaunch_index] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        sysHDRBuffer[sysLaunch_index]    = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    #if 0
    if (isnan(sysOutputBuffer[sysLaunch_index].x) || isnan(sysOutputBuffer[sysLaunch_index].y) || isnan(sysOutputBuffer[sysLaunch_index].z) || (sysOutputBuffer[sysLaunch_index].x < 0.f || sysOutputBuffer[sysLaunch_index].y < 0.f || sysOutputBuffer[sysLaunch_index].z < 0.f))
    {
        rtPrintf("%d %d out=%f * %f + %f / (%f + %f)\n", sysLaunch_index.x,sysLaunch_index.y,
            sysOutputBuffer[sysLaunch_index].x,
            sysSampleWeightedSum[sysLaunch_index], 
            sysCurrResultBuffer[sysLaunch_index].x, 
            sysSampleWeightedSum[sysLaunch_index], 
            sysCurrWeightedSumBuffer[sysLaunch_index]);
    }
#endif // 
    

	/*clear current buffer for next iteration*/
	sysCurrResultBuffer[sysLaunch_index] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
	sysCurrWeightedSumBuffer[sysLaunch_index] = 0.0f;

	/* Perform "degamma" operation converting linear color to sRGB for diplaying in RenderView. */
	sysOutputBuffer[sysLaunch_index] = convertFromLinearTosRGB(sysOutputBuffer[sysLaunch_index]);
}


RT_PROGRAM void Exception_Default()
{
	const unsigned int code = rtGetExceptionCode();
	rtPrintf("[Exception]Caught exception 0x%X at launch index (%d,%d)\n", code, sysLaunch_index.x, sysLaunch_index.y);
	rtPrintExceptionDetails();
	sysOutputBuffer[sysLaunch_index] = make_float4(1000.0f, 0.0f, 0.0f, 1.0f);
}


