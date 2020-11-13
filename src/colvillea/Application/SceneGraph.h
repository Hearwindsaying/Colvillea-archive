#pragma once

#define  CL_CHECK_MEMORY_LEAKS
#ifdef CL_CHECK_MEMORY_LEAKS
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#define CL_CHECK_MEMORY_LEAKS_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
#define new CL_CHECK_MEMORY_LEAKS_NEW
#endif

#include <optix.h>
#include <optixu/optixpp_namespace.h>
#include <optixu_matrix_namespace.h>

#include <algorithm>
#include <map>
#include <memory>

#include "colvillea/Module/Integrator/DirectLighting.h"
#include "colvillea/Module/Integrator/AnalyticalDirectLighting.h"
#include "colvillea/Module/Integrator/PathTracing.h"
#include "colvillea/Module/Geometry/TriangleMesh.h"
#include "colvillea/Module/Geometry/Quad.h"
#include "colvillea/Module/Sampler/HaltonSampler.h"
#include "colvillea/Module/Sampler/SobolSampler.h"
#include "colvillea/Module/Sampler/IndependentSampler.h"
#include "colvillea/Module/Sampler/FiniteSampler.h"

#include "colvillea/Module/Filter/BoxFilter.h"
#include "colvillea/Module/Filter/GaussianFilter.h"

#include "colvillea/Module/Camera/Camera.h"
#include "colvillea/Application/GlobalDefs.h"

class Application;
class BSDF;


/**
 * @brief SceneGraph class gather integrator, shapes, materialPool,
 * sampler, lights information and provide helper utility to build 
 * up scene graph for OptiX. 
 * Light and Material is delegated to LightPool and MaterialPool
 * respectively for better management.
 */
class SceneGraph
{
public:
    SceneGraph(Application *application, const std::map<std::string, optix::Program> &programsMap, const optix::Context context, const unsigned int &filmWidth, const unsigned int &filmHeight) :
        m_programsMap(programsMap), m_context(context),
        m_filmWidth(filmWidth), m_filmHeight(filmHeight), m_application(application)
	{
		this->initializeGraph();
        this->initializeLoadingIntegrators();
        this->initializeLoadingDefaultSampler();
        this->initializeLoadingFilters();
	}

private:
    /*******************************************************************/
    /*           Initialization functions called by constructor        */
    /*******************************************************************/

    /**
     * @brief Initialize OptiX Graph nodes that should be set once.
     */
    void initializeGraph()
    {
        /* Create GeometryGroups, containing GeometryTriangles and Geometry, respectively. */
        auto& context = this->m_context;
        this->m_topGeometryGroup_GeometryTriangles = context->createGeometryGroup();
        this->m_topGeometryGroup_Geometry          = context->createGeometryGroup();

        /* Create Top Group containing GeometryGroups. */
        this->m_topGroup = context->createGroup();

        /* Disable anyhits for GeometryGroups. */
        this->m_topGeometryGroup_GeometryTriangles->setFlags(RTinstanceflags::RT_INSTANCE_FLAG_DISABLE_ANYHIT);
        this->m_topGeometryGroup_Geometry->setFlags(RTinstanceflags::RT_INSTANCE_FLAG_DISABLE_ANYHIT);


        /* Create Accelerations for GeometryGroups and Top Group. */
        optix::Acceleration geometryTrianglesAccel = context->createAcceleration("Trbvh");
                            geometryTrianglesAccel->setProperty("chunk_size", "-1");
                            geometryTrianglesAccel->setProperty("vertex_buffer_name", "vertexBuffer");
                            geometryTrianglesAccel->setProperty("index_buffer_name",  "indexBuffer");
        optix::Acceleration geometryAccel = context->createAcceleration("Trbvh");
                            geometryAccel->setProperty("chunk_size", "-1");
                            geometryAccel->setProperty("vertex_buffer_name", "vertexBuffer");
                            geometryAccel->setProperty("index_buffer_name",  "indexBuffer");
        optix::Acceleration groupAccel = context->createAcceleration("Trbvh");
        

        /* Set accelerations. */
        this->m_topGeometryGroup_GeometryTriangles->setAcceleration(geometryTrianglesAccel);
        this->m_topGeometryGroup_Geometry->setAcceleration(geometryAccel);
        this->m_topGroup->setAcceleration(groupAccel);


        /* Add GeometryTriangles GeometryGroup and Geometry GeometryGroup to top Group. */
        this->m_topGroup->addChild(this->m_topGeometryGroup_GeometryTriangles);
        this->m_topGroup->addChild(this->m_topGeometryGroup_Geometry);

        this->m_context["sysTopObject"]->set(this->m_topGroup);
        this->m_context["sysTopShadower"]->set(this->m_topGroup);
    }

    /**
     * @brief Creating integrators and add to integrators map for later use.
     * It is slightly different from Samplers that we load all integrators
     * initially. 
     * @note Default behavior: If no integrator is specified in hard code, 
     * DirectLighting integrator will be used.
     * @todo The default parameters for creating integrators should be done with 
     * config file.
     */
    void initializeLoadingIntegrators()
    {
        std::shared_ptr<Integrator> integrator = DirectLighting::createIntegrator(this->m_context, this->m_programsMap);
        TW_ASSERT(this->m_integratorsMap.insert({ IntegratorType::DirectLighting, integrator }).second);
        this->m_integrator = integrator;

        /* Default values. */
        constexpr bool enableRoussianRoulette = true;
        constexpr int  maxDepth = 5;
        integrator = PathTracing::createIntegrator(this->m_context, this->m_programsMap, enableRoussianRoulette, maxDepth);
        TW_ASSERT(this->m_integratorsMap.insert({ IntegratorType::PathTracing, integrator }).second);

        integrator = AnalyticalDirectLighting::createIntegrator(this->m_context, this->m_programsMap);
        TW_ASSERT(this->m_integratorsMap.insert({ IntegratorType::AnalyticalDirectLighting, integrator }).second);
    }

    /**
     * @brief Load a default sampler which could be overrided in hard code.
     * @note Default behavior: If no sampler is specified in hard code,
     * IndependentSampler will be used.
     * @todo The default parameters for creating integrators should be done with
     * config file.
     */
    void initializeLoadingDefaultSampler()
    {
        /* SceneGraph::createSampler() create corresponding sampler and apply to
         * |m_sampler|. */
        this->createSampler(CommonStructs::SamplerType::IndependentSampler);
    }

    /**
     * @brief Creating filters and add to filters map for later use.
     * It is slightly different from Samplers that we load all filters
     * initially.
     * @note Default behavior: If no filter is specified in hard code,
     * BoxFilter with radius = 0.5f will be used.
     * @todo The default parameters for creating integrators should be done with
     * config file.
     */
    void initializeLoadingFilters()
    {
        /* Default values. */
        constexpr float radius = 0.5f;
        std::shared_ptr<Filter> filter = BoxFilter::createBoxFilter(radius); 
        TW_ASSERT(this->m_filtersMap.insert({ CommonStructs::FilterType::BoxFilter, filter }).second);
        this->updateCurrentFilter(filter);

        constexpr float gaussianAlpha = 2.0f;
        filter = GaussianFilter::createGaussianFilter(radius, gaussianAlpha);
        TW_ASSERT(this->m_filtersMap.insert({ CommonStructs::FilterType::GaussianFilter, filter }).second);
    }

public:
    /************************************************************************/
    /*                 Scene configuration creating functions               */
    /************************************************************************/ 

	/**
	 * @brief Create TriangleMesh and add to SceneGraph shape
	 * pool. A simple function to encapsulate constructor and
	 * loadShape().
	 * 
	 * @param[in] meshFilename wavefront obj filename with path
	 * @param[in] materialIndex material index to materialBuffer
	 * @param[in] bsdf
	 * 
	 * @todo Remove materialIndex and use MaterialPool::m_bsdfs.size() instead.
	 */
    void createTriangleMesh(const std::string & meshFileName, int materialIndex, const std::shared_ptr<BSDF> &bsdf);

    /**
     * @brief Create a single quad and add to SceneGraph shape
     * pool. A simple function to encapsulate constructor and
     * loadShape().
     *
     * @param[in] materialIndex material index to materialBuffer
     * @param[in] position
     * @param[in] rotation      XYZ rotation angle in radian
     * @param[in] scale         Z-component is zero
     * @param[in] flipNormal    flip quad's normal
     */
    std::shared_ptr<Quad> createQuad(SceneGraph *sceneGraph, int32_t materialIndex, const optix::float3 &position, const optix::float3 &rotation, const optix::float3 &scale, const std::shared_ptr<BSDF> &bsdf, bool flipNormal = false);

    /**
     * @brief Create a quad for quadLight and add to SceneGraph shape
     * pool. A simple function to encapsulate constructor and
     * loadShape().
     *
     * @param[in] materialIndex  material index to materialBuffer
     * @param[in] position
     * @param[in] rotation      XYZ rotation angle in radian
     * @param[in] scale         Z-component is zero
     * @param[in] quadLightIndex index to |quadLightBuffer|
     * @param[in] flipNormal     flip quad's normal
     */
    std::shared_ptr<Quad> createQuad(int32_t materialIndex, const optix::float3 &position, const optix::float3 &rotation, const optix::float3 &scale, int32_t quadLightIndex, const std::shared_ptr<BSDF> &bsdf, bool flipNormal = false);


    /**
     * @brief Remove a Geometry from graph.
     * @param[in] geometryShape GeometryShape to be removed
     * @note This is for Geometry only, not for GeometryTriangles.
     */
    void removeGeometry(const std::shared_ptr<GeometryShape> &geometryShape)
    {
        /* Remove child node from OptiX Graph. */
        this->m_topGeometryGroup_Geometry->removeChild(geometryShape->getGeometryInstance());

        auto findItr = std::find_if(this->m_shapes_Geometry.cbegin(), this->m_shapes_Geometry.cend(), 
            [&geometryShape](const auto& geometryShapePtr)
        {
            return geometryShapePtr.get() == geometryShape.get();
        });

        /* Erase GeometryShape. */
        TW_ASSERT(findItr != this->m_shapes_Geometry.cend());
        this->m_shapes_Geometry.erase(findItr);

        this->rebuildGeometry();
    }

    /**
     * @brief Remove a GeometryTriangles from graph.
     * @param[in] geometryTrianglesShape GeometryTrianglesShape to be removed
     * @note This is for GeometryTriangles only, not for Geometry.
     */
    void removeGeometryTriangles(const std::shared_ptr<GeometryTrianglesShape> &geometryTrianglesShape)
    {
#ifndef CL_USE_OLD_TRIMESH
        /* Remove child node from OptiX Graph. */
        this->m_topGeometryGroup_GeometryTriangles->removeChild(geometryTrianglesShape->getGeometryInstance());
        
        auto findItr = std::find_if(this->m_shapes_GeometryTriangles.cbegin(), this->m_shapes_GeometryTriangles.cend(),
            [&geometryTrianglesShape](const auto& geometryTrianglesShapePtr)
        {
            return geometryTrianglesShapePtr.get() == geometryTrianglesShape.get();
        });

        /* Erase GeometryTrianglesShape. */
        TW_ASSERT(findItr != this->m_shapes_GeometryTriangles.cend());
        this->m_shapes_GeometryTriangles.erase(findItr);

        this->rebuildGeometryTriangles();
#else
        __debugbreak();
#endif
    }

    /**
     * @brief Expect to rebuild |m_topGeometryGroup_Geometry| Acceleration Structure.
     */
    void rebuildGeometry()
    {
        this->m_topGeometryGroup_Geometry->getAcceleration()->markDirty();

        /* Update top group as well. */
        this->m_topGroup->getAcceleration()->markDirty();

        std::cout << "[Info] AS of m_topGeometryGroup_Geometry and m_topGroup has been marked dirty." << std::endl;
    }

    /**
     * @brief Expect to rebuild |m_topGeometryGroup_GeometryTriangles| Acceleration Structure.
     */
    void rebuildGeometryTriangles()
    {
#ifdef CL_USE_OLD_TRIMESH
        __debugbreak();
#endif
        this->m_topGeometryGroup_GeometryTriangles->getAcceleration()->markDirty();

        /* Update top group as well. */
        this->m_topGroup->getAcceleration()->markDirty();
        std::cout << "[Info] AS of m_topGeometryGroup_GeometryTriangles and m_topGroup has been marked dirty." << std::endl;
    }



    /**
     * @brief Create a sampler object and use in rendering.
     * |m_samplersMap| stores all types of samplers supported in renderer.
     * 
     * @note This method could be used for changing sampler type in Colvillea.
     * If the expected sampler is not created, it's created and added to the
     * |m_samplersMap| for later use. Otherwise, the sampler will be got from
     * |m_samplersMap|. In either case, |m_sampler| will be updated to store
     * current used sampler.
     * 
     * @param[in] samplerType  the expected sampler type for rendering
     */
    void createSampler(const CommonStructs::SamplerType &samplerType)
    {
        /* Search |m_samplersMap| for the sampler we want. */
        auto samplerItr = this->m_samplersMap.find(samplerType);
        if (samplerItr == this->m_samplersMap.end())
        {
            optix::int2 filmResolution = optix::make_int2(this->m_filmWidth, this->m_filmHeight);
            switch (samplerType)
            {
            case CommonStructs::SamplerType::HaltonQMCSampler:
            {
#ifdef USE_HALTON_SAMPLER
                this->m_sampler = HaltonSampler::createHaltonSampler(this->m_context, filmResolution);
#else
                std::cout << "[Info] An issue is found when using OptiX 6.5 to implement Halton QMC sampler using fast permuation table. Currently this sampler will fallback to Sobol QMC Sampler" << std::endl;
                this->m_sampler = SobolSampler::createSobolSampler(this->m_context, filmResolution);
#endif
                
            } 
                break;
            case CommonStructs::SamplerType::SobolQMCSampler:
                this->m_sampler = SobolSampler::createSobolSampler(this->m_context, filmResolution);
                break;
            case CommonStructs::SamplerType::IndependentSampler:
                this->m_sampler = IndependentSampler::createIndependentSampler(this->m_context);
            case CommonStructs::SamplerType::FiniteSequenceSampler:
                this->m_sampler = FiniteSampler::createFiniteSampler(this->m_context);
            default:
                std::cerr << "[Error] Expected sampler is not supported." << std::endl;
                break;
            }
            /* Insert newly created sampler type into samplers map. */
            TW_ASSERT(this->m_samplersMap.insert({ samplerType, this->m_sampler }).second);
        }
        else
        {
            /* Update current sampler. */
            this->m_sampler = samplerItr->second;
        }

        /* Setup sampler index for GPU program. */
        this->m_context["sysSamplerType"]->setInt(toUnderlyingValue(samplerType));
    }


    /**
     * @brief Create a camera object and 
     * add to scene graph. It could be used as a input parameter 
     * for instantiating a CameraController class.
     */
    void createCamera(
        const optix::Matrix4x4 & cam2world, float fov, float filmWidth, float filmHeight/*, std::function<void()> resetRenderParam = std::function<void()>()*/, float focalDistance, float lensRadius)
    {
//         if (!resetRenderParam)
//         {
//             std::cout << "[Info] resetRenderParam is specified by user" << std::endl;
//         }
//         else
//         {
//             std::cout << "[Info] resetRenderParam is empty." << std::endl;
//         }

        this->m_camera = std::make_shared<Camera>(this->m_context, this->m_programsMap, /*resetRenderParam,*/this->m_application, cam2world, fov, filmWidth, filmHeight, focalDistance, lensRadius);
    }


    


    /************************************************************************/
    /*                         Getters & Setters                            */
    /************************************************************************/
    std::shared_ptr<Camera> getCamera() const
    {
        return this->m_camera;
    }

    /**
     * @brief Return currently used integrator.
     */
    std::shared_ptr<Integrator> getIntegrator() const
    {
        return this->m_integrator;
    }

    /**
     * @brief Get DirectLighting Integrator from integrators map.
     */
    std::shared_ptr<DirectLighting> getDirectLighting() const
    {
        auto integratorItr = this->m_integratorsMap.find(IntegratorType::DirectLighting);
        TW_ASSERT(integratorItr != this->m_integratorsMap.end());
        return std::static_pointer_cast<DirectLighting>(integratorItr->second);
    }

    /**
     * @brief Get DirectLighting Integrator from integrators map.
     */
    std::shared_ptr<PathTracing> getPathTracing() const
    {
        auto integratorItr = this->m_integratorsMap.find(IntegratorType::PathTracing);
        TW_ASSERT(integratorItr != this->m_integratorsMap.end());
        return std::static_pointer_cast<PathTracing>(integratorItr->second);
    }

    /**
     * @brief Change current used integrator.
     * @param[in] integratorType  Expected integrator type to use
     */
    void changeIntegrator(IntegratorType integratorType)
    {
        auto integratorItr = this->m_integratorsMap.find(integratorType);
        TW_ASSERT(integratorItr != this->m_integratorsMap.end());
        this->m_integrator = integratorItr->second;

        for (const auto& shape : this->m_shapes_GeometryTriangles)
        {
            shape->changeIntegrator(this->m_integrator->getIntegratorMaterial());
        }

        /* Iterate all GeometryShape for adding to |m_topGeometryGroup_Geometry|. */
        for (const auto& shape : this->m_shapes_Geometry)
        {
            shape->changeIntegrator(this->m_integrator->getIntegratorMaterial());
        }
    }

    /**
     * @brief Getter for currently used sampler.
     */
    std::shared_ptr<Sampler> getSampler() const
    {
        return this->m_sampler;
    }

    /**
     * @brief Getter for currently used filter.
     */
    std::shared_ptr<Filter> getFilter() const
    {
        return this->m_filter;
    }

    /**
     * @brief Get Box Filter from filters map.
     */
    std::shared_ptr<BoxFilter> getBoxFilter() const
    {
        auto filterItr = this->m_filtersMap.find(CommonStructs::FilterType::BoxFilter);
        TW_ASSERT(filterItr != this->m_filtersMap.end());
        return std::static_pointer_cast<BoxFilter>(filterItr->second);
    }

    /**
     * @brief Get Gaussian Filter from filters map.
     */
    std::shared_ptr<GaussianFilter> getGaussianFilter() const
    {
        auto filterItr = this->m_filtersMap.find(CommonStructs::FilterType::GaussianFilter);
        TW_ASSERT(filterItr != this->m_filtersMap.end());
        return std::static_pointer_cast<GaussianFilter>(filterItr->second);
    }

    /**
     * @brief Switch to the expected type of filter.
     * @param[in] filterType  Expected filter type to use
     */
    void changeFilter(CommonStructs::FilterType filterType)
    { 
        auto filterItr = this->m_filtersMap.find(filterType);
        TW_ASSERT(filterItr != this->m_filtersMap.end());

        /* Update current filter to expected filter instance. */
        this->updateCurrentFilter(filterItr->second);
    }

    /**
     * @brief Get |m_shapes_Geometry|.
     */
    const std::vector<std::shared_ptr<GeometryShape>> &getShapes_Geometry() const
    {
        return this->m_shapes_Geometry;
    }

    /**
     * @brief Get |m_shapes_GeometryTriangles|.
     */
    const std::vector<std::shared_ptr<GeometryTrianglesShape>> &getShapes_GeometryTriangles() const
    {
        return this->m_shapes_GeometryTriangles;
    }


private:
    /************************************************************************/
    /*                             Update functions                         */
    /************************************************************************/
    void updateCurrentFilter(const std::shared_ptr<Filter> &filter)
    {
        this->m_filter = filter;

        /* Activate filter: update Context variables. */
        this->m_context["sysFilterType"]->setInt(toUnderlyingValue(filter->getFilterType()));
        CommonStructs::GPUFilter gpuFilter = this->m_filter->getCommonStructsGPUFilter();
        this->m_context["sysGPUFilter"]->setUserData(sizeof(CommonStructs::GPUFilter), &gpuFilter);
    }

private:
    Application *m_application;
    const std::map<std::string, optix::Program> &m_programsMap;
    optix::Context                               m_context;
    unsigned int m_filmWidth, m_filmHeight;

    std::map<CommonStructs::SamplerType, std::shared_ptr<Sampler>> m_samplersMap;
    std::map<IntegratorType, std::shared_ptr<Integrator>>          m_integratorsMap;
    std::map<CommonStructs::FilterType, std::shared_ptr<Filter>>   m_filtersMap;

    /// Current used integrator
    std::shared_ptr<Integrator> m_integrator;
    /// Current used sampler
    std::shared_ptr<Sampler>    m_sampler;  
    /// Current used filter
    std::shared_ptr<Filter>     m_filter;
    /// Current camera (only one camera instance is supported)
    std::shared_ptr<Camera>     m_camera;   


    std::vector<std::shared_ptr<GeometryTrianglesShape>> m_shapes_GeometryTriangles;
    std::vector<std::shared_ptr<GeometryShape>>          m_shapes_Geometry;

	optix::GeometryGroup  m_topGeometryGroup_GeometryTriangles;
    optix::GeometryGroup  m_topGeometryGroup_Geometry;
    optix::Group          m_topGroup;
};