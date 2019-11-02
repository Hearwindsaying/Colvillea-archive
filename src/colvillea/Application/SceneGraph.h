#pragma once

#include <optix.h>
#include <optixu/optixpp_namespace.h>
#include <optixu_matrix_namespace.h>

#include <map>
#include <memory>

#include "colvillea/Module/Integrator/DirectLighting.h"
#include "colvillea/Module/Integrator/PathTracing.h"
#include "colvillea/Module/Geometry/TriangleMesh.h"
#include "colvillea/Module/Geometry/Quad.h"
#include "colvillea/Module/Sampler/HaltonSampler.h"
#include "colvillea/Module/Sampler/SobolSampler.h"
#include "colvillea/Module/Sampler/IndependentSampler.h"

#include "colvillea/Module/Filter/BoxFilter.h"
#include "colvillea/Module/Filter/GaussianFilter.h"

#include "colvillea/Module/Camera/Camera.h"
#include "colvillea/Application/GlobalDefs.h"

class Application;



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
        optix::Acceleration geometryAccel     = context->createAcceleration("Trbvh");
        optix::Acceleration groupAccel = context->createAcceleration("Trbvh");
        

        /* Set accelerations. */
        this->m_topGeometryGroup_GeometryTriangles->setAcceleration(geometryTrianglesAccel);
        this->m_topGeometryGroup_Geometry->setAcceleration(geometryAccel);
        this->m_topGroup->setAcceleration(groupAccel);


        /* Add GeometryTriangles GeometryGroup and Geometry GeometryGroup to top Group. */
        this->m_topGroup->addChild(this->m_topGeometryGroup_GeometryTriangles);
        this->m_topGroup->addChild(this->m_topGeometryGroup_Geometry);
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
     * BoxFilter with radius = 1.0f will be used.
     * @todo The default parameters for creating integrators should be done with
     * config file.
     */
    void initializeLoadingFilters()
    {
        /* Default values. */
        constexpr float radius = 1.f;
        std::shared_ptr<Filter> filter = BoxFilter::createBoxFilter(this->m_context);
        TW_ASSERT(this->m_filtersMap.insert({ CommonStructs::FilterType::BoxFilter, filter }).second);
        this->updateCurrentFilter(filter);

        constexpr float gaussianAlpha = 0.25f;
        filter = GaussianFilter::createGaussianFilter(this->m_context, radius);
        TW_ASSERT(this->m_filtersMap.insert({ CommonStructs::FilterType::GaussianFilter, filter }).second);
    }

public:
    /************************************************************************/
    /*                 Scene configuration creating functions               */
    /************************************************************************/ 

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
     * @param[in]  Expected integrator type to use
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
	 * @brief Create TriangleMesh and add to SceneGraph shape
	 * pool. A simple function to encapsulate constructor and
	 * loadShape().
	 * 
	 * @param[in] meshFilename wavefront obj filename with path
	 * @param[in] materialIndex material index to materialBuffer
	 * 
	 * @note Note that this operation doesn't invoke buildGraph()
	 * which is necessary to call explicitly eventually.
	 */
	void createTriangleMesh(const std::string & meshFileName, const int materialIndex)
	{
        /* Converting unique_ptr to shared_ptr. */
        std::shared_ptr<TriangleMesh> triMesh = TriangleMesh::createTriangleMesh(this->m_context, this->m_programsMap, meshFileName, this->m_integrator->getIntegratorMaterial(), materialIndex);

        m_shapes_GeometryTriangles.push_back(triMesh);
	}

    /**
     * @brief Create a single quad and add to SceneGraph shape
     * pool. A simple function to encapsulate constructor and
     * loadShape().
     *
     * @param[in] materialIndex material index to materialBuffer
     * @param[in] objectToWorld transform matrix that does not have
     * a z-component scale
     * @param[in] flipNormal    flip quad's normal
     *
     * @note Note that this operation doesn't invoke buildGraph()
     * which is necessary to call explicitly eventually.
     */
    std::shared_ptr<Quad> createQuad(const int materialIndex, const optix::Matrix4x4 &objectToWorld, bool flipNormal = false)
    {
        //todo:assert that quad is not assigned with Emissive BSDF.//todo:delete emissive?
        //todo:review copy of Quad
        std::shared_ptr<Quad> quad = Quad::createQuad(this->m_context, this->m_programsMap, objectToWorld, this->m_integrator->getIntegratorMaterial(), materialIndex);
        if(flipNormal)
            quad->flipGeometryNormal();
        m_shapes_Geometry.push_back(quad);

        return quad;
    }

    /**
     * @brief Create a quad for quadLight and add to SceneGraph shape
     * pool. A simple function to encapsulate constructor and
     * loadShape().
     *
     * @param[in] materialIndex  material index to materialBuffer
     * @param[in] objectToWorld  transform matrix that does not have
     * a z-component scale
     * @param[in] quadLightIndex index to |quadLightBuffer|
     * @param[in] flipNormal     flip quad's normal
     *
     * @note Note that this operation doesn't invoke buildGraph()
     * which is necessary to call explicitly eventually.
     */
    std::shared_ptr<Quad> createQuad(const int materialIndex, const optix::Matrix4x4 &objectToWorld, int quadLightIndex, bool flipNormal = false)
    {
        //todo:assert that quad is not assigned with Emissive BSDF.//todo:delete emissive?
        //todo:review copy of Quad
        std::shared_ptr<Quad> quad = Quad::createQuad(this->m_context, this->m_programsMap, objectToWorld, quadLightIndex, this->m_integrator->getIntegratorMaterial(), materialIndex);
        if(flipNormal)
            quad->flipGeometryNormal();
        m_shapes_Geometry.push_back(quad);

        return quad;
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
                //this->m_sampler = HaltonSampler::createHaltonSampler(this->m_context, filmResolution);
                std::cout << "[Info] An issue is found when using OptiX 6.5 to implement Halton QMC sampler using fast permuation table. Currently this sampler will fallback to Sobol QMC Sampler" << std::endl;
                this->m_sampler = SobolSampler::createSobolSampler(this->m_context, filmResolution);
            } 
                break;
            case CommonStructs::SamplerType::SobolQMCSampler:
                this->m_sampler = SobolSampler::createSobolSampler(this->m_context, filmResolution);
                break;
            case CommonStructs::SamplerType::IndependentSampler:
                this->m_sampler = IndependentSampler::createIndependentSampler(this->m_context);
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
     * @brief Getter for currently used sampler.
     */
    std::shared_ptr<Sampler> getSampler() const 
    {
        return this->m_sampler;
    }

    /**
     * @brief Create a camera object and 
     * add to scene graph. It could be used as a input parameter 
     * for instantiating a CameraController class.
     */
    void createCamera(
        const optix::Matrix4x4 & cam2world, float fov, float filmWidth, float filmHeight/*, std::function<void()> resetRenderParam = std::function<void()>()*/)
    {
//         if (!resetRenderParam)
//         {
//             std::cout << "[Info] resetRenderParam is specified by user" << std::endl;
//         }
//         else
//         {
//             std::cout << "[Info] resetRenderParam is empty." << std::endl;
//         }

        this->m_camera = std::make_shared<Camera>(this->m_context, this->m_programsMap, /*resetRenderParam,*/this->m_application, cam2world, fov, filmWidth, filmHeight);
    }

	/**
	 * @brief Collect all related nodes and build graph for OptiX.
	 * This function should be called once all shapes are created.
	 * 
	 * @note In current implementation, it's not supported instancing
	 * so we gather all geometryInstance to share a parent of 
	 * geometryGroup and one acceleration structure without any
	 * transform nodes availiable.
	 */
	void buildGraph()
	{
		/* Iterate all GeometryTrianglesShape for adding to |m_topGeometryGroup_GeometryTriangles|. */
		for (const auto& shape : this->m_shapes_GeometryTriangles)
		{
			this->m_topGeometryGroup_GeometryTriangles->addChild(shape->getGeometryInstance());
		}

        /* Iterate all GeometryShape for adding to |m_topGeometryGroup_Geometry|. */
        for (const auto& shape : this->m_shapes_Geometry)
        {
            this->m_topGeometryGroup_Geometry->addChild(shape->getGeometryInstance());
        }


        this->m_context["sysTopObject"]->set(this->m_topGroup);
        this->m_context["sysTopShadower"]->set(this->m_topGroup);
	}



    /************************************************************************/
    /*                         Getters & Setters                            */
    /************************************************************************/
    std::shared_ptr<Camera> getCamera() const
    {
        return this->m_camera;
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