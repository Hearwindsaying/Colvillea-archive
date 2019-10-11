#pragma once

#include <optix.h>
#include <optixu/optixpp_namespace.h>
#include <optixu_matrix_namespace.h>

#include <map>
#include <memory>

#include "../Module/Integrator/DirectLighting.h"
#include "../Module/Integrator/PathTracing.h"
#include "../Module/Geometry/TriangleMesh.h"
#include "../Module/Geometry/Quad.h"
#include "../Module/Light/PointLight.h"
#include "../Module/Light/HDRILight.h"
#include "../Module/Light/QuadLight.h"
#include "../Module/Sampler/HaltonSampler.h"
#include "../Module/Sampler/SobolSampler.h"
#include "../Module/Camera/Camera.h"
#include "GlobalDefs.h"

class Application;



/**
 * @brief SceneGraph class gather integrator, shapes, materialPool,
 * sampler, lights information and provide helper utility to build 
 * up scene graph for OptiX.
 */
class SceneGraph
{
public:
    SceneGraph(Application *application, const std::map<std::string, optix::Program> &programsMap, const optix::Context context, const unsigned int &filmWidth, const unsigned int &filmHeight) :
        m_HDRILight(nullptr), m_programsMap(programsMap), m_context(context),
        m_filmWidth(filmWidth), m_filmHeight(filmHeight), m_application(application)
	{
		this->initGraph();
        this->initScene();
	}

private:
    /*******************************************************************/
    /*           Initialization functions called by constructor        */
    /*******************************************************************/

    /**
     * @brief Initialize context-wide information in scene aspects,
     * such as light, sampler, etc. This function is invoked in ctor
     * in SceneGraph and should never be called elsewhere anytime.
     *
     * @note This function is quite tricky and according to its description,
     * it should initialize callable program group as well?
     *
     * @see SceneGraph::SceneGraph()
     */
    void initScene()
    {
        /* Light initialization work that should be done once for rendering,
        independent of individual light. */
        PointLight::initPointLight(this->m_context);
        HDRILight::initHDRILight(this->m_context, this->m_programsMap);
        QuadLight::initQuadLight(this->m_context);
    }

    /**
     * @brief Initialize OptiX Graph nodes that should be set once.
     */
    void initGraph()
    {
        /* Create top geometryGroup and top Acceleration structure. */
        auto& context = this->m_context;
        this->m_topGeometryGroup = context->createGeometryGroup();

        this->m_topAcceleration = context->createAcceleration("Trbvh");//todo:use enum
        this->m_topAcceleration->setProperty("chunk_size", "-1");
        this->m_topAcceleration->setProperty("vertex_buffer_name", "vertexBuffer");
        this->m_topAcceleration->setProperty("index_buffer_name", "indexBuffer");

        this->m_topGeometryGroup->setAcceleration(this->m_topAcceleration);
    }

public:
    /************************************************************************/
    /*                 Scene configuration creating functions               */
    /************************************************************************/ 

    /**
     * @brief Create an integrator and add to sceneGraph.
     * 
     * @see DirectLighting::Integrator()
     */
    void createDirectLightingIntegrator()
    {
        this->m_integrator = DirectLighting::createIntegrator(this->m_context, this->m_programsMap);
    }

    /**
     * @brief Create an integrator and add to sceneGraph.
     *
     * @see PathTracing::Integrator()
     */
    void createPathTracingIntegrator(bool enableRoussianRoulette, int maxDepth)
    {
        this->m_integrator = PathTracing::createIntegrator(this->m_context, this->m_programsMap, enableRoussianRoulette, maxDepth);
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

		m_shapes.push_back(triMesh);
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
        m_shapes.push_back(quad);

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
        m_shapes.push_back(quad);

        return quad;
    }

    /**
     * @brief Create HDRILight object and add to SceneGraph.
     * In current implmentation, only one instance to HDRILight
     * is hold which means changing HDRI is permitted but it
     * will destroy previous existing HDRILight. Recomputation
     * for initialization of HDRILight needs to be done again.
     * 
     * @param HDRIFilename filename including path to HDRI.
     */
    void createHDRILight(const std::string & HDRIFilename, const optix::Matrix4x4 &lightToWorld)
    {
        this->m_HDRILight = std::make_shared<HDRILight>(this->m_application, this->m_context, this->m_programsMap, HDRIFilename, lightToWorld);

        this->m_HDRILight->loadLight(); 
    }

    /**
     * @brief Create a pointLight object and add to SceneGraph.
     * 
     * @param lightToWorld matrix representing position, only
     * translation component is needed for an ideal point light.
     * @param intensity representing both color and intensity,
     * whose components could be larger than 1.f.
     * 
     * @note Only adding light is supported. //todo:use LightPool
     */
    void createPointLight(const optix::Matrix4x4 &lightToWorld, const optix::float3 &intensity)
    {
        std::shared_ptr<PointLight> pointLight = std::make_shared<PointLight>(this->m_context, this->m_programsMap, intensity, lightToWorld);
        pointLight->loadLight();

        this->m_pointLights.push_back(pointLight);


        /* Setup pointLightBuffer for GPU Program */
        auto pointLightBuffer = this->m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, this->m_pointLights.size());
        pointLightBuffer->setElementSize(sizeof(CommonStructs::PointLight));
        auto pointLightArray = static_cast<CommonStructs::PointLight *>(pointLightBuffer->map());
        for (auto itr = this->m_pointLights.cbegin(); itr != this->m_pointLights.cend();  ++itr)
        {
            pointLightArray[itr - this->m_pointLights.cbegin()] = (*itr)->getCommonStructsLight();
        }
        this->m_context["pointLightBuffer"]->setBuffer(pointLightBuffer);

        pointLightBuffer->unmap();
    }

    /**
     * @brief Create a quadLight object and add to SceneGraph.
     *
     * @param[in] lightToWorld  matrix representing position, only
     * translation component is needed for an ideal point light.
     * @param[in] intensity     representing both color and intensity,
     * whose components could be larger than 1.f.
     * @param[in] materialIndex index to materialBuffer, call 
     * MaterialPool::createEmissiveMaterial() to create the material
     * for quadlight.
     * @param[in] flipNormal    flip light's geometry if necessary
     *
     * @note Only adding light is supported. //todo:use LightPool
     * @see MaterialPool::createEmissiveMaterial()
     */
    void createQuadLight(const optix::Matrix4x4 &lightToWorld, const optix::float3 &intensity, const int materialIndex, bool flipNormal = false)
    {
        std::shared_ptr<QuadLight> quadLight = std::make_shared<QuadLight>(this->m_context, this->m_programsMap, intensity,
            this->createQuad(materialIndex, lightToWorld, this->m_quadLights.size() /* size() is the index we want for the current creating quad */, flipNormal));
        quadLight->loadLight();
        this->m_quadLights.push_back(quadLight);

        /* Setup quadLightBuffer for GPU Program */
        auto quadLightBuffer = this->m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, this->m_quadLights.size());
        quadLightBuffer->setElementSize(sizeof(CommonStructs::QuadLight));
        auto quadLightArray = static_cast<CommonStructs::QuadLight *>(quadLightBuffer->map());
        for (auto itr = this->m_quadLights.cbegin(); itr != this->m_quadLights.cend(); ++itr)
        {
            quadLightArray[itr - this->m_quadLights.cbegin()] = (*itr)->getCommonStructsLight();
        }
        this->m_context["quadLightBuffer"]->setBuffer(quadLightBuffer);

        quadLightBuffer->unmap();
    }


    /**
     * @brief Create a sampler object and use in rendering. Only
     * one instance of smapler object is permitted currently.
     */
    void createSampler(const CommonStructs::SamplerType &samplerType)
    {
        optix::int2 filmResolution = optix::make_int2(this->m_filmWidth, this->m_filmHeight);
        switch (samplerType)
        {
            case CommonStructs::SamplerType::HaltonQMCSampler:
                this->m_sampler = std::make_unique<HaltonSampler>(this->m_context, filmResolution);
                break;
            case CommonStructs::SamplerType::SobolQMCSampler:
                this->m_sampler = std::make_unique<SobolSampler>(this->m_context, filmResolution);
                break;
            default:
                std::cerr << "[Error] Expected sampler is not supported." << std::endl;
                break;
        }

        /* Initialize sampler. */
        this->m_sampler->initSampler();

        /* Setup sampler index for GPU program. */
        this->m_context["sysSamplerType"]->setInt(toUnderlyingValue(samplerType));
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
		/* Iterate all shapes for adding to geometryGroup. */
		for (const auto& shape : this->m_shapes)
		{
			this->m_topGeometryGroup->addChild(shape->getGeometryInstance());
		}

        this->m_context["sysTopObject"]->set(this->m_topGeometryGroup);
        this->m_context["sysTopShadower"]->set(this->m_topGeometryGroup);
	}



    /************************************************************************/
    /*                         Getters & Setters                            */
    /************************************************************************/
    std::shared_ptr<Camera> getCamera() const
    {
        return this->m_camera;
    }
    std::shared_ptr<HDRILight> getHDRILight() const
    {
        return this->m_HDRILight;
    }


private:
    Application *m_application;
    const std::map<std::string, optix::Program> &m_programsMap;
    optix::Context                               m_context;
    unsigned int m_filmWidth, m_filmHeight;

	std::vector<std::shared_ptr<Shape>>          m_shapes;

	//adding sampler, materialPool, lights etc. here:
    std::shared_ptr<HDRILight> m_HDRILight; // todo: We could only hold one instance of HDRILight
    std::vector<std::shared_ptr<PointLight>>    m_pointLights;
    std::vector<std::shared_ptr<QuadLight>>     m_quadLights; 

    std::unique_ptr<Integrator> m_integrator;
    std::unique_ptr<Sampler>    m_sampler;   // todo:We could only hold one instance
    std::shared_ptr<Camera>     m_camera;    // todo:We could only hold one instance

	optix::GeometryGroup  m_topGeometryGroup;
	optix::Acceleration   m_topAcceleration;
};