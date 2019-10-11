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
#include "../Module/Sampler/HaltonSampler.h"
#include "../Module/Sampler/SobolSampler.h"
#include "../Module/Camera/Camera.h"
#include "GlobalDefs.h"

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
		this->initGraph();
	}

private:
    /*******************************************************************/
    /*           Initialization functions called by constructor        */
    /*******************************************************************/

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


private:
    Application *m_application;
    const std::map<std::string, optix::Program> &m_programsMap;
    optix::Context                               m_context;
    unsigned int m_filmWidth, m_filmHeight;

	std::vector<std::shared_ptr<Shape>>          m_shapes;

    std::unique_ptr<Integrator> m_integrator;
    std::unique_ptr<Sampler>    m_sampler;   // todo:We could only hold one instance
    std::shared_ptr<Camera>     m_camera;    // todo:We could only hold one instance

	optix::GeometryGroup  m_topGeometryGroup;
	optix::Acceleration   m_topAcceleration;
};