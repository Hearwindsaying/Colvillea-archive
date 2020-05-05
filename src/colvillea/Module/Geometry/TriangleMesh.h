#pragma once
//#include "../../Application/Application.h"

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include "colvillea/Module/Geometry/GeometryTrianglesShape.h"
#include "colvillea/Module/Geometry/GeometryShape.h"

#include "tinyobjloader/tiny_obj_loader.h"

class Application;
class SceneGraph;

/**
 * @brief TriangleMesh class serves as an auxiliary host class
 *  to help loading wavefront .obj mesh into device.
 * 
 * @note This TriangleMesh doesn't employ GeometryTriangles inside.
 * It could be used as a fallback for Megakernel execution strategy.
 * It seems that a performance issue could be in OptiX's new RTX strategy
 * introduced since OptiX 6.0. So OrdinaryTriangleMesh is provided as well.
 * Use TriangleMesh for RTX acceleration.
 * 
 * @see TriangleMesh
 */
class OrdinaryTriangleMesh : public GeometryShape
{
private:
    struct MeshBuffer
    {
        optix::Buffer vertexIndexBuffer;   // specify index to vertex position
        optix::Buffer normalIndexBuffer;   // specify index to vertex normal
        optix::Buffer texcoordIndexBuffer; // specify index to vertex texcoords
        optix::Buffer vertexBuffer;        // vertex position
        optix::Buffer normalBuffer;
        optix::Buffer texcoordBuffer;

        int   *vertexIndexArray;
        int   *normalIndexArray;
        int   *texcoordIndexArray;

        float *vertexArray;
        float *normalArray;
        float *texcoordArray;
    };


public:
    /**
     * @brief Factory method for creating a trianglemesh instance.
     *
     * @param[in] context
     * @param[in] programsMap      map to store Programs
     * @param[in] filename         obj filename
     * @param[in] integrator       integrator of optix::Material type
     * @param[in] materialIndex    material index in |materialBuffer|
     */
    static std::unique_ptr<OrdinaryTriangleMesh> createOrdinaryTriangleMesh(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const std::string &filename, optix::Material integrator, const int materialIndex)
    {
        std::unique_ptr<OrdinaryTriangleMesh> triangleMesh = std::make_unique<OrdinaryTriangleMesh>(context, programsMap, filename, integrator, materialIndex);
        triangleMesh->initializeShape();
        return triangleMesh;
    }

    /**
     * @brief Constructor for TriangleMesh.
     * @param filename filename with path
     */
    OrdinaryTriangleMesh(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const std::string &filename, optix::Material integrator, const int materialIndex) :
        GeometryShape(context, programsMap, "TriangleMesh", integrator, materialIndex, "TriangleMesh", IEditableObject::IEditableObjectType::QuadGeometry /*hack*/),
        m_filename(filename), m_verticesCount(-1), hasNormals(false), hasTexcoords(false)
    {

    }

    std::string getTriangleMeshFilename() const
    {
        return this->m_filename;
    }


    void initializeShape() override;

private:
    /**
     * @note This is not override to GeometryTrianglesShape::setupGeometry().
     */
    void setupGeometry(const MeshBuffer &meshBuffer);

    /**
     * @brief Parse TriangleMesh from wavefront obj file using
     * Tinyobjloader.
     */
    void parseTriangleMesh(tinyobj::attrib_t &outAttrib, std::vector<tinyobj::shape_t> &outShapes, std::vector<tinyobj::material_t> &outMaterials);

    /**
     * @brief Allocate geometry buffers related to TriangleMesh,
     * such as indices, positions, normals and texture coordinates.
     */
    MeshBuffer allocateGeometryBuffers(const tinyobj::attrib_t &attrib);

    /**
     * @brief Fill up allocated geometry buffers by transferring
     * data from Tinyobjloader to OptiX buffers.
     */
    void packGeometryBuffers(MeshBuffer &meshBuffer, const tinyobj::attrib_t &attrib, const std::vector<tinyobj::shape_t> &shapes);

    /**
     * @brief Do some necessary unmap work for geometry buffers.
     */
    void unmapGeometryBuffers(MeshBuffer &meshBuffer);

private:
    std::string m_filename;

    uint32_t m_verticesCount;

    bool hasNormals, hasTexcoords;
};

/** @brief TriangleMesh class serves as an auxiliary host class
 *  to help loading wavefront .obj mesh into device.
 *  
 *  @note This TriangleMesh employs GeometryTriangles inside to 
 * leverage RTX acceleration.
 */
class TriangleMesh : public GeometryTrianglesShape
{
private:
	struct MeshBuffer
	{
		optix::Buffer vertexIndexBuffer;   // specify index to vertex position
		optix::Buffer normalIndexBuffer;   // specify index to vertex normal
		optix::Buffer texcoordIndexBuffer; // specify index to vertex texcoords
		optix::Buffer vertexBuffer;        // vertex position
		optix::Buffer normalBuffer;
		optix::Buffer texcoordBuffer;

		int   *vertexIndexArray;
		int   *normalIndexArray;
		int   *texcoordIndexArray;

		float *vertexArray;
		float *normalArray;
		float *texcoordArray;
	};


public:
    /**
     * @brief Factory method for creating a trianglemesh instance.
     *
     * @param[in] context
     * @param[in] programsMap      map to store Programs
     * @param[in] filename         obj filename
     * @param[in] integrator       integrator of optix::Material type
     * @param[in] materialIndex    material index in |materialBuffer|
     */
    static std::unique_ptr<TriangleMesh> createTriangleMesh(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const std::string &filename, optix::Material integrator, const int materialIndex)
    {
        std::unique_ptr<TriangleMesh> triangleMesh = std::make_unique<TriangleMesh>(context, programsMap, filename, integrator, materialIndex);
        triangleMesh->initializeShape();
        return triangleMesh;
    }

	/**
	 * @brief Constructor for TriangleMesh.
	 * @param filename filename with path
	 */
	TriangleMesh(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const std::string &filename, optix::Material integrator, const int materialIndex):
		GeometryTrianglesShape(context, programsMap, "TriangleMesh", integrator, materialIndex, IEditableObject::IEditableObjectType::TriangleMeshGeometry),
        m_filename(filename),m_verticesCount(-1),hasNormals(false),hasTexcoords(false)
	{

	}

    std::string getTriangleMeshFilename() const
    {
        return this->m_filename;
    }

	
	void initializeShape() override;

private:
    /**
     * @note This is not override to GeometryTrianglesShape::setupGeometry().
     */
    void setupGeometry(const MeshBuffer &meshBuffer); 

	/**
	 * @brief Parse TriangleMesh from wavefront obj file using
	 * Tinyobjloader.
	 */
	void parseTriangleMesh(tinyobj::attrib_t &outAttrib, std::vector<tinyobj::shape_t> &outShapes, std::vector<tinyobj::material_t> &outMaterials);

	/**
	 * @brief Allocate geometry buffers related to TriangleMesh,
	 * such as indices, positions, normals and texture coordinates.
	 */
	MeshBuffer allocateGeometryBuffers(const tinyobj::attrib_t &attrib);

	/**
	 * @brief Fill up allocated geometry buffers by transferring
	 * data from Tinyobjloader to OptiX buffers.
	 */
	void packGeometryBuffers(MeshBuffer &meshBuffer, const tinyobj::attrib_t &attrib, const std::vector<tinyobj::shape_t> &shapes);

	/**
	 * @brief Do some necessary unmap work for geometry buffers.
	 */
	void unmapGeometryBuffers(MeshBuffer &meshBuffer);

private:
	std::string m_filename;

	uint32_t m_verticesCount;

    bool hasNormals, hasTexcoords;
};

/**
 * @brief Triangle Soup for simple individual structured triangle
 * mesh. Useful for testing simple meshes.
 * 
 * @note This TriangleSoup also employs GeometryTriangles inside to
 * leverage RTX acceleration.
 */
class TriangleSoup : public GeometryTrianglesShape
{
public:
    static std::unique_ptr<TriangleSoup> createTriangleSoup(SceneGraph *sceneGraph, optix::Context context, const std::map<std::string, optix::Program> &programsMap, optix::Material integrator, const int materialIndex, const std::vector<optix::float3> &vertices)
    {
        std::unique_ptr<TriangleSoup> triangleSoup = std::make_unique<TriangleSoup>(sceneGraph, context, programsMap, vertices, integrator, materialIndex);
        triangleSoup->initializeShape();
        return triangleSoup;
    }

    TriangleSoup(SceneGraph *sceneGraph, optix::Context context, const std::map<std::string, optix::Program> &programsMap, const std::vector<optix::float3> &vertices, optix::Material integrator, const int materialIndex) :
        GeometryTrianglesShape(context, programsMap, "TriangleSoup", integrator, materialIndex, IEditableObject::IEditableObjectType::TriangleSoupGeometry) 
    {
        this->vertices = vertices;
        this->m_sceneGraph = sceneGraph;

        /* Default transform parameters. */
        this->m_position = optix::make_float3(0.f);
        this->m_rotationRad = optix::make_float3(0.f);
        this->m_scale = optix::make_float3(1.f);
    }

    void initializeShape() override
    {
        /* Collect primitive count information. */
        this->m_primitiveCount = this->vertices.size() / 3;
        TW_ASSERT(this->m_primitiveCount * 3 == this->vertices.size());

        this->m_geometryTriangles = this->m_context->createGeometryTriangles();

        optix::Buffer vertexBuffer = this->m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, this->vertices.size());
        void *vertexBufferData = vertexBuffer->map();
        memcpy(vertexBufferData, this->vertices.data(), sizeof(optix::float3) * this->vertices.size());
        this->m_geometryTriangles["vertexBuffer"]->setBuffer(vertexBuffer);
        this->m_geometryTriangles->setVertices(this->vertices.size(), vertexBuffer, RT_FORMAT_FLOAT3);
        vertexBuffer->unmap();

#pragma region DEPRECATED
        /* Setup null buffers (if reusing TriangleMesh program). */
        /*optix::Buffer nullBufferf3 = this->m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 0);
        optix::Buffer nullBufferf2 = this->m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, 0);
        optix::Buffer nullBufferi3 = this->m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT3, 0);
        this->m_geometryTriangles["normalBuffer"]->setBuffer(nullBufferf3);
        this->m_geometryTriangles["texcoordBuffer"]->setBuffer(nullBufferf2);
        this->m_geometryTriangles["vertexIndexBuffer"]->setBuffer(nullBufferi3);
        this->m_geometryTriangles["normalIndexBuffer"]->setBuffer(nullBufferi3);
        this->m_geometryTriangles["texcoordIndexBuffer"]->setBuffer(nullBufferi3);*/
#pragma endregion DEPRECATED

        GeometryTrianglesShape::setupGeometry();
        GeometryTrianglesShape::setupGeometryInstance(this->m_integrator);

        this->setMaterialIndex(this->m_materialIndex);
    }

    optix::float3 getPosition() const
    {
        return this->m_position;
    }

    void setPosition(const optix::float3 &position)
    {
        this->m_position = position;
        this->updateMatrixParameter();
    }

    optix::float3 getRotation() const
    {
        return this->m_rotationRad;
    }

    void setRotation(const optix::float3 &rotation)
    {
        this->m_rotationRad = rotation;
        this->updateMatrixParameter();
    }

    optix::float3 getScale() const
    {
        return this->m_scale;
    }

    void setScale(const optix::float3 &scale)
    {
        TW_ASSERT(this->m_scale.z == 1.f && scale.z == 1.f);
        if (scale.z != 1.f)
        {
            std::cout << "[Info] Quad shape scale's z-component is not zero!" << std::endl;
        }

        this->m_scale = scale;
        this->updateMatrixParameter();
    }

private:
    void updateMatrixParameter();


private:
    std::vector<optix::float3> vertices;

    /// Record user-friendly transform elements.
    optix::float3 m_rotationRad;
    optix::float3 m_position;
    optix::float3 m_scale;

    /// Pointer to SceneGraph
    SceneGraph *m_sceneGraph;
};