#pragma once
//#include "../../Application/Application.h"

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include "colvillea/Module/Geometry/GeometryTrianglesShape.h"
#include "colvillea/Module/Geometry/GeometryShape.h"

#include "tinyobjloader/tiny_obj_loader.h"

class Application;

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
		GeometryTrianglesShape(context, programsMap, "TriangleMesh", integrator, materialIndex),
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