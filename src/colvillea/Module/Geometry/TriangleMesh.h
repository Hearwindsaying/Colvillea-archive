#pragma once
//#include "../../Application/Application.h"

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include "Shape.h"
#include "../../../tinyobjloader/tiny_obj_loader.h"

class Application;

/** @brief TriangleMesh class serves as an auxiliary host class
 *  to help loading wavefront .obj mesh into device.
 */
class TriangleMesh : public Shape
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
	 * @brief Constructor for TriangleMesh.
	 * @param filename filename with path
	 */
	TriangleMesh(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const std::string &filename):
		Shape(context, programsMap, "TriangleMesh"),m_filename(filename),m_verticesCount(-1),hasNormals(false),hasTexcoords(false)
	{

	}

	
	void loadShape(optix::Material integrator, const int materialIndex) override;

private:
    void setupGeometry(const MeshBuffer &meshBuffer); //note: not override to setupGeometry

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

	//////////////////////////////////////////////////////////////////////////
	//Old Code


	//void loadTriangleMesh(const std::string & objFilename);

	void attachOpacityMaskProfile();

	/*we can safely use optix::Material as return type benefit by high level optix wrapper class using reference counting.*/
	//optix::Material getMaterial() const{ return this->material; };
	//optix::GeometryInstance getGeometryInstance() const{ return this->geometryInstance; };

	//friend class Application;
private:
	//void scanTriangleMesh();
	//void allocateBuffers();//this function also send Mesh to Buffers(use map to binding)
	//void extractImplMesh();
	//void initGeometry();
	//void unmapBuffers();//this function also unmap Buffers

	//bool validateMesh() const;

private:
// 	struct Mesh
// 	{
// 		int32_t             num_vertices;   // number of triangle vertices
// 		float*              positions;      // vertex position array (len num_vertices)
// 
// 		bool                has_normals;    //
// 		float*              normals;        // vertex normal array (len 0 or num_vertices)
// 
// 		bool                has_texcoords;  //
// 		float*              texcoords;      // vertex uv array (len 0 or num_vertices)
// 
// 
// 		int32_t             num_triangles;  // number of triangles
// 		int32_t*            tri_indices;    // index array into positions, normals, texcoords
// 
// 	public:
// 		Mesh() :num_triangles(0), num_vertices(0), positions(nullptr), normals(nullptr), texcoords(nullptr), tri_indices(nullptr), has_normals(false), has_texcoords(false) { }
// 	};

private:
	//const Application * application;

	//Mesh mesh;//never get accessed to this struct once finish loading objMesh
	//std::vector<tinyobj::shape_t> m_shapes;
	//std::vector<tinyobj::material_t> m_materials;//useless obj parsing output in current implementation

// 	optix::Buffer triangleIndicesBuffer;
// 	optix::Buffer positionsBuffer;
// 	optix::Buffer normalsBuffer;
// 	optix::Buffer texcoordsBuffer;

	//optix::Material material;
	//optix::Geometry geometry;
	//optix::GeometryInstance geometryInstance;

	int alphaTextureID;
};