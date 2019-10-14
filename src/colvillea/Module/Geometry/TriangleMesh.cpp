#include "colvillea/Module/Geometry/TriangleMesh.h"

#include <algorithm>
#include <chrono>
#include <locale>

#include "colvillea/Application/TWAssert.h"
#include "colvillea/Device/Toolkit/CommonStructs.h"



void TriangleMesh::initializeShape()
{
	/* First, parse wavefront obj file. */
	tinyobj::attrib_t                attrib;
	std::vector<tinyobj::shape_t>    shapes;
	std::vector<tinyobj::material_t> materials;
	this->parseTriangleMesh(attrib, shapes, materials);

	/* Second, allocate necessary buffers. */
	MeshBuffer meshBuffer = this->allocateGeometryBuffers(attrib);

	/* Third, transfer data to buffers. */
	this->packGeometryBuffers(meshBuffer, attrib, shapes);

	/* Fourth, unpack buffers. */
	this->unmapGeometryBuffers(meshBuffer);

	/* Finally, create nodes for SceneGraph and initialize. */
	this->setupGeometry(meshBuffer);
	this->setupGeometryInstance(this->m_integrator);

	this->setMaterialIndex(this->m_materialIndex);
    this->m_geometryInstance["reverseOrientation"]->setInt(0);
    this->m_geometryInstance["quadLightIndex"]->setInt(-1); //todo:fix this
}

void TriangleMesh::setupGeometry(const MeshBuffer &meshBuffer)
{
    this->m_geometry = this->m_context->createGeometry();

	this->m_geometry["vertexBuffer"]->setBuffer(meshBuffer.vertexBuffer);
	this->m_geometry["normalBuffer"]->setBuffer(meshBuffer.normalBuffer);
	this->m_geometry["texcoordBuffer"]->setBuffer(meshBuffer.texcoordBuffer);
	this->m_geometry["vertexIndexBuffer"]->setBuffer(meshBuffer.vertexIndexBuffer);
	this->m_geometry["normalIndexBuffer"]->setBuffer(meshBuffer.normalIndexBuffer);
	this->m_geometry["texcoordIndexBuffer"]->setBuffer(meshBuffer.texcoordIndexBuffer);

	Shape::setupGeometry();
}

void TriangleMesh::parseTriangleMesh(tinyobj::attrib_t &outAttrib, std::vector<tinyobj::shape_t> &outShapes, std::vector<tinyobj::material_t> &outMaterials)
{
	auto getFileExtension = [](const std::string &filename)->std::string
	{
		// Get the filename extension                                                
		std::string::size_type extension_index = filename.find_last_of(".");
		std::string ext = extension_index != std::string::npos ?
			filename.substr(extension_index + 1) :
			std::string();
		std::locale loc;
		for (std::string::size_type i = 0; i < ext.length(); ++i)
			ext[i] = std::tolower(ext[i], loc);
		return ext;
	};
	if (getFileExtension(m_filename) != std::string("obj"))
		std::cerr << "[Error] TriangleMesh::parseTriangleMesh() parsing unsupported file type: " << m_filename << std::endl;
	
	std::cout << "[Info] Start parsing file:" << this->m_filename << std::endl;
	auto currentTime = std::chrono::system_clock::now();


	/* Load obj by Tinyobjloader. */
	std::string warn;
	std::string err;
	bool ret = tinyobj::LoadObj(&outAttrib, &outShapes, &outMaterials, &warn, &err, this->m_filename.c_str());

	if (!warn.empty()) 
	{
		std::cout << "[Warning] " << warn << std::endl;
	}

	if (!err.empty()) 
	{
		std::cerr << "[Error] " << err << std::endl;
	}

	if (!ret)
	{
		std::cerr << "[Error] Failed loading obj from Tinyobjloader." << std::endl;
		__debugbreak(); //todo: cope with this
		return;
	}

	if (outShapes.size() != 1)
	{
		std::cerr << "[Error] Multiple shape in a single obj file! Failed loading." << std::endl;
		__debugbreak(); //todo: cope with this
		return;
	}


	/* Collect primitive count information. */
	this->m_primitiveCount = outShapes[0].mesh.num_face_vertices.size();
	this->m_verticesCount = outAttrib.vertices.size() / 3;

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - currentTime;
	std::cout << "[Info] End parsing. Time consumed: " << (diff).count() << "s" << std::endl;
}

TriangleMesh::MeshBuffer TriangleMesh::allocateGeometryBuffers(const tinyobj::attrib_t &attrib)
{
	MeshBuffer meshBuffer;

	optix::Context context = this->m_context;

	this->hasNormals   = !attrib.normals.empty();
    this->hasTexcoords = !attrib.texcoords.empty();

	meshBuffer.vertexIndexBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT3, this->m_primitiveCount);
	meshBuffer.normalIndexBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT3, hasNormals ? this->m_primitiveCount : 0);
	meshBuffer.texcoordIndexBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT3, hasTexcoords ? this->m_primitiveCount : 0);

	meshBuffer.vertexBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, this->m_verticesCount);
	meshBuffer.normalBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3,
		hasNormals ? attrib.normals.size() / 3 : 0);
	meshBuffer.texcoordBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2,
		hasTexcoords ? attrib.texcoords.size() / 2 : 0);

	meshBuffer.vertexIndexArray = reinterpret_cast<int32_t*>(meshBuffer.vertexIndexBuffer->map());
	meshBuffer.normalIndexArray = reinterpret_cast<int32_t*>(meshBuffer.normalIndexBuffer->map());
	meshBuffer.texcoordIndexArray = reinterpret_cast<int32_t*>(meshBuffer.texcoordIndexBuffer->map());

	meshBuffer.vertexArray = reinterpret_cast<float*>(meshBuffer.vertexBuffer->map());
	meshBuffer.normalArray = reinterpret_cast<float*>(hasNormals ? meshBuffer.normalBuffer->map() : nullptr);
	meshBuffer.texcoordArray = reinterpret_cast<float*>(hasTexcoords ? meshBuffer.texcoordBuffer->map() : nullptr);

	return meshBuffer;
}

void TriangleMesh::packGeometryBuffers(MeshBuffer &meshBuffer, const tinyobj::attrib_t &attrib, const std::vector<tinyobj::shape_t> &shapes)
{
	std::cout << "[Info] Start packing geometry buffers:" << std::endl;
	auto currentTime = std::chrono::system_clock::now();

	RTsize vertexBufferSize;
	meshBuffer.vertexBuffer->getSize(vertexBufferSize);
	TW_ASSERT(attrib.vertices.size() == vertexBufferSize * 3);
	memcpy(meshBuffer.vertexArray, attrib.vertices.data(), sizeof(float) * attrib.vertices.size());

	RTsize normalBufferSize;
	meshBuffer.normalBuffer->getSize(normalBufferSize);
	TW_ASSERT(attrib.normals.size() == normalBufferSize * 3);//todo:review
	memcpy(meshBuffer.normalArray, attrib.normals.data(), sizeof(float) * attrib.normals.size());

	RTsize texcoordBufferSize;
	meshBuffer.texcoordBuffer->getSize(texcoordBufferSize);
	TW_ASSERT(attrib.texcoords.size() == texcoordBufferSize * 2);//todo:review
	memcpy(meshBuffer.texcoordArray, attrib.texcoords.data(), sizeof(float) * attrib.texcoords.size());

	/* AoS to SoA conversion, todo:using SSE acceleration. */
	TW_ASSERT(shapes[0].mesh.indices.size() / 3 == this->m_primitiveCount);
	for (int64_t i = 0; i < shapes[0].mesh.indices.size() / 3; ++i)
	{
		meshBuffer.vertexIndexArray[i * 3 + 0] = shapes[0].mesh.indices[i * 3 + 0].vertex_index;
		meshBuffer.vertexIndexArray[i * 3 + 1] = shapes[0].mesh.indices[i * 3 + 1].vertex_index;
		meshBuffer.vertexIndexArray[i * 3 + 2] = shapes[0].mesh.indices[i * 3 + 2].vertex_index;

        if (hasNormals)//todo:Review
        {
            meshBuffer.normalIndexArray[i * 3 + 0] = shapes[0].mesh.indices[i * 3 + 0].normal_index;
            meshBuffer.normalIndexArray[i * 3 + 1] = shapes[0].mesh.indices[i * 3 + 1].normal_index;
            meshBuffer.normalIndexArray[i * 3 + 2] = shapes[0].mesh.indices[i * 3 + 2].normal_index;
        }
		
        if (hasTexcoords)//todo:Review
        {
            meshBuffer.texcoordIndexArray[i * 3 + 0] = shapes[0].mesh.indices[i * 3 + 0].texcoord_index;
            meshBuffer.texcoordIndexArray[i * 3 + 1] = shapes[0].mesh.indices[i * 3 + 1].texcoord_index;
            meshBuffer.texcoordIndexArray[i * 3 + 2] = shapes[0].mesh.indices[i * 3 + 2].texcoord_index;
        }
		
	}

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - currentTime;
	std::cout << "[Info] End packing geometry buffers. Time consumed: " << (diff).count() << "s" << std::endl;
}

void TriangleMesh::unmapGeometryBuffers(MeshBuffer & meshBuffer)
{
	meshBuffer.vertexIndexBuffer->unmap();
	meshBuffer.normalIndexBuffer->unmap();
	meshBuffer.texcoordIndexBuffer->unmap();

	meshBuffer.vertexBuffer->unmap();
    if(this->hasNormals)
        meshBuffer.normalBuffer->unmap();
    if(this->hasTexcoords)
        meshBuffer.texcoordBuffer->unmap();
}
