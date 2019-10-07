#include "TriangleMesh.h"
#include "../../Application/TWAssert.h"
#include "../../Device/Toolkit/CommonStructs.h"

#include <locale>
#include <algorithm>
#include <chrono>




void TriangleMesh::loadShape(optix::Material integrator, const int materialIndex)
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
	this->setupGeometryInstance(integrator);

	this->setMaterialIndex(materialIndex);
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


//////////////////////////////////////////////////////////////////////////
//OLD CODE
// 
// void TriangleMesh::loadTriangleMesh(const std::string & objFilename)
// {
// 	//Assert that trianglemesh file is obj format
// 	auto fileIsOBJ = [](const std::string &filename)->bool
// 	{
// 		// Get the filename extension                                                
// 		std::string::size_type extension_index = filename.find_last_of(".");
// 		std::string ext = extension_index != std::string::npos ?
// 			filename.substr(extension_index + 1) :
// 			std::string();
// 		std::locale loc;
// 		for (std::string::size_type i = 0; i < ext.length(); ++i)
// 			ext[i] = std::tolower(ext[i], loc);
// 		return ext == "obj";
// 	};
// 	if(!fileIsOBJ(objFilename))
// 		throw std::runtime_error("MeshLoader: Unsupported file type for '" + objFilename + "'");
// 
// 	//Loadobj using tinyobjloader
// 	if(!this->m_shapes.empty())
// 		std::cout << "[Warning]reusing meshloader for loading obj file: " << objFilename << std::endl;
// 
// 	std::string err;
// 	bool ret = tinyobj::LoadObj(this->m_shapes, this->m_materials, err, objFilename.c_str());
// 
// 	if (!err.empty())
// 		std::cerr << err << std::endl;
// 
// 	if (!ret)
// 		throw std::runtime_error("MeshLoader: " + err);
// 
// 	//Iterate over all shapes and sum up number of vertices and triangles
// 	uint64_t num_groups_with_normals = 0;
// 	uint64_t num_groups_with_texcoords = 0;
// 	for (std::vector<tinyobj::shape_t>::const_iterator it = this->m_shapes.begin();
// 		it < this->m_shapes.end();
// 		++it)
// 	{
// 		const tinyobj::shape_t & shape = *it;
// 
// 		this->mesh.num_triangles += static_cast<int32_t>(shape.mesh.indices.size()) / 3;
// 		this->mesh.num_vertices += static_cast<int32_t>(shape.mesh.positions.size()) / 3;
// 
// 		if (!shape.mesh.normals.empty())
// 			++num_groups_with_normals;
// 
// 		if (!shape.mesh.texcoords.empty())
// 			++num_groups_with_texcoords;
// 	}
// 
// 	// We ignore normals and texcoords unless they are present for all shapes
// 	if (num_groups_with_normals != 0)
// 	{
// 		if (num_groups_with_normals != m_shapes.size())
// 			std::cerr << "MeshLoader - WARNING: mesh '" << objFilename
// 			<< "' has normals for some groups but not all.  "
// 			<< "Ignoring all normals." << std::endl;
// 		else
// 			this->mesh.has_normals = true;
// 	}
// 
// 	if (num_groups_with_texcoords != 0)
// 	{
// 		if (num_groups_with_texcoords != m_shapes.size())
// 			std::cerr << "MeshLoader - WARNING: mesh '" << objFilename
// 			<< "' has texcoords for some groups but not all.  "
// 			<< "Ignoring all texcoords." << std::endl;
// 		else
// 			this->mesh.has_texcoords = true;
// 	}
// 
// 	//allocate Mesh for storage
// 	//note that we use map() mapping buffers to pointers so that we don't need to allocate memory for storage(also never delete the pointers in [mesh] using destructor
// 	//this->allocateMesh();
// 
// 	//allocate Buffers for device
// 	this->allocateBuffers();
// 
// 	//extract tinyobjloader mesh structure into Mesh
// 	this->extractImplMesh();
// 
// 	//initialize geometry related node
// 	this->initGeometry();
// 
// 	//cleanup
// 	this->unmapBuffers();
// }

// void TriangleMesh::attachOpacityMaskProfile()
// {
// 	auto &programsMap = this->application->getProgramsMap();
// 
// 	/*find the corresponding anyhit program with opacity cutout support*/
// 	auto programItr = programsMap.find("AnyHit_ShadowRay_TriangleMesh_Cutout");
// 	TW_ASSERT(programItr != programsMap.end());
// 	optix::Program anyHitProgram = programItr->second;
// 
// 	programItr = programsMap.find("AnyHit_RDPT_TriangleMesh_Cutout");
// 	TW_ASSERT(programItr != programsMap.end());
// 	optix::Program anyHitRDPTProgram = programItr->second;
// 
// 	//review:updated material automatically?
// 	this->material->setAnyHitProgram(toUnderlyingValue(CommonStructs::RayType::Shadow), anyHitProgram);
// 	this->material->setAnyHitProgram(PRDType::RADIANCE, anyHitRDPTProgram);
// 	this->material->setAnyHitProgram(toUnderlyingValue(CommonStructs::RayType::Detection), anyHitRDPTProgram);
// }

// 
// void TriangleMesh::allocateBuffers()
// {
// 	optix::Context context = application->getContext();
// 	this->triangleIndicesBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT3, this->m_primitiveCount);
// 	this->positionsBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, this->m_verticesCount);
// 	this->normalsBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3,
// 		this->mesh.has_normals ? this->mesh.num_vertices : 0);
// 	this->texcoordsBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2,
// 		this->mesh.has_texcoords ? this->mesh.num_vertices : 0);
// 
// 	this->mesh.tri_indices = reinterpret_cast<int32_t*>(this->triangleIndicesBuffer->map());
// 	this->mesh.positions = reinterpret_cast<float*>(this->positionsBuffer->map());
// 	this->mesh.normals = reinterpret_cast<float*>(this->mesh.has_normals ? this->normalsBuffer->map() : 0);
// 	this->mesh.texcoords = reinterpret_cast<float*>(this->mesh.has_texcoords ? this->texcoordsBuffer->map() : 0);
// }

//void TriangleMesh::extractImplMesh()
//{
//	if (!this->validateMesh())
//	{
//		std::cerr << "MeshLoader - ERROR: Attempted to load mesh '" 
//			<< "' into invalid mesh struct:" << std::endl;
//		return;
//	}
//
//	uint32_t vrt_offset = 0;//per vertex
//	uint32_t tri_offset = 0;//per triangle
//	//Iterate all shapes:
//	for (std::vector<tinyobj::shape_t>::const_iterator it = this->m_shapes.begin();
//		it < this->m_shapes.end();
//		++it)
//	{
//		const tinyobj::shape_t & shape = *it;
//		//Iterate all vertices
//		for (uint64_t i = 0; i < shape.mesh.positions.size() / 3; ++i)
//		{
//			//Get ith vertex position
//			const float x = shape.mesh.positions[i * 3 + 0];
//			const float y = shape.mesh.positions[i * 3 + 1];
//			const float z = shape.mesh.positions[i * 3 + 2];
//
//			/*mesh.bbox_min[0] = std::min<float>(mesh.bbox_min[0], x);
//			mesh.bbox_min[1] = std::min<float>(mesh.bbox_min[1], y);
//			mesh.bbox_min[2] = std::min<float>(mesh.bbox_min[2], z);
//
//			mesh.bbox_max[0] = std::max<float>(mesh.bbox_max[0], x);
//			mesh.bbox_max[1] = std::max<float>(mesh.bbox_max[1], y);
//			mesh.bbox_max[2] = std::max<float>(mesh.bbox_max[2], z);*/
//
//			//Put all vertices(shared vertices pool) in current shape into aggregate array
//			mesh.positions[vrt_offset * 3 + i * 3 + 0] = x;
//			mesh.positions[vrt_offset * 3 + i * 3 + 1] = y;
//			mesh.positions[vrt_offset * 3 + i * 3 + 2] = z;
//		}
//
//		//Put all normals(shared normals pool) in current shape into aggregate array
//		if (mesh.has_normals)
//			for (uint64_t i = 0; i < shape.mesh.normals.size(); ++i)
//				mesh.normals[vrt_offset * 3 + i] = shape.mesh.normals[i];
//
//		//Put all texcoords(shared texcoords pool) in current shape into aggregate array
//		if (mesh.has_texcoords)
//			for (uint64_t i = 0; i < shape.mesh.texcoords.size(); ++i)
//				mesh.texcoords[vrt_offset * 2 + i] = shape.mesh.texcoords[i];
//
//		//Put all tri_indices(shared tri_indices pool) in current shape into aggregate array
//		for (uint64_t i = 0; i < shape.mesh.indices.size() / 3; ++i)
//		{
//			//manually handle index offset
//			mesh.tri_indices[tri_offset * 3 + i * 3 + 0] = shape.mesh.indices[i * 3 + 0] + vrt_offset;
//			mesh.tri_indices[tri_offset * 3 + i * 3 + 1] = shape.mesh.indices[i * 3 + 1] + vrt_offset;
//			mesh.tri_indices[tri_offset * 3 + i * 3 + 2] = shape.mesh.indices[i * 3 + 2] + vrt_offset;
//		}
//
//		vrt_offset += static_cast<uint32_t>(shape.mesh.positions.size()) / 3;
//		tri_offset += static_cast<uint32_t>(shape.mesh.indices.size()) / 3;
//	}
//}
//
//void TriangleMesh::initGeometry()
//{
//	try
//	{
//		auto &programsMap = this->application->getProgramsMap();
//
//		auto programItr = programsMap.find("BoundingBox_TriangleMesh");
//		TW_ASSERT(programItr != programsMap.end());
//		optix::Program boundingBoxProgram = programItr->second;
//
//		programItr = programsMap.find("Intersect_TriangleMesh");
//		TW_ASSERT(programItr != programsMap.end());
//		optix::Program intersectProgram = programItr->second;
//
//		this->geometry = this->application->getContext()->createGeometry();
//		this->geometry->setPrimitiveCount(this->mesh.num_triangles);
//		this->geometry->setBoundingBoxProgram(boundingBoxProgram);
//		this->geometry->setIntersectionProgram(intersectProgram);
//		this->geometry["vertex_buffer"]->setBuffer(this->positionsBuffer);
//		this->geometry["normal_buffer"]->setBuffer(this->normalsBuffer);
//		this->geometry["texcoord_buffer"]->setBuffer(this->texcoordsBuffer);
//		this->geometry["index_buffer"]->setBuffer(this->triangleIndicesBuffer);
//
//#define USE_DIRECTLIGHTING
//#ifdef USE_DIRECTLIGHTING
//		programItr = programsMap.find("ClosestHit_DirectLighting");
//#else
//		programItr = programsMap.find("ClosestHit_PathTracing");
//#endif // USE_DIRECTLIGHTING
//		TW_ASSERT(programItr != programsMap.end());
//		optix::Program closestHitProgram = programItr->second;
//
//#ifndef USE_DIRECTLIGHTING
//		programItr = programsMap.find("ClosestHit_PTRay_PathTracing");
//		TW_ASSERT(programItr != programsMap.end());
//		optix::Program closestHit_PT_Program = programItr->second;
//#endif
//
//		programItr = programsMap.find("AnyHit_ShadowRay_TriangleMesh");
//		TW_ASSERT(programItr != programsMap.end());
//		optix::Program anyHitProgram = programItr->second;
//
//		this->material = this->application->getContext()->createMaterial();
//		this->material->setClosestHitProgram(PRDType::RADIANCE, closestHitProgram);
//		this->material->setAnyHitProgram(PRDType::SHADOW, anyHitProgram);
//#ifndef USE_DIRECTLIGHTING
//		this->material->setClosestHitProgram(PRDType::PT, closestHit_PT_Program);
//#endif
//
//		this->geometryInstance = this->application->getContext()->createGeometryInstance();
//		this->geometryInstance->setGeometry(this->geometry);
//		this->geometryInstance->addMaterial(this->material);
//		
//	}
//	catch (optix::Exception& e)
//	{
//		std::cerr << e.getErrorString() << std::endl;
//	}
//	
//}
//
//void TriangleMesh::unmapBuffers()
//{
//	this->triangleIndicesBuffer->unmap();
//	this->positionsBuffer->unmap();
//	if (this->mesh.has_normals)
//		this->normalsBuffer->unmap();
//	if (this->mesh.has_texcoords)
//		this->texcoordsBuffer->unmap();
//}
//
//bool TriangleMesh::validateMesh() const
//{
//	if (this->mesh.num_vertices == 0)
//	{
//		std::cerr << "Mesh not valid: num_vertices = 0" << std::endl;
//		return false;
//	}
//	if (this->mesh.positions == 0)
//	{
//		std::cerr << "Mesh not valid: positions = NULL" << std::endl;
//		return false;
//	}
//	if (this->mesh.num_triangles == 0)
//	{
//		std::cerr << "Mesh not valid: num_triangles = 0" << std::endl;
//		return false;
//	}
//	if (this->mesh.tri_indices == 0)
//	{
//		std::cerr << "Mesh not valid: tri_indices = NULL" << std::endl;
//		return false;
//	}
//	if (this->mesh.has_normals && !this->mesh.normals)
//	{
//		std::cerr << "Mesh has normals, but normals is NULL" << std::endl;
//		return false;
//	}
//	if (this->mesh.has_texcoords && !this->mesh.texcoords)
//	{
//		std::cerr << "Mesh has texcoords, but texcoords is NULL" << std::endl;
//		return false;
//	}
//	return true;
//}
