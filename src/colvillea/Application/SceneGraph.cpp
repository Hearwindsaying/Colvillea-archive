#define  CL_CHECK_MEMORY_LEAKS
#ifdef CL_CHECK_MEMORY_LEAKS
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#define CL_CHECK_MEMORY_LEAKS_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
#define new CL_CHECK_MEMORY_LEAKS_NEW
#endif

#include "colvillea/Application/SceneGraph.h"

#include "colvillea/Module/Material/MaterialPool.h"

void SceneGraph::createTriangleMesh(const std::string & meshFileName, int materialIndex, const std::shared_ptr<BSDF> &bsdf)
{
#ifndef CL_USE_OLD_TRIMESH
    /* Converting unique_ptr to shared_ptr. */
    std::shared_ptr<TriangleMesh> triMesh = TriangleMesh::createTriangleMesh(this->m_context, this->m_programsMap, meshFileName, this->m_integrator->getIntegratorMaterial(), materialIndex);
    /* Bind BSDF's corresponding shape. */
    bsdf->setShape(triMesh);

    this->m_shapes_GeometryTriangles.push_back(triMesh);

    /* Update OptiX Graph. */
    this->m_topGeometryGroup_GeometryTriangles->addChild(triMesh->getGeometryInstance());
    this->rebuildGeometryTriangles();
#else
    /* Converting unique_ptr to shared_ptr. */
    std::shared_ptr<OrdinaryTriangleMesh> triMesh = OrdinaryTriangleMesh::createOrdinaryTriangleMesh(this->m_context, this->m_programsMap, meshFileName, this->m_integrator->getIntegratorMaterial(), materialIndex);
    /* Bind BSDF's corresponding shape. */
    bsdf->setShape(triMesh);

    this->m_shapes_Geometry.push_back(triMesh);

    /* Update OptiX Graph. */
    this->m_topGeometryGroup_Geometry->addChild(triMesh->getGeometryInstance());
    this->rebuildGeometry();
#endif
}

std::shared_ptr<Quad> SceneGraph::createQuad(SceneGraph *sceneGraph, int32_t materialIndex, const optix::float3 &position, const optix::float3 &rotation, const optix::float3 &scale, const std::shared_ptr<BSDF> &bsdf, bool flipNormal)
{
    //todo:review sceneGraph param (bad)
    //todo:assert that quad is not assigned with Emissive BSDF.//todo:delete emissive?
    //todo:review copy of Quad
    std::shared_ptr<Quad> quad = Quad::createQuad(sceneGraph, this->m_context, this->m_programsMap, position, rotation, scale, this->m_integrator->getIntegratorMaterial(), materialIndex);
    /* Bind BSDF's corresponding shape. */
    bsdf->setShape(quad);
    if (flipNormal)
        quad->flipGeometryNormal();
    this->m_shapes_Geometry.push_back(quad);

    /* Update OptiX Graph. */
    this->m_topGeometryGroup_Geometry->addChild(quad->getGeometryInstance());
    this->rebuildGeometry();

    return quad;
}

std::shared_ptr<Quad> SceneGraph::createQuad(int32_t materialIndex, const optix::float3 &position, const optix::float3 &rotation, const optix::float3 &scale, int32_t quadLightIndex, const std::shared_ptr<BSDF> &bsdf, bool flipNormal)
{
    //todo:assert that quad is not assigned with Emissive BSDF.//todo:delete emissive?
    //todo:review copy of Quad
    std::shared_ptr<Quad> quad = Quad::createQuad(this->m_context, this->m_programsMap, position, rotation, scale, quadLightIndex, this->m_integrator->getIntegratorMaterial(), materialIndex);
    /* Shared BSDF for QuadLight's underlying Quad. */
    //bsdf->setShape(quad);

    if (flipNormal)
        quad->flipGeometryNormal();
    this->m_shapes_Geometry.push_back(quad);

    /* Update OptiX Graph. */
    this->m_topGeometryGroup_Geometry->addChild(quad->getGeometryInstance());
    this->rebuildGeometry();

    return quad;
}

void SceneGraph::createTriangleSoup(int32_t materialIndex, const std::shared_ptr<BSDF> &bsdf, const std::vector<optix::float3> &vertices)
{
    /* Converting unique_ptr to shared_ptr. */
    std::shared_ptr<TriangleSoup> triSoup = TriangleSoup::createTriangleSoup(this, this->m_context, this->m_programsMap, this->m_integrator->getIntegratorMaterial(), materialIndex, vertices);
    /* Bind BSDF's corresponding shape. */
    bsdf->setShape(triSoup);

    this->m_shapes_GeometryTriangles.push_back(triSoup);

    /* Update OptiX Graph. */
    this->m_topGeometryGroup_GeometryTriangles->addChild(triSoup->getGeometryInstance());
    this->rebuildGeometryTriangles();
}


std::shared_ptr<Sphere> SceneGraph::createSphere(SceneGraph * sceneGraph, int32_t materialIndex, const optix::float3 & center, const float radius, const std::shared_ptr<BSDF>& bsdf)
{
    std::shared_ptr<Sphere> sphere = Sphere::createSphere(sceneGraph, this->m_context, this->m_programsMap, center, radius, this->m_integrator->getIntegratorMaterial(), materialIndex);
    /* Bind BSDF's corresponding shape. */
    bsdf->setShape(sphere);

    this->m_shapes_Geometry.push_back(sphere);

    /* Update OptiX Graph. */
    this->m_topGeometryGroup_Geometry->addChild(sphere->getGeometryInstance());
    this->rebuildGeometry();

    return sphere;
}

std::shared_ptr<Sphere> SceneGraph::createSphere(int32_t materialIndex, const optix::float3 & center, const float radius, int32_t sphereLightIndex, const std::shared_ptr<BSDF>& bsdf)
{
    std::shared_ptr<Sphere> sphere = Sphere::createSphere(this->m_context, this->m_programsMap, center, radius, sphereLightIndex, this->m_integrator->getIntegratorMaterial(), materialIndex);
    /* Shared BSDF for SphereLight's underlying Sphere. */
    //bsdf->setShape(quad);

    this->m_shapes_Geometry.push_back(sphere);

    /* Update OptiX Graph. */
    this->m_topGeometryGroup_Geometry->addChild(sphere->getGeometryInstance());
    this->rebuildGeometry();

    return sphere;
}