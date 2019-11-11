#include "colvillea/Application/SceneGraph.h"

#include "colvillea/Module/Material/MaterialPool.h"

void SceneGraph::createTriangleMesh(const std::string & meshFileName, int materialIndex, const std::shared_ptr<BSDF> &bsdf)
{
    /* Converting unique_ptr to shared_ptr. */
    std::shared_ptr<TriangleMesh> triMesh = TriangleMesh::createTriangleMesh(this->m_context, this->m_programsMap, meshFileName, this->m_integrator->getIntegratorMaterial(), materialIndex);
    /* Bind BSDF's corresponding shape. */
    bsdf->setShape(triMesh);

    this->m_shapes_GeometryTriangles.push_back(triMesh);

    /* Update OptiX Graph. */
    this->m_topGeometryGroup_GeometryTriangles->addChild(triMesh->getGeometryInstance());
    this->rebuildGeometryTriangles();
}

std::shared_ptr<Quad> SceneGraph::createQuad(SceneGraph *sceneGraph, const int materialIndex, const optix::float3 &position, const optix::float3 &rotation, const optix::float3 &scale, const std::shared_ptr<BSDF> &bsdf, bool flipNormal)
{
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

std::shared_ptr<Quad> SceneGraph::createQuad(const int materialIndex, const optix::float3 &position, const optix::float3 &rotation, const optix::float3 &scale, int quadLightIndex, const std::shared_ptr<BSDF> &bsdf, bool flipNormal)
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