#include "colvillea/Module/Geometry/Quad.h"

#include "colvillea/Application/SceneGraph.h"

void Quad::updateMatrixParameter()
{
    TW_ASSERT(this->m_geometryInstance);

    optix::Matrix4x4 objectToWorld =
        optix::Matrix4x4::translate(this->m_position) *
        optix::Matrix4x4::rotate(this->m_rotationRad.x, optix::make_float3(1.f, 0.f, 0.f)) *
        optix::Matrix4x4::rotate(this->m_rotationRad.y, optix::make_float3(0.f, 1.f, 0.f)) *
        optix::Matrix4x4::rotate(this->m_rotationRad.z, optix::make_float3(0.f, 0.f, 1.f)) *
        optix::Matrix4x4::scale(this->m_scale);

    std::cout << TwUtil::getXScale(objectToWorld) << " " << TwUtil::getYScale(objectToWorld) << std::endl;

    this->m_geometryInstance["objectToWorld"]->setMatrix4x4fv(false, objectToWorld.getData());
    this->m_geometryInstance["worldToObject"]->setMatrix4x4fv(false, objectToWorld.inverse().getData());

    /* Rebuild Geometry if it's not an AreaLight.
     * Note that AreaLight's setScale() ensure invoking
     * -- LightPool's function for calling rebuildGeometry. */
    if (!this->m_isAreaLight)
    {
        TW_ASSERT(this->m_quadLightIndex == -1);
        this->m_sceneGraph->rebuildGeometry();
    }
}