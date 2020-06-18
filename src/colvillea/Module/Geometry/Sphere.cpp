#include "colvillea/Module/Geometry/Sphere.h"

#include "colvillea/Application/SceneGraph.h"

void Sphere::updateMatrixParameter()
{
    TW_ASSERT(this->m_geometryInstance);

    optix::Matrix4x4 objectToWorld =
        optix::Matrix4x4::translate(this->m_position) *
        optix::Matrix4x4::scale(optix::make_float3(this->m_radius));

    this->m_geometryInstance["objectToWorld"]->setMatrix4x4fv(false, objectToWorld.getData());
    this->m_geometryInstance["worldToObject"]->setMatrix4x4fv(false, objectToWorld.inverse().getData());
    this->m_geometryInstance["center"]->setFloat(this->m_position);
    this->m_geometryInstance["radius"]->setFloat(this->m_radius);


    /* Rebuild Geometry if it's not an AreaLight.
     * Note that AreaLight's setScale() ensure invoking
     * -- LightPool's function for calling rebuildGeometry. */
    if (!this->m_isAreaLight)
    {
        TW_ASSERT(this->m_sphereLightIndex == -1);
        this->m_sceneGraph->rebuildGeometry();
    }
}