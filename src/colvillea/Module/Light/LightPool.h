#pragma once

#include <vector>

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include "../../Device/Toolkit/CommonStructs.h"
#include "../../Application/TWAssert.h"


/**
 * @brief A simple pool used for creating, containing and
 * managing lights. This is analogous to MaterialPool.
 * todo: add a pool interface?
 * 
 */
//class LightPool
//{
//public:
//    LightPool(const std::map<std::string, optix::Program> &programsMap, const optix::Context context) 
//        : m_programsMap(programsMap), m_context(context)
//    {
//
//    }
//
//private:
//    const std::map<std::string, optix::Program> &m_programsMap;
//    optix::Context     m_context;
//
//    optix::Buffer      m_quadLightBuffer;
//    optix::Buffer      m_pointLightBuffer;
//};