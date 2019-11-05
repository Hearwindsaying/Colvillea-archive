#pragma once
/* This file is host side only. */ 

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include <map>

#include "colvillea/Application/GlobalDefs.h"
#include "colvillea/Application/TWAssert.h"


class Application;

/**
 * @brief Light class serves as a base class 
 * for all supported Light types in Colvillea.
 * This base class provides interfaces to 
 * initialize Bindless Callable program group.?
 * Any specific light type should own a common
 * struct to represent data that might be used
 * in GPU kernel.
 * 
 * @note Light class is not responsible for the
 * work that initialize Bindless Callable program
 * group, which is context related and should be
 * loaded once. todo:possibly delegate to lightPool??
 * See also 
 * Application::loadCallableProgramGroup().
 * 
 * todo: design decoupling:loading Light programs.
 */
class Light : public IEditableObject
{
public:
    Light(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const std::string &lightClassName, const std::string &lightObjectName, IEditableObject::IEditableObjectType objectType) : 
        IEditableObject(lightObjectName, objectType),
        m_context(context), m_programsMap(programsMap)
    {    
        std::cout << "[Info] Derived class name from Light is: " << lightClassName << std::endl;

        /* Load Sample_Ld and lightPdf programs. */
        auto programItr = this->m_programsMap.find("Sample_Ld_" + lightClassName);
        TW_ASSERT(programItr != this->m_programsMap.end());
        this->m_sample_Ld = programItr->second;

        programItr = this->m_programsMap.find("LightPdf_" + lightClassName);
        TW_ASSERT(programItr != this->m_programsMap.end());
        this->m_lightPdf = programItr->second;
    }

    //virtual void getCommonStructsLight()

protected:
    /**
     * @brief Preprocess job for creating a specific
     * light, such as computing light distribution
     * for HDRILight and setting light buffers.
     */
    virtual void preprocess()
    {

    }

protected:
    optix::Context m_context;
    std::map<std::string, optix::Program> m_programsMap;


    optix::Program     m_sample_Ld;
    optix::Program     m_lightPdf;

    //light properties go here:

};