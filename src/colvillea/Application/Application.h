#pragma once


#include <optix.h>
#include <optixu/optixpp_namespace.h>
#include <optixu_matrix_namespace.h>

#include <iostream>
#include <iomanip>
#include <map>
#include <memory>

#define  CL_CHECK_MEMORY_LEAKS
#ifdef CL_CHECK_MEMORY_LEAKS
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#define CL_CHECK_MEMORY_LEAKS_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
#define new CL_CHECK_MEMORY_LEAKS_NEW
#endif

class SceneGraph;
struct GLFWwindow;

class CameraController;

class Application
{
public:
    struct OptixUsageReportLogger
    {
        void log(int lvl, const char* tag, const char* msg)
        {
            std::cout << "[" << lvl << "][" << std::left << std::setw(12) << tag << "] " << msg;
        }
    };

public:
    /**
     * @brief Create application and initialize ray tracer.
     * The initialization job doesn't include sceneGraph.
     * Call Application::InitializeSceneGraph() once sceneGraph
     * is ready.
     * 
     * @see Application::InitializeSceneGraph()
     */
    Application(GLFWwindow* glfwWindow, const uint32_t filmWidth, const uint32_t filmHeight, const int optixReportLevel, const uint32_t optixStackSize);

    /**
     * @brief Initialize sceneGraph for application and validate.
     */
    void InitializeSceneGraph(std::unique_ptr<SceneGraph> &sceneGraph);
    

    /**
     * @brief Create a SceneGraph object to contain all
     * information related to rendering.
     */
     //void createSceneGraph();

    void resetRenderParams()
    {
        m_resetRenderParamsNotification = true;
    }

    optix::Context getContext() const
    { 
        return this->m_context; 
    }

    const std::map<std::string, optix::Program> &getProgramsMap() const 
    { 
        return this->m_programsMap; 
    }

    /**
     * @brief Apply camera transformation and launch GPU kernel
     * to render. Display result in RenderView after filtering.
     */
    void render();

    /**
     * @brief Use widgets from Dear ImGui to draw User Interface.
     */
    void drawWidget();

    /**
     * @brief Get *.ptx files from sampleConfig.h
     * @param[in] program Program name to get ptx.
     */
    std::string getPTXFilepath(const std::string &program);

    /**
     * @brief Destroy context and release resouces.
     */
    ~Application();

private:
    /**
     * @brief Fetch and display some GPU device information
     * without context.
     * 
     * @note This function should be called once inside
     * constructor.
     * 
     * @ref OptiX Advanced Samples
     */
    void outputDeviceInfo();

    /**
     * @brief Create programs from *.ptx file and add to
     * programs map for access later.
     *
     * @note This function should be called once inside
     * setupContext().
     *
     * @see Application::setupContext()
     */
    void createProgramsFromPTX();

	/**
	 * @brief Setup OptiX context, including setting exception
	 * program, miss program, entries information and system-wide
	 * parameters.
	 * 
	 * @note This function should be called once.
	 */
	void initializeContext();

    /**
     * @brief Setup Dear ImGui context. Load style information
     * for GUI.
     *
     * @note This function should be called once.
     */
    void initializeImGui(GLFWwindow *glfwWindow);

    /**
     * @brief Setup RenderView for displaying output buffer
     * from ray tracing, including OpenGL status machine 
     * setup, generating textures for PBO.
     *
     * @note This function should be called once.
     */
    void initializeRenderView();


    /**
     * @brief Load and setup bindless callable program group for
     * BSDFs and Lights. Perhaps add samplers and cameras in the 
     * future.
     * 
     * @note This function should be called once.
     */
    void initializeCallableProgramGroup();


    /**
     * @brief Setup output buffers accounting for progressive
     * filtering. Note that reconstruction and filtering are
     * closely related to the render procedure (filtering is
     * considered as a post-processing stage). Consequently,
     * Application is in charge of creating output buffers,
     * weighting buffers and launching post-processing stage
     * for progressive fitering.
     *
     * @note This function should be called once.
     */
    void initializeOutputBuffers();


    

    /**
     * @brief Handle input event from mouse and send to CameraController
     * in order to manipulate RenderView via mouse buttons.
     */
    void handleInputEvent(bool dispatchMouseInput);

    

    /**
     * @brief Draw RenderView using OpenGL & Dear ImGui.
     */
    void drawRenderView();

private:

	std::map<std::string, optix::Program> m_programsMap;
    std::unique_ptr<CameraController>     m_cameraController;

    optix::Context m_context;
    RTsize         m_stackSize;
    int            m_optixReportLevel;
    OptixUsageReportLogger m_optixUsageReportLogger;

    optix::Buffer  m_sysOutputBuffer; 
    optix::Buffer  m_sysHDRBuffer;
    unsigned int   m_renderViewTexture; //todo:GLuint
    unsigned int   m_glPBO;
    unsigned int   m_filmWidth, m_filmHeight;

    unsigned int   m_sysIterationIndex;

    std::unique_ptr<SceneGraph> m_sceneGraph;

    bool m_resetRenderParamsNotification = true;
};