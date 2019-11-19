#pragma once

#include <chrono>
#include <functional>
#include <iostream>
#include <iomanip>
#include <map>
#include <memory>

#include <optix.h>
#include <optixu/optixpp_namespace.h>
#include <optixu_matrix_namespace.h>

//#define  CL_CHECK_MEMORY_LEAKS
//#ifdef CL_CHECK_MEMORY_LEAKS
//#define _CRTDBG_MAP_ALLOC
//#include <stdlib.h>
//#include <crtdbg.h>
//#define CL_CHECK_MEMORY_LEAKS_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
//#define new CL_CHECK_MEMORY_LEAKS_NEW
//#endif

class SceneGraph;
class LightPool;
class IEditableObject;

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
     * Call Application::buildSceneGraph() once sceneGraph
     * is ready.
     * 
     * @see Application::buildSceneGraph()
     */
    Application(GLFWwindow* glfwWindow, uint32_t filmWidth, uint32_t filmHeight, int optixReportLevel);

    /**
     * @brief Destroy context and release resouces.
     */
    ~Application();

    /**
     * @brief Build sceneGraph for application and validate.
     * This should be called when scene is ready for rendering
     * before calling Application::render().
     */
    void buildSceneGraph(std::shared_ptr<SceneGraph> &sceneGraph);

    /**
     * @brief Use widgets from Dear ImGui to draw User Interface.
     */
    void drawWidget();

    /**
     * @brief Apply camera transformation and launch GPU kernel
     * to render. Display result in RenderView after filtering.
     */
    void render();

private:
    /**
     * @brief Handle input event from mouse and send to CameraController
     * in order to manipulate RenderView via mouse buttons.
     * This function is called by Application::drawRenderView().
     */
    void handleInputEvent(bool dispatchMouseInput);

    /**
     * @brief Draw RenderView using OpenGL & Dear ImGui.
     * This function is called by Application::render().
     */
    void drawRenderView();

    /**
     * @brief Draw DockSpace.
     * 
     * @note This is from Dear ImGui Demo.
     */
    void drawDockSpace();

    /**
     * @brief Draw Settings window.
     */
    void drawSettings();

    /**
     * @brief Draw Inspector window.
     */
    void drawInspector();

    /**
     * @brief Draw Hierarchy window.
     */
    void drawHierarchy();

    /**
     * @brief Draw Material Hierarchy window.
     */
    void drawMaterialHierarchy();

public:
    /**
     * @brief Force to relaunch progressive rendering.
     * Note that this will clear up data accmulated before
     * and current iteration goes from 0 again.
     * It could be useful when we have edited scene such as
     * moving camera and changing SceneGraph parameters and
     * under that circumstance, we want to simply discard
     * all previous rendered data and start rendering again.
     * 
     * @see Camera::updateCameraMatrices()
     */
    void resetRenderParams()
    {
        m_resetRenderParamsNotification = true;
    }

public:
    /************************************************************************/
    /*                         Getters & Setters                            */
    /************************************************************************/ 
    optix::Context getContext() const
    { 
        return this->m_context; 
    }

    const std::map<std::string, optix::Program> &getProgramsMap() const 
    { 
        return this->m_programsMap; 
    }

    /**
     * @brief Specify a preprocessing function that needs to be done
     * before launching rendering kernel but right after everything
     * is prepared for rendering (all variables in GPU programs are
     * resolved). This can be useful when HDRILight would like to
     * do some prefiltering job, which can use this to specify a 
     * HDRILight::preprocess() member function.
     * 
     * @see HDRILight::preprocess()
     */
    void setPreprocessFunc(std::function<void()> preprocessFunc)
    {
        this->m_preprocessFunc = preprocessFunc;
    }

private:
    /************************************************************************/
    /*              Helper functions for initialization                     */
    /************************************************************************/

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
     * @brief Get *.ptx files from sampleConfig.h
     * @param[in] program Program name to get ptx.
     */
    std::string getPTXFilepath(const std::string &program);

    /**
     * @brief Create programs from *.ptx file and add to
     * programs map for access later.
     *
     * @note This function should be called once inside
     * setupContext().
     *
     * @see Application::initializeContext()
     */
    void createProgramsFromPTX();



    /*******************************************************************/
    /*           Initialization functions called by constructor        */
    /*******************************************************************/

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
	 * @brief Setup OptiX context, including setting exception
	 * program, miss program, entries information and system-wide
	 * parameters.
	 * 
	 * @note This function should be called once.
	 */
	void initializeContext();

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
     * @brief Load and setup bindless callable program group for
     * BSDFs and Lights. Perhaps add samplers and cameras in the 
     * future.
     * 
     * @note This function should be called once.
     */
    void initializeCallableProgramGroup();

private:
    /// OptiX environment settings
    optix::Context m_context;
    int            m_optixReportLevel;
    OptixUsageReportLogger m_optixUsageReportLogger;

    /// OptiX programs
    std::map<std::string, optix::Program> m_programsMap;

    /// Render buffers
    optix::Buffer  m_sysOutputBuffer; 
    optix::Buffer  m_sysHDRBuffer;
    unsigned int   m_renderViewTexture;
    unsigned int   m_glPBO;
    unsigned int   m_filmWidth, m_filmHeight;
    bool           m_resetRenderParamsNotification;

    friend class LightPool;
    friend class MaterialPool;

    /// Scene related variables
    unsigned int                      m_sysIterationIndex;
    std::shared_ptr<SceneGraph>       m_sceneGraph;
    std::shared_ptr<LightPool>        m_lightPool;
    std::shared_ptr<MaterialPool>     m_materialPool;
    std::unique_ptr<CameraController> m_cameraController;

    /// Function object to store preprocessor using OptiX launch 
    std::function<void()> m_preprocessFunc;

    /// Current holding IEditableObject:
    std::shared_ptr<IEditableObject> m_currentHierarchyNode;

    std::chrono::time_point<std::chrono::system_clock> currentTime;
};