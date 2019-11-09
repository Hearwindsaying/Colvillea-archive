#include "colvillea/Application/Application.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <random>
#include <string>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <imgui/imgui_stdlib.h>

// About Desktop OpenGL function loaders:
//  Modern desktop OpenGL doesn't have a standard portable header file to load OpenGL function pointers.
//  Helper libraries are often used for this purpose! We use glew here.

#include <gl/glew.h>

// Not to include glfw3.h before ImageLoader inclusion to
// prevent from macro redefinition.
//#include <GLFW/glfw3.h>

// [Win32] Our example includes a copy of glfw3.lib pre-compiled with VS2010 to maximize ease of testing and compatibility with old VS compilers.
// To link with VS2010-era libraries, VS2015+ requires linking with legacy_stdio_definitions.lib, which we do using this pragma.
// Your own project should not be affected, as you are likely to link with a newer binary of GLFW that is adequate for your version of Visual Studio.
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

/* For TinyFileDialogs. */
#include "tinyfiledialogs/tinyfiledialogs.h"

#include "colvillea/Application/GlobalDefs.h"
#include "colvillea/Application/SceneGraph.h"
#include "colvillea/Application/TWAssert.h"

#include "colvillea/Device/Toolkit/CommonStructs.h"
#include "colvillea/Module/Camera/CameraController.h"
#include "colvillea/Module/Image/ImageLoader.h"
#include "colvillea/Module/Light/LightPool.h"
#include "colvillea/Module/Material/MaterialPool.h"


#include <src/sampleConfig.h>

using namespace optix;

Application::Application(GLFWwindow* glfwWindow, const uint32_t filmWidth, const uint32_t filmHeight, const int optixReportLevel) : 
    m_filmWidth(filmWidth), m_filmHeight(filmHeight), 
    m_optixReportLevel(optixReportLevel),
    m_sysIterationIndex(0),m_resetRenderParamsNotification(true),
    m_sceneGraph(nullptr), m_cameraController(nullptr),m_lightPool(nullptr),m_materialPool(nullptr)
{
    /* Output OptiX Device information. */
    this->outputDeviceInfo();

    /* Initialize ImGui, OpenGL context for RenderView, 
     * -- setup OptiX context, filtering buffers and 
     * -- callable program gropus. */
    this->initializeImGui(glfwWindow);
    this->initializeRenderView();

    try
    {
        this->initializeContext();
        this->initializeOutputBuffers();
        this->initializeCallableProgramGroup();
    }
    catch (Exception& e)
    {
        std::cerr << e.getErrorString() << std::endl;
        DebugBreak();
    }
}

Application::~Application()
{
    this->m_context->destroy();
    std::cout << "[Info] Context has been destroyed." << std::endl;

    /* Don't destroy OpenGL objects inside a destructor.
     * -- See also: https://sourceforge.net/p/glfw/discussion/247562/thread/95ae0d7a/. */
    /*ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();*/
}


void Application::buildSceneGraph(std::shared_ptr<SceneGraph> &sceneGraph)
{
    this->m_sceneGraph = std::move(sceneGraph);
    this->m_cameraController = std::make_unique<CameraController>(this->m_sceneGraph->getCamera(), this->m_filmWidth, this->m_filmHeight);

    try
    {
        /* Before launching rendering kernel, maybe some necessary preprocessing launches
         * -- need to be done. */
        if (this->m_preprocessFunc)
        {
            this->m_preprocessFunc();
        }
        else
        {
            this->m_context->validate();
        }

    }
    catch (optix::Exception& e)
    {
        std::cerr << e.getErrorString() << std::endl;
        DebugBreak();
    }
}

void Application::drawWidget()
{
    this->drawSettings();
    this->drawInspector();
    this->drawHierarchy();

    static uint32_t frame_count = 0; // todo:use iteration index


    //{
    //    static const ImGuiWindowFlags window_flags = 0;

    //    //ImGui::SetNextWindowPos(ImVec2(2.0f, 40.0f));
    //    ImGui::SetNextWindowSize(ImVec2(550, 680), ImGuiCond_FirstUseEver);
    //    if (!ImGui::Begin("Hierarchy", NULL, window_flags))
    //    {
    //        // Early out if the window is collapsed, as an optimization.
    //        ImGui::End();
    //        return;
    //    }
    //    ImGui::PushItemWidth(-140);

    //    ImGui::Spacing();

    //    ImGui::End();
    //}

    {
        static const ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;

        //ImGui::SetNextWindowPos(ImVec2(-1.0f, 826.0f));
        //ImGui::SetNextWindowSize(ImVec2(550, 200), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("System", NULL, window_flags))
        {
            // Early out if the window is collapsed, as an optimization.
            ImGui::End();
            return;
        }
        ImGui::PushItemWidth(-140);

        ImGui::Spacing();


        //////////////////////////////////////////////////////////////////////////
        //System Module:
        ImGui::Text("Current Iterations:%d", this->m_sysIterationIndex);

        auto GetFPS = [](const int frame)->float
        {
            static double fps = -1.0;
            static unsigned last_frame_count = 0;
            static auto last_update_time = std::chrono::system_clock::now();
            static decltype(last_update_time) current_time = last_update_time;
            current_time = std::chrono::system_clock::now();
            std::chrono::duration<double> fp_ms = current_time - last_update_time;
            if (fp_ms.count() > 0.5)
            {
                fps = (frame_count - last_frame_count) / fp_ms.count();
                last_frame_count = frame_count;
                last_update_time = current_time;
            }

            return static_cast<float>(fps);
        };

        auto GetCurrentDateTime = []() ->std::string
        {
            time_t     now = time(0);
            struct tm  tstruct;
            char       buf[80];
            tstruct = *localtime(&now);

            strftime(buf, sizeof(buf), "%Y-%m-%d %H-%M-%S", &tstruct);

            return buf;
        };

        float currfps = GetFPS(frame_count++);
        ImGui::Text("FPS(frame per second):%.2f\nAverage Rendering Time(per launch):%.5fms", currfps, 1000.f / currfps);

        if (ImGui::Button("Save Current Result"))
        {
            ImageLoader::saveHDRBufferToImage(this->m_sysHDRBuffer, (GetCurrentDateTime() + ".exr").c_str());
            //             double currentRenderingTime = sutil::currentTime();
            //             double renderingTimeElapse = currentRenderingTime - this->startRenderingTime;
            //             std::cout << "[Info]currentFrame:" << sysIterationIndex << " time elapsed:" << renderingTimeElapse << std::endl;
        }

        ImGui::End();
    }
}

void Application::render()
{
    try
    {
        if (this->m_resetRenderParamsNotification)
        {
            this->m_sysIterationIndex = 0;
            this->m_context->launch(toUnderlyingValue(RayGenerationEntryType::InitFilter), this->m_filmWidth, this->m_filmHeight);
            this->m_resetRenderParamsNotification = false;
        }

        this->m_context->launch(toUnderlyingValue(RayGenerationEntryType::Render), this->m_filmWidth, this->m_filmHeight);
        this->m_context->launch(toUnderlyingValue(RayGenerationEntryType::Filter), this->m_filmWidth, this->m_filmHeight);

        this->drawRenderView();

        this->m_context["sysIterationIndex"]->setUint(this->m_sysIterationIndex++);
    }
    catch (optix::Exception& e)
    {
        std::cerr << e.getErrorString() << std::endl;
        DebugBreak();
    }
}

void Application::handleInputEvent(bool dispatchMouseInput)
{
    ImGuiIO const& io = ImGui::GetIO();
    const ImVec2 mousePosition = ImGui::GetMousePos();

    CameraController::InputMouseActionType mouseAction;

    if (dispatchMouseInput) /* Only allow camera interactions to begin when interacting with the GUI.  However, release operation is not affected. */
    {
        if (ImGui::IsMouseDown(0))
        {
            mouseAction = static_cast<CameraController::InputMouseActionType>(toUnderlyingValue(CameraController::InputMouseActionType::LeftMouse) | toUnderlyingValue(CameraController::InputMouseActionType::Down));
            this->m_cameraController->handleInputGUIEvent(mouseAction, make_int2(mousePosition.x, mousePosition.y));
        }

        else if (ImGui::IsMouseDown(1))
        {
            mouseAction = static_cast<CameraController::InputMouseActionType>(toUnderlyingValue(CameraController::InputMouseActionType::RightMouse) | toUnderlyingValue(CameraController::InputMouseActionType::Down));
            this->m_cameraController->handleInputGUIEvent(mouseAction, make_int2(mousePosition.x, mousePosition.y));
        }

        else if (ImGui::IsMouseDown(2))
        {
            mouseAction = static_cast<CameraController::InputMouseActionType>(toUnderlyingValue(CameraController::InputMouseActionType::MiddleMouse) | toUnderlyingValue(CameraController::InputMouseActionType::Down));
            this->m_cameraController->handleInputGUIEvent(mouseAction, make_int2(mousePosition.x, mousePosition.y));
        }

        else if (ImGui::IsMouseReleased(0))
        {
            mouseAction = static_cast<CameraController::InputMouseActionType>(toUnderlyingValue(CameraController::InputMouseActionType::LeftMouse) | toUnderlyingValue(CameraController::InputMouseActionType::Release));
            this->m_cameraController->handleInputGUIEvent(mouseAction, make_int2(mousePosition.x, mousePosition.y));
        }

        else if (ImGui::IsMouseReleased(1))
        {
            mouseAction = static_cast<CameraController::InputMouseActionType>(toUnderlyingValue(CameraController::InputMouseActionType::RightMouse) | toUnderlyingValue(CameraController::InputMouseActionType::Release));
            this->m_cameraController->handleInputGUIEvent(mouseAction, make_int2(mousePosition.x, mousePosition.y));
        }

        else if (ImGui::IsMouseReleased(2))
        {
            mouseAction = static_cast<CameraController::InputMouseActionType>(toUnderlyingValue(CameraController::InputMouseActionType::MiddleMouse) | toUnderlyingValue(CameraController::InputMouseActionType::Release));
            this->m_cameraController->handleInputGUIEvent(mouseAction, make_int2(mousePosition.x, mousePosition.y));
        }
    }

    else
    {
        if (ImGui::IsMouseReleased(0))
        {
            mouseAction = static_cast<CameraController::InputMouseActionType>(toUnderlyingValue(CameraController::InputMouseActionType::LeftMouse) | toUnderlyingValue(CameraController::InputMouseActionType::Release));
            this->m_cameraController->handleInputGUIEvent(mouseAction, make_int2(mousePosition.x, mousePosition.y));
        }

        else if (ImGui::IsMouseReleased(1))
        {
            mouseAction = static_cast<CameraController::InputMouseActionType>(toUnderlyingValue(CameraController::InputMouseActionType::RightMouse) | toUnderlyingValue(CameraController::InputMouseActionType::Release));
            this->m_cameraController->handleInputGUIEvent(mouseAction, make_int2(mousePosition.x, mousePosition.y));
        }

        else if (ImGui::IsMouseReleased(2))
        {
            mouseAction = static_cast<CameraController::InputMouseActionType>(toUnderlyingValue(CameraController::InputMouseActionType::MiddleMouse) | toUnderlyingValue(CameraController::InputMouseActionType::Release));
            this->m_cameraController->handleInputGUIEvent(mouseAction, make_int2(mousePosition.x, mousePosition.y));
        }
    }
}

void Application::drawRenderView()
{
    RTformat buffer_format = this->m_sysOutputBuffer->getFormat();
    TW_ASSERT(buffer_format == RT_FORMAT_FLOAT4);

    const unsigned pboId = this->m_sysOutputBuffer->getGLBOId();
    TW_ASSERT(pboId);

    /* Draw RenderView widget. */
    static const ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoCollapse;
    //ImGui::SetNextWindowPos(ImVec2(733.0f, 58.0f));
    //ImGui::SetNextWindowSize(ImVec2(1489.0f, 810.0f));
    static bool showWindow = true;
    if (!ImGui::Begin("RenderView", &showWindow, window_flags))
    {
        // Early out if the window is collapsed, as an optimization.
        ImGui::End();
        return;
    }

    /* Handle mouse input event for RenderView.
     * -- Note that there is a bug when there are multiple windows inside
     * -- RenderView window (dragging titlebar will also lead to camera changing. 
     * todo: fix this setting |isTitleBarHovering| to true for other windows. */
    if(!ImGui::IsItemHovered())
        this->handleInputEvent(ImGui::IsWindowHovered());

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, this->m_renderViewTexture);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboId);
    //todo:using glTexSubImage2D
    //glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (GLsizei)this->filmWidth, (GLsizei)this->filmHeight, GL_RGBA, GL_FLOAT, (void*)0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, static_cast<GLsizei>(this->m_filmWidth), static_cast<GLsizei>(this->m_filmHeight), 0, GL_RGBA, GL_FLOAT, nullptr); // RGBA32F from byte offset 0 in the pixel unpack buffer.
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    ImGui::Image((void*)(intptr_t)this->m_renderViewTexture, ImVec2(this->m_filmWidth, this->m_filmHeight), ImVec2(0, 1), ImVec2(1, 0)); /* flip UV Coordinates due to the inconsistence(vertically invert) */

    ImGui::End();
}

void Application::drawSettings()
{
    static const ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;

    if (!ImGui::Begin("Settings", NULL, window_flags))
    {
        // Early out if the window is collapsed, as an optimization.
        ImGui::End();
        return;
    }

    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_CollapsingHeader))
    {
        if (ImGui::BeginPopupContextItem())
        {
            if (ImGui::MenuItem("Save Transform"))
            {
                this->m_cameraController->cameraInfoToFile();
            }
            if (ImGui::MenuItem("Load Transform"))
            {
                this->m_cameraController->cameraInfoFromFile();
            }

            /* End of CollapsingHeader. */
            ImGui::EndPopup();
        }

        auto cInfo = this->m_cameraController->getCameraInfo();
        ImGui::Indent();

        /* Eye Location Group: */
        ImGui::Text("            Eye Location");
        ImGui::SameLine(178.f);
        ImGui::AlignTextToFramePadding();
        ImGui::Text(" X\n");
        ImGui::SameLine(200.f);
        ImGui::SetNextItemWidth(85);
        if (ImGui::DragFloat("##Eye Location X", &cInfo.eye.x, 1.0f, -100.0f, 100.0f))
        {
            this->m_cameraController->setCameraInfo(cInfo);
        }
        
        ImGui::Text("            ");
        ImGui::SameLine(178.f);
        ImGui::AlignTextToFramePadding();
        ImGui::Text(" Y\n");
        ImGui::SameLine(200.f);
        ImGui::SetNextItemWidth(85);
        if (ImGui::DragFloat("##Eye Location Y", &cInfo.eye.y, 1.0f, -100.0f, 100.0f))
        {
            this->m_cameraController->setCameraInfo(cInfo);
        }

        ImGui::Text("            ");
        ImGui::SameLine(178.f);
        ImGui::AlignTextToFramePadding();
        ImGui::Text(" Z\n");
        ImGui::SameLine(200.f);
        ImGui::SetNextItemWidth(85);
        if (ImGui::DragFloat("##Eye Location Z", &cInfo.eye.z, 1.0f, -100.0f, 100.0f))
        {
            this->m_cameraController->setCameraInfo(cInfo);
        }

        /* LookAt Destination Group: */
        ImGui::Text("Destination Location ");
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        ImGui::Text("X\n");
        ImGui::SameLine(200.f);
        ImGui::SetNextItemWidth(85);
        if (ImGui::DragFloat("##Destination Location X", &cInfo.lookAtDestination.x, 1.0f, -100.0f, 100.0f))
        {
            this->m_cameraController->setCameraInfo(cInfo);
        }

        ImGui::Text("            ");
        ImGui::SameLine(178.f);
        ImGui::AlignTextToFramePadding();
        ImGui::Text(" Y\n");
        ImGui::SameLine(200.f);
        ImGui::SetNextItemWidth(85);
        if (ImGui::DragFloat("##Destination Location Y", &cInfo.lookAtDestination.y, 1.0f, -100.0f, 100.0f))
        {
            this->m_cameraController->setCameraInfo(cInfo);
        }

        ImGui::Text("            ");
        ImGui::SameLine(178.f);
        ImGui::AlignTextToFramePadding();
        ImGui::Text(" Z\n");
        ImGui::SameLine(200.f);
        ImGui::SetNextItemWidth(85);
        if (ImGui::DragFloat("##Destination Location Z", &cInfo.lookAtDestination.z, 1.0f, -100.0f, 100.0f))
        {
            this->m_cameraController->setCameraInfo(cInfo);
        }
        ImGui::Unindent();

    }
    /* Enable popup menu even if the Camera module is collapsed. */
    if (ImGui::BeginPopupContextItem())
    {
        if (ImGui::MenuItem("Save Transform"))
        {
            this->m_cameraController->cameraInfoToFile();
        }
        if (ImGui::MenuItem("Load Transform"))
        {
            this->m_cameraController->cameraInfoFromFile();
        }

        /* End of CollapsingHeader. */
        ImGui::EndPopup();
    }
    

    /* Draw Sampler setting. */
    if (ImGui::CollapsingHeader("Sampling", ImGuiTreeNodeFlags_CollapsingHeader))
    {
        int currentSamplerIdx = toUnderlyingValue(this->m_sceneGraph->getSampler()->getSamplerType());

        ImGui::AlignTextToFramePadding();
        ImGui::Text("                             Sampler"); ImGui::SameLine(200.f);
        ImGui::SetNextItemWidth(165);
        if (ImGui::Combo("##Sampler", &currentSamplerIdx, "Halton QMC\0Sobol QMC\0Independent\0\0"))
        {
            this->m_sceneGraph->createSampler(static_cast<CommonStructs::SamplerType>(currentSamplerIdx));
            this->resetRenderParams();
        }

        int currentIntegratorIdx = toUnderlyingValue(this->m_sceneGraph->getIntegrator()->getIntegratorType());
        ImGui::AlignTextToFramePadding();
        ImGui::Text("                          Integrator"); ImGui::SameLine(200.f);
        ImGui::SetNextItemWidth(165);
        if (ImGui::Combo("##Integrator", &currentIntegratorIdx, "Direct Lighting\0Path Tracing\0\0"))
        {
            /* Switch integrator. */
            this->m_sceneGraph->changeIntegrator(static_cast<IntegratorType>(currentIntegratorIdx));
            this->resetRenderParams();
        }

        if (currentIntegratorIdx == toUnderlyingValue(IntegratorType::PathTracing))
        {
            std::shared_ptr<PathTracing> ptIntegrator = this->m_sceneGraph->getPathTracing();
            int  maxDepth               = ptIntegrator->getMaxDepth();
            bool enableRoussianRoulette = ptIntegrator->getEnableRoussianRoulette();

            ImGui::AlignTextToFramePadding();
            ImGui::Text("            Max Light Bounces"); ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(165);
            if (ImGui::InputInt("##Max Light Bounces", &maxDepth, 1, 1))
            {
                if (maxDepth <= 0)
                    maxDepth = 1;
                ptIntegrator->setMaxDepth(maxDepth);
                this->resetRenderParams();
            }

            ImGui::AlignTextToFramePadding();
            ImGui::Text("             Roussian Roulette"); ImGui::SameLine(200.f);
            if (ImGui::Checkbox("##RoussianRoulette", &enableRoussianRoulette))
            {
                ptIntegrator->setEnableRoussianRoulette(enableRoussianRoulette);
                this->resetRenderParams();
            }
        }
    }


    /* Draw Reconstruction Filter setting. */
    if (ImGui::CollapsingHeader("Reconstruction Filter", ImGuiTreeNodeFlags_CollapsingHeader))
    {
        int currentFilterTypeIdx = toUnderlyingValue(this->m_sceneGraph->getFilter()->getFilterType());

        ImGui::AlignTextToFramePadding();
        ImGui::Text("                                  Filter"); ImGui::SameLine(200.f);
        ImGui::SetNextItemWidth(165);
        if (ImGui::Combo("##Filter", &currentFilterTypeIdx, "Box\0Gaussian\0\0"))
        {
            /* Switch filter. */
            this->m_sceneGraph->changeFilter(static_cast<CommonStructs::FilterType>(currentFilterTypeIdx));
            this->resetRenderParams();
        }

        if (currentFilterTypeIdx == toUnderlyingValue(CommonStructs::FilterType::BoxFilter))
        {
            BoxFilter *boxFilter = this->m_sceneGraph->getBoxFilter().get();
            float filterRadius = boxFilter->getRadius();
            ImGui::AlignTextToFramePadding();
            ImGui::Text("                                Radius"); ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(165);
            if (ImGui::InputFloat("##Radius", &filterRadius, 0.1f, 0.1f))
            {
                if (filterRadius <= 0.5f)
                    filterRadius = 0.5f;
                boxFilter->setRadius(filterRadius);
                /* We need to update device variables (this work isn't included in setRadius(). */
                this->m_sceneGraph->changeFilter(static_cast<CommonStructs::FilterType>(currentFilterTypeIdx));
                this->resetRenderParams();
            }
        }
        else if (currentFilterTypeIdx == toUnderlyingValue(CommonStructs::FilterType::GaussianFilter))
        {
            GaussianFilter *gaussianFilter = this->m_sceneGraph->getGaussianFilter().get();
            float filterRadius = gaussianFilter->getRadius();
            float filterAlpha  = gaussianFilter->getAlpha();

            ImGui::AlignTextToFramePadding();
            ImGui::Text("                                Radius"); ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(165);
            if (ImGui::InputFloat("##Radius", &filterRadius, 0.1f, 0.1f))
            {
                if (filterRadius <= 0.5f)
                    filterRadius = 0.5f;
                gaussianFilter->setRadius(filterRadius);
                /* We need to update device variables (this work isn't included in setRadius(). */
                this->m_sceneGraph->changeFilter(static_cast<CommonStructs::FilterType>(currentFilterTypeIdx));
                this->resetRenderParams();
            }

            ImGui::AlignTextToFramePadding();
            ImGui::Text("                  Gaussian Alpha"); ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(165);
            if (ImGui::InputFloat("##Gaussian Alpha", &filterAlpha, 0.05f, 0.05f))
            {
                if (filterAlpha <= 0.05f)
                    filterAlpha = 0.05f;
                gaussianFilter->setAlpha(filterAlpha);
                /* We need to update device variables (this work isn't included in setRadius().
                 * todo: update context variables in setRadius() */
                this->m_sceneGraph->changeFilter(static_cast<CommonStructs::FilterType>(currentFilterTypeIdx));
                this->resetRenderParams();
            }
        }

    }

    ImGui::End();
}

void Application::drawInspector()
{
    static const ImGuiWindowFlags window_flags_inspector = ImGuiWindowFlags_None;
    if (!ImGui::Begin("Inspector", NULL, window_flags_inspector))
    {
        ImGui::End();
        return;
    }

    /* Nothing has been selected: */
    if (!this->m_currentHierarchyNode)
    {
        ImGui::End();
        return;
    }

    IEditableObject::IEditableObjectType objectType = this->m_currentHierarchyNode->getObjectType();
    if (objectType == IEditableObject::IEditableObjectType::HDRILight)
    {
        std::shared_ptr<HDRILight> hdriLight = std::static_pointer_cast<HDRILight>(this->m_currentHierarchyNode);
        TW_ASSERT(hdriLight);

        std::string hdriLightName = hdriLight->getName();
        if (ImGui::InputText("##Object Name", &hdriLightName))
        {
            hdriLight->setName(hdriLightName);
        }

        ImGui::Separator();

        if (ImGui::CollapsingHeader("General", ImGuiTreeNodeFlags_CollapsingHeader))
        {
            bool enableHDRIProbe = hdriLight->getEnableHDRILight();
            std::string HDRIFilename = hdriLight->getHDRIFilename();

            ImGui::AlignTextToFramePadding();
            ImGui::Text("                               Enable"); ImGui::SameLine(200.f);
            if (ImGui::Checkbox("##Enable HDRIProbe", &enableHDRIProbe))
            {
                hdriLight->setEnableHDRILight(enableHDRIProbe);
                this->resetRenderParams();
            }

            ImGui::AlignTextToFramePadding();
            ImGui::Text("                        HDR Image"); ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(165);
            ImGui::Button(HDRIFilename.c_str()/*, ImVec2(220.f, 0.0f)*/);
            ImGui::SameLine();
            if (ImGui::Button("M"))
            {
                static char const * lFilterPatterns[2] = { "*.exr" };
                const char * HDRIFilename_c_str = tinyfd_openFileDialog("Select a HDR Image file", "", 1, lFilterPatterns, "HDR Files(*.exr)", 0);
                if (HDRIFilename_c_str != NULL)
                {
                    hdriLight->setHDRIFilename(HDRIFilename_c_str);
                    this->resetRenderParams();
                }
            }
        }

        if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_CollapsingHeader))
        {
            ImGui::AlignTextToFramePadding();
            ImGui::Text("                         Rotation");
            ImGui::SameLine(178.f);
            ImGui::AlignTextToFramePadding();
            ImGui::Text(" X\n");
            ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(85);

            float3 rotationAngle = hdriLight->getLightRotation();
            /* Note that SliderAngle manipulates angle in radian. */
            if (ImGui::SliderAngle("##HDRI Rotation X", &rotationAngle.x, 0.0f, 360.0f))
            {
                hdriLight->setLightRotation(rotationAngle);
                this->resetRenderParams();
            }

            ImGui::Text("            ");
            ImGui::SameLine(178.f);
            ImGui::AlignTextToFramePadding();
            ImGui::Text(" Y\n");
            ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(85);
            if (ImGui::SliderAngle("##HDRI Rotation Y", &rotationAngle.y, 0.0f, 360.0f))
            {
                hdriLight->setLightRotation(rotationAngle);
                this->resetRenderParams();
            }

            ImGui::Text("            ");
            ImGui::SameLine(178.f);
            ImGui::AlignTextToFramePadding();
            ImGui::Text(" Z\n");
            ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(85);
            if (ImGui::SliderAngle("##HDRI Rotation Z", &rotationAngle.z, 0.0f, 360.0f))
            {
                hdriLight->setLightRotation(rotationAngle);
                this->resetRenderParams();
            }
        }
    }
    else if (objectType == IEditableObject::IEditableObjectType::PointLight)
    {
        std::shared_ptr<PointLight> pointLight = std::static_pointer_cast<PointLight>(this->m_currentHierarchyNode);
        TW_ASSERT(pointLight);

        std::string pointLightName = pointLight->getName();
        if (ImGui::InputText("##Object Name", &pointLightName))
        {
            pointLight->setName(pointLightName);
        }

        ImGui::Separator();

        if (ImGui::CollapsingHeader("General##Point Light", ImGuiTreeNodeFlags_CollapsingHeader))
        {
            optix::float3 color = pointLight->getLightColor();
            float intensity = pointLight->getLightIntensity();

            ImGui::AlignTextToFramePadding();
            ImGui::Text("                                  Color"); ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(165);
            if (ImGui::ColorEdit3("##Point Light Color", static_cast<float *>(&color.x), ImGuiColorEditFlags__OptionsDefault))
            {
                pointLight->setLightColor(color);
                this->resetRenderParams();
            }

            ImGui::AlignTextToFramePadding();
            ImGui::Text("                             Intensity"); ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(165);
            if (ImGui::InputFloat("##Light Intensity", &intensity, 1, 1))
            {
                if (intensity < 0.0f)
                    intensity = 0.0f;
                pointLight->setLightIntensity(intensity);
                this->resetRenderParams();
            }
        }

        if (ImGui::CollapsingHeader("Transform##Point Light", ImGuiTreeNodeFlags_CollapsingHeader))
        {
            optix::float3 pointLightLocation = pointLight->getLightPosition();

            ImGui::AlignTextToFramePadding();
            ImGui::Text("                         Location");
            ImGui::SameLine(178.f);
            ImGui::AlignTextToFramePadding();
            ImGui::Text(" X\n");
            ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(85);
            if (ImGui::DragFloat("##PointLight Location X", &pointLightLocation.x, 1.0f, -100.0f, 100.0f))
            {
                pointLight->setLightPosition(pointLightLocation);
                this->resetRenderParams();
            }

            ImGui::Text("            ");
            ImGui::SameLine(178.f);
            ImGui::AlignTextToFramePadding();
            ImGui::Text(" Y\n");
            ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(85);
            if (ImGui::DragFloat("##PointLight Location Y", &pointLightLocation.y, 1.0f, -100.0f, 100.0f))
            {
                pointLight->setLightPosition(pointLightLocation);
                this->resetRenderParams();
            }

            ImGui::Text("            ");
            ImGui::SameLine(178.f);
            ImGui::AlignTextToFramePadding();
            ImGui::Text(" Z\n");
            ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(85);
            if (ImGui::DragFloat("##PointLight Location Z", &pointLightLocation.z, 1.0f, -100.0f, 100.0f))
            {
                pointLight->setLightPosition(pointLightLocation);
                this->resetRenderParams();
            }
        }


        ImGui::BeginGroup();

        ImGui::BeginChild("RemoveObjectSpaceChild", ImVec2(0, -ImGui::GetFrameHeightWithSpacing())); // Leave room for 1 line below us
        ImGui::EndChild();

        /* Last line of Inspector window. */
        if (ImGui::Button("Remove Object")) 
        {
            this->m_lightPool->removePointLight(pointLight);
            this->m_currentHierarchyNode.reset();
            this->resetRenderParams();
        }

        ImGui::EndGroup();
    }
    else if (objectType == IEditableObject::IEditableObjectType::QuadLight)
    {
        std::shared_ptr<QuadLight> quadLight = std::static_pointer_cast<QuadLight>(this->m_currentHierarchyNode);
        TW_ASSERT(quadLight);

        std::string quadLightName = quadLight->getName();
        if (ImGui::InputText("##Object Name", &quadLightName))
        {
            quadLight->setName(quadLightName);
        }

        ImGui::Separator();

        if (ImGui::CollapsingHeader("General##Quad Light", ImGuiTreeNodeFlags_CollapsingHeader))
        {
            optix::float3 color = quadLight->getLightColor();
            float intensity = quadLight->getLightIntensity();

            ImGui::AlignTextToFramePadding();
            ImGui::Text("                                  Color"); ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(165);
            if (ImGui::ColorEdit3("##Quad Light Color", static_cast<float *>(&color.x), ImGuiColorEditFlags__OptionsDefault))
            {
                quadLight->setLightColor(color);
                this->resetRenderParams();
            }

            ImGui::AlignTextToFramePadding();
            ImGui::Text("                             Intensity"); ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(165);
            if (ImGui::InputFloat("##Light Intensity", &intensity, 1, 1))
            {
                if (intensity < 0.0f)
                    intensity = 0.0f;
                quadLight->setLightIntensity(intensity);
                this->resetRenderParams();
            }
        }

        if (ImGui::CollapsingHeader("Transform##Quad Light", ImGuiTreeNodeFlags_CollapsingHeader))
        {
            float3 quadLightLocation = quadLight->getPosition();
            float3 quadLightRotation = quadLight->getRotation();
            float3 quadLightScale    = quadLight->getScale();

            ImGui::AlignTextToFramePadding();
            ImGui::Text("                         Location");
            ImGui::SameLine(178.f);
            ImGui::AlignTextToFramePadding();
            ImGui::Text(" X\n");
            ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(85);
            if (ImGui::DragFloat("##QuadLight Location X", &quadLightLocation.x, 1.0f, -100.0f, 100.0f))
            {
                quadLight->setPosition(quadLightLocation);
                this->resetRenderParams();
            }

            ImGui::Text("            ");
            ImGui::SameLine(178.f);
            ImGui::AlignTextToFramePadding();
            ImGui::Text(" Y\n");
            ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(85);
            if (ImGui::DragFloat("##QuadLight Location Y", &quadLightLocation.y, 1.0f, -100.0f, 100.0f))
            {
                quadLight->setPosition(quadLightLocation);
                this->resetRenderParams();
            }

            ImGui::Text("            ");
            ImGui::SameLine(178.f);
            ImGui::AlignTextToFramePadding();
            ImGui::Text(" Z\n");
            ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(85);
            if (ImGui::DragFloat("##QuadLight Location Z", &quadLightLocation.z, 1.0f, -100.0f, 100.0f))
            {
                quadLight->setPosition(quadLightLocation);
                this->resetRenderParams();
            }

            ImGui::AlignTextToFramePadding();
            ImGui::Text("                         Rotation");
            ImGui::SameLine(178.f);
            ImGui::AlignTextToFramePadding();
            ImGui::Text(" X\n");
            ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(85);
            if (ImGui::SliderAngle("##QuadLight Rotation X", &quadLightRotation.x, 0.0f, 360.0f))
            {
                quadLight->setRotation(quadLightRotation);
                this->resetRenderParams();
            }

            ImGui::Text("            ");
            ImGui::SameLine(178.f);
            ImGui::AlignTextToFramePadding();
            ImGui::Text(" Y\n");
            ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(85);
            if (ImGui::SliderAngle("##QuadLight Rotation Y", &quadLightRotation.y, 0.0f, 360.0f))
            {
                quadLight->setRotation(quadLightRotation);
                this->resetRenderParams();
            }

            ImGui::Text("            ");
            ImGui::SameLine(178.f);
            ImGui::AlignTextToFramePadding();
            ImGui::Text(" Z\n");
            ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(85);
            if (ImGui::SliderAngle("##QuadLight Rotation Z", &quadLightRotation.z, 0.0f, 360.0f))
            {
                quadLight->setRotation(quadLightRotation);
                this->resetRenderParams();
            }


            ImGui::Text("");
            ImGui::SameLine(135.5f);
            ImGui::AlignTextToFramePadding();
            ImGui::Text("Scale");
            ImGui::SameLine(178.f);
            ImGui::AlignTextToFramePadding();
            ImGui::Text(" X\n");
            ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(85);
            if (ImGui::DragFloat("##QuadLight Scale X", &quadLightScale.x, 1.0f, 0.1f, 100.0f))
            {
                quadLight->setScale(quadLightScale);
                this->resetRenderParams();
            }

            ImGui::Text("            ");
            ImGui::SameLine(178.f);
            ImGui::AlignTextToFramePadding();
            ImGui::Text(" Y\n");
            ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(85);
            if (ImGui::DragFloat("##QuadLight Scale Y", &quadLightScale.y, 1.0f, 0.1f, 100.0f))
            {
                quadLight->setScale(quadLightScale);
                this->resetRenderParams();
            }

        }

        /* Public Remove Button */
        ImGui::BeginGroup();

        ImGui::BeginChild("RemoveObjectSpaceChild", ImVec2(0, -ImGui::GetFrameHeightWithSpacing())); // Leave room for 1 line below us
        ImGui::EndChild();

        /* Last line of Inspector window. */
        if (ImGui::Button("Remove Object"))
        {
            this->m_lightPool->removeQuadLight(quadLight);
            this->m_currentHierarchyNode.reset();
            this->resetRenderParams();
        }

        ImGui::EndGroup();
    }
    else if (objectType == IEditableObject::IEditableObjectType::QuadGeometry)
    {
        std::shared_ptr<Quad> quad = std::static_pointer_cast<Quad>(this->m_currentHierarchyNode);
        TW_ASSERT(quad);


        std::string quadName = quad->getName();
        if (ImGui::InputText("##Object Name", &quadName))
        {
            quad->setName(quadName);
        }

        ImGui::Separator();

        if (ImGui::CollapsingHeader("General##Quad Geometry", ImGuiTreeNodeFlags_CollapsingHeader))
        {
            ImGui::AlignTextToFramePadding();
            ImGui::Text("                                 Mesh"); ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(165);
            ImGui::Button("Quad", ImVec2(165.f, 0.0f));
            ImGui::SameLine();
            ImGui::Button("M");
        }
        if (ImGui::CollapsingHeader("Transform##Quad Geometry", ImGuiTreeNodeFlags_CollapsingHeader))
        {
            float3 quadLocation = quad->getPosition();
            float3 quadRotation = quad->getRotation();
            float3 quadScale    = quad->getScale();

            ImGui::AlignTextToFramePadding();
            ImGui::Text("                         Location");
            ImGui::SameLine(178.f);
            ImGui::AlignTextToFramePadding();
            ImGui::Text(" X\n");
            ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(85);
            if (ImGui::DragFloat("##Quad Geometry Location X", &quadLocation.x, 1.0f, -100.0f, 100.0f))
            {
                quad->setPosition(quadLocation);
                this->resetRenderParams();
            }

            ImGui::Text("            ");
            ImGui::SameLine(178.f);
            ImGui::AlignTextToFramePadding();
            ImGui::Text(" Y\n");
            ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(85);
            if (ImGui::DragFloat("##Quad Geometry Location Y", &quadLocation.y, 1.0f, -100.0f, 100.0f))
            {
                quad->setPosition(quadLocation);
                this->resetRenderParams();
            }

            ImGui::Text("            ");
            ImGui::SameLine(178.f);
            ImGui::AlignTextToFramePadding();
            ImGui::Text(" Z\n");
            ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(85);
            if (ImGui::DragFloat("##Quad Geometry Location Z", &quadLocation.z, 1.0f, -100.0f, 100.0f))
            {
                quad->setPosition(quadLocation);
                this->resetRenderParams();
            }

            //---------------------------------------------------------------------
            ImGui::AlignTextToFramePadding();
            ImGui::Text("                         Rotation");
            ImGui::SameLine(178.f);
            ImGui::AlignTextToFramePadding();
            ImGui::Text(" X\n");
            ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(85);
            if (ImGui::SliderAngle("##Quad Geometry Rotation X", &quadRotation.x, 0.0f, 360.0f))
            {
                quad->setRotation(quadRotation);
                this->resetRenderParams();
            }

            ImGui::Text("            ");
            ImGui::SameLine(178.f);
            ImGui::AlignTextToFramePadding();
            ImGui::Text(" Y\n");
            ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(85);
            if (ImGui::SliderAngle("##Quad Geometry Rotation Y", &quadRotation.y, 0.0f, 360.0f))
            {
                quad->setRotation(quadRotation);
                this->resetRenderParams();
            }

            ImGui::Text("            ");
            ImGui::SameLine(178.f);
            ImGui::AlignTextToFramePadding();
            ImGui::Text(" Z\n");
            ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(85);
            if (ImGui::SliderAngle("##Quad Geometry Rotation Z", &quadRotation.z, 0.0f, 360.0f))
            {
                quad->setRotation(quadRotation);
                this->resetRenderParams();
            }


            ImGui::Text("");
            ImGui::SameLine(135.5f);
            ImGui::AlignTextToFramePadding();
            ImGui::Text("Scale");
            ImGui::SameLine(178.f);
            ImGui::AlignTextToFramePadding();
            ImGui::Text(" X\n");
            ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(85);
            if (ImGui::DragFloat("##Quad Geometry Scale X", &quadScale.x, 1.0f, 0.1f, 100.0f))
            {
                quad->setScale(quadScale);
                this->resetRenderParams();
            }

            ImGui::Text("            ");
            ImGui::SameLine(178.f);
            ImGui::AlignTextToFramePadding();
            ImGui::Text(" Y\n");
            ImGui::SameLine(200.f);
            ImGui::SetNextItemWidth(85);
            if (ImGui::DragFloat("##Quad Geometry Scale Y", &quadScale.y, 1.0f, 0.1f, 100.0f))
            {
                quad->setScale(quadScale);
                this->resetRenderParams();
            }
        }
        
        /* Public Remove Button */
        ImGui::BeginGroup();

        ImGui::BeginChild("RemoveObjectSpaceChild", ImVec2(0, -ImGui::GetFrameHeightWithSpacing())); // Leave room for 1 line below us
        ImGui::EndChild();

        /* Last line of Inspector window. */
        if (ImGui::Button("Remove Object"))
        {
            this->m_sceneGraph->removeGeometry(quad);
            this->m_currentHierarchyNode.reset();
            this->resetRenderParams();
        }

        ImGui::EndGroup();
    }
    else
    {
        std::cerr << "[Error] Error object type" << std::endl;
    }


    

    

    ImGui::End();
}

void Application::drawHierarchy()
{
    static const ImGuiWindowFlags window_flags_inspector = ImGuiWindowFlags_MenuBar;
    if (!ImGui::Begin("Hierarchy", NULL, window_flags_inspector))
    {
        ImGui::End();
        return;
    }

    /* Menu for creating lights and geometries. */
    if (ImGui::BeginMenuBar())
    {
        if (ImGui::BeginMenu("Create"))
        {
            if (ImGui::BeginMenu("Light"))
            {
                if (ImGui::MenuItem("PointLight"))
                {
                    this->m_lightPool->createPointLight(optix::make_float3(0.0f), optix::make_float3(1.0f), 5.0f);
                    this->resetRenderParams();
                }
                if (ImGui::MenuItem("QuadLight"))
                {
                    this->m_lightPool->createQuadLight(make_float3(0.f), make_float3(0.f), make_float3(1.f, 1.f, 1.f), make_float3(1.f), 5.f, this->m_materialPool->getEmissiveMaterial(), false);
                    this->resetRenderParams();
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Geometry"))
            {
                ImGui::MenuItem("Quad");
                ImGui::MenuItem("TriangleMesh");
                ImGui::EndMenu();
            }

            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }


    if (ImGui::TreeNode("Light"))
    {
        static ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;

        LightPool *lightPool = this->m_lightPool.get();
        TW_ASSERT(lightPool);

        /* HDRILights */
        HDRILight *hdriLight = lightPool->getHDRILight().get();
        TW_ASSERT(hdriLight);

        ImGuiTreeNodeFlags hdriLight_TreeNode_flag = (this->m_currentHierarchyNode ? 
            ((this->m_currentHierarchyNode->getId() == hdriLight->getId()) ? 
             (base_flags | ImGuiTreeNodeFlags_Selected) : base_flags)
            : base_flags);
        ImGui::TreeNodeEx((void*)(intptr_t)hdriLight->getId(), hdriLight_TreeNode_flag, hdriLight->getName().c_str());
        if (ImGui::IsItemClicked())
        {
            this->m_currentHierarchyNode = lightPool->getHDRILight();
        }

        /* PointLights */
        for (auto pointLightItr = lightPool->getPointLights().cbegin(); pointLightItr != lightPool->getPointLights().cend(); ++pointLightItr)
        {
            ImGuiTreeNodeFlags pointLight_TreeNode_flag = (this->m_currentHierarchyNode ? 
                ((this->m_currentHierarchyNode->getId() == (*pointLightItr)->getId()) ? 
                 (base_flags | ImGuiTreeNodeFlags_Selected) : base_flags)
                : base_flags);

            ImGui::TreeNodeEx((void*)(intptr_t)((*pointLightItr)->getId()), pointLight_TreeNode_flag, (*pointLightItr)->getName().c_str());
            if (ImGui::IsItemClicked())
            {
                this->m_currentHierarchyNode = *pointLightItr;
            }    
        }

        /* QuadLights */
        for (auto quadLightItr = lightPool->getQuadLights().cbegin(); quadLightItr != lightPool->getQuadLights().cend(); ++quadLightItr)
        {
            ImGuiTreeNodeFlags quadLight_TreeNode_flag = (this->m_currentHierarchyNode ?
                ((this->m_currentHierarchyNode->getId() == (*quadLightItr)->getId()) ?
                (base_flags | ImGuiTreeNodeFlags_Selected) : base_flags)
                : base_flags);
            ImGui::TreeNodeEx((void*)(intptr_t)((*quadLightItr)->getId()), quadLight_TreeNode_flag, (*quadLightItr)->getName().c_str());
            if (ImGui::IsItemClicked())
            {
                this->m_currentHierarchyNode = *quadLightItr;
            }
        }



        ImGui::TreePop();
    }

    ImGui::Separator();

    if (ImGui::TreeNode("Geometry"))
    {
        static ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;

        /* Quad Shapes. 
         *  Todo: modify GUI implementation to support various GeometryShape types. 
         *        -- We assumed that there are only quads in GeometryShape in current implementation. */
        const std::vector<std::shared_ptr<GeometryShape>> &shapes_Geometry = this->m_sceneGraph->getShapes_Geometry();

        for (auto quadShapePtrItr = shapes_Geometry.cbegin(); quadShapePtrItr != shapes_Geometry.cend(); ++quadShapePtrItr)
        {
            /* If this Quad shape is used to be an underlying shape for AreaLight, never show 
             * -- it up in Hierarchy. Because once the special Quad (used for AreaLight) is 
             * -- removed by user, QuadLight is ill-defined and thus leading to corruption. */
            if(std::static_pointer_cast<Quad>(*quadShapePtrItr)->isAreaLight())
                continue;

            ImGuiTreeNodeFlags quadShape_TreeNode_flag = (this->m_currentHierarchyNode ?
                ((this->m_currentHierarchyNode->getId() == (*quadShapePtrItr)->getId()) ?
                (base_flags | ImGuiTreeNodeFlags_Selected) : base_flags)
                : base_flags);
            ImGui::TreeNodeEx((void*)(intptr_t)((*quadShapePtrItr)->getId()), quadShape_TreeNode_flag, (*quadShapePtrItr)->getName().c_str());
            if (ImGui::IsItemClicked())
            {
                this->m_currentHierarchyNode = *quadShapePtrItr;
            }
        }

        /* GeometryTriangles Shapes. */
        const std::vector<std::shared_ptr<GeometryTrianglesShape>> &shapes_GeometryTriangles = this->m_sceneGraph->getShapes_GeometryTriangles();

        for (auto geometryTrianglesShapePtrItr = shapes_GeometryTriangles.cbegin(); geometryTrianglesShapePtrItr != shapes_GeometryTriangles.cend(); ++geometryTrianglesShapePtrItr)
        {
            ImGuiTreeNodeFlags geometryTrianglesShape_TreeNode_flag = (this->m_currentHierarchyNode ?
                ((this->m_currentHierarchyNode->getId() == (*geometryTrianglesShapePtrItr)->getId()) ?
                (base_flags | ImGuiTreeNodeFlags_Selected) : base_flags)
                : base_flags);
            ImGui::TreeNodeEx((void*)(intptr_t)((*geometryTrianglesShapePtrItr)->getId()), geometryTrianglesShape_TreeNode_flag, (*geometryTrianglesShapePtrItr)->getName().c_str());
            if (ImGui::IsItemClicked())
            {
                this->m_currentHierarchyNode = *geometryTrianglesShapePtrItr;
            }
        }



        ImGui::TreePop();
    }

    ImGui::End();
}

/************************************************************************/
/*              Helper functions for initialization                     */
/************************************************************************/ 

void Application::outputDeviceInfo()
{
    unsigned int optixVersion;
    RT_CHECK_ERROR_NO_CONTEXT(rtGetVersion(&optixVersion));

    unsigned int major = optixVersion / 1000; // Check major with old formula.
    unsigned int minor;
    unsigned int micro;
    if (3 < major) // New encoding since OptiX 4.0.0 to get two digits micro numbers?
    {
        major = optixVersion / 10000;
        minor = (optixVersion % 10000) / 100;
        micro = optixVersion % 100;
    }
    else // Old encoding with only one digit for the micro number.
    {
        minor = (optixVersion % 1000) / 10;
        micro = optixVersion % 10;
    }
    std::cout << "OptiX " << major << "." << minor << "." << micro << std::endl;

    unsigned int numberOfDevices = 0;
    RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetDeviceCount(&numberOfDevices));
    std::cout << "Number of Devices = " << numberOfDevices << std::endl << std::endl;

    for (unsigned int i = 0; i < numberOfDevices; ++i)
    {
        char name[256];
        RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_NAME, sizeof(name), name));
        std::cout << "Device " << i << ": " << name << std::endl;

        int computeCapability[2] = { 0, 0 };
        RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(computeCapability), &computeCapability));
        std::cout << "  Compute Support: " << computeCapability[0] << "." << computeCapability[1] << std::endl;

        RTsize totalMemory = 0;
        RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_TOTAL_MEMORY, sizeof(totalMemory), &totalMemory));
        std::cout << "  Total Memory: " << (unsigned long long) totalMemory << std::endl;

        int clockRate = 0;
        RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_CLOCK_RATE, sizeof(clockRate), &clockRate));
        std::cout << "  Clock Rate: " << clockRate << " kHz" << std::endl;

        int maxThreadsPerBlock = 0;
        RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, sizeof(maxThreadsPerBlock), &maxThreadsPerBlock));
        std::cout << "  Max. Threads per Block: " << maxThreadsPerBlock << std::endl;

        int smCount = 0;
        RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, sizeof(smCount), &smCount));
        std::cout << "  Streaming Multiprocessor Count: " << smCount << std::endl;

        int executionTimeoutEnabled = 0;
        RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_EXECUTION_TIMEOUT_ENABLED, sizeof(executionTimeoutEnabled), &executionTimeoutEnabled));
        std::cout << "  Execution Timeout Enabled: " << executionTimeoutEnabled << std::endl;

        int maxHardwareTextureCount = 0;
        RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_MAX_HARDWARE_TEXTURE_COUNT, sizeof(maxHardwareTextureCount), &maxHardwareTextureCount));
        std::cout << "  Max. Hardware Texture Count: " << maxHardwareTextureCount << std::endl;

        int tccDriver = 0;
        RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_TCC_DRIVER, sizeof(tccDriver), &tccDriver));
        std::cout << "  TCC Driver enabled: " << tccDriver << std::endl;

        int cudaDeviceOrdinal = 0;
        RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL, sizeof(cudaDeviceOrdinal), &cudaDeviceOrdinal));
        std::cout << "  CUDA Device Ordinal: " << cudaDeviceOrdinal << std::endl;

        int rtcoreVersion = 0;
        RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_RTCORE_VERSION, sizeof(rtcoreVersion), &rtcoreVersion));
        std::cout << "  RTCore version: " << rtcoreVersion << std::endl;
    }

    //todo: check for the issue that OptiX doesn't get correct RT_GLOBAL_ATTRIBUTE_ENABLE_RTX back to enableRTX.
    int enableRTX = 1;
    RT_CHECK_ERROR_NO_CONTEXT(rtGlobalSetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(enableRTX), &enableRTX));
    RT_CHECK_ERROR_NO_CONTEXT(rtGlobalGetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(enableRTX), &enableRTX));

    std::cout << " RTX Execution enabled: " << enableRTX << std::endl;
}

std::string Application::getPTXFilepath(const std::string & program)
{
    return std::string(SAMPLES_PTX_DIR) + std::string("\\colvillea_generated_") + program + std::string(".cu.ptx");
}

void Application::createProgramsFromPTX()
{
    auto loadProgram = [this](const std::string &file, const std::initializer_list<std::string> programs) -> void
    {
        for (const auto& program : programs)
        {
            this->m_programsMap[program] = this->m_context->createProgramFromPTXFile(this->getPTXFilepath(file), program);
        }
        
    };

    /* Load RayTracingPipeline programs. */
    loadProgram("PinholeCamera", { "Exception_Default",
                                   "RayGeneration_PinholeCamera", 
                                   "RayGeneration_InitializeFilter",
                                   "RayGeneration_Filter" });

    loadProgram("SphericalSkybox", { "Miss_Default" });

    loadProgram("TriangleMesh", { "Attributes_TriangleMesh" });
    loadProgram("Quad",         { "BoundingBox_Quad", "Intersect_Quad" });

    loadProgram("DirectLighting", { "ClosestHit_DirectLighting" });
    loadProgram("PathTracing",    { "ClosestHit_PathTracing", "ClosestHit_PTRay_PathTracing" });

    loadProgram("HitProgram", { "ClosestHit_ShadowRay_GeometryTriangles" });


    /* Load Bindless Callable programs. */
    loadProgram("PointLight", { "Sample_Ld_Point", "LightPdf_Point" });
    loadProgram("HDRILight",  { "Sample_Ld_HDRI", "LightPdf_HDRI", 
                                "RayGeneration_PrefilterHDRILight" });
    loadProgram("QuadLight",  { "Sample_Ld_Quad", "LightPdf_Quad" });

    auto loadProgramMaterial = [loadProgram](const std::initializer_list<std::string> materials)->void
    {
        for (const auto& material : materials)
        {
            loadProgram(material, { "Pdf_" + material, "Eval_f_" + material, "Sample_f_" + material });
        }
    };
    loadProgramMaterial({ "Lambert","RoughMetal","RoughDielectric","SmoothGlass","Plastic","SmoothMirror","FrostedMetal" });


}



/*******************************************************************/
/*           Initialization functions called by constructor        */
/*******************************************************************/

void Application::initializeImGui(GLFWwindow *glfwWindow)
{
    /* Create ImGui context. */
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;       // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;           // Enable Docking
    //io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         // Enable Multi-Viewport / Platform Windows
    //io.ConfigViewportsNoAutoMerge = true;
    //io.ConfigViewportsNoTaskBarIcon = true;
    
    /* ImGui window could be moved only while dragging titlebar. */
    io.ConfigWindowsMoveFromTitleBarOnly = true;

    /* Load font. */
    ImFont *imfont = io.Fonts->AddFontFromFileTTF("../../../data/droidsans.ttf", 16.0f);
    TW_ASSERT(imfont);

    /* Setup Platform/Renderer bindings. */
    ImGui_ImplGlfw_InitForOpenGL(glfwWindow, true);
    ImGui_ImplOpenGL3_Init(nullptr);

#pragma region ImGui_Style_Region
    ImGuiStyle& style = ImGui::GetStyle();

    const float r = 1.0f;
    const float g = 1.0f;
    const float b = 1.0f;

    style.Colors[ImGuiCol_Text] = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
    style.Colors[ImGuiCol_WindowBg] = ImVec4(r * 0.2f, g * 0.2f, b * 0.2f, 0.6f);
    style.Colors[ImGuiCol_ChildWindowBg] = ImVec4(r * 0.2f, g * 0.2f, b * 0.2f, 1.0f);
    style.Colors[ImGuiCol_PopupBg] = ImVec4(r * 0.2f, g * 0.2f, b * 0.2f, 1.0f);
    style.Colors[ImGuiCol_Border] = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
    style.Colors[ImGuiCol_BorderShadow] = ImVec4(r * 0.0f, g * 0.0f, b * 0.0f, 0.4f);
    style.Colors[ImGuiCol_FrameBg] = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
    style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
    style.Colors[ImGuiCol_FrameBgActive] = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    style.Colors[ImGuiCol_TitleBg] = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
    style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(r * 0.2f, g * 0.2f, b * 0.2f, 0.2f);
    style.Colors[ImGuiCol_TitleBgActive] = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    style.Colors[ImGuiCol_MenuBarBg] = ImVec4(r * 0.2f, g * 0.2f, b * 0.2f, 1.0f);
    style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(r * 0.2f, g * 0.2f, b * 0.2f, 0.2f);
    style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
    style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
    style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    style.Colors[ImGuiCol_CheckMark] = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    style.Colors[ImGuiCol_SliderGrab] = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
    style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    style.Colors[ImGuiCol_Button] = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
    style.Colors[ImGuiCol_ButtonHovered] = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
    style.Colors[ImGuiCol_ButtonActive] = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    style.Colors[ImGuiCol_Header] = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
    style.Colors[ImGuiCol_HeaderHovered] = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
    style.Colors[ImGuiCol_HeaderActive] = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    /*style.Colors[ImGuiCol_Column] = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
    style.Colors[ImGuiCol_ColumnHovered] = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
    style.Colors[ImGuiCol_ColumnActive] = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);*/
    style.Colors[ImGuiCol_ResizeGrip] = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
    style.Colors[ImGuiCol_ResizeGripHovered] = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    style.Colors[ImGuiCol_ResizeGripActive] = ImVec4(r * 1.0f, g * 1.0f, b * 1.0f, 1.0f);
    /*style.Colors[ImGuiCol_CloseButton] = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
    style.Colors[ImGuiCol_CloseButtonHovered] = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
    style.Colors[ImGuiCol_CloseButtonActive] = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);*/
    style.Colors[ImGuiCol_PlotLines] = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 1.0f);
    style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(r * 1.0f, g * 1.0f, b * 1.0f, 1.0f);
    style.Colors[ImGuiCol_PlotHistogram] = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 1.0f);
    style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(r * 1.0f, g * 1.0f, b * 1.0f, 1.0f);
    style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(r * 0.5f, g * 0.5f, b * 0.5f, 1.0f);
    style.Colors[ImGuiCol_ModalWindowDarkening] = ImVec4(r * 0.2f, g * 0.2f, b * 0.2f, 0.2f);
    style.Colors[ImGuiCol_DragDropTarget] = ImVec4(r * 1.0f, g * 1.0f, b * 0.0f, 1.0f); // Yellow
    style.Colors[ImGuiCol_NavHighlight] = ImVec4(r * 1.0f, g * 1.0f, b * 1.0f, 1.0f);
    style.Colors[ImGuiCol_NavWindowingHighlight] = ImVec4(r * 1.0f, g * 1.0f, b * 1.0f, 1.0f);
#pragma endregion ImGui_Style_Region
}

void Application::initializeRenderView()
{
    /* Create PBO for the fast OptiX sysOutputBuffer to texture transfer. */

    /* 1.Generate a new buffer object with glGenBuffers(). */
    glGenBuffers(1, &this->m_glPBO);
    TW_ASSERT(this->m_glPBO != 0); // Buffer size must be > 0 or OptiX can't create a buffer from it.

    /* 2.Bind the buffer object with glBindBuffer(). */
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, this->m_glPBO);

    /* 3.Copy pixel data to the buffer object with glBufferData(). */
    glBufferData(GL_PIXEL_UNPACK_BUFFER, this->m_filmWidth * this->m_filmHeight * sizeof(float) * 4, nullptr, GL_STREAM_READ); // RGBA32F from byte offset 0 in the pixel unpack buffer.

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    /* Generate texture storing PBO data, used in RenderView. */
    glGenTextures(1, &this->m_renderViewTexture);
    TW_ASSERT(this->m_renderViewTexture != 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, this->m_renderViewTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_2D, 0);

    /* DAR ImGui has been changed to push the GL_TEXTURE_BIT so that this works. */
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
}

void Application::initializeContext()
{
    /* Create context. */
    this->m_context = Context::create();

    /* Setup context parameters. */
    this->m_context->setMaxTraceDepth(2);
    this->m_context->setMaxCallableProgramDepth(1);
    this->m_context->setPrintEnabled(true);

    if (this->m_optixReportLevel > 0)
    {
        this->m_context->setUsageReportCallback(
            [](int lvl, const char * tag, const char * msg, void * cbdata)
        {
            Application::OptixUsageReportLogger* logger = reinterpret_cast<Application::OptixUsageReportLogger*>(cbdata);
            logger->log(lvl, tag, msg);
        }, this->m_optixReportLevel, &this->m_optixUsageReportLogger);
    }
    
    /* Disable exceptions. */
    /* this->m_context->setExceptionEnabled(RT_EXCEPTION_ALL, true); */

    /* Setup launch entries. */
    this->m_context->setEntryPointCount(toUnderlyingValue(RayGenerationEntryType::CountOfType));

    /* Setup number of ray type. */
    this->m_context->setRayTypeCount(toUnderlyingValue(CommonStructs::RayType::CountOfType));

    // todo:pack scene related paramters(materialBuffer) into a struct
    this->m_context["sysSceneEpsilon"]->setFloat(1e-4f);

    /* Create gpu programs from PTX files and add to programs map. */
    this->createProgramsFromPTX();

    /* Set programs for exception, miss and filters (ray generation). */
    auto programItr = this->m_programsMap.find("Exception_Default");
    TW_ASSERT(programItr != this->m_programsMap.end());
    this->m_context->setExceptionProgram(toUnderlyingValue(RayGenerationEntryType::Render), programItr->second);

    programItr = this->m_programsMap.find("Miss_Default");
    TW_ASSERT(programItr != this->m_programsMap.end());
    this->m_context->setMissProgram(toUnderlyingValue(RayGenerationEntryType::Render), programItr->second);

    programItr = this->m_programsMap.find("RayGeneration_InitializeFilter");
    TW_ASSERT(programItr != this->m_programsMap.end());
    this->m_context->setRayGenerationProgram(toUnderlyingValue(RayGenerationEntryType::InitFilter), programItr->second);

    programItr = this->m_programsMap.find("RayGeneration_Filter");
    TW_ASSERT(programItr != this->m_programsMap.end());
    this->m_context->setRayGenerationProgram(toUnderlyingValue(RayGenerationEntryType::Filter), programItr->second);

    programItr = this->m_programsMap.find("RayGeneration_PrefilterHDRILight");
    TW_ASSERT(programItr != this->m_programsMap.end());
    this->m_context->setRayGenerationProgram(toUnderlyingValue(RayGenerationEntryType::HDRI), programItr->second);
}

void Application::initializeOutputBuffers()
{
    /* Setup progressive rendering iteration index, which is used in RandomSampler, progressive filtering. */
    this->m_context["sysIterationIndex"]->setUint(0);

    /* Setup output buffer for RenderView. */
    this->m_sysOutputBuffer = this->m_context->createBufferFromGLBO(RT_BUFFER_OUTPUT, this->m_glPBO);
    this->m_sysOutputBuffer->setFormat(RT_FORMAT_FLOAT4); // RGBA32F Buffer for output, todo:review format
    this->m_sysOutputBuffer->setSize(this->m_filmWidth, this->m_filmHeight);
    this->m_context["sysOutputBuffer"]->set(this->m_sysOutputBuffer);

    /* Setup HDR output buffer for exporting result to OpenEXR. */
    this->m_sysHDRBuffer = this->m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, this->m_filmWidth, this->m_filmHeight);
    this->m_context["sysHDRBuffer"]->set(this->m_sysHDRBuffer);

    /* Setup reconstruction and filtering buffers. */
    Buffer weightedSumBuffer = this->m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, this->m_filmWidth, this->m_filmHeight);
    this->m_context["sysSampleWeightedSum"]->setBuffer(weightedSumBuffer);

    /* Setup current radiance and weightedSum buffers for filtering.
     * Note that in current progressive filtering algorithm, it stores elements
     * of the sum of weighted radiance(f(dx,dy)*Li) with respect to the current iteration. */
    Buffer sysCurrResultBuffer = this->m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, this->m_filmWidth, this->m_filmHeight);
    this->m_context["sysCurrResultBuffer"]->setBuffer(sysCurrResultBuffer);

    /*weightedSum buffer with respect to the current iteration*/
    Buffer sysCurrWeightedSumBuffer = this->m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, this->m_filmWidth, this->m_filmHeight);
    this->m_context["sysCurrWeightedSumBuffer"]->setBuffer(sysCurrWeightedSumBuffer);
}

void Application::initializeCallableProgramGroup()
{
    /* Initialize BSDF program group. */
    Buffer bsdfPdfProgramBuffer = this->m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, toUnderlyingValue(CommonStructs::BSDFType::CountOfType));
    Buffer bsdfEvalfProgramBuffer = this->m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, toUnderlyingValue(CommonStructs::BSDFType::CountOfType));
    Buffer bsdfSamplefProgramBuffer = this->m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, toUnderlyingValue(CommonStructs::BSDFType::CountOfType));


    int* bsdfPdfProgramBufferData = static_cast<int *>(bsdfPdfProgramBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));
    int* bsdfEvalfProgramBufferData = static_cast<int *>(bsdfEvalfProgramBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));
    int* bsdfSamplefProgramBufferData = static_cast<int *>(bsdfSamplefProgramBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));

    // todo:convert bsdfType to string for searching.
    auto loadBSDFPrograms = [this, &bsdfPdfProgramBufferData, &bsdfEvalfProgramBufferData, &bsdfSamplefProgramBufferData](CommonStructs::BSDFType bsdfType, const std::string &programName /* [in] bsdfType in string, no postfix such as _Pdf */)
    {
        //for (const auto& programName : programNames)
        //{
            auto programItr = this->m_programsMap.find("Pdf_" + programName);
            TW_ASSERT(programItr != this->m_programsMap.end());
            bsdfPdfProgramBufferData[toUnderlyingValue(bsdfType)] = programItr->second->getId();

            programItr = this->m_programsMap.find("Eval_f_" + programName);
            TW_ASSERT(programItr != this->m_programsMap.end());
            bsdfEvalfProgramBufferData[toUnderlyingValue(bsdfType)] = programItr->second->getId();

            programItr = this->m_programsMap.find("Sample_f_" + programName);
            TW_ASSERT(programItr != this->m_programsMap.end());
            bsdfSamplefProgramBufferData[toUnderlyingValue(bsdfType)] = programItr->second->getId();
       // }
    };

    // todo:using enumerating to loop over all types instead of handling them.
    loadBSDFPrograms(CommonStructs::BSDFType::Lambert,         "Lambert");
    loadBSDFPrograms(CommonStructs::BSDFType::RoughMetal,      "RoughMetal");
    loadBSDFPrograms(CommonStructs::BSDFType::RoughDielectric, "RoughDielectric");
    loadBSDFPrograms(CommonStructs::BSDFType::SmoothGlass,     "SmoothGlass");
    loadBSDFPrograms(CommonStructs::BSDFType::Plastic,         "Plastic");

    /* Note that this is on purpose that uses Lambert BSDF for emissive material. */
    loadBSDFPrograms(CommonStructs::BSDFType::Emissive, "Lambert");

    loadBSDFPrograms(CommonStructs::BSDFType::SmoothMirror, "SmoothMirror");
    loadBSDFPrograms(CommonStructs::BSDFType::FrostedMetal, "FrostedMetal"); //todo:add to scene graph

    bsdfPdfProgramBuffer->unmap();
    bsdfEvalfProgramBuffer->unmap();
    bsdfSamplefProgramBuffer->unmap();

    this->m_context["Pdf"]->setBuffer(bsdfPdfProgramBuffer);
    this->m_context["Eval_f"]->setBuffer(bsdfEvalfProgramBuffer);
    this->m_context["Sample_f"]->setBuffer(bsdfSamplefProgramBuffer);


    /* Initialize Light program group. */
    Buffer sampleLdProgramBuffer = this->m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, toUnderlyingValue(CommonStructs::LightType::CountOfType));
    int* sampleLdProgramBufferData = static_cast<int *>(sampleLdProgramBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));

    Buffer lightPdfProgramBuffer = this->m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, toUnderlyingValue(CommonStructs::LightType::CountOfType));
    int* lightPdfProgramBufferData = static_cast<int *>(lightPdfProgramBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));

    // todo:convert bsdfType to string for searching.
    auto loadLightPrograms = [this, &sampleLdProgramBufferData, &lightPdfProgramBufferData](CommonStructs::LightType lightType, const std::string &programName /* [in] lightType in string, no prefix such as Sample_Ld_ */)
    {
        auto programItr = this->m_programsMap.find("Sample_Ld_" + programName);
        TW_ASSERT(programItr != this->m_programsMap.end());
        sampleLdProgramBufferData[toUnderlyingValue(lightType)] = programItr->second->getId();

        programItr = this->m_programsMap.find("LightPdf_" + programName);
        TW_ASSERT(programItr != this->m_programsMap.end());
        lightPdfProgramBufferData[toUnderlyingValue(lightType)] = programItr->second->getId();
    };

    loadLightPrograms(CommonStructs::LightType::PointLight, "Point");
    loadLightPrograms(CommonStructs::LightType::QuadLight,  "Quad");//todo:fix name 
    loadLightPrograms(CommonStructs::LightType::HDRILight,  "HDRI");

    sampleLdProgramBuffer->unmap();
    lightPdfProgramBuffer->unmap();
    this->m_context["Sample_Ld"]->setBuffer(sampleLdProgramBuffer);
    this->m_context["LightPdf"]->setBuffer(lightPdfProgramBuffer);
}

