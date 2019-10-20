#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw_gl2.h> 


#include <GL/glew.h>
#if defined( _WIN32 )
#include <GL/wglew.h>
#endif

#include <GLFW/glfw3.h>

#include "colvillea/Application/Application.h"
#include "colvillea/Device/Toolkit/CommonStructs.h"
#include "colvillea/Module/Material/MaterialPool.h"
#include "colvillea/Module/Light/LightPool.h"
#include "colvillea/Application/SceneGraph.h"

#include <iostream>
#include <locale>
#include <string>

using namespace optix;

/* Create Cornellbox. */
void create_CornellBoxScene(std::shared_ptr<SceneGraph> &sceneGraph, std::unique_ptr<LightPool> &lightPool, std::unique_ptr<Application> &application, std::unique_ptr<MaterialPool> &materialPool, const std::string & basePath)
{
    /* Create integator. */

    /* For finding a better position to look at CornellBox. */
    //lightPool->createHDRILight(basePath + "HDRI\\uffizi-large.hdr", Matrix4x4::identity());

    sceneGraph->createDirectLightingIntegrator();
    //sceneGraph->createPathTracingIntegrator(true, 5);

    /* Create sampler. */
    sceneGraph->createSampler(CommonStructs::SamplerType::SobolQMCSampler);// define USE_HALTON_SAMPLER to enable Halton

    /* Create triangle mesh. */
    sceneGraph->createTriangleMesh(
        basePath + "Cornell\\green.obj",
        materialPool->createLambertMaterial(optix::make_float4(0.63f, 0.065f, 0.05f, 1.f)));
    sceneGraph->createTriangleMesh(
        basePath + "Cornell\\red.obj",
        materialPool->createLambertMaterial(optix::make_float4(0.14f, 0.45f, 0.091f, 1.f)));
    sceneGraph->createTriangleMesh(
        basePath + "Cornell\\grey.obj",
        materialPool->createLambertMaterial(optix::make_float4(0.725f, 0.71f, 0.68f, 1.f)));

    /* Create light. */
    lightPool->createQuadLight(
         Matrix4x4::translate(make_float3(0.f, 0.f, 11.7f)) * 
         Matrix4x4::scale(make_float3(3.25f, 2.625f, 1.f)),
        make_float3(17.f, 12.f, 4.f), materialPool->createEmissiveMaterial(), true);
}

/* Create test scene. */
void create_TestScene(std::shared_ptr<SceneGraph> &sceneGraph, std::unique_ptr<LightPool> &lightPool, std::unique_ptr<Application> &application, std::unique_ptr<MaterialPool> &materialPool, const std::string & basePath)
{
    sceneGraph->createDirectLightingIntegrator();
    //sceneGraph->createPathTracingIntegrator(true, 5);
    lightPool->createHDRILight(basePath + "HDRI\\uffizi-large.hdr", Matrix4x4::identity());
    sceneGraph->createSampler(CommonStructs::SamplerType::SobolQMCSampler);// todo:ifdef USE_HALTON_SAMPLER to enable Halton

    /* TriangleMesh is created with the help of MaterialPool. */
    //std::unique_ptr<MaterialPool> materialPool = std::make_unique<MaterialPool>(application->getProgramsMap(), application->getContext());

    //sceneGraph->createTriangleMesh(
    //    "D:\\Project\\Twilight\\GraphicsRes\\Colvillea\\sofa.obj", 
    //    materialPool->createLambertMaterial(optix::make_float4(.8f)));

    //auto quadShape = sceneGraph->createQuad(
    //    materialPool->createRoughMetalMaterial(0.03f, make_float4(1.66f,0.95151f,0.7115f,0.f), make_float4(8.0406f,6.3585f,5.1380f,0.f)), 
    //    /*materialPool->createLambertMaterial(optix::make_float4(0.8f)),*/
    //       /*Matrix4x4::translate(make_float3(0.f,0.f,4.f)) * */Matrix4x4::rotate(TwUtil::deg2rad(45.f), make_float3(1.f,0.f,0.f)));
    //quadShape->flipGeometryNormal();

    auto quadShapeB = sceneGraph->createQuad(
        materialPool->createRoughMetalMaterial(0.03f, make_float4(1.66f, 0.95151f, 0.7115f, 0.f), make_float4(8.0406f, 6.3585f, 5.1380f, 0.f)),
        /*materialPool->createLambertMaterial(optix::make_float4(0.8f)),*/
        Matrix4x4::identity());
    //quadShapeB->flipGeometryNormal();

    //sceneGraph->createTriangleMesh("D:\\Project\\Twilight\\GraphicsRes\\Colvillea\\roughmetal.obj",
    //    materialPool->createRoughMetalMaterial(0.03f, make_float4(1.66f, 0.95151f, 0.7115f, 0.f), 
    //    make_float4(8.0406f, 6.3585f, 5.1380f, 0.f)));
    //sceneGraph->createTriangleMesh("D:\\Project\\Twilight\\GraphicsRes\\Colvillea\\sphere.obj", 
    //    /*materialPool->createRoughMetalMaterial(0.03f, make_float4(1.66f, 0.95151f, 0.7115f, 0.f), make_float4(8.0406f, 6.3585f, 5.1380f, 0.f))*/
    //    materialPool->createLambertMaterial(optix::make_float4(0.8f))
    //    /*materialPool->createSmoothGlassMaterial(1.46f)*/);
    int emissiveMaterial = materialPool->createEmissiveMaterial();

    lightPool->createQuadLight(
        Matrix4x4::translate(make_float3(-0.5f, 0.f, 1.5f))
        /**Matrix4x4::scale(make_float3(0.5f,0.5f,1.f))
        *Matrix4x4::rotate(TwUtil::deg2rad(-45.f), make_float3(1.f, 0.f, 0.f))*/,
        make_float3(4.f, 0.f, 0.f), emissiveMaterial, true);
    lightPool->createQuadLight(
        Matrix4x4::translate(make_float3(0.5f, 0.f, 1.5f))
        /**Matrix4x4::scale(make_float3(0.5f, 0.5f, 1.f))
        *Matrix4x4::rotate(TwUtil::deg2rad(45.f), make_float3(1.f, 0.f, 0.f))*/,
        make_float3(0.f, 0.f, 4.f), emissiveMaterial, true);
}



int main(int argc, char *argv[])
{
    //_CrtSetBreakAlloc(3231);

    /* Setup GLFW/GLEW context. */
    auto glfwErrorFunc = [](int error, const char* description)
    {
        std::cerr << "Error: " << error << ": " << description << std::endl; 
    };
    glfwSetErrorCallback(glfwErrorFunc);

    if (!glfwInit())
    {
        glfwErrorFunc(1, "GLFW failed to initialize.");
        return 1;
    }

    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
    GLFWwindow* glfwWindow = glfwCreateWindow(1920, 1080, "Colvillea 0.1.0", NULL, NULL);

    if (!glfwWindow)
    {
        glfwErrorFunc(2, "glfwCreateWindow() failed.");
        glfwTerminate();
        return 2;
    }

    glfwMakeContextCurrent(glfwWindow);

    if (glewInit() != GL_NO_ERROR)
    {
        glfwErrorFunc(3, "GLEW failed to initialize.");
        glfwTerminate();
        return 3;
    }

    auto getExampleDirectoryPath = []()->std::string
    {
        const std::string filename = __FILE__;
        std::string::size_type extension_index = filename.find_last_of("\\");
        std::string filePath = extension_index != std::string::npos ?
            filename.substr(0, extension_index) :
            std::string();

        return filePath + "\\..\\..\\..\\examples\\";
    };

    const std::string examplesBasePath = getExampleDirectoryPath();

    
    /* Create scene. */ 

    /* Parameters shared by camera and application. */
    //const uint32_t filmWidth = 1280, filmHeight = 720;
    const uint32_t filmWidth = 1024, filmHeight = 1024;
    const float fov = 60.f;

    /* Create application instance first.*/
    std::unique_ptr<Application> application = std::make_unique<Application>(glfwWindow, filmWidth, filmHeight, 0, 1200);

    /* Create sceneGraph instance. */
    std::shared_ptr<SceneGraph> sceneGraph = std::make_unique<SceneGraph>(application.get(), application->getProgramsMap(), application->getContext(), filmWidth, filmHeight);

    /* Create materialPool instance. */
    std::unique_ptr<MaterialPool> materialPool = std::make_unique<MaterialPool>(application->getProgramsMap(), application->getContext());

    /* Create lightPool instance. */
    std::unique_ptr<LightPool> lightPool = std::make_unique<LightPool>(application.get(), application->getProgramsMap(), application->getContext(), sceneGraph);

    /* Create scene using sceneGraph::createXXX methods. */
    sceneGraph->createCamera(Matrix4x4::identity(), fov, filmWidth, filmHeight);
    create_CornellBoxScene(sceneGraph, lightPool, application, materialPool, examplesBasePath); /* left scene configurations are created... */
    //create_TestScene(sceneGraph, lightPool, application, materialPool, examplesBasePath);

    /* Finally initialize scene and prepare for launch. */
    application->buildSceneGraph(sceneGraph);

    /* Main loop for Colvillea. */
    while (!glfwWindowShouldClose(glfwWindow))
    {
        glfwPollEvents();

        ImGui_ImplGlfwGL2_NewFrame();

        application->drawWidget();
        application->render();

        ImGui::Render();
        ImGui_ImplGlfwGL2_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(glfwWindow);
    }

    glfwTerminate();

    _CrtDumpMemoryLeaks();
    return 0;
}