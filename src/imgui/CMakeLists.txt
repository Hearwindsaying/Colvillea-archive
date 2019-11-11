
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_compile_definitions("GLEW_STATIC")

#see also:https://cmake.org/Bug/view.php?id=15239
SET( CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /NODEFAULTLIB:\"msvcrt.lib\"" )

add_library( imgui STATIC
    imconfig.h
    imgui.cpp
    imgui.h
    imgui_demo.cpp
    imgui_draw.cpp
    imgui_impl_glfw.cpp
    imgui_impl_glfw.h
	imgui_impl_opengl3.cpp
	imgui_impl_opengl3.h
    imgui_internal.h
	imgui_widgets.cpp
    imstb_rectpack.h
    imstb_textedit.h
    imstb_truetype.h

	imgui_stdlib.h
	imgui_stdlib.cpp
	
	${CMAKE_CURRENT_SOURCE_DIR}/../gl/glew.c
	${CMAKE_CURRENT_SOURCE_DIR}/../gl/glew.h
	${CMAKE_CURRENT_SOURCE_DIR}/../gl/wglew.h
	
	${CMAKE_CURRENT_SOURCE_DIR}/../glfw/glfw3.h
	${CMAKE_CURRENT_SOURCE_DIR}/../glfw/glfw3native.h
    )


	
target_link_libraries( imgui ${CMAKE_CURRENT_SOURCE_DIR}/../../dependencies/glfw3.lib ${OPENGL_gl_LIBRARY})

