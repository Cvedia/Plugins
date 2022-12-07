message("Configuring MNN plugin")

# Add source to this project's executable.
add_library (MNNPlugin MODULE "src/mnncore.cpp" "src/mnninferencehandler.cpp")

set_target_properties(MNNPlugin 
                        PROPERTIES 
                        PREFIX ""
                        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Plugins/$<0:>")

target_include_directories(MNNPlugin PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

find_library(MNN_LIB NAMES MNN HINTS ../../../bin/${CVEDIA_BIN_ARCH}/ REQUIRED)
message(STATUS "MNN library found at ${MNN_LIB}")

if (NOT WIN32)
    find_library(MNN_EXPRESS_LIB NAMES MNN_Express HINTS ../../../bin/${CVEDIA_BIN_ARCH}/ REQUIRED)
    message(STATUS "MNN_Express library found at ${MNN_EXPRESS_LIB}")
else()
    set(MNN_EXPRESS_LIB "")
endif()

target_link_libraries(MNNPlugin PRIVATE RTCORE ${MNN_LIB} ${MNN_EXPRESS_LIB})

install2(${MNN_LIB})
if (NOT WIN32)
    install2(${MNN_EXPRESS_LIB})
endif()

install(TARGETS MNNPlugin DESTINATION ${CMAKE_INSTALL_PREFIX}/Plugins)
