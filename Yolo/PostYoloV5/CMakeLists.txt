message("Configuring postyolov5 plugin")

# Add source to this project's executable.
add_library (postyolov5 MODULE "src/postyolov5.cpp" "include/postyolov5.h")

set_target_properties(postyolov5 
                        PROPERTIES 
                        PREFIX ""
                        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Plugins/$<0:>")

target_include_directories(postyolov5
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/../common/
)

target_link_libraries (postyolov5 PUBLIC RTCORE)

install(TARGETS postyolov5 DESTINATION ${CMAKE_INSTALL_PREFIX}/Plugins)
