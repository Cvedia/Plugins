message("Configuring postyolov6 plugin")

# Add source to this project's executable.
add_library (postyolov6 MODULE "src/postyolov6.cpp" "include/postyolov6.h")

set_target_properties(postyolov6 
                        PROPERTIES 
                        PREFIX ""
                        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Plugins/$<0:>")

target_include_directories(postyolov6
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/../common/
)

target_link_libraries (postyolov6 PUBLIC RTCORE)

install(TARGETS postyolov6 DESTINATION ${CMAKE_INSTALL_PREFIX}/Plugins)
