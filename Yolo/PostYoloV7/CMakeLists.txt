message("Configuring postyolov7 plugin")

# Add source to this project's executable.
add_library (postyolov7 MODULE "src/postyolov7.cpp" "include/postyolov7.h")

set_target_properties(postyolov7 
                        PROPERTIES 
                        PREFIX ""
                        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Plugins/$<0:>")

target_include_directories(postyolov7
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/../common/
)

target_link_libraries (postyolov7 PUBLIC RTCORE)

install(TARGETS postyolov7 DESTINATION ${CMAKE_INSTALL_PREFIX}/Plugins)
