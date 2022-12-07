message("Configuring postyolox plugin")

# Add source to this project's executable.
add_library (postyolox MODULE "src/postyolox.cpp" "include/postyolox.h")

set_target_properties(postyolox 
                        PROPERTIES 
                        PREFIX ""
                        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Plugins/$<0:>")

target_include_directories(postyolox
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries (postyolox PUBLIC RTCORE)

install(TARGETS postyolox DESTINATION ${CMAKE_INSTALL_PREFIX}/Plugins)
