message("Configuring postyolov4 plugin")

# Add source to this project's executable.
add_library (postyolov4 MODULE "src/postyolov4.cpp")

set_target_properties(postyolov4 
                        PROPERTIES 
                        PREFIX ""
                        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Plugins/$<0:>")

target_include_directories(postyolov4
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries (postyolov4 PUBLIC RTCORE)

install(TARGETS postyolov4 DESTINATION ${CMAKE_INSTALL_PREFIX}/Plugins)
