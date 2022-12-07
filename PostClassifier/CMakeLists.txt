message("Configuring postclassifier plugin")

# Add source to this project's executable.
add_library (postclassifier MODULE "src/postclassifier.cpp")

set_target_properties(postclassifier 
                        PROPERTIES 
                        PREFIX ""
                        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Plugins/$<0:>")

target_include_directories(postclassifier
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries (postclassifier PUBLIC RTCORE)

install(TARGETS postclassifier DESTINATION ${CMAKE_INSTALL_PREFIX}/Plugins)
