# find and initiate all the examples under this folder
file(GLOB_RECURSE EXAMPLEFILES "*.cpp")


foreach (proj_source ${EXAMPLEFILES})
    get_filename_component(proj_name ${proj_source} NAME_WLE)

    message(STATUS "Compiling project ${proj_name}")

    add_executable(${proj_name} ${proj_source})

    # other dependencies
    target_link_libraries(${proj_name}
            CLI11::CLI11
            spdlog::spdlog
            igl::core
            igl_restricted::triangle
            libshell
            polyscope)

    # Ensure headers are properly included
    target_sources(${proj_name} PRIVATE ${OPTIMIZATION_SRC_FILES})
endforeach ()