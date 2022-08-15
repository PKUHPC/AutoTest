set(_external_target_name libconfig)

if (CMAKE_VERSION VERSION_LESS 3.2)
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "")
else()
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "UPDATE_DISCONNECTED 1")
endif()

include(cmake/local/DownloadProject.cmake)
download_project(PROJ                ${_external_target_name}
                 GIT_REPOSITORY      https://github.com/hyperrealm/libconfig.git
                 GIT_TAG             v1.7.3
                 GIT_PROGRESS        TRUE
                 ${UPDATE_DISCONNECTED_IF_AVAILABLE}
                 PREFIX "${AITISA_API_EXTERNAL_DIR}/${_external_target_name}"
                 )


if(NOT TARGET config)
    add_subdirectory(
            ${${_external_target_name}_SOURCE_DIR}
            ${${_external_target_name}_BINARY_DIR}
            EXCLUDE_FROM_ALL)
endif()

if(CMAKE_HOST_WIN32)
    set(libname "libconfig")
else()
    set(libname "config")
endif()

add_library(aitisa_api::libconfig ALIAS ${libname}++)

unset(_external_target_name)