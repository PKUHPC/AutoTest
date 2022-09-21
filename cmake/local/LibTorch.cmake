set(_external_target_name libtorch)

if (CMAKE_VERSION VERSION_LESS 3.2)
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "")
else ()
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "UPDATE_DISCONNECTED 1")
endif ()

include(cmake/local/DownloadProject.cmake)
download_project(PROJ ${_external_target_name}
        URL https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
        DOWNLOAD_NO_EXTRACT true
        ${UPDATE_DISCONNECTED_IF_AVAILABLE}
        PREFIX "${AITISA_API_EXTERNAL_DIR}/${_external_target_name}"
        )

file(ARCHIVE_EXTRACT
        INPUT ${AITISA_API_EXTERNAL_DIR}/${_external_target_name}/${_external_target_name}-src/libtorch-shared-with-deps-latest.zip
        DESTINATION ${AITISA_API_EXTERNAL_DIR}/${_external_target_name}/${_external_target_name}-src)
