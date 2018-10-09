set(NV_TENSORRT_ROOT "" CACHE PATH "TENSORRT root folder")

set(TENSORRT_LIB_NAME "libnvinfer.so")

find_path(NV_TENSORRT_INCLUDE NvInfer.h
    PATHS ${NV_TENSORRT_ROOT} $ENV{NV_TENSORRT_ROOT} ${CUDA_TOOLKIT_INCLUDE}
    DOC "Path to TensorRT include directory." )

get_filename_component(__libpath_hist ${CUDA_CUDART_LIBRARY} PATH)
find_library(LIB_NVINFER NAMES libnvinfer.so
    PATHS ${NV_TENSORRT_ROOT} $ENV{NV_TENSORRT_ROOT} ${NV_TENSORRT_INCLUDE} ${__libpath_hist} ${__libpath_hist}/../lib
    DOC "Path to TensorRT library.")
find_library(LIB_NVCAFFE_PARSER NAMES libnvcaffe_parser.so
    PATHS ${NV_TENSORRT_ROOT} $ENV{NV_TENSORRT_ROOT} ${NV_TENSORRT_INCLUDE} ${__libpath_hist} ${__libpath_hist}/../lib
    DOC "Path to TensorRT library.")
find_library(LIB_NVINFER_PLUGIN NAMES libnvinfer_plugin.so
    PATHS ${NV_TENSORRT_ROOT} $ENV{NV_TENSORRT_ROOT} ${NV_TENSORRT_INCLUDE} ${__libpath_hist} ${__libpath_hist}/../lib
    DOC "Path to TensorRT library.")
find_library(LIB_NVPARSER NAMES libnvparsers.so
    PATHS ${NV_TENSORRT_ROOT} $ENV{NV_TENSORRT_ROOT} ${NV_TENSORRT_INCLUDE} ${__libpath_hist} ${__libpath_hist}/../lib
    DOC "Path to TensorRT library.")


if(NV_TENSORRT_INCLUDE AND LIB_NVINFER)
    set(HAVE_TENSORRT  TRUE)
    set(TENSORRT_FOUND TRUE)

    file(READ ${NV_TENSORRT_INCLUDE}/NvInfer.h NV_TENSORRT_VERSION_FILE_CONTENTS)

    # NV_TENSORRT v4 and beyond
    string(REGEX MATCH "define NV_TENSORRT_MAJOR * +([0-9]+)"
           NV_TENSORRT_MAJOR "${NV_TENSORRT_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define NV_TENSORRT_MAJOR * +([0-9]+)" "\\1"
           NV_TENSORRT_MAJOR "${NV_TENSORRT_MAJOR}")
    string(REGEX MATCH "define NV_TENSORRT_MINOR * +([0-9]+)"
           NV_TENSORRT_MINOR "${NV_TENSORRT_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define NV_TENSORRT_MINOR * +([0-9]+)" "\\1"
           NV_TENSORRT_MINOR "${NV_TENSORRT_MINOR}")
    string(REGEX MATCH "define NV_TENSORRT_PATCH * +([0-9]+)"
           NV_TENSORRT_VERSION_PATCH "${NV_TENSORRT_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define NV_TENSORRT_PATCH * +([0-9]+)" "\\1"
           NV_TENSORRT_VERSION_PATCH "${NV_TENSORRT_VERSION_PATCH}")

    if (NOT NV_TENSORRT_MAJOR)
      set(NV_TENSORRT_VERSION "???")
    else ()
      set(NV_TENSORRT_VERSION "${NV_TENSORRT_MAJOR}.${NV_TENSORRT_MINOR}.${NV_TENSORRT_VERSION_PATCH}")
    endif()

    message(STATUS "Found TensorRT: ver. ${NV_TENSORRT_VERSION} found (include: ${NV_TENSORRT_INCLUDE}, library: ${TENSORRT_LIBRARY})")

    string(COMPARE LESS "${NV_TENSORRT_MAJOR}" 5 NvTensorRTVersionIncompatible)
    if(NvTensorRTVersionIncompatible)
      message(FATAL_ERROR "TensorRT version > 5  is required.")
    endif()

    set(NV_TENSORRT_VERSION "${NV_TENSORRT_VERSION}")
    mark_as_advanced(NV_TENSORRT_INCLUDE TENSORRT_LIBRARY NV_TENSORRT_ROOT)
else(NV_TENSORRT_INCLUDE AND TENSORRT_LIBRARY)
    message(STATUS "TensorRT not found")
endif()
