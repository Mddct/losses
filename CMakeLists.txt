cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(ctcdecoder)

set(CMAKE_VERBOSE_MAKEFILE on)

set(CMAKE_CXX_STANDARD 14)

include(FetchContent)
include(ExternalProject)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(FETCHCONTENT_QUIET off)

get_filename_component(fc_base "fc_base" REALPATH BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(WENET_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/wenet/runtime/core/cmake")
list(APPEND CMAKE_MODULE_PATH "${WENET_CMAKE_PATH}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -fPIC")
include(openfst)
include(glog)

add_subdirectory(src/pybind11)
include_directories(
  src/wenet/runtime/core
  )
add_library(ctc_prefix_beam_search STATIC
  src/wenet/runtime/core/decoder/ctc_prefix_beam_search.cc
  src/wenet/runtime/core/decoder/context_graph.cc
  src/wenet/runtime/core/utils/utils.cc
  src/wenet/runtime/core/utils/string.cc
  )
target_link_libraries(ctc_prefix_beam_search PUBLIC fst)

# ctc decoder
pybind11_add_module(_ctcdecoder src/ctcdecoder.cc)
target_link_libraries(_ctcdecoder PRIVATE ctc_prefix_beam_search)
