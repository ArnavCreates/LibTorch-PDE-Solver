cmake_minimum_required(VERSION 3.20)
project(LibTorchPDESolver VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Ensure Release build by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build configuration" FORCE)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall -Wextra
        -Wno-deprecated-declarations
        -Wno-unused-parameter
        -Wno-unused-variable
        -O3
    )
elseif(MSVC)
    add_compile_options(/W4 /O2)
endif()

if(WIN32 AND NOT Torch_DIR)
    set(_MSYS2_PREFIX "$ENV{MSYS2_PREFIX}")
    if(NOT _MSYS2_PREFIX AND DEFINED ENV{MSYS2_ROOT})
        set(_MSYS2_PREFIX "$ENV{MSYS2_ROOT}/ucrt64")
    endif()
    if(NOT _MSYS2_PREFIX AND CMAKE_CXX_COMPILER MATCHES "ucrt64")
        get_filename_component(_MSYS2_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
        get_filename_component(_MSYS2_PREFIX "${_MSYS2_BIN_DIR}" DIRECTORY)
    endif()
    if(_MSYS2_PREFIX)
        file(GLOB TORCH_CMAKE_DIRS
            "${_MSYS2_PREFIX}/lib/python*/site-packages/torch/share/cmake/Torch"
        )
    endif()
    if(TORCH_CMAKE_DIRS)
        list(SORT TORCH_CMAKE_DIRS)
        list(GET TORCH_CMAKE_DIRS -1 Torch_DIR)
    endif()
endif()

find_package(Torch REQUIRED)
message(STATUS "Found Torch: ${TORCH_LIBRARIES}")

# 2. OpenMP
find_package(OpenMP REQUIRED)

# 3. HDF5
find_package(HDF5 REQUIRED COMPONENTS C)
message(STATUS "Found HDF5: ${HDF5_LIBRARIES}")

# 4. GoogleTest
find_package(GTest REQUIRED)
message(STATUS "Found GTest: ${GTEST_LIBRARIES}")

# Core PDE solver library
add_library(pde_solver_core STATIC
    src/pinn_network.cpp
    src/autograd_residual.cpp
    src/fdm_gauss_seidel.cpp
    src/conservation_validator.cpp
    src/roofline_profiler.cpp
    src/hdf5_serializer.cpp
)

target_include_directories(pde_solver_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_include_directories(pde_solver_core SYSTEM PUBLIC
        ${TORCH_INCLUDE_DIRS}
        ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(pde_solver_core
    PUBLIC
        ${TORCH_LIBRARIES}
        OpenMP::OpenMP_CXX
        ${HDF5_LIBRARIES}
)

if(TORCH_CXX_FLAGS)
    separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(pde_solver_core PUBLIC ${TORCH_CXX_FLAGS_LIST})
endif()

# Main CLI solver executable
add_executable(pde_solver_main
    src/main_solver.cpp
)

target_link_libraries(pde_solver_main
    PRIVATE
        pde_solver_core
)

# C++ Unit test executable
enable_testing()
add_executable(pde_unit_tests
    tests/cpp/test_pde_components.cpp
)

target_link_libraries(pde_unit_tests
    PRIVATE
        pde_solver_core
        GTest::gtest
        GTest::gtest_main
)

add_test(NAME PdeUnitTests COMMAND pde_unit_tests)
