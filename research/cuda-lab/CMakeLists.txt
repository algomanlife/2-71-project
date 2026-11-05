cmake_minimum_required(VERSION 3.18)
project(cuda-lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Target architecture: sm_120 (Blackwell) by default; override for other
# GPUs: cmake -DCMAKE_CUDA_ARCHITECTURES=89 ..
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

# Standalone: compiles independently of the project's search application.
# secp_ref.cuh is verbatim-extracted project arithmetic (see file header).

add_executable(cuda-lab-bench src/main.cu src/kernels.cu)
target_include_directories(cuda-lab-bench PRIVATE include)

add_executable(cuda-lab-tests tests/test_vectors.cu src/kernels.cu)
target_include_directories(cuda-lab-tests PRIVATE include)

enable_testing()
add_test(NAME public_vectors COMMAND cuda-lab-tests)

# isolated experimental variants (baseline benches unchanged; see
# experiments/modmul_foldmul.cu for the hypothesis and decision rules)
add_executable(cuda-lab-exp experiments/modmul_foldmul.cu src/kernels.cu)
target_include_directories(cuda-lab-exp PRIVATE include)
