# 2^71 project — build
# CUDA 13 / Blackwell (sm_120) ready. See docs/compatibility/CUDA13_BLACKWELL.md
TARGET      := project-2-71
SRC         := project_2_71.cu CUDAHash.cu
OBJ         := $(SRC:.cu=.o)
CC          := nvcc

# --- Toolchain detection (safe when nvcc is absent) -----------------------
CUDA_MAJOR := $(shell nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]\+\)\..*/\1/p' | head -n1)
CUDA_MINOR := $(shell nvcc --version 2>/dev/null | sed -n 's/.*release [0-9]\+\.\([0-9]\+\).*/\1/p' | head -n1)
CUDA_VER   := $(if $(CUDA_MAJOR),$(CUDA_MAJOR)$(CUDA_MINOR),0)

# --- GPU detection (safe on GPU-less machines; override with GPU_ARCH=) ---
GPU_ARCH ?= $(shell command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. ' || true)

# --- Architecture selection -------------------------------------------------
# CUDA 13 minimum supported arch is sm_75 (Turing); pre-Turing is dropped.
# sm_120 (Blackwell GeForce / RTX PRO Blackwell) requires CUDA >= 12.8.
BASE_ARCHS := 75 86 89
ifeq ($(shell test $(CUDA_VER) -ge 128 && echo yes),yes)
  BASE_ARCHS += 120
endif
SM_ARCHS ?= $(BASE_ARCHS) $(GPU_ARCH)
SM_ARCHS_SORTED := $(sort $(SM_ARCHS))
# Embed PTX for the newest arch so future GPUs can JIT.
PTX_ARCH ?= $(lastword $(SM_ARCHS_SORTED))
GENCODE := $(foreach arch,$(SM_ARCHS_SORTED),-gencode arch=compute_$(arch),code=sm_$(arch))
GENCODE += -gencode arch=compute_$(PTX_ARCH),code=compute_$(PTX_ARCH)

NVCC_FLAGS := -O3 -rdc=true -use_fast_math --ptxas-options=-O3 $(GENCODE)
CXXFLAGS   := -std=c++17
LDFLAGS    := -lcudadevrt -cudart=static

.PHONY: all info ptxas-info clean

all: info $(TARGET)

info:
	@echo "=== 2^71 project build ==="
	@echo "CUDA toolkit  : $(if $(CUDA_MAJOR),$(CUDA_MAJOR).$(CUDA_MINOR),NOT FOUND)"
	@echo "Detected GPU  : $(if $(GPU_ARCH),compute $(GPU_ARCH),none - set GPU_ARCH= to override"
	@echo "Architectures : $(SM_ARCHS_SORTED) (PTX fallback: compute_$(PTX_ARCH))"
	@echo "nvcc          : $(NVCC_FLAGS) $(CXXFLAGS)"

$(TARGET): $(OBJ)
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.cu
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) -c $< -o $@

# Register / spill report — evidence for audit hypothesis H2.
# Output goes to stderr; object discarded.
ptxas-info:
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) -Xptxas -v -c project_2_71.cu -o /dev/null
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) -Xptxas -v -c CUDAHash.cu -o /dev/null

clean:
	rm -f $(TARGET) $(OBJ)
