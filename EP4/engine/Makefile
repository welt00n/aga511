NVCC ?= /usr/local/cuda/bin/nvcc
FLAGS = -O2 -arch=native -std=c++17 -Xcompiler -Wall
SOURCES = main.cu options.cu scene.cu observers.cu image.cu report.cu
HEADERS = real.cuh rng.cuh geometry.cuh scene.cuh observers.cuh photon.cuh image.cuh files.cuh options.cuh run.cuh report.cuh

all: star star32

star: $(SOURCES) $(HEADERS)
	$(NVCC) $(FLAGS) $(SOURCES) -o star

star32: $(SOURCES) $(HEADERS)
	$(NVCC) $(FLAGS) -DSTAR_FP32 $(SOURCES) -o star32

test: star
	./star --test-rng

clean:
	rm -f star star32

.PHONY: all test clean
