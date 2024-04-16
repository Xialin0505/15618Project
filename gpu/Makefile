EXECUTABLE := dijkstra_parallel
CU_FILES   := dijkstra_parallel.cu
CU_DEPS    :=
CC_FILES   := 


all: $(EXECUTABLE)

###########################################################

CXX=g++ -m64
CXXFLAGS=-O3 -Wall
LDFLAGS=-L/usr/local/cuda-11.7/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc


.PHONY: dirs clean

all: $(EXECUTABLE)

default: $(EXECUTABLE)

clean:
	rm -rf $(EXECUTABLE) *.txt

$(EXECUTABLE): $(CU_FILES)
	$(NVCC) $(NVCCFLAGS) -o $(EXECUTABLE) $(CU_FILES)

