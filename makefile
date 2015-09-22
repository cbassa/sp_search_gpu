# CUDA PATH
CUDAPATH = /opt/cuda

# Compiling flags
CFLAGS = -I$(CUDAPATH)/samples/common/inc

# Linking flags
LFLAGS = -lm -L$(CUDAPATH)/lib64 -lcufft

# Compiler
NVCC = $(CUDAPATH)/bin/nvcc

sp_search_gpu: sp_search_gpu.cu
	$(NVCC) $(CFLAGS) -o sp_search_gpu sp_search_gpu.cu $(LFLAGS)

clean:
	rm -f *.o
	rm -f *~
