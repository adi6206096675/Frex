#include <cuda_runtime.h>
#include <iostream>

// This is the Frex hardware-level memory allocator.
// Later, this will be upgraded to use the cudaCtrlMem VMM API for paging.

extern "C" { // Use C linkage so our C++ core can easily call these without name mangling

    // Allocate physical memory directly on the GPU
    void* frex_cuda_malloc(size_t size) {
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, size);
        
        if (err != cudaSuccess) {
            std::cerr << "[Frex Hardware Error] CUDA Allocation failed: " 
                      << cudaGetErrorString(err) << std::endl;
            return nullptr;
        }
        return ptr;
    }

    // Free physical memory from the GPU
    void frex_cuda_free(void* ptr) {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }

    // A simple kernel to test that the GPU is actually executing Frex code
    __global__ void frex_warmup_kernel() {
        int thread_id = threadIdx.x;
        if (thread_id == 0) {
            printf("[Frex CUDA] GPU Hardware initialized and responding.\n");
        }
    }

    void run_frex_warmup() {
        // Launch the kernel with 1 block and 1 thread
        frex_warmup_kernel<<<1, 1>>>();
        cudaDeviceSynchronize(); // Wait for the GPU to finish
    }
}