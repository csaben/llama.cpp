// Measure a dependent request and response with device-local polling and peer stores.
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(call) do { auto err = (call); if (err != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(err)); std::exit(1); } } while (0)

using flag = cuda::atomic_ref<unsigned, cuda::thread_scope_system>;

__global__ void exchange(unsigned * src, unsigned * dst, unsigned * ready, unsigned * next_ready,
                         int receive, int send, unsigned steps, bool responder, int * failed) {
    __shared__ int expired;
    if (threadIdx.x == 0) {
        expired = 0;
    }
    __syncthreads();
    for (unsigned step = 1; step <= steps; ++step) {
        const unsigned expected = responder ? step : step - 1;
        if (threadIdx.x == 0) {
            const auto start = clock64();
            while (flag(*ready).load(cuda::memory_order_acquire) != expected) {
                if (clock64() - start > 1000000000ULL) {
                    expired = 1;
                    *failed = 1;
                    break;
                }
            }
        }
        __syncthreads();
        if (expired) {
            return;
        }
        for (int i = threadIdx.x; i < send; i += blockDim.x) {
            unsigned value = src[i % receive];
            dst[i] = value + 1;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            flag(*next_ready).store(step, cuda::memory_order_release);
        }
        __syncthreads();
    }
}

int main() {
    constexpr int request = 2304, max_words = 4*request, steps = 4096;
    unsigned *data[2], *ready[2];
    int *failed[2];
    cudaStream_t stream[2];
    for (int d = 0; d < 2; ++d) {
        CK(cudaSetDevice(d));
        CK(cudaDeviceEnablePeerAccess(1-d, 0));
        CK(cudaStreamCreateWithFlags(&stream[d], cudaStreamNonBlocking));
        CK(cudaMalloc(&data[d], max_words*sizeof(unsigned)));
        CK(cudaMalloc(&ready[d], sizeof(unsigned)));
        CK(cudaMalloc(&failed[d], sizeof(int)));
    }
    for (int reply : {request, max_words, max_words, request}) {
        for (int d = 0; d < 2; ++d) {
            CK(cudaSetDevice(d));
            CK(cudaMemsetAsync(data[d], 0, max_words*sizeof(unsigned), stream[d]));
            CK(cudaMemsetAsync(ready[d], 0, sizeof(unsigned), stream[d]));
            CK(cudaMemsetAsync(failed[d], 0, sizeof(int), stream[d]));
            CK(cudaStreamSynchronize(stream[d]));
        }
        auto start = std::chrono::steady_clock::now();
        for (int d : {1, 0}) {
            CK(cudaSetDevice(d));
            exchange<<<1, 256, 0, stream[d]>>>(data[d], data[1-d], ready[d], ready[1-d],
                    d ? request : reply, d ? reply : request, steps, d == 1, failed[d]);
            CK(cudaGetLastError());
        }
        for (int d = 0; d < 2; ++d) {
            CK(cudaSetDevice(d));
            CK(cudaStreamSynchronize(stream[d]));
        }
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()-start).count()/steps;
        for (int d = 0; d < 2; ++d) {
            CK(cudaSetDevice(d));
            int error;
            unsigned seq;
            CK(cudaMemcpy(&error, failed[d], sizeof(error), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(&seq, ready[d], sizeof(seq), cudaMemcpyDeviceToHost));
            if (error || seq != steps) {
                std::fprintf(stderr, "device %d timeout %d sequence %u\n", d, error, seq);
                return 2;
            }
            std::vector<unsigned> result(d ? request : reply);
            CK(cudaMemcpy(result.data(), data[d], result.size()*sizeof(unsigned), cudaMemcpyDeviceToHost));
            for (unsigned v : result) {
                if (v != unsigned(2*steps-d)) {
                    std::fprintf(stderr, "device %d incorrect payload %u\n", d, v);
                    return 3;
                }
            }
        }
        std::printf("request=%zu reply=%zu roundtrip_us=%.3f checked=%d steps\n", request*sizeof(unsigned), reply*sizeof(unsigned), us, steps);
    }
    for (int d = 0; d < 2; ++d) {
        CK(cudaSetDevice(d));
        CK(cudaFree(data[d]));
        CK(cudaFree(ready[d]));
        CK(cudaFree(failed[d]));
        CK(cudaStreamDestroy(stream[d]));
    }
}
