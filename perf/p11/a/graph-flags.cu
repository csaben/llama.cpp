// Compare a CUDA event at each exchange with two rank-local chains joined once.
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(call) do { auto err = (call); if (err != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(err)); std::exit(1); } } while (0)
using flag = cuda::atomic_ref<unsigned, cuda::thread_scope_system>;

struct buffers {
    unsigned * data;
    unsigned * ready;
    unsigned * seq;
    int * failed;
};

template<bool flags>
__global__ void transfer(buffers local, buffers remote, int n, bool responder) {
    __shared__ unsigned seq;
    __shared__ int failed;
    if constexpr (flags) {
        if (threadIdx.x == 0) {
            failed = *local.failed;
            seq = *local.seq + 1;
            if (!failed) {
                auto start = clock64();
                while (flag(*local.ready).load(cuda::memory_order_acquire) != seq - !responder) {
                    if (clock64() - start > 500000000ULL) {
                        *local.failed = failed = 1;
                        break;
                    }
                }
            }
        }
        __syncthreads();
        if (failed) { return; }
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        remote.data[i] = local.data[i] + 1;
    }
    if constexpr (flags) {
        __syncthreads();
        if (threadIdx.x == 0) {
            *local.seq = seq;
            flag(*remote.ready).store(seq, cuda::memory_order_release);
        }
    }
}

int main() {
    constexpr int n = 2304, visits = 48, repeats = 100;
    buffers b[2];
    cudaStream_t st[2];
    cudaEvent_t ev[2];
    std::vector<unsigned> initial(n);
    for (int i = 0; i < n; ++i) { initial[i] = i; }
    for (int d = 0; d < 2; ++d) {
        CK(cudaSetDevice(d));
        CK(cudaDeviceEnablePeerAccess(1-d, 0));
        CK(cudaStreamCreateWithFlags(&st[d], cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&ev[d], cudaEventDisableTiming));
        CK(cudaMalloc(&b[d].data, n*4));
        CK(cudaMalloc(&b[d].seq, 4));
        CK(cudaMalloc(&b[d].ready, 4));
        CK(cudaMalloc(&b[d].failed, 4));
    }
    for (bool flags : {false, true, true, false}) {
        for (int d = 0; d < 2; ++d) {
            CK(cudaSetDevice(d));
            CK(cudaMemsetAsync(b[d].seq, 0, 4, st[d]));
            CK(cudaMemsetAsync(b[d].ready, 0, 4, st[d]));
            CK(cudaMemsetAsync(b[d].failed, 0, 4, st[d]));
            CK(cudaMemcpyAsync(b[d].data, initial.data(), n*4, cudaMemcpyHostToDevice, st[d]));
            CK(cudaStreamSynchronize(st[d]));
        }
        CK(cudaSetDevice(0));
        CK(cudaStreamBeginCapture(st[0], cudaStreamCaptureModeGlobal));
        if (flags) {
            CK(cudaEventRecord(ev[0], st[0]));
            CK(cudaStreamWaitEvent(st[1], ev[0]));
        }
        for (int i = 0; i < visits; ++i) {
            for (int d = 0; d < 2; ++d) {
                CK(cudaSetDevice(d));
                if (flags) {
                    transfer<true><<<1, 256, 0, st[d]>>>(b[d], b[1-d], n, d == 1);
                } else {
                    transfer<false><<<1, 256, 0, st[d]>>>(b[d], b[1-d], n, d == 1);
                    CK(cudaEventRecord(ev[d], st[d]));
                    CK(cudaStreamWaitEvent(st[1-d], ev[d]));
                }
            }
        }
        if (flags) {
            CK(cudaEventRecord(ev[1], st[1]));
            CK(cudaStreamWaitEvent(st[0], ev[1]));
        }
        CK(cudaSetDevice(0));
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        CK(cudaStreamEndCapture(st[0], &graph));
        CK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        CK(cudaGraphLaunch(exec, st[0]));
        CK(cudaStreamSynchronize(st[0]));
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i) { CK(cudaGraphLaunch(exec, st[0])); }
        CK(cudaStreamSynchronize(st[0]));
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()-start).count()/repeats;
        std::vector<unsigned> out(n);
        for (int d = 0; d < 2; ++d) {
            CK(cudaSetDevice(d));
            int failed;
            CK(cudaMemcpy(&failed, b[d].failed, 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(out.data(), b[d].data, n*4, cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i) {
                if (failed || out[i] != initial[i]+2*(repeats+1)*visits-d) {
                    std::fprintf(stderr, "flags=%d d=%d failed=%d index=%d value=%u\n", flags, d, failed, i, out[i]);
                    return 2;
                }
            }
        }
        std::printf("flags=%d us/token=%.2f us/roundtrip=%.2f words_checked=%d\n", flags, us, us/visits, 2*n);
        CK(cudaSetDevice(0));
        CK(cudaGraphExecDestroy(exec));
        CK(cudaGraphDestroy(graph));
    }
    for (int d = 0; d < 2; ++d) {
        CK(cudaSetDevice(d));
        CK(cudaFree(b[d].data));
        CK(cudaFree(b[d].seq));
        CK(cudaFree(b[d].ready));
        CK(cudaFree(b[d].failed));
        CK(cudaEventDestroy(ev[d]));
        CK(cudaStreamDestroy(st[d]));
    }
}
