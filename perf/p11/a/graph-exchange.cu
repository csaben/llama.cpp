// Compare CUDA graph fork/join latency with one-way peer writes at each of 48 visits.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(call) do { auto error = (call); if (error != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); std::exit(1); } } while (0)

__global__ void copy_words(const int * src, int * dst, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i] + 1;
    }
}

int main() {
    constexpr int n = 2304, visits = 48, repeats = 100;
    int *buf[2][2];
    cudaStream_t stream[2];
    cudaEvent_t done[2];
    for (int d = 0; d < 2; ++d) {
        CK(cudaSetDevice(d));
        CK(cudaDeviceEnablePeerAccess(1-d, 0));
        CK(cudaStreamCreateWithFlags(&stream[d], cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&done[d], cudaEventDisableTiming));
        for (int i = 0; i < 2; ++i) {
            CK(cudaMalloc(&buf[d][i], n*sizeof(int)));
            CK(cudaMemset(buf[d][i], 0, n*sizeof(int)));
        }
        CK(cudaDeviceSynchronize());
    }
    for (int peer : {0, 1, 1, 0}) {
        CK(cudaSetDevice(0));
        CK(cudaMemsetAsync(buf[0][0], 0, n*sizeof(int), stream[0]));
        CK(cudaStreamSynchronize(stream[0]));
        CK(cudaStreamBeginCapture(stream[0], cudaStreamCaptureModeGlobal));
        for (int i = 0; i < visits; ++i) {
            copy_words<<<9, 256, 0, stream[0]>>>(buf[0][0], buf[peer][1], n);
            if (peer) {
                CK(cudaEventRecord(done[0], stream[0]));
                CK(cudaSetDevice(1));
                CK(cudaStreamWaitEvent(stream[1], done[0]));
            }
            copy_words<<<9, 256, 0, stream[peer]>>>(buf[peer][1], buf[0][0], n);
            if (peer) {
                CK(cudaEventRecord(done[1], stream[1]));
                CK(cudaSetDevice(0));
                CK(cudaStreamWaitEvent(stream[0], done[1]));
            }
        }
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        CK(cudaStreamEndCapture(stream[0], &graph));
        CK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        for (int i = 0; i < 10; ++i) {
            CK(cudaGraphLaunch(exec, stream[0]));
        }
        CK(cudaStreamSynchronize(stream[0]));
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i) {
            CK(cudaGraphLaunch(exec, stream[0]));
        }
        CK(cudaStreamSynchronize(stream[0]));
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()-start).count()/repeats;
        std::vector<int> out(n);
        CK(cudaMemcpy(out.data(), buf[0][0], n*sizeof(int), cudaMemcpyDeviceToHost));
        for (int v : out) {
            if (v != 2*visits*(repeats+10)) {
                std::fprintf(stderr, "wrong value %d\n", v);
                return 2;
            }
        }
        std::printf("peer=%d us/token=%.2f us/roundtrip=%.2f checked=%d words\n", peer, us, us/visits, n);
        CK(cudaGraphExecDestroy(exec));
        CK(cudaGraphDestroy(graph));
    }
    for (int d = 0; d < 2; ++d) {
        CK(cudaSetDevice(d));
        CK(cudaEventDestroy(done[d]));
        CK(cudaStreamDestroy(stream[d]));
        for (int * ptr : buf[d]) {
            CK(cudaFree(ptr));
        }
    }
}
