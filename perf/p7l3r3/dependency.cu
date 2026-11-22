#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

#define CHECK(expr) do { cudaError_t e = (expr); if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void empty_node() {}

__device__ int acquire(const int * p) {
    int value;
    asm volatile("ld.acquire.gpu.global.s32 %0,[%1];" : "=r"(value) : "l"(p) : "memory");
    return value;
}

__device__ void release(int * p, int value) {
    asm volatile("st.release.gpu.global.s32 [%0],%1;" :: "l"(p), "r"(value) : "memory");
}

enum schedule { grid, ring, broadcast };

template<schedule mode>
__global__ void dependency(int edges, int * flags, int * done) {
    const int id = blockIdx.x, blocks = gridDim.x;
    if constexpr (mode == grid) {
        for (int i = 0; i < edges; ++i) { cooperative_groups::this_grid().sync(); }
    } else if constexpr (mode == ring) {
        if (!threadIdx.x) {
            for (int i = id; i < edges; i += blocks) {
                if (i) { while (acquire(flags + id*32) < i) {} }
                release(flags + ((id + 1)%blocks)*32, i + 1);
            }
        }
    } else {
        // A single CTA publishes work. Each worker must consume it before the next publication.
        for (int i = 1; i <= edges; ++i) {
            if (!threadIdx.x) {
                if (id == 0) {
                    if (i > 1) { while (acquire(done) != (i - 1)*(blocks - 1)) {} }
                    release(flags, i);
                } else {
                    while (acquire(flags) != i) {}
                    asm volatile("red.release.gpu.global.add.s32 [%0],1;" :: "l"(done) : "memory");
                }
            }
            __syncthreads();
        }
    }
}

template<class Launch>
float elapsed(cudaStream_t stream, Launch launch) {
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    launch();
    std::vector<float> samples;
    for (int i = 0; i < 7; ++i) {
        CHECK(cudaEventRecord(a, stream));
        for (int rep = 0; rep < 16; ++rep) { launch(); }
        CHECK(cudaEventRecord(b, stream));
        CHECK(cudaEventSynchronize(b));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, a, b));
        samples.push_back(1000*ms/16);
    }
    CHECK(cudaEventDestroy(a));
    CHECK(cudaEventDestroy(b));
    std::sort(samples.begin(), samples.end());
    return samples[3];
}

float graph_time(cudaStream_t s, int blocks, int edges) {
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i <= edges; ++i) { empty_node<<<blocks, 128, 0, s>>>(); }
    CHECK(cudaStreamEndCapture(s, &graph));
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    float us = elapsed(s, [&]{ CHECK(cudaGraphLaunch(exec, s)); });
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    return us;
}

template<schedule mode>
float resident_time(cudaStream_t s, int blocks, int edges, int * flags, int * done) {
    void * args[] = { &edges, &flags, &done };
    float us = elapsed(s, [&]{
        CHECK(cudaMemsetAsync(flags, 0, (blocks*32 + 1)*sizeof(int), s));
        CHECK(cudaLaunchCooperativeKernel((void *) dependency<mode>, blocks, 128, args, 0, s));
    });
    int observed;
    if constexpr (mode != grid) {
        const int * result = mode == ring ? flags + (edges%blocks)*32 : done;
        CHECK(cudaMemcpy(&observed, result, sizeof(int), cudaMemcpyDeviceToHost));
        if (observed != (mode == ring ? edges : edges*(blocks - 1))) { exit(2); }
    }
    return us;
}

int main() {
    int devices;
    CHECK(cudaGetDeviceCount(&devices));
    char pci[32];
    CHECK(cudaDeviceGetPCIBusId(pci, sizeof(pci), 0));
    if (devices != 1 || std::string(pci) != "0000:01:00.0") { return 3; }
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("%s %s SMs=%d; CUDA event median of 7 batches x 16 replays\n", pci, prop.name, prop.multiProcessorCount);
    cudaStream_t s;
    CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    int * flags;
    CHECK(cudaMalloc(&flags, (prop.multiProcessorCount*32 + 1)*sizeof(int)));
    for (int blocks : {1, 2, 16, prop.multiProcessorCount}) {
        int * done = flags + blocks*32;
        for (int mode = -1; mode <= 2; ++mode) {
            if (blocks == 1 && mode >= 0) { continue; }
            float t[4];
            const int n[] = {32, 128, 512, 1024};
            for (int i = 0; i < 4; ++i) {
                if (mode == -1) { t[i] = graph_time(s, blocks, n[i]); }
                if (mode == grid) { t[i] = resident_time<grid>(s, blocks, n[i], flags, done); }
                if (mode == ring) { t[i] = resident_time<ring>(s, blocks, n[i], flags, done); }
                if (mode == broadcast) { t[i] = resident_time<broadcast>(s, blocks, n[i], flags, done); }
            }
            const char * names[] = {"graph", "grid", "ring", "broadcast_ack"};
            printf("%-14s blocks %2d us at 32/128/512/1024: %9.2f %9.2f %9.2f %9.2f slope(1024-128) %.4f us/edge\n",
                    names[mode + 1], blocks, t[0], t[1], t[2], t[3], (t[3]-t[1])/896);
            fflush(stdout);
        }
    }
    CHECK(cudaFree(flags));
    CHECK(cudaStreamDestroy(s));
}
