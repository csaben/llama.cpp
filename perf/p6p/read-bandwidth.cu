#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <initializer_list>
#include <cstdlib>

#define CHECK(op) do { cudaError_t err = (op); if (err != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #op, cudaGetErrorString(err)); std::exit(1); } } while (0)

__global__ void fill(uint4 * data, int count) {
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < count; i += blockDim.x*gridDim.x) {
        unsigned v = (unsigned) i*2654435761u;
        data[i] = make_uint4(v, v ^ (v >> 9), ~v, v*1597334677u);
    }
}

__global__ void read_stream(const uint4 * data, int count, uint32_t * out) {
    uint4 acc = {};
    const int tid = blockIdx.x*blockDim.x + threadIdx.x;
    for (int i = tid; i < count; i += blockDim.x*gridDim.x) {
        uint4 v = data[i];
        acc.x ^= v.x;
        acc.y ^= v.y;
        acc.z ^= v.z;
        acc.w ^= v.w;
    }
    out[tid] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}

int main() {
    int devices, sm;
    CHECK(cudaGetDeviceCount(&devices));
    if (devices != 1) { return 1; }
    CHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0));
    char bus[32];
    CHECK(cudaDeviceGetPCIBusId(bus, sizeof(bus), 0));
    std::printf("CUDA device 0, PCI %s, %d SMs\n", bus, sm);
    const int count = (1 << 30)/sizeof(uint4);
    uint4 * data;
    uint32_t * out;
    CHECK(cudaMalloc(&data, (size_t) count*sizeof(*data)));
    CHECK(cudaMalloc(&out, sm*16*256*sizeof(*out)));
    fill<<<sm*16, 256>>>(data, count);
    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    for (int waves : {4, 8, 16}) {
        read_stream<<<sm*waves, 256>>>(data, count, out);
        CHECK(cudaEventRecord(start));
        for (int i = 0; i < 128; ++i) { read_stream<<<sm*waves, 256>>>(data, count, out); }
        CHECK(cudaEventRecord(end));
        CHECK(cudaEventSynchronize(end));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, end));
        std::printf("%d blocks/SM, %.1f GB/s\n", waves, 128.0*count*sizeof(*data)/(ms*1e6));
    }
    CHECK(cudaFree(data));
    CHECK(cudaFree(out));
}
