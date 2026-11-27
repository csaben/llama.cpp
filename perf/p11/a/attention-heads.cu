// A decode attention sharding probe with resident Q8_0 KV, 24 query heads and eight KV heads.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { auto err = (x); if (err != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(err)); std::exit(1); } } while (0)

constexpr int hd = 128, tile = 64, groups = 8, nh = 3*groups;
struct block_q8 { half scale; int8_t values[32]; };
static_assert(sizeof(block_q8) == 34);

__host__ __device__ static uint32_t mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    return x;
}

static __global__ void init_kv(block_q8 * k, block_q8 * v, int count, int offset) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    k[i].scale = __float2half(1.0f/64);
    v[i].scale = __float2half(1.0f/64);
    for (int c = 0; c < 32; ++c) {
        uint32_t value = mix(uint32_t(i + offset)*32u + c);
        k[i].values[c] = int(value & 127) - 64;
        v[i].values[c] = int((value >> 12) & 127) - 64;
    }
}

static __global__ void send_q(const float * q, float * remote, int count) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < count) { remote[i] = q[i]; }
}

// Each warp computes one query/key dot. The block reduces 64 positions into a stable softmax partial.
static __global__ void scores(const block_q8 * k, const block_q8 * v, const float * q, float * partial, int depth) {
    __shared__ float score[tile];
    const int t = threadIdx.x, lane = t % 32, warp = t / 32;
    const int head = blockIdx.y, first = blockIdx.x*tile;
    for (int i = warp; i < tile; i += 4) {
        const int row = ((head/3)*depth + first + i)*(hd/32);
        float sum = 0;
        for (int block = 0; block < hd/32; ++block) {
            sum += q[head*hd+block*32+lane]*float(k[row+block].values[lane])*__half2float(k[row+block].scale);
        }
        for (int delta = 16; delta > 0; delta /= 2) { sum += __shfl_down_sync(0xffffffff, sum, delta); }
        if (lane == 0) { score[i] = sum*rsqrtf(float(hd)); }
    }
    __syncthreads();
    float maximum = score[0];
    for (int i = 1; i < tile; ++i) { maximum = fmaxf(maximum, score[i]); }
    __syncthreads();
    if (t < tile) { score[t] = expf(score[t]-maximum); }
    __syncthreads();
    float numerator = 0, denominator = 0;
    for (int i = 0; i < tile; ++i) {
        const block_q8 & value = v[((head/3)*depth+first+i)*(hd/32)+t/32];
        numerator += score[i]*(float(value.values[t%32])*__half2float(value.scale));
        denominator += score[i];
    }
    float * out = partial + (head*gridDim.x+blockIdx.x)*(hd+2);
    out[t] = numerator;
    if (t == 0) { out[hd] = maximum; out[hd+1] = denominator; }
}

static __global__ void finish(const float * partial, float * output, int splits) {
    const int head = blockIdx.x, c = threadIdx.x;
    const float * data = partial + head*splits*(hd+2);
    float maximum = data[hd];
    for (int i = 1; i < splits; ++i) { maximum = fmaxf(maximum, data[i*(hd+2)+hd]); }
    float numerator = 0, denominator = 0;
    for (int i = 0; i < splits; ++i) {
        const float * p = data + i*(hd+2);
        const float w = expf(p[hd]-maximum);
        numerator += w*p[c];
        denominator += w*p[hd+1];
    }
    output[head*hd+c] = numerator/denominator;
}

int main(int argc, char ** argv) {
    if (argc != 2) { return 1; }
    const int depth = std::atoi(argv[1]);
    if (depth < tile || depth % tile || depth > 131072) { return 1; }
    const int splits = depth/tile;
    cudaStream_t streams[2];
    cudaEvent_t events[2];
    block_q8 * k[2], * v[2];
    float * q[2], * partial[2], * out;
    std::vector<float> input(nh*hd), expected(nh*hd), actual(nh*hd);
    for (int i = 0; i < nh*hd; ++i) { input[i] = (int(mix(i+49) & 255)-128)/128.0f; }

    for (int d = 0; d < 2; ++d) {
        CK(cudaSetDevice(d));
        CK(cudaDeviceEnablePeerAccess(1-d, 0));
        CK(cudaStreamCreateWithFlags(&streams[d], cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&events[d], cudaEventDisableTiming));
        const int heads = d ? nh/2 : nh;
        const int n = heads/3*depth*(hd/32);
        CK(cudaMalloc(&k[d], n*sizeof(block_q8)));
        CK(cudaMalloc(&v[d], n*sizeof(block_q8)));
        CK(cudaMalloc(&q[d], heads*hd*sizeof(float)));
        CK(cudaMalloc(&partial[d], heads*splits*(hd+2)*sizeof(float)));
        init_kv<<<(n+255)/256,256,0,streams[d]>>>(k[d], v[d], n, d ? n : 0);
        CK(cudaStreamSynchronize(streams[d]));
    }
    CK(cudaSetDevice(0));
    CK(cudaMalloc(&out, nh*hd*sizeof(float)));
    CK(cudaMemcpy(q[0], input.data(), nh*hd*sizeof(float), cudaMemcpyHostToDevice));

    for (int leg = 0; leg < 4; ++leg) {
        const bool split = leg == 1 || leg == 2;
        CK(cudaSetDevice(0));
        CK(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal));
        if (split) {
            send_q<<<(nh/2*hd+255)/256,256,0,streams[0]>>>(q[0]+nh/2*hd, q[1], nh/2*hd);
            CK(cudaEventRecord(events[0], streams[0]));
            CK(cudaSetDevice(1));
            CK(cudaStreamWaitEvent(streams[1], events[0]));
            scores<<<dim3(splits, nh/2), hd, 0, streams[1]>>>(k[1], v[1], q[1], partial[1], depth);
            finish<<<nh/2, hd, 0, streams[1]>>>(partial[1], out+nh/2*hd, splits);
            CK(cudaEventRecord(events[1], streams[1]));
            CK(cudaSetDevice(0));
        }
        const int heads = split ? nh/2 : nh;
        scores<<<dim3(splits, heads), hd, 0, streams[0]>>>(k[0], v[0], q[0], partial[0], depth);
        finish<<<heads, hd, 0, streams[0]>>>(partial[0], out, splits);
        if (split) { CK(cudaStreamWaitEvent(streams[0], events[1])); }
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        CK(cudaStreamEndCapture(streams[0], &graph));
        CK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        for (int i = 0; i < 8; ++i) { CK(cudaGraphLaunch(exec, streams[0])); }
        CK(cudaStreamSynchronize(streams[0]));
        const int repeats = depth <= 16384 ? 128 : 32;
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i) { CK(cudaGraphLaunch(exec, streams[0])); }
        CK(cudaStreamSynchronize(streams[0]));
        const double us = std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/repeats;
        CK(cudaMemcpy(actual.data(), out, nh*hd*sizeof(float), cudaMemcpyDeviceToHost));
        if (leg == 0) { expected = actual; }
        if (!std::all_of(actual.begin(), actual.end(), [](float x) { return std::isfinite(x); }) ||
                std::memcmp(actual.data(), expected.data(), nh*hd*sizeof(float))) { return 2; }
        std::printf("depth=%d leg=%d split=%d us=%.3f rows=%d repeats=%d\n", depth, leg, split, us, nh, repeats);
        CK(cudaGraphExecDestroy(exec));
        CK(cudaGraphDestroy(graph));
    }
    // Independently check channels at opposite ends of the head split.
    for (int head : {0, nh/2, nh-1}) {
        double num = 0, den = 0;
        for (int pos = 0; pos < depth; ++pos) {
            double dot = 0;
            for (int c = 0; c < hd; ++c) {
                const uint32_t value = mix(((head/3)*depth+pos)*hd+c);
                dot += input[head*hd+c]*(int(value&127)-64)/64.0;
            }
            const double w = std::exp(dot/std::sqrt(double(hd)));
            const uint32_t value = mix(((head/3)*depth+pos)*hd+91);
            num += w*(int((value>>12)&127)-64)/64.0;
            den += w;
        }
        if (std::abs(expected[head*hd+91]-num/den) > 2e-5) { return 3; }
    }
    CK(cudaFree(out));
    for (int d = 0; d < 2; ++d) {
        CK(cudaSetDevice(d));
        CK(cudaFree(q[d])); CK(cudaFree(k[d])); CK(cudaFree(v[d])); CK(cudaFree(partial[d]));
        CK(cudaEventDestroy(events[d])); CK(cudaStreamDestroy(streams[d]));
    }
    std::puts("All 3072 output floats matched on every leg; CPU softmax samples passed.");
}
