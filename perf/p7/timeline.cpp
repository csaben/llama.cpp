// LD_PRELOAD probe for a single-stream CUDA server. See the Phase 7 profiling command in RESULTS.md.
#include <cuda_runtime_api.h>
#include <chrono>
#include <cstdio>
#include <dlfcn.h>

static int64_t clock_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

#define TIME_API(fn, params, args) \
    extern "C" cudaError_t fn params { \
        static auto call = reinterpret_cast<decltype(&fn)>(dlsym(RTLD_NEXT, #fn)); \
        const int64_t start = clock_ns(); \
        const cudaError_t rc = call args; \
        fprintf(stderr, "p7api %s %lld %lld\n", #fn, (long long) start, (long long) (clock_ns() - start)); \
        return rc; \
    }

TIME_API(cudaEventSynchronize, (cudaEvent_t event), (event))
TIME_API(cudaDeviceSynchronize, (), ())
TIME_API(cudaMemcpy, (void * dst, const void * src, size_t size, cudaMemcpyKind kind), (dst, src, size, kind))

struct time_pair {
    cudaEvent_t first, last;
    int64_t submitted;
};
static thread_local time_pair times[32] = {};

static void report(time_pair & pair) {
    float ms;
    if (pair.submitted && cudaEventQuery(pair.last) == cudaSuccess &&
            cudaEventElapsedTime(&ms, pair.first, pair.last) == cudaSuccess) {
        fprintf(stderr, "p7graph %lld %.4f\n", (long long) pair.submitted, ms);
        pair.submitted = 0;
    }
}

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    static auto call = reinterpret_cast<decltype(&cudaStreamSynchronize)>(dlsym(RTLD_NEXT, "cudaStreamSynchronize"));
    const int64_t start = clock_ns();
    const auto rc = call(stream);
    const int64_t ns = clock_ns() - start;
    fprintf(stderr, "p7api cudaStreamSynchronize %lld %lld\n", (long long) start, (long long) ns);
    if (ns > 5000000) {
        for (auto & pair : times) {
            report(pair);
        }
    }
    return rc;
}

extern "C" cudaError_t cudaGraphLaunch(cudaGraphExec_t graph, cudaStream_t stream) {
    static auto call = reinterpret_cast<decltype(&cudaGraphLaunch)>(dlsym(RTLD_NEXT, "cudaGraphLaunch"));
    static thread_local unsigned next = 0;
    auto & pair = times[next++ % 32];
    if (!pair.last) {
        cudaEventCreate(&pair.first);
        cudaEventCreate(&pair.last);
    } else {
        report(pair);
    }
    pair.submitted = clock_ns();
    cudaEventRecord(pair.first, stream);
    const auto rc = call(graph, stream);
    cudaEventRecord(pair.last, stream);
    return rc;
}
