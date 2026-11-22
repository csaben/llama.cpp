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

TIME_API(cudaStreamSynchronize, (cudaStream_t stream), (stream))
TIME_API(cudaEventSynchronize, (cudaEvent_t event), (event))
TIME_API(cudaDeviceSynchronize, (), ())
TIME_API(cudaMemcpy, (void * dst, const void * src, size_t size, cudaMemcpyKind kind), (dst, src, size, kind))

extern "C" cudaError_t cudaGraphLaunch(cudaGraphExec_t graph, cudaStream_t stream) {
    static auto call = reinterpret_cast<decltype(&cudaGraphLaunch)>(dlsym(RTLD_NEXT, "cudaGraphLaunch"));
    struct time_pair {
        cudaEvent_t first, last;
        int64_t submitted;
    };
    static thread_local time_pair times[32] = {};
    static thread_local unsigned next = 0;
    auto & pair = times[next++ % 32];
    if (!pair.last) {
        cudaEventCreate(&pair.first);
        cudaEventCreate(&pair.last);
    } else {
        float ms;
        if (cudaEventQuery(pair.last) == cudaSuccess && cudaEventElapsedTime(&ms, pair.first, pair.last) == cudaSuccess) {
            fprintf(stderr, "p7graph %lld %.4f\n", (long long) pair.submitted, ms);
        }
    }
    pair.submitted = clock_ns();
    cudaEventRecord(pair.first, stream);
    const auto rc = call(graph, stream);
    cudaEventRecord(pair.last, stream);
    return rc;
}
