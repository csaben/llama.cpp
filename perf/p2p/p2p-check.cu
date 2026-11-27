// P2P check: small-exchange latency, direct peer-write kernel, and bandwidth, with peer access OFF vs ON.
#include <cstdio>
#include <cuda_runtime.h>
#include <chrono>
#include <string>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("%s: %s\n",#x,cudaGetErrorString(e)); return 1;}}while(0)
__global__ void peer_write(float* dst, const float* src, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dst[i]=src[i]; }
static double now(){ return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int run(bool peer){
  const size_t small=9*1024, big=256u<<20; float *a0,*b0,*a1,*b1; cudaStream_t s0;
  CK(cudaSetDevice(0)); CK(cudaMalloc(&a0,big)); CK(cudaMalloc(&b0,big)); CK(cudaStreamCreate(&s0));
  CK(cudaSetDevice(1)); CK(cudaMalloc(&a1,big)); CK(cudaMalloc(&b1,big));
  if(peer){ CK(cudaSetDevice(0)); CK(cudaDeviceEnablePeerAccess(1,0)); CK(cudaSetDevice(1)); CK(cudaDeviceEnablePeerAccess(0,0)); }
  CK(cudaSetDevice(0));
  int N=2000; for(int i=0;i<200;i++) cudaMemcpyPeerAsync(a1,1,a0,0,small,s0); cudaStreamSynchronize(s0);
  double t=now(); for(int i=0;i<N;i++){ cudaMemcpyPeerAsync(a1,1,a0,0,small,s0); cudaMemcpyPeerAsync(b0,0,a1,1,small,s0);} cudaStreamSynchronize(s0);
  double rt=(now()-t)/N;
  double kw=-1;
  if(peer){ int n=small/4; for(int i=0;i<200;i++) peer_write<<<(n+255)/256,256,0,s0>>>(a1,a0,n); cudaStreamSynchronize(s0);
    t=now(); for(int i=0;i<N;i++) peer_write<<<(n+255)/256,256,0,s0>>>(a1,a0,n); cudaStreamSynchronize(s0); kw=(now()-t)/N; }
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0,s0); for(int i=0;i<10;i++) cudaMemcpyPeerAsync(a1,1,a0,0,big,s0); cudaEventRecord(e1,s0); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms,e0,e1); double gbs=10.0*big/(ms*1e6);
  printf("peer access %-3s | 9 KB round trip %6.1f us | 9 KB kernel peer-write %s | 256 MB copy %5.1f GB/s\n",
    peer?"ON":"OFF", rt, kw<0?"   n/a  ":(std::to_string(kw).substr(0,5)+" us").c_str(), gbs);
  if(peer){ cudaSetDevice(0); cudaDeviceDisablePeerAccess(1); cudaSetDevice(1); cudaDeviceDisablePeerAccess(0); }
  cudaSetDevice(0); cudaFree(a0); cudaFree(b0); cudaSetDevice(1); cudaFree(a1); cudaFree(b1); return 0;
}
int main(){ int c01,c10; cudaDeviceCanAccessPeer(&c01,0,1); cudaDeviceCanAccessPeer(&c10,1,0); printf("canAccessPeer 0->1:%d 1->0:%d\n",c01,c10);
  if(run(false)) return 1; return run(true); }
