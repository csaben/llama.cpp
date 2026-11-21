#include <cstdio>
#include <cuda_runtime.h>
#include <chrono>
int main(){
  int a=0,b=0; cudaDeviceCanAccessPeer(&a,0,1); cudaDeviceCanAccessPeer(&b,1,0); printf("P2P 0->1:%d 1->0:%d\n",a,b);
  const size_t n=2304*4; // one hidden-state vector in fp32 (9 KB)
  float *d0,*d1; cudaSetDevice(0); cudaMalloc(&d0,n); cudaSetDevice(1); cudaMalloc(&d1,n);
  cudaStream_t s; cudaStreamCreate(&s);
  for(int it=0;it<200;it++){ cudaMemcpyPeerAsync(d0,0,d1,1,n,s); }
  cudaStreamSynchronize(s);
  auto t=std::chrono::high_resolution_clock::now(); int N=2000;
  for(int it=0;it<N;it++){ cudaMemcpyPeerAsync(d0,0,d1,1,n,s); cudaMemcpyPeerAsync(d1,1,d0,0,n,s);} cudaStreamSynchronize(s);
  double us=std::chrono::duration<double,std::micro>(std::chrono::high_resolution_clock::now()-t).count()/N;
  printf("round-trip exchange of a 9KB vector (stream-ordered): %.1f us\n",us);
}
