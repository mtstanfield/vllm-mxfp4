// wmma_peak.hip -- sustained v_wmma_f32_16x16x16_fp8_fp8 rate on gfx1201 with no memory traffic: each wave keeps
// ACC independent 16x16 accumulators (the MXFP4 prefill kernel uses TM*TN = 16) and issues ITER x ACC WMMAs.
// Usage: wmma_peak <waves per WG> <WGs> <iters>   (FLOP = WGs * waves * iters * ACC * 8192)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
typedef int int2_t __attribute__((ext_vector_type(2)));
typedef float floatx8 __attribute__((ext_vector_type(8)));

template <int ACC>
__global__ void peak(float *out, int iters, int seed) {
  floatx8 acc[ACC];
  for (int i = 0; i < ACC; ++i) for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;
  int2_t a, b;
  a[0] = 0x38383838 ^ (threadIdx.x & seed); a[1] = 0x30303030;
  b[0] = 0x38303830; b[1] = 0x38383838 ^ (threadIdx.x & seed);
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < ACC; ++i) acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, acc[i]);
  }
  float s = 0.f;
  for (int i = 0; i < ACC; ++i) for (int e = 0; e < 8; ++e) s += acc[i][e];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

int main(int argc, char **argv) {
  const int waves = argc > 1 ? atoi(argv[1]) : 8, wgs = argc > 2 ? atoi(argv[2]) : 1024;
  const int iters = argc > 3 ? atoi(argv[3]) : 4096;
  float *out; hipMalloc(&out, (size_t)wgs * waves * 32 * 4);
  hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
  for (int acc : {4, 8, 16}) {
    auto run = [&]() {
      if (acc == 4) hipLaunchKernelGGL(peak<4>, dim3(wgs), dim3(waves * 32), 0, 0, out, iters, 0);
      else if (acc == 8) hipLaunchKernelGGL(peak<8>, dim3(wgs), dim3(waves * 32), 0, 0, out, iters, 0);
      else hipLaunchKernelGGL(peak<16>, dim3(wgs), dim3(waves * 32), 0, 0, out, iters * 4 / 4, 0);
    };
    run(); hipDeviceSynchronize();
    hipEventRecord(e0);
    for (int r = 0; r < 5; ++r) run();
    hipEventRecord(e1); hipEventSynchronize(e1);
    float ms; hipEventElapsedTime(&ms, e0, e1);
    const double flop = 5.0 * wgs * waves * (double)iters * acc * 8192.0;
    printf("ACC=%2d waves/WG=%d WGs=%d: %.1f ms, %.1f TFLOP/s\n", acc, waves, wgs, ms, flop / (ms * 1e-3) / 1e12);
  }
  return 0;
}
