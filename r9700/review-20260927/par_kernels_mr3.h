// ------------------------------------------------------------------ conflict-free (ownership layout) + multi-row
// (local, 2026-09-28; needs par_kernels_mr.h). The stock per-token chain is LDS-bank-conflict-bound: with the
// checkpoint's random Givens pairs, a layer's reads and writes hit random banks (mr_check.py: sharing the records
// across rows alone buys ~5%). The PG producer pq_rotate_quant3 already fixed that for per-GROUP quant with the
// ownership layout (R3/INIT tables from pq_build_rot3: a lane's two pairs sit at slots {l, 32+l, 64+l, 96+l} = bank l,
// and the writes into the next layer's layout are four perfect matchings). Its fma expressions per pair are the
// stock chain's, so the rotated values are bit-identical. Here that core runs on R interleaved rows per group with
// the per-TOKEN amax/encode of pq_mr_encode -> codes and scales byte-identical to pq_rotate_tokquant /
// pq_ew_rot_tok (mr_check.py gates it).

__device__ __forceinline__ float pq_sel4(float a0, float a1, float a2, float a3, int k) {
  const float lo = (k & 1) ? a1 : a0, hi = (k & 1) ? a3 : a2;
  return (k & 2) ? hi : lo;
}

// nv: R rows x this lane's 4 channels (c0..c0+3) of the group, the producer's values (bf16-exact floats).
template <int R>
__device__ __forceinline__ void pq_mr3_rotate_rows(const float (&nv)[R][4], float cs0, float cs1, float cs2,
                                                   float cs3, const unsigned long long (&rec)[PQ_KROT_MAX][2],
                                                   unsigned long long init, int krot, float (*s_xw)[PQ_GROUP],
                                                   int lane, int c0, int nr, unsigned int (&out)[R][2],
                                                   float (&wamax)[R]) {
#pragma unroll
  for (int r = 0; r < R; ++r) {                        // channel order -> layer-0 layout (4 matched scatter writes)
    const float v0 = nv[r][0] * cs0, v1 = nv[r][1] * cs1, v2 = nv[r][2] * cs2, v3 = nv[r][3] * cs3;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const unsigned int e = (unsigned int)(init >> (16 * w)) & 0xFFFFu;
      s_xw[r][e & 127u] = pq_sel4(v0, v1, v2, v3, (int)(e >> 7));
    }
  }
  __asm__ volatile("s_waitcnt lgkmcnt(0)");
#pragma unroll
  for (int l = 0; l < PQ_KROT_MAX; ++l) {
    if (l < krot) {
      const unsigned long long r0 = rec[l][0], r1 = rec[l][1];
      const float ca = __half2float(__ushort_as_half((unsigned short)(r0 & 0xFFFFu)));
      const float sa = __half2float(__ushort_as_half((unsigned short)((r0 >> 16) & 0xFFFFu)));
      const float cb = __half2float(__ushort_as_half((unsigned short)(r1 & 0xFFFFu)));
      const float sb = __half2float(__ushort_as_half((unsigned short)((r1 >> 16) & 0xFFFFu)));
      const unsigned int e0 = (unsigned int)(r0 >> 32) & 0xFFFFu, e1 = (unsigned int)(r0 >> 48) & 0xFFFFu;
      const unsigned int e2 = (unsigned int)(r1 >> 32) & 0xFFFFu, e3 = (unsigned int)(r1 >> 48) & 0xFFFFu;
      float a0[R], a1[R], a2[R], a3[R];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        a0[r] = s_xw[r][lane]; a1[r] = s_xw[r][32 + lane];
        a2[r] = s_xw[r][64 + lane]; a3[r] = s_xw[r][96 + lane];
      }
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const float xi = a0[r], xj = a1[r], xk = a2[r], xl = a3[r];
        const float y0 = fmaf(ca, xi, sa * xj), y1 = fmaf(ca, xj, -sa * xi);
        const float y2 = fmaf(cb, xk, sb * xl), y3 = fmaf(cb, xl, -sb * xk);
        s_xw[r][e0 & 127u] = pq_sel4(y0, y1, y2, y3, (int)(e0 >> 7));
        s_xw[r][e1 & 127u] = pq_sel4(y0, y1, y2, y3, (int)(e1 >> 7));
        s_xw[r][e2 & 127u] = pq_sel4(y0, y1, y2, y3, (int)(e2 >> 7));
        s_xw[r][e3 & 127u] = pq_sel4(y0, y1, y2, y3, (int)(e3 >> 7));
      }
      __asm__ volatile("s_waitcnt lgkmcnt(0)");
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {                        // the last layer wrote channel order
    const float4 f = *(const float4 *)&s_xw[r][c0];
    if (r < nr) wamax[r] = fmaxf(wamax[r], fmaxf(fmaxf(fabsf(f.x), fabsf(f.y)), fmaxf(fabsf(f.z), fabsf(f.w))));
    out[r][0] = pq_mr_bf16x2(f.x, f.y);
    out[r][1] = pq_mr_bf16x2(f.z, f.w);
  }
  __asm__ volatile("s_waitcnt lgkmcnt(0)");           // s_x reuse by the next group
}

__device__ __forceinline__ void pq_mr3_load_group(const unsigned short *__restrict__ R3,
                                                  const unsigned short *__restrict__ INIT,
                                                  const __half *__restrict__ CS, int p, int g, int K, int G,
                                                  int krot, int lane, unsigned long long (&rec)[PQ_KROT_MAX][2],
                                                  unsigned long long &init, float &cs0, float &cs1, float &cs2,
                                                  float &cs3) {
  const unsigned long long *__restrict__ Rb =
      (const unsigned long long *)R3 + ((size_t)p * krot) * (K / 2) + (size_t)g * 64;
#pragma unroll
  for (int r = 0; r < PQ_KROT_MAX; ++r) {
    const int rc = r < krot ? r : krot - 1;
    rec[r][0] = Rb[(size_t)rc * (K / 2) + lane];
    rec[r][1] = Rb[(size_t)rc * (K / 2) + lane + 32];
  }
  init = ((const unsigned long long *)INIT)[((size_t)p * G + g) * 32 + lane];
  const uint2_t csv = *(const uint2_t *)(CS + (size_t)p * K + (size_t)g * PQ_GROUP + lane * 4);
  cs0 = __half2float(__ushort_as_half((unsigned short)(csv[0] & 0xFFFFu)));
  cs1 = __half2float(__ushort_as_half((unsigned short)(csv[0] >> 16)));
  cs2 = __half2float(__ushort_as_half((unsigned short)(csv[1] & 0xFFFFu)));
  cs3 = __half2float(__ushort_as_half((unsigned short)(csv[1] >> 16)));
}

#include "par_kernels_mr4.h"   // local 2026-09-28 day: select-free read-routed core (V4 = true)

template <int W, int R, int GPW, bool TILED, bool V4 = false>
__global__ __launch_bounds__(W * 32) void pq_rotate_tokquant3_mr(
    const __bf16 *__restrict__ X, const unsigned short *__restrict__ R3, const unsigned short *__restrict__ INIT,
    const __half *__restrict__ CS, unsigned char *__restrict__ A, float *__restrict__ AS, int M, int K, int krot) {
  __shared__ __align__(16) float s_x[W][R][PQ_GROUP];
  __shared__ float s_amax[R][W];
  const int p = blockIdx.y, m0 = blockIdx.x * R;
  const int nr = min(R, M - m0);
  const int G = K / PQ_GROUP;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int c0 = lane * 4;
  float wamax[R];
  unsigned int vals[R][GPW][2];
#pragma unroll
  for (int r = 0; r < R; ++r) wamax[r] = 0.f;
#pragma unroll
  for (int gi = 0; gi < GPW; ++gi) {
    const int g = wave + gi * W;
    if (g < G) {
      unsigned long long rec[PQ_KROT_MAX][2], init;
      float cs0, cs1, cs2, cs3;
      pq_mr3_load_group(R3, INIT, CS, p, g, K, G, krot, lane, rec, init, cs0, cs1, cs2, cs3);
      float nv[R][4];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const int m = m0 + (r < nr ? r : 0);
        const uint2_t xv = *(const uint2_t *)(X + (size_t)m * K + (size_t)g * PQ_GROUP + c0);
        nv[r][0] = __uint_as_float(xv[0] << 16); nv[r][1] = __uint_as_float(xv[0] & 0xFFFF0000u);
        nv[r][2] = __uint_as_float(xv[1] << 16); nv[r][3] = __uint_as_float(xv[1] & 0xFFFF0000u);
      }
      unsigned int outg[R][2];
      if constexpr (V4) pq_mr4_rotate_rows<R>(nv, cs0, cs1, cs2, cs3, rec, init, krot, s_x[wave], lane, nr, outg, wamax);
      else pq_mr3_rotate_rows<R>(nv, cs0, cs1, cs2, cs3, rec, init, krot, s_x[wave], lane, c0, nr, outg, wamax);
#pragma unroll
      for (int r = 0; r < R; ++r) { vals[r][gi][0] = outg[r][0]; vals[r][gi][1] = outg[r][1]; }
    }
  }
  const int Mt = (M + 15) >> 4;
  unsigned char *Ap = TILED ? A + (size_t)p * Mt * 16 * K : A + (size_t)p * M * K;
  pq_mr_encode<W, R, GPW, TILED>(wamax, vals, s_amax, Ap, AS + (size_t)p * M, m0, nr, M, K, G, wave, lane, tid);
}

// == pq_ew_rot_tok<MODE> with the conflict-free core (producer math verbatim). Partition 0 only (P = 1).
template <int MODE, int W, int R, int GPW, bool TILED, bool WHS, bool V4 = false>
__global__ __launch_bounds__(W * 32) void pq_ew_rot_tok3_mr(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps,
    const unsigned short *__restrict__ R3, const unsigned short *__restrict__ INIT, const __half *__restrict__ CS,
    __bf16 *__restrict__ HS, unsigned char *__restrict__ A, float *__restrict__ AS,
    int M, int N, int krot) {
  __shared__ __align__(16) float s_x[W][R][PQ_GROUP];
  __shared__ float s_amax[R][W];
  const int m0 = blockIdx.x * R;
  const int nr = min(R, M - m0);
  const int G = N / PQ_GROUP;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int c0 = lane * 4;
  float wamax[R];
  unsigned int vals[R][GPW][2];
#pragma unroll
  for (int r = 0; r < R; ++r) wamax[r] = 0.f;
#pragma unroll
  for (int gi = 0; gi < GPW; ++gi) {
    const int g = wave + gi * W;
    if (g < G) {
      unsigned long long rec[PQ_KROT_MAX][2], init;
      float cs0, cs1, cs2, cs3;
      pq_mr3_load_group(R3, INIT, CS, 0, g, N, G, krot, lane, rec, init, cs0, cs1, cs2, cs3);
      const int kb = g * PQ_GROUP + c0;
      float nvr[R][4];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const int m = m0 + (r < nr ? r : 0);
        float *nv = nvr[r];
        if constexpr (MODE == 0) {
          const uint2_t vg = *(const uint2_t *)(X + (size_t)m * 2 * N + kb);
          const uint2_t vu = *(const uint2_t *)(X + (size_t)m * 2 * N + N + kb);
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const float g0f = __uint_as_float(vg[h] << 16), g1f = __uint_as_float(vg[h] & 0xFFFF0000u);
            const float u0 = __uint_as_float(vu[h] << 16), u1 = __uint_as_float(vu[h] & 0xFFFF0000u);
            const float t0 = (float)(__bf16)(g0f / (1.f + expf(-g0f)));
            const float t1 = (float)(__bf16)(g1f / (1.f + expf(-g1f)));
            nv[2 * h] = (float)(__bf16)(t0 * u0);
            nv[2 * h + 1] = (float)(__bf16)(t1 * u1);
          }
        } else if constexpr (MODE == 1) {
          const uint2_t vx = *(const uint2_t *)(X + (size_t)m * N + kb);
          const uint2_t vz = *(const uint2_t *)(Y + (size_t)m * ys + kb);
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const float x0 = __uint_as_float(vx[h] << 16), x1 = __uint_as_float(vx[h] & 0xFFFF0000u);
            const float z0 = __uint_as_float(vz[h] << 16), z1 = __uint_as_float(vz[h] & 0xFFFF0000u);
            const float s0 = (float)(__bf16)(1.f / (1.f + expf(-z0)));
            const float s1 = (float)(__bf16)(1.f / (1.f + expf(-z1)));
            nv[2 * h] = (float)(__bf16)(x0 * s0);
            nv[2 * h + 1] = (float)(__bf16)(x1 * s1);
          }
        } else {
          const uint2_t vx = *(const uint2_t *)(X + (size_t)m * N + kb);
          const uint2_t vz = *(const uint2_t *)(Y + (size_t)m * ys + kb);
          const uint2_t vw = *(const uint2_t *)(Wn + c0);
          float xv[4] = {__uint_as_float(vx[0] << 16), __uint_as_float(vx[0] & 0xFFFF0000u),
                         __uint_as_float(vx[1] << 16), __uint_as_float(vx[1] & 0xFFFF0000u)};
          float ssq = 0.f;
#pragma unroll
          for (int e = 0; e < 4; ++e) ssq = fmaf(xv[e], xv[e], ssq);
#pragma unroll
          for (int off = 16; off >= 1; off >>= 1) ssq += __shfl_xor(ssq, off, 32);
          const float inv = rsqrtf(ssq * (1.f / 128.f) + eps);
          const float zv[4] = {__uint_as_float(vz[0] << 16), __uint_as_float(vz[0] & 0xFFFF0000u),
                               __uint_as_float(vz[1] << 16), __uint_as_float(vz[1] & 0xFFFF0000u)};
          const float wv[4] = {__uint_as_float(vw[0] << 16), __uint_as_float(vw[0] & 0xFFFF0000u),
                               __uint_as_float(vw[1] << 16), __uint_as_float(vw[1] & 0xFFFF0000u)};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float sg = zv[e] / (1.f + expf(-zv[e]));
            nv[e] = (float)(__bf16)(((xv[e] * inv) * wv[e]) * sg);
          }
        }
        if constexpr (WHS) {
          if (r < nr)
            *(uint2_t *)(HS + (size_t)m * N + kb) = uint2_t{pq_mr_bf16x2(nv[0], nv[1]), pq_mr_bf16x2(nv[2], nv[3])};
        }
      }
      unsigned int outg[R][2];
      if constexpr (V4) pq_mr4_rotate_rows<R>(nvr, cs0, cs1, cs2, cs3, rec, init, krot, s_x[wave], lane, nr, outg, wamax);
      else pq_mr3_rotate_rows<R>(nvr, cs0, cs1, cs2, cs3, rec, init, krot, s_x[wave], lane, c0, nr, outg, wamax);
#pragma unroll
      for (int r = 0; r < R; ++r) { vals[r][gi][0] = outg[r][0]; vals[r][gi][1] = outg[r][1]; }
    }
  }
  pq_mr_encode<W, R, GPW, TILED>(wamax, vals, s_amax, A, AS, m0, nr, M, N, G, wave, lane, tid);
}
