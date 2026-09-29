// ------------------------------------------------------------------ multi-row per-token producers (local, 2026-09-28)
//
// The stock per-token producers (pq_rotate_tokquant, pq_ew_rot_tok) run ONE workgroup per (row, partition), and every
// row's workgroup re-reads its groups' rotation records: krot x 512 B per 128-channel group, 160 KB per row at K=5120,
// krot 8 -- ten times the row's own bytes. At prefill M the kernels are therefore L2-bound on records (~1.9 TB/s,
// rot_bench.py) while moving the row data at a third of DRAM bandwidth.
//
// Here a workgroup owns R consecutive rows. Wave w still owns groups g = w, w+W, ... (the stock mapping), but loads a
// group's records ONCE and applies them to all R rows. Each row's rotated values stay in registers as packed bf16 --
// each wave encodes exactly the groups it rotated, so nothing is parked in LDS -- and the token amax is reduced per row
// through LDS. Arithmetic is the stock kernels' own, in the same order: channel scale, the krot Givens layers through
// the wave's s_x, amax over the fp32 rotated values, encode from the bf16-rounded ones. Codes and scales are therefore
// byte-identical (mr_check.py gates it). The prefill band only (M > 64): at decode M the stock kernels' short
// per-wave chain is what matters. WHS=false skips the ew producer's bf16 HS write, which the stream consumer
// (paroquant_mxfp4_linear_pre) never reads -- 35 KB per row on the silu-mul site.

__device__ __forceinline__ unsigned int pq_mr_bf16x2(float a, float b) {
  return (unsigned int)__bfloat16_as_ushort((__bf16)a) | ((unsigned int)__bfloat16_as_ushort((__bf16)b) << 16);
}

// pq_tok_rotate_park with the bf16 row kept in registers instead of LDS (same fmaf chain, same waits).
__device__ __forceinline__ void pq_mr_rotate(const float (&nv)[4], const float (&cs)[4],
                                             const unsigned long long (&rec)[PQ_KROT_MAX][2], int krot,
                                             float *s_xw, int c0, unsigned int (&out)[2], float &wamax) {
  float v0 = nv[0] * cs[0], v1 = nv[1] * cs[1], v2 = nv[2] * cs[2], v3 = nv[3] * cs[3];
  s_xw[c0 + 0] = v0; s_xw[c0 + 1] = v1; s_xw[c0 + 2] = v2; s_xw[c0 + 3] = v3;
  __asm__ volatile("s_waitcnt lgkmcnt(0)");
#pragma unroll
  for (int r = 0; r < PQ_KROT_MAX; ++r) {
    if (r < krot) {
#pragma unroll
      for (int t2 = 0; t2 < 2; ++t2) {
        const unsigned long long rv = rec[r][t2];
        const unsigned int ij = (unsigned int)(rv & 0xFFFFu);
        const float c = __half2float(__ushort_as_half((unsigned short)((rv >> 16) & 0xFFFFu)));
        const float sn = __half2float(__ushort_as_half((unsigned short)((rv >> 32) & 0xFFFFu)));
        const int i = ij & 0xFF, j = ij >> 8;
        const float xi = s_xw[i], xj = s_xw[j];
        s_xw[i] = fmaf(c, xi, sn * xj);
        s_xw[j] = fmaf(c, xj, -sn * xi);
      }
      __asm__ volatile("s_waitcnt lgkmcnt(0)");
    }
  }
  v0 = s_xw[c0 + 0]; v1 = s_xw[c0 + 1]; v2 = s_xw[c0 + 2]; v3 = s_xw[c0 + 3];
  wamax = fmaxf(wamax, fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3))));
  out[0] = pq_mr_bf16x2(v0, v1);
  out[1] = pq_mr_bf16x2(v2, v3);
  __asm__ volatile("s_waitcnt lgkmcnt(0)");           // s_x reuse by the next row / group
}

// R rows' chains through one group, INTERLEAVED: the rows share the group's pair indices, so their chains are
// independent -- each layer issues every row's LDS reads before any fma/write and waits once, hiding the round trip
// R-fold (a per-row sequence is pure LDS latency at the occupancy these registers allow). Within a layer a lane's two
// pairs are disjoint from each other and from every other lane's (the layer is a perfect matching), so reading all
// four values before writing is the stock order's result exactly. s_xw: [R][128] floats, this wave's.
template <int R>
__device__ __forceinline__ void pq_mr_rotate_rows(float (&nv)[R][4], const float (&cs)[4],
                                                  const unsigned long long (&rec)[PQ_KROT_MAX][2], int krot,
                                                  float *s_xw, int c0, int nr, unsigned int (&out)[R][2],
                                                  float (&wamax)[R]) {
#pragma unroll
  for (int r = 0; r < R; ++r) {
    float *s = s_xw + r * PQ_GROUP;
    s[c0 + 0] = nv[r][0] * cs[0]; s[c0 + 1] = nv[r][1] * cs[1];
    s[c0 + 2] = nv[r][2] * cs[2]; s[c0 + 3] = nv[r][3] * cs[3];
  }
  __asm__ volatile("s_waitcnt lgkmcnt(0)");
#pragma unroll
  for (int l = 0; l < PQ_KROT_MAX; ++l) {
    if (l < krot) {
      int ii[2], jj[2];
      float cc[2], ss[2];
#pragma unroll
      for (int t2 = 0; t2 < 2; ++t2) {
        const unsigned long long rv = rec[l][t2];
        const unsigned int ij = (unsigned int)(rv & 0xFFFFu);
        cc[t2] = __half2float(__ushort_as_half((unsigned short)((rv >> 16) & 0xFFFFu)));
        ss[t2] = __half2float(__ushort_as_half((unsigned short)((rv >> 32) & 0xFFFFu)));
        ii[t2] = ij & 0xFF; jj[t2] = ij >> 8;
      }
      float xi[R][2], xj[R][2];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const float *s = s_xw + r * PQ_GROUP;
#pragma unroll
        for (int t2 = 0; t2 < 2; ++t2) { xi[r][t2] = s[ii[t2]]; xj[r][t2] = s[jj[t2]]; }
      }
#pragma unroll
      for (int r = 0; r < R; ++r) {
        float *s = s_xw + r * PQ_GROUP;
#pragma unroll
        for (int t2 = 0; t2 < 2; ++t2) {
          s[ii[t2]] = fmaf(cc[t2], xi[r][t2], ss[t2] * xj[r][t2]);
          s[jj[t2]] = fmaf(cc[t2], xj[r][t2], -ss[t2] * xi[r][t2]);
        }
      }
      __asm__ volatile("s_waitcnt lgkmcnt(0)");
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const float *s = s_xw + r * PQ_GROUP;
    const float v0 = s[c0 + 0], v1 = s[c0 + 1], v2 = s[c0 + 2], v3 = s[c0 + 3];
    if (r < nr) wamax[r] = fmaxf(wamax[r], fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3))));
    out[r][0] = pq_mr_bf16x2(v0, v1);
    out[r][1] = pq_mr_bf16x2(v2, v3);
  }
  __asm__ volatile("s_waitcnt lgkmcnt(0)");           // s_x reuse by the next group
}

// per-row amax (block-wide) -> scale -> encode the wave's own groups from registers
template <int W, int R, int GPW, bool TILED>
__device__ __forceinline__ void pq_mr_encode(float (&wamax)[R], const unsigned int (&vals)[R][GPW][2],
                                             float (*s_amax)[W], unsigned char *__restrict__ Ap,
                                             float *__restrict__ ASp, int m0, int nr, int M, int K, int G,
                                             int wave, int lane, int tid) {
  const int c0 = lane * 4;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    float a = wamax[r];
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) a = fmaxf(a, __shfl_xor(a, off, 32));
    if (lane == 0) s_amax[r][wave] = a;
  }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < R; ++r) {
    if (r >= nr) break;
    float amax = 0.f;
#pragma unroll
    for (int w = 0; w < W; ++w) amax = fmaxf(amax, s_amax[r][w]);
    const float scale = pq_qscale<false>(amax);
    const float inv = 1.f / scale;
    const int m = m0 + r;
    if (tid == 0) ASp[m] = scale;
#pragma unroll
    for (int gi = 0; gi < GPW; ++gi) {
      const int g = wave + gi * W;
      if (g < G) {
        const int c = g * PQ_GROUP + c0;
        float rs = 0.f;
        const unsigned int w0 = vals[r][gi][0], w1 = vals[r][gi][1];
        const float u0 = __uint_as_float(w0 << 16) * inv, u1 = __uint_as_float(w0 & 0xFFFF0000u) * inv;
        const float u2 = __uint_as_float(w1 << 16) * inv, u3 = __uint_as_float(w1 & 0xFFFF0000u) * inv;
        const unsigned char b0 = pq_qenc<false>(u0, rs), b1 = pq_qenc<false>(u1, rs);
        const unsigned char b2 = pq_qenc<false>(u2, rs), b3 = pq_qenc<false>(u3, rs);
        const unsigned int packed = (unsigned int)b0 | ((unsigned int)b1 << 8) |
                                    ((unsigned int)b2 << 16) | ((unsigned int)b3 << 24);
        if constexpr (TILED) *(unsigned int *)(Ap + pq_tiled_off(m, c, K)) = packed;
        else *(unsigned int *)(Ap + (size_t)m * K + c) = packed;
      }
    }
  }
}

// == pq_rotate_tokquant<W', TILED> (any W': the per-element chain does not depend on W) on rows m0..m0+R-1.
// Grid (ceil(M/R), P), block W*32, no dynamic LDS. A: TILED ? [P, Mt*16*K] : [P, M, K]. AS [P, M].
template <int W, int R, int GPW, bool TILED>
__global__ __launch_bounds__(W * 32) void pq_rotate_tokquant_mr(
    const __bf16 *__restrict__ X, const unsigned short *__restrict__ T, const __half *__restrict__ CS,
    unsigned char *__restrict__ A, float *__restrict__ AS, int M, int K, int krot) {
  __shared__ float s_x[W][R * PQ_GROUP];
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
      unsigned long long rec[PQ_KROT_MAX][2];
      float cs[4];
      pq_load_group(T, CS, p, g, K, krot, lane, rec, cs);
      float nv[R][4];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const int m = m0 + (r < nr ? r : 0);           // rows past M reload row m0 (never encoded)
        const uint2_t xv = *(const uint2_t *)(X + (size_t)m * K + (size_t)g * PQ_GROUP + c0);
        nv[r][0] = __uint_as_float(xv[0] << 16); nv[r][1] = __uint_as_float(xv[0] & 0xFFFF0000u);
        nv[r][2] = __uint_as_float(xv[1] << 16); nv[r][3] = __uint_as_float(xv[1] & 0xFFFF0000u);
      }
      unsigned int outg[R][2];
      pq_mr_rotate_rows<R>(nv, cs, rec, krot, s_x[wave], c0, nr, outg, wamax);
#pragma unroll
      for (int r = 0; r < R; ++r) { vals[r][gi][0] = outg[r][0]; vals[r][gi][1] = outg[r][1]; }
    }
  }
  const int Mt = (M + 15) >> 4;
  unsigned char *Ap = TILED ? A + (size_t)p * Mt * 16 * K : A + (size_t)p * M * K;
  pq_mr_encode<W, R, GPW, TILED>(wamax, vals, s_amax, Ap, AS + (size_t)p * M, m0, nr, M, K, G, wave, lane, tid);
}

// == pq_ew_rot_tok<MODE, W', TILED> on rows m0..m0+R-1 (producer math copied verbatim). WHS writes HS as stock.
template <int MODE, int W, int R, int GPW, bool TILED, bool WHS>
__global__ __launch_bounds__(W * 32) void pq_ew_rot_tok_mr(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps,
    const unsigned short *__restrict__ T, const __half *__restrict__ CS,
    __bf16 *__restrict__ HS, unsigned char *__restrict__ A, float *__restrict__ AS,
    int M, int N, int krot) {
  __shared__ float s_x[W][R * PQ_GROUP];
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
      unsigned long long rec[PQ_KROT_MAX][2];
      float cs[4];
      pq_load_group(T, CS, 0, g, N, krot, lane, rec, cs);
      const int kb = g * PQ_GROUP + c0;
      float nvr[R][4];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        {
          const int m = m0 + (r < nr ? r : 0);         // rows past M recompute row m0 (never written)
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
      }
      unsigned int outg[R][2];
      pq_mr_rotate_rows<R>(nvr, cs, rec, krot, s_x[wave], c0, nr, outg, wamax);
#pragma unroll
      for (int r = 0; r < R; ++r) { vals[r][gi][0] = outg[r][0]; vals[r][gi][1] = outg[r][1]; }
    }
  }
  pq_mr_encode<W, R, GPW, TILED>(wamax, vals, s_amax, A, AS, m0, nr, M, N, G, wave, lane, tid);
}
