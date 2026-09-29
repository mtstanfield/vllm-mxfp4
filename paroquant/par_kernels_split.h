// local 2026-09-28: split-token per-token producers for the decode band.
//
// The stock producers (pq_rotate_tokquant / pq_ew_rot_tok) run ONE workgroup per (token, partition). At decode M <= 16
// that is at most 16 x P workgroups on a 64-CU part, and each wave walks G/W groups serially: ew0 (silu-mul -> down,
// N=17408) is 136 groups over 32 waves = 5 rotation chains back to back, 12.3 us per call, 64 calls per step.
// Here S workgroups share a token. Each rotates GPS = W groups (one chain per wave) exactly as the stock kernel does
// (same fetch, same pq_tok_rotate_park chain, parked bf16 in LDS), copies its parked groups to a global scratch row
// and folds its fp32 amax into a per-(p, m) atomicMax; the LAST-arriving workgroup of the token (atomic counter)
// encodes the whole row against the token scale and resets the counter/amax for the next launch. Same arithmetic,
// same amax (a max is order-free, and non-negative floats order as their bit patterns), same encode -> codes, scales
// (and HS) byte-identical to the stock kernels. No inter-workgroup wait, so no co-residency requirement.
#pragma once

#define PQ_SPLIT_W 16          // waves per workgroup = groups per split

// Park -> global: the WG's parked rows (LDS) to the scratch row, 8 B per lane per group.
template <int W>
__device__ __forceinline__ void pq_split_publish(const __bf16 *s_row, __bf16 *__restrict__ grow, int g0, int ng,
                                                 int wave, int lane) {
  for (int gi = wave; gi < ng; gi += W) {
    const int c = gi * PQ_GROUP + lane * 4;
    *(uint2_t *)(grow + (size_t)g0 * PQ_GROUP + c) = *(const uint2_t *)(s_row + c);
  }
}

// Block amax -> global atomicMax, then the arrival count. Returns true in every thread of the last-arriving WG.
template <int W>
__device__ __forceinline__ bool pq_split_arrive(float wamax, float *s_amax, int *s_last, unsigned int *gamax,
                                                int *gcnt, int S, int wave, int lane, int tid) {
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) wamax = fmaxf(wamax, __shfl_xor(wamax, off, 32));
  if (lane == 0) s_amax[wave] = wamax;
  __threadfence();                                   // this thread's scratch stores before the arrival
  __syncthreads();
  if (tid == 0) {
    float a = 0.f;
#pragma unroll
    for (int w = 0; w < W; ++w) a = fmaxf(a, s_amax[w]);
    atomicMax(gamax, __float_as_uint(a));
    __threadfence();
    *s_last = (atomicAdd(gcnt, 1) == S - 1);
  }
  __syncthreads();
  if (!*s_last) return false;
  __threadfence();                                   // acquire: the other WGs' scratch rows and amax
  return true;
}

// Last WG: token scale, encode the whole row from the scratch (== pq_tok_encode_row<.., false> on the same bf16
// values with the same scale), reset the token's slot.
template <int W>
__device__ __forceinline__ void pq_split_encode(const __bf16 *__restrict__ grow, unsigned char *__restrict__ arow,
                                                float *__restrict__ as_out, unsigned int *gamax, int *gcnt, int G,
                                                int wave, int lane, int tid) {
  const float amax = __uint_as_float(atomicOr(gamax, 0u));
  const float scale = pq_qscale<false>(amax);
  const float inv = 1.f / scale;
  if (tid == 0) *as_out = scale;
  const int c0 = lane * 4;
  for (int g = wave; g < G; g += W) {
    const int c = g * PQ_GROUP + c0;
    const uint2_t v = *(const uint2_t *)(grow + c);
    const float u0 = __uint_as_float(v[0] << 16) * inv, u1 = __uint_as_float(v[0] & 0xFFFF0000u) * inv;
    const float u2 = __uint_as_float(v[1] << 16) * inv, u3 = __uint_as_float(v[1] & 0xFFFF0000u) * inv;
    float rs = 0.f;
    const unsigned char b0 = pq_qenc<false>(u0, rs), b1 = pq_qenc<false>(u1, rs);
    const unsigned char b2 = pq_qenc<false>(u2, rs), b3 = pq_qenc<false>(u3, rs);
    *(unsigned int *)(arow + c) = (unsigned int)b0 | ((unsigned int)b1 << 8) | ((unsigned int)b2 << 16) |
                                  ((unsigned int)b3 << 24);
  }
  __syncthreads();
  if (tid == 0) { *gamax = 0u; *gcnt = 0; }
}

// == pq_rotate_tokquant<W', false> (row-major A [P, M, K], AS [P, M]). Grid (S, M, P), S = ceil(G / W).
// SCR bf16 [P, M, K] scratch; GAMAX / GCNT [P * M] zero-initialised, left zeroed.
template <int W>
__global__ __launch_bounds__(W * 32) void pq_rotate_tokquant_split(
    const __bf16 *__restrict__ X, const unsigned short *__restrict__ T, const __half *__restrict__ CS,
    unsigned char *__restrict__ A, float *__restrict__ AS, __bf16 *__restrict__ SCR, unsigned int *__restrict__ GAMAX,
    int *__restrict__ GCNT, int M, int K, int krot) {
  __shared__ __align__(16) __bf16 s_row[W * PQ_GROUP];
  __shared__ float s_x[W][PQ_GROUP];
  __shared__ float s_amax[W];
  __shared__ int s_last;
  const int s = blockIdx.x, m = blockIdx.y, p = blockIdx.z, S = gridDim.x;
  const int G = K / PQ_GROUP;
  const int g0 = s * W, ng = min(W, G - g0);
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int c0 = lane * 4;
  const size_t pm = (size_t)p * M + m;
  const __bf16 *__restrict__ xrow = X + (size_t)m * K;
  float wamax = 0.f;
  if (wave < ng) {
    const int g = g0 + wave;
    unsigned long long rec[PQ_KROT_MAX][2];
    float cs[4], nv[4];
    pq_load_group(T, CS, p, g, K, krot, lane, rec, cs);
    const uint2_t xv = *(const uint2_t *)(xrow + (size_t)g * PQ_GROUP + c0);
    nv[0] = __uint_as_float(xv[0] << 16); nv[1] = __uint_as_float(xv[0] & 0xFFFF0000u);
    nv[2] = __uint_as_float(xv[1] << 16); nv[3] = __uint_as_float(xv[1] & 0xFFFF0000u);
    pq_tok_rotate_park(nv, cs, rec, krot, s_x[wave], c0, s_row + (size_t)wave * PQ_GROUP, wamax);
  }
  __bf16 *grow = SCR + pm * K;
  pq_split_publish<W>(s_row, grow, g0, ng, wave, lane);
  if (!pq_split_arrive<W>(wamax, s_amax, &s_last, GAMAX + pm, GCNT + pm, S, wave, lane, tid)) return;
  pq_split_encode<W>(grow, A + pm * K, AS + pm, GAMAX + pm, GCNT + pm, G, wave, lane, tid);
}

// == pq_ew_rot_tok<MODE, W', false> (HS [M, N] written per group as the stock fetch does, A [M, N], AS [M]).
// Grid (S, M).
template <int MODE, int W>
__global__ __launch_bounds__(W * 32) void pq_ew_rot_tok_split(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys, const __bf16 *__restrict__ Wn, float eps,
    const unsigned short *__restrict__ T, const __half *__restrict__ CS, __bf16 *__restrict__ HS,
    unsigned char *__restrict__ A, float *__restrict__ AS, __bf16 *__restrict__ SCR, unsigned int *__restrict__ GAMAX,
    int *__restrict__ GCNT, int M, int N, int krot) {
  __shared__ __align__(16) __bf16 s_row[W * PQ_GROUP];
  __shared__ float s_x[W][PQ_GROUP];
  __shared__ float s_amax[W];
  __shared__ int s_last;
  const int s = blockIdx.x, m = blockIdx.y, S = gridDim.x;
  const int G = N / PQ_GROUP;
  const int g0 = s * W, ng = min(W, G - g0);
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int c0 = lane * 4;
  float wamax = 0.f;
  if (wave < ng) {
    const int g = g0 + wave;
    unsigned long long rec[PQ_KROT_MAX][2];
    float cs[4], nv[4];
    pq_load_group(T, CS, 0, g, N, krot, lane, rec, cs);
    const int kb = g * PQ_GROUP + c0;
    // ---- the stock fetch body, verbatim (pq_ew_rot_tok) ----
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
    {
      unsigned int h0 = (unsigned int)__bfloat16_as_ushort((__bf16)nv[0]) | ((unsigned int)__bfloat16_as_ushort((__bf16)nv[1]) << 16);
      unsigned int h1 = (unsigned int)__bfloat16_as_ushort((__bf16)nv[2]) | ((unsigned int)__bfloat16_as_ushort((__bf16)nv[3]) << 16);
      *(uint2_t *)(HS + (size_t)m * N + kb) = uint2_t{h0, h1};
    }
    pq_tok_rotate_park(nv, cs, rec, krot, s_x[wave], c0, s_row + (size_t)wave * PQ_GROUP, wamax);
  }
  __bf16 *grow = SCR + (size_t)m * N;
  pq_split_publish<W>(s_row, grow, g0, ng, wave, lane);
  if (!pq_split_arrive<W>(wamax, s_amax, &s_last, GAMAX + m, GCNT + m, S, wave, lane, tid)) return;
  pq_split_encode<W>(grow, A + (size_t)m * N, AS + m, GAMAX + m, GCNT + m, G, wave, lane, tid);
}

// == pq_add_rms_rot_tok<W, false> (stream 1: residual add + Gemma RMSNorm + rotate + token quant; HS/RO [M, K] from
// p == 0, A [P, M, K], AS [P, M]). Grid (S, M, P), S = ceil(G / GPS). W MUST equal the stock launch's W for the same M
// (pq_tok_waves: 32 at M <= 16): every workgroup repeats the stock row pass with the same block shape, so the ssq
// reduction -- and with it inv, HS and every code -- is bit-identical; only split 0 writes RO.
template <int W, int GPS>
__global__ __launch_bounds__(W * 32) void pq_add_rms_rot_tok_split(
    const __bf16 *__restrict__ Y, const __bf16 *__restrict__ RES, const __bf16 *__restrict__ Wn, float eps,
    const unsigned short *__restrict__ T, const __half *__restrict__ CS, __bf16 *__restrict__ HS,
    __bf16 *__restrict__ RO, unsigned char *__restrict__ A, float *__restrict__ AS, __bf16 *__restrict__ SCR,
    unsigned int *__restrict__ GAMAX, int *__restrict__ GCNT, int M, int K, int krot) {
  __shared__ __align__(16) __bf16 s_row[GPS * PQ_GROUP];
  __shared__ float s_red[W];
  __shared__ float s_amax[W];
  __shared__ float s_x[W][PQ_GROUP];
  __shared__ int s_last;
  const int s = blockIdx.x, m = blockIdx.y, p = blockIdx.z, S = gridDim.x;
  const int G = K / PQ_GROUP;
  const int g0 = s * GPS, ng = min(GPS, G - g0);
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int c0 = lane * 4;
  const size_t pm = (size_t)p * M + m;
  // ---- row pass, verbatim from pq_add_rms_rot_tok (RO only from split 0) ----
  const uint4_t *__restrict__ y4 = (const uint4_t *)(Y + (size_t)m * K);
  const uint4_t *__restrict__ r4 = (const uint4_t *)(RES + (size_t)m * K);
  uint4_t *__restrict__ o4 = (uint4_t *)(RO + (size_t)m * K);
  const int KV = K >> 3;
  float ssq = 0.f;
  for (int i = tid; i < KV; i += W * 32) {
    const uint4_t vy = y4[i], vr = r4[i];
    uint4_t vo;
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      const float a0 = __uint_as_float(vy[h] << 16) + __uint_as_float(vr[h] << 16);
      const float a1 = __uint_as_float(vy[h] & 0xFFFF0000u) + __uint_as_float(vr[h] & 0xFFFF0000u);
      ssq = fmaf(a0, a0, ssq);
      ssq = fmaf(a1, a1, ssq);
      const __bf16 b0 = (__bf16)a0, b1 = (__bf16)a1;
      vo[h] = (unsigned int)__bfloat16_as_ushort(b0) | ((unsigned int)__bfloat16_as_ushort(b1) << 16);
    }
    if (p == 0 && s == 0) o4[i] = vo;
  }
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) ssq += __shfl_xor(ssq, off, 32);
  if (lane == 0) s_red[wave] = ssq;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int w = 0; w < W; ++w) tot += s_red[w];
  const float inv = rsqrtf(tot / (float)K + eps);
  float wamax = 0.f;
  if (wave < ng) {
    const int g = g0 + wave;
    unsigned long long rec[PQ_KROT_MAX][2];
    float cs[4], nv[4];
    pq_load_group(T, CS, p, g, K, krot, lane, rec, cs);
    const int kb = g * PQ_GROUP + c0;
    const uint2_t vy = *(const uint2_t *)(Y + (size_t)m * K + kb);
    const uint2_t vr = *(const uint2_t *)(RES + (size_t)m * K + kb);
    const uint2_t vw = *(const uint2_t *)(Wn + kb);
    unsigned int hsw[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float a0 = __uint_as_float(vy[h] << 16) + __uint_as_float(vr[h] << 16);
      const float a1 = __uint_as_float(vy[h] & 0xFFFF0000u) + __uint_as_float(vr[h] & 0xFFFF0000u);
      const float w0 = __uint_as_float(vw[h] << 16) + 1.f, w1 = __uint_as_float(vw[h] & 0xFFFF0000u) + 1.f;
      const __bf16 n0 = (__bf16)((a0 * inv) * w0), n1 = (__bf16)((a1 * inv) * w1);
      nv[2 * h] = (float)n0; nv[2 * h + 1] = (float)n1;
      hsw[h] = (unsigned int)__bfloat16_as_ushort(n0) | ((unsigned int)__bfloat16_as_ushort(n1) << 16);
    }
    if (p == 0) *(uint2_t *)(HS + (size_t)m * K + kb) = uint2_t{hsw[0], hsw[1]};
    pq_tok_rotate_park(nv, cs, rec, krot, s_x[wave], c0, s_row + (size_t)wave * PQ_GROUP, wamax);
  }
  __bf16 *grow = SCR + pm * K;
  pq_split_publish<W>(s_row, grow, g0, ng, wave, lane);
  if (!pq_split_arrive<W>(wamax, s_amax, &s_last, GAMAX + pm, GCNT + pm, S, wave, lane, tid)) return;
  pq_split_encode<W>(grow, A + pm * K, AS + pm, GAMAX + pm, GCNT + pm, G, wave, lane, tid);
}
