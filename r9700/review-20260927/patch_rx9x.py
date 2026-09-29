"""libr4d b9e42ab-rx9 -> rx9x (run inside a copy of the rx9 tree):
  1. gdn_chunk_scan: EXACT decay -- every decay weight is a factor <= 1 (gv = e^{g_last - g_t} on the state path,
     sA = scale * e^{g_i - g_j} under the causal mask, S0 scaled by e^{g_last}); replaces the midpoint split whose
     e^80 clamp attenuates chunks with a gate span past ~176 (PERFORMANCE.md: wikitext 8.3706 vs reference 8.3335).
  2. gdn_chunk_scan: fp16 / bf16 initial and final state (the single-GPU profile's narrow SSM cache), new entry points
     ..._f16state / ..._bf16state, registered AHEAD of the fp32 row so select(state_dtype=...) finds them while a
     caller that passes no state_dtype still gets the fp32 kernel.
"""
import pathlib

K = pathlib.Path("r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip")
s = K.read_text()


def rep(src, old, new, n=1):
    assert src.count(old) == n, (src.count(old), old[:90])
    return src.replace(old, new)


s = rep(s, '#include "r4d_gdn_wmma.h"\n',
        '#include "r4d_gdn_wmma.h"\n#include "r4d_common.h"   // rx9x: f32_to_bf16 for the bf16 state\n')

# ---- 1. exact decay ----------------------------------------------------------------------------------------------
s = rep(s, "    const float cref = 0.5f * (gcs[(size_t)c0 * H + hv] + gl);\n", "")
s = rep(s, """      s.gv[tid] = ok ? __expf(fminf(cref - ggr, 80.0f)) : 0.f;
      s.rs[tid] = ok ? scale * __expf(fminf(ggr - cref, 80.0f)) : 0.f;""",
"""      // rx9x EXACT DECAY: the split above is replaced -- every weight is a factor <= 1 instead:
      //   gv[t] = e^{g_last - g_t} (state path)    rs[t] = g_t (raw; sA takes e^{g_i - g_j} per element)
      s.gv[tid] = ok ? __expf(gl - ggr) : 0.f;
      s.rs[tid] = ok ? ggr : 0.f;""")
s = rep(s, """    // sA = rs * tril(P): one multiply per element, no exp
    {
      const int ii = mt * 16 + lo;            // now constant per lane: one rs load""",
"""    // sA = scale * e^{g_i - g_j} * tril(P) (rx9x EXACT DECAY: the exponent is <= 0 under the mask)
    {
      const int ii = mt * 16 + lo;            // constant per lane""")
s = rep(s, "        for (int e = 0; e < 8; ++e) a[e] = (j0 + e <= ii) ? p[i][e] * r : 0.f;\n",
"""        for (int e = 0; e < 8; ++e)
          a[e] = (ii < nval && j0 + e <= ii) ? p[i][e] * (scale * __expf(r - s.rs[j0 + e])) : 0.f;
""")
s = rep(s, """      float gvv[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) gvv[e] = s.gv[mt * 16 + 8 * hi + e];
#pragma unroll
      for (int i = 0; i < NTV; ++i) {
        unsigned short* dp = &s.D[(ntb + i) * 16 + lo][mt * 16 + 8 * hi];
        *(uint2*)dp       = F2BF4I(d[i][0]*gvv[0], d[i][1]*gvv[1],
                                   d[i][2]*gvv[2], d[i][3]*gvv[3]);
        *(uint2*)(dp + 4) = F2BF4I(d[i][4]*gvv[4], d[i][5]*gvv[5],
                                   d[i][6]*gvv[6], d[i][7]*gvv[7]);
      }""",
"""#pragma unroll
      for (int i = 0; i < NTV; ++i) {          // rx9x EXACT DECAY: V'' = A@W unscaled (the O path)
        unsigned short* dp = &s.D[(ntb + i) * 16 + lo][mt * 16 + 8 * hi];
        *(uint2*)dp       = F2BF4I(d[i][0], d[i][1], d[i][2], d[i][3]);
        *(uint2*)(dp + 4) = F2BF4I(d[i][4], d[i][5], d[i][6], d[i][7]);
      }""")
s = rep(s, """    // ---- S = lam * (e^c * S + V'^T @ K)   (B needs K as [dim][token] -> strided fragT)
    {
      const float ec = __expf(cref), lam = __expf(gl - cref);""",
"""    // rx9x EXACT DECAY: every wave has finished reading V'' for O; scale its token columns by e^{g_last - g_t}
    // (<= 1, 0 past the chunk end) for the state path
    BAR();
    for (int idx = tid; idx < BV * (BT / 4); idx += NTHR) {
      const int vv = idx / (BT / 4), t4 = (idx % (BT / 4)) * 4;
      uint2* p4 = (uint2*)&s.D[vv][t4];
      const uint2 w = *p4;
      *p4 = f2bf4(bf2f((unsigned short)(w.x & 0xffffu)) * s.gv[t4],     bf2f((unsigned short)(w.x >> 16)) * s.gv[t4 + 1],
                  bf2f((unsigned short)(w.y & 0xffffu)) * s.gv[t4 + 2], bf2f((unsigned short)(w.y >> 16)) * s.gv[t4 + 3]);
    }
    BAR();

    // ---- S = e^{g_last} * S + V'^T @ K   (B needs K as [dim][token] -> strided fragT)
    {
      const float ec = __expf(gl);""")
s = rep(s, """#pragma unroll
      for (int t = 0; t < NST; ++t)
#pragma unroll
        for (int e = 0; e < 8; ++e) St[t][e] *= lam;
""", "")

# ---- 2. state dtype template ----------------------------------------------------------------------------------------
s = rep(s, """__global__ __launch_bounds__(NTHR) void r4d_gdn_chunk_scan_kernel(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v, const unsigned short* __restrict__ Amat,
    const float* __restrict__ gcs, const float* __restrict__ beta,
    const float* __restrict__ h0, unsigned short* __restrict__ o,
    float* __restrict__ ht, const int* __restrict__ cu, int H, int Hg, float scale)
{""",
"""// rx9x: initial / final state in fp32 (SD 0), fp16 (1, round to nearest even) or bf16 (2, RNE); 8 values per call,
// offsets are multiples of 8 elements (16 B aligned for the 16-bit states)
template <int SD>
__device__ __forceinline__ void rx9x_st_load8(const void* base, size_t off, float* x) {
  if constexpr (SD == 0) {
    const float* p = (const float*)base + off;
    const float4 a = *(const float4*)p, b = *(const float4*)(p + 4);
    x[0] = a.x; x[1] = a.y; x[2] = a.z; x[3] = a.w; x[4] = b.x; x[5] = b.y; x[6] = b.z; x[7] = b.w;
  } else {
    const uint4 w = *(const uint4*)((const unsigned short*)base + off);
    const uint32_t u[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if constexpr (SD == 1) {
        x[2 * i]     = (float)__builtin_bit_cast(_Float16, (unsigned short)(u[i] & 0xffffu));
        x[2 * i + 1] = (float)__builtin_bit_cast(_Float16, (unsigned short)(u[i] >> 16));
      } else {
        x[2 * i]     = __builtin_bit_cast(float, u[i] << 16);
        x[2 * i + 1] = __builtin_bit_cast(float, u[i] & 0xffff0000u);
      }
    }
  }
}
template <int SD>
__device__ __forceinline__ void rx9x_st_store8(void* base, size_t off, const float* x) {
  if constexpr (SD == 0) {
    float* p = (float*)base + off;
    *(float4*)p       = make_float4(x[0], x[1], x[2], x[3]);
    *(float4*)(p + 4) = make_float4(x[4], x[5], x[6], x[7]);
  } else {
    uint32_t u[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if constexpr (SD == 1) {
        u[i] = (uint32_t)__builtin_bit_cast(unsigned short, (_Float16)x[2 * i]) |
               ((uint32_t)__builtin_bit_cast(unsigned short, (_Float16)x[2 * i + 1]) << 16);
      } else {
        u[i] = (uint32_t)f32_to_bf16(x[2 * i]) | ((uint32_t)f32_to_bf16(x[2 * i + 1]) << 16);
      }
    }
    *(uint4*)((unsigned short*)base + off) = make_uint4(u[0], u[1], u[2], u[3]);
  }
}

template <int SD>
__global__ __launch_bounds__(NTHR) void r4d_gdn_chunk_scan_kernel(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v, const unsigned short* __restrict__ Amat,
    const float* __restrict__ gcs, const float* __restrict__ beta,
    const void* __restrict__ h0, unsigned short* __restrict__ o,
    void* __restrict__ ht, const int* __restrict__ cu, int H, int Hg, float scale)
{""")
s = rep(s, """  {
    const float* p = h0 + ((size_t)(nq * H + hv) * VD) * KD;
#pragma unroll
    for (int t = 0; t < NST; ++t) {
      const float4 a = *(const float4*)(p + SROW(t));
      const float4 b = *(const float4*)(p + SROW(t) + 4);
      St[t][0]=a.x; St[t][1]=a.y; St[t][2]=a.z; St[t][3]=a.w;
      St[t][4]=b.x; St[t][5]=b.y; St[t][6]=b.z; St[t][7]=b.w;
    }
  }""",
"""  {
    const size_t sb = ((size_t)(nq * H + hv) * VD) * KD;
#pragma unroll
    for (int t = 0; t < NST; ++t) {
      float x[8];
      rx9x_st_load8<SD>(h0, sb + SROW(t), x);
#pragma unroll
      for (int e = 0; e < 8; ++e) St[t][e] = x[e];
    }
  }""")
s = rep(s, """  float* p = ht + ((size_t)(nq * H + hv) * VD) * KD;
#pragma unroll
  for (int t = 0; t < NST; ++t) {
    *(float4*)(p + SROW(t))     = make_float4(St[t][0], St[t][1], St[t][2], St[t][3]);
    *(float4*)(p + SROW(t) + 4) = make_float4(St[t][4], St[t][5], St[t][6], St[t][7]);
  }
}""",
"""  const size_t sbo = ((size_t)(nq * H + hv) * VD) * KD;
#pragma unroll
  for (int t = 0; t < NST; ++t) {
    float x[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) x[e] = St[t][e];
    rx9x_st_store8<SD>(ht, sbo + SROW(t), x);
  }
}""")
s = rep(s, """extern "C" int r4d_gdn_chunk_scan_k128_v128_c64_bf16(
    const void* q, const void* k, const void* v, const void* A, const void* g,
    const void* beta, const void* h0, void* o, void* ht, const void* cu,
    int N, int H, int Hg, int K, int V, int bt, float scale, void* stream)
{
  if (K != KD || V != VD || bt != BT) return -1;
  dim3 grid(V / BV, H, N), blk(NTHR);
  r4d_gdn_chunk_scan_kernel<<<grid, blk, 0, (hipStream_t)stream>>>(
      (const unsigned short*)q, (const unsigned short*)k, (const unsigned short*)v,
      (const unsigned short*)A, (const float*)g, (const float*)beta, (const float*)h0,
      (unsigned short*)o, (float*)ht, (const int*)cu, H, Hg, scale);
  return (int)hipGetLastError();
}""",
"""template <int SD>
static int rx9x_chunk_scan(const void* q, const void* k, const void* v, const void* A, const void* g,
                           const void* beta, const void* h0, void* o, void* ht, const void* cu,
                           int N, int H, int Hg, int K, int V, int bt, float scale, void* stream) {
  if (K != KD || V != VD || bt != BT) return -1;
  dim3 grid(V / BV, H, N), blk(NTHR);
  r4d_gdn_chunk_scan_kernel<SD><<<grid, blk, 0, (hipStream_t)stream>>>(
      (const unsigned short*)q, (const unsigned short*)k, (const unsigned short*)v,
      (const unsigned short*)A, (const float*)g, (const float*)beta, h0,
      (unsigned short*)o, ht, (const int*)cu, H, Hg, scale);
  return (int)hipGetLastError();
}

extern "C" int r4d_gdn_chunk_scan_k128_v128_c64_bf16(
    const void* q, const void* k, const void* v, const void* A, const void* g,
    const void* beta, const void* h0, void* o, void* ht, const void* cu,
    int N, int H, int Hg, int K, int V, int bt, float scale, void* stream)
{
  return rx9x_chunk_scan<0>(q, k, v, A, g, beta, h0, o, ht, cu, N, H, Hg, K, V, bt, scale, stream);
}
extern "C" int r4d_gdn_chunk_scan_k128_v128_c64_bf16_f16state(
    const void* q, const void* k, const void* v, const void* A, const void* g,
    const void* beta, const void* h0, void* o, void* ht, const void* cu,
    int N, int H, int Hg, int K, int V, int bt, float scale, void* stream)
{
  return rx9x_chunk_scan<1>(q, k, v, A, g, beta, h0, o, ht, cu, N, H, Hg, K, V, bt, scale, stream);
}
extern "C" int r4d_gdn_chunk_scan_k128_v128_c64_bf16_bf16state(
    const void* q, const void* k, const void* v, const void* A, const void* g,
    const void* beta, const void* h0, void* o, void* ht, const void* cu,
    int N, int H, int Hg, int K, int V, int bt, float scale, void* stream)
{
  return rx9x_chunk_scan<2>(q, k, v, A, g, beta, h0, o, ht, cu, N, H, Hg, K, V, bt, scale, stream);
}""")
K.write_text(s)

# ---- header -----------------------------------------------------------------------------------------------------------
Hh = pathlib.Path("r4d.h")
h = Hh.read_text()
h = rep(h, """        float scale, void* stream);
void r4d_gdn_dims(int* head_k, int* head_v, int* chunk);""",
"""        float scale, void* stream);
// rx9x: the same scan on a 16-bit initial / final state (h0, ht [N, H, V, K] fp16 or bf16); fp32 internally
int r4d_gdn_chunk_scan_k128_v128_c64_bf16_f16state(
        const void* q, const void* k, const void* v, const void* A,
        const void* g, const void* beta, const void* h0, void* o, void* ht,
        const void* cu, int N, int H, int Hg, int K, int V, int bt,
        float scale, void* stream);
int r4d_gdn_chunk_scan_k128_v128_c64_bf16_bf16state(
        const void* q, const void* k, const void* v, const void* A,
        const void* g, const void* beta, const void* h0, void* o, void* ht,
        const void* cu, int N, int H, int Hg, int K, int V, int bt,
        float scale, void* stream);
void r4d_gdn_dims(int* head_k, int* head_v, int* chunk);""")
Hh.write_text(h)

# ---- registry: narrow rows first ------------------------------------------------------------------------------------
R = pathlib.Path("r4d_registry.hip")
r = R.read_text()
r = rep(r, """static const R4DConstraint cGdnConvPrep[] = {""",
"""static const R4DConstraint cGdnChunkScanF16St[] = {
    C_EQ("head_k", 128), C_EQ("head_v", 128), C_EQ("chunk", 64), C_IN("state_dtype", "fp16"),
};
static const R4DConstraint cGdnChunkScanBf16St[] = {
    C_EQ("head_k", 128), C_EQ("head_v", 128), C_EQ("chunk", 64), C_IN("state_dtype", "bf16"),
};
static const R4DConstraint cGdnConvPrep[] = {""")
r = rep(r, """    {"gdn_chunk_scan_k128_v128_c64_bf16", "gdn", "gdn_chunk_scan",""",
"""    // rx9x: 16-bit state rows AHEAD of the fp32 one: select() returns the first match, and these need
    // state_dtype given, so a caller that does not pass it still resolves to the fp32 kernel below
    {"gdn_chunk_scan_k128_v128_c64_bf16_f16state", "gdn", "gdn_chunk_scan",
     "gated delta net chunked scan (exact decay), F16 initial/final state",
     "head_k 128, head_v 128, chunk 64, varlen, state resident in WMMA accumulators",
     "bf16 q/k/v/A, fp32 gate/beta, bf16 out, fp16 initial and final state (fp32 inside)",
     ROW(cGdnChunkScanF16St)},
    {"gdn_chunk_scan_k128_v128_c64_bf16_bf16state", "gdn", "gdn_chunk_scan",
     "gated delta net chunked scan (exact decay), BF16 initial/final state",
     "head_k 128, head_v 128, chunk 64, varlen, state resident in WMMA accumulators",
     "bf16 q/k/v/A, fp32 gate/beta, bf16 out, bf16 initial and final state (fp32 inside)",
     ROW(cGdnChunkScanBf16St)},
    {"gdn_chunk_scan_k128_v128_c64_bf16", "gdn", "gdn_chunk_scan",""")
R.write_text(r)

# ---- module: one wrapper per state dtype ----------------------------------------------------------------------------
M = pathlib.Path("r4d_module.hip")
m = M.read_text()
m = rep(m, """static void gdn_chunk_scan(int64_t q, int64_t k, int64_t v, int64_t A, int64_t g, int64_t beta,
                           int64_t h0, int64_t o, int64_t ht, int64_t cu, int num_seqs,
                           int num_v_heads, int num_k_heads, int head_k, int head_v, int chunk,
                           double scale, int64_t stream) {
  int rc = r4d_gdn_chunk_scan_k128_v128_c64_bf16(""",
"""typedef int (*r4d_chunk_scan_fn)(const void*, const void*, const void*, const void*, const void*, const void*,
                                 const void*, void*, void*, const void*, int, int, int, int, int, int, float, void*);
template <r4d_chunk_scan_fn FN>
static void gdn_chunk_scan_impl(int64_t q, int64_t k, int64_t v, int64_t A, int64_t g, int64_t beta,
                                int64_t h0, int64_t o, int64_t ht, int64_t cu, int num_seqs,
                                int num_v_heads, int num_k_heads, int head_k, int head_v, int chunk,
                                double scale, int64_t stream) {
  int rc = FN(""")
m = rep(m, """  m.def("gdn_chunk_scan_k128_v128_c64_bf16", &gdn_chunk_scan);""",
"""  m.def("gdn_chunk_scan_k128_v128_c64_bf16", &gdn_chunk_scan_impl<r4d_gdn_chunk_scan_k128_v128_c64_bf16>);
  m.def("gdn_chunk_scan_k128_v128_c64_bf16_f16state",
        &gdn_chunk_scan_impl<r4d_gdn_chunk_scan_k128_v128_c64_bf16_f16state>);
  m.def("gdn_chunk_scan_k128_v128_c64_bf16_bf16state",
        &gdn_chunk_scan_impl<r4d_gdn_chunk_scan_k128_v128_c64_bf16_bf16state>);""")
M.write_text(m)
print("rx9x patched")
