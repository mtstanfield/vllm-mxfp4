// ------------------------------------------------------------------ select-free rotation core ("rot4")
// (local, 2026-09-28 day; needs par_kernels.h for pq_rot3::euler_split and par_kernels_mr3.h for the kernels.)
// The rot3 ownership layout keeps every LDS access conflict-free by making each round's WRITES four perfect matchings
// -- but a lane then has to pick which of its four outputs goes to each write, three v_cndmask per write, per row, per
// round: an ablation that drops those selects (wrong output, same traffic) runs the producers 20-36% faster.
// rot4 routes on the READ side instead. After every round lane l writes its outputs to fixed slots {l, 32+l, 64+l,
// 96+l} (bank l, conflict-free, no selects). A round's 64 pairs are then edges between the banks holding their two
// elements: a 4-regular multigraph on 32 banks, which always splits into two 2-factors (Petersen; here: orient along
// Euler circuits, then split the in/out bipartite graph with the same alternating-circuit helper rot3 uses). In a
// 2-factor every bank is exactly once a tail and once a head, so "read the tails" and "read the heads" are each
// conflict-free; lane l takes the 2-factor-A edge whose tail is bank l as its pair a and the 2-factor-B one as pair b.
// Orientation: an edge read head-first swaps the pair's roles, which the table absorbs by negating the stored sine --
// fmaf(c, xj, (-s) * xi) and fmaf(c, xi, -(-s) * xj) are the stock expressions for (j', i') term by term, so every
// rotated value is bit-identical to the stock chain. Layer 0 is a fixed conflict-free write too (channel 4l+k ->
// slot 32k+l); only the final gather back to channel order reads through a table (once per group, may conflict).
// Tables keep rot3's shapes: R4 [P, krot, K/2, 4] u16 = {cos, +-sin, head-slot | tail-slot << 8... } see below,
// FIN [P, K/128, 32, 4] u16 = the slots of channels 4l..4l+3 after the last round.
namespace pq_rot4 {
// Orient the 64 undirected edges (U[e], V[e]) of a 4-regular multigraph on 32 vertices (loops allowed) so that every
// vertex has in = out = 2: walk closed trails (Hierholzer) and keep the direction each edge was first walked in.
static inline void orient(const std::vector<int> &U, const std::vector<int> &V, std::vector<char> &flip) {
  const int n = 32, m = (int)U.size();
  std::vector<std::vector<int>> adj(n);
  for (int e = 0; e < m; ++e) { adj[U[e]].push_back(e); adj[V[e]].push_back(e); }   // a loop sits twice in adj[u]
  std::vector<char> used(m, 0);
  std::vector<int> pos(n, 0);
  flip.assign(m, 0);
  for (int s = 0; s < n; ++s) {
    std::vector<int> st{s};
    while (!st.empty()) {
      const int v = st.back();
      bool moved = false;
      while (pos[v] < (int)adj[v].size()) {
        const int e = adj[v][pos[v]++];
        if (used[e]) continue;
        used[e] = 1;
        flip[e] = (U[e] != v);                            // walked V -> U
        st.push_back(U[e] == v ? V[e] : U[e]);
        moved = true;
        break;
      }
      if (!moved) st.pop_back();
    }
  }
}
}  // namespace pq_rot4

// T [P, krot, K/2, 4] u16 {ij, cos, sin, -} -> R4 [P, krot, K/2, 4] u16: record t = l (pair a of lane l) and t = l+32
// (pair b) hold {cos, sin (negated if the pair is read j-first), first-slot | second-slot << 8, 0}; FIN [P, K/128, 32,
// 4] u16. Returns the number of (p, g, round) tables that failed the checks (0 on success).
static inline int pq_build_rot4(const unsigned short *T, int P, int krot, int K, unsigned short *R4, unsigned short *FIN) {
  const int G = K / 128, HK = K / 2;
  int failures = 0;
  for (int p = 0; p < P; ++p)
    for (int g = 0; g < G; ++g) {
      int pos[128];
      for (int l = 0; l < 32; ++l)
        for (int k = 0; k < 4; ++k) pos[4 * l + k] = 32 * k + l;
      for (int r = 0; r < krot; ++r) {
        auto rec = [&](int t) { return &T[(((size_t)p * krot + r) * HK + (size_t)g * 64 + t) * 4]; };
        std::vector<int> U(64), V(64), I(64), J(64);
        for (int t = 0; t < 64; ++t) {
          const unsigned short ij = rec(t)[0];
          I[t] = ij & 0xFF; J[t] = ij >> 8;
          U[t] = pos[I[t]] & 31; V[t] = pos[J[t]] & 31;
        }
        std::vector<char> flip;
        pq_rot4::orient(U, V, flip);
        std::vector<pq_rot3::Edge> E, A, B;
        for (int t = 0; t < 64; ++t) {
          const int tail = flip[t] ? V[t] : U[t], head = flip[t] ? U[t] : V[t];
          E.push_back({tail, head, t, flip[t]});
        }
        pq_rot3::euler_split(E, A, B);
        int npos[128];
        for (int c = 0; c < 128; ++c) npos[c] = -1;
        bool ok = A.size() == 32 && B.size() == 32;
        int lane_t[2][32], seen_h[2] = {0, 0}, seen_t[2] = {0, 0};
        for (int h = 0; h < 2 && ok; ++h)
          for (const pq_rot3::Edge &e : (h ? B : A)) {
            seen_t[h] |= 1 << e.src; seen_h[h] |= 1 << e.dst;
            lane_t[h][e.src] = e.k | (e.addr << 8);
          }
        ok = ok && seen_t[0] == -1 && seen_h[0] == -1 && seen_t[1] == -1 && seen_h[1] == -1;
        if (!ok) { ++failures; continue; }
        for (int h = 0; h < 2; ++h)
          for (int l = 0; l < 32; ++l) {
            const int t = lane_t[h][l] & 0xFF, fl = lane_t[h][l] >> 8;
            const int first = fl ? J[t] : I[t], second = fl ? I[t] : J[t];
            if ((pos[first] & 31) != l) ++failures;          // the tail read must be bank l (checked, never expected)
            const unsigned short *src = rec(t);
            unsigned short *dst = &R4[(((size_t)p * krot + r) * HK + (size_t)g * 64 + (h ? l + 32 : l)) * 4];
            dst[0] = src[1];
            dst[1] = fl ? (unsigned short)(src[2] ^ 0x8000u) : src[2];
            dst[2] = (unsigned short)(pos[first] | (pos[second] << 8));
            dst[3] = 0;
            npos[first] = 64 * h + l; npos[second] = 64 * h + 32 + l;
          }
        for (int c = 0; c < 128; ++c) { if (npos[c] < 0) ++failures; pos[c] = npos[c]; }
      }
      for (int l = 0; l < 32; ++l)
        for (int k = 0; k < 4; ++k) FIN[(((size_t)p * G + g) * 32 + l) * 4 + k] = (unsigned short)pos[4 * l + k];
    }
  return failures;
}

// The rotation of R rows of one group with rot4 tables: same inputs/outputs as pq_mr3_rotate_rows (fin = the FIN word
// of this lane, loaded where rot3 loads INIT).
template <int R>
__device__ __forceinline__ void pq_mr4_rotate_rows(const float (&nv)[R][4], float cs0, float cs1, float cs2,
                                                   float cs3, const unsigned long long (&rec)[PQ_KROT_MAX][2],
                                                   unsigned long long fin, int krot, float (*s_xw)[PQ_GROUP],
                                                   int lane, int nr, unsigned int (&out)[R][2], float (&wamax)[R]) {
#pragma unroll
  for (int r = 0; r < R; ++r) {                        // channel 4*lane+k -> slot 32k+lane: bank lane, no selects
    s_xw[r][lane] = nv[r][0] * cs0; s_xw[r][32 + lane] = nv[r][1] * cs1;
    s_xw[r][64 + lane] = nv[r][2] * cs2; s_xw[r][96 + lane] = nv[r][3] * cs3;
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
      const unsigned int i0 = (unsigned int)(r0 >> 32) & 127u, j0 = (unsigned int)(r0 >> 40) & 127u;
      const unsigned int i1 = (unsigned int)(r1 >> 32) & 127u, j1 = (unsigned int)(r1 >> 40) & 127u;
      float a0[R], a1[R], a2[R], a3[R];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        a0[r] = s_xw[r][i0]; a1[r] = s_xw[r][j0];
        a2[r] = s_xw[r][i1]; a3[r] = s_xw[r][j1];
      }
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const float xi = a0[r], xj = a1[r], xk = a2[r], xl = a3[r];
        s_xw[r][lane] = fmaf(ca, xi, sa * xj);
        s_xw[r][32 + lane] = fmaf(ca, xj, -sa * xi);
        s_xw[r][64 + lane] = fmaf(cb, xk, sb * xl);
        s_xw[r][96 + lane] = fmaf(cb, xl, -sb * xk);
      }
      __asm__ volatile("s_waitcnt lgkmcnt(0)");
    }
  }
  const unsigned int f0 = (unsigned int)fin & 127u, f1 = (unsigned int)(fin >> 16) & 127u;
  const unsigned int f2 = (unsigned int)(fin >> 32) & 127u, f3 = (unsigned int)(fin >> 48) & 127u;
#pragma unroll
  for (int r = 0; r < R; ++r) {                        // back to channel order: 4l..4l+3
    const float x0 = s_xw[r][f0], x1 = s_xw[r][f1], x2 = s_xw[r][f2], x3 = s_xw[r][f3];
    if (r < nr) wamax[r] = fmaxf(wamax[r], fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(x2), fabsf(x3))));
    out[r][0] = pq_mr_bf16x2(x0, x1);
    out[r][1] = pq_mr_bf16x2(x2, x3);
  }
  __asm__ volatile("s_waitcnt lgkmcnt(0)");           // s_x reuse by the next group
}
