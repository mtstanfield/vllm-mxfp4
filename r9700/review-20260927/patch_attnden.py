"""libr4d rx9x -> rx9y: PVDEN -- with the fp8 P leg (PV8), sum the denominator over the P values the PV WMMA actually
uses (the e4m3-rounded ones) instead of the pre-quantization p. PV8 flushes p < 2^-10 (relative to the reference max) to
zero in the numerator while the old denominator still counted it: with value vectors that share a component the attention
output shrinks (harness, 128k keys, logit spread 2 / 3 nats: norm ratio 0.980 / 0.946). New R4D_ATTN_FP8 values:
6 = PV8 + PVDEN, 7 = QK8 + PV8 + PVDEN (production's 3 plus the fix). Modes 0-3 compile to the same kernels as before."""
import pathlib

K = pathlib.Path("r4d_attn_prefill_h256_gqa6.hip")
s = K.read_text()


def rep(src, old, new, n=1):
    assert src.count(old) == n, (src.count(old), old[:90])
    return src.replace(old, new)


assert "#define O_PVDEN" not in s
s = rep(s, "#define O_PV8    64\n", """#define O_PV8    64
// PVDEN (rx9y): with PV8, the denominator sums the e4m3-ROUNDED p the PV WMMA uses. Without it a p that rounds to 0 in
// the numerator still counts in the denominator, so the output shrinks by the flushed tail mass (2-5% at 128k keys with a
// shared value component).
#define O_PVDEN 128
""")
s = rep(s, "    constexpr int PV8   = ((OPT & O_PV8) && !KVP) ? 1 : 0;\n",
        "    constexpr int PV8   = ((OPT & O_PV8) && !KVP) ? 1 : 0;\n"
        "    constexpr int PVDEN = (PV8 && (OPT & O_PVDEN)) ? 1 : 0;\n")
s = rep(s, """                pf[e] = p;
                if (!DOT2) lsum += p;
            }
            if (!DOT2) l_i += lsum;

            if constexpr (PV8) {
                const v2i32_r4d p8 = (v2i32_r4d){(int)pk_fp8x4(pf[0], pf[1], pf[2], pf[3]),
                                                 (int)pk_fp8x4(pf[4], pf[5], pf[6], pf[7])};
""", """                pf[e] = p;
                if (!DOT2 && !PVDEN) lsum += p;
            }
            if (!DOT2 && !PVDEN) l_i += lsum;

            if constexpr (PV8) {
                const v2i32_r4d p8 = (v2i32_r4d){(int)pk_fp8x4(pf[0], pf[1], pf[2], pf[3]),
                                                 (int)pk_fp8x4(pf[4], pf[5], pf[6], pf[7])};
                if constexpr (PVDEN) {       // rx9y: the denominator of exactly the P the PV WMMA uses
                    const v2f a0 = __builtin_amdgcn_cvt_pk_f32_fp8(p8.x, false);
                    const v2f a1 = __builtin_amdgcn_cvt_pk_f32_fp8(p8.x, true);
                    const v2f a2 = __builtin_amdgcn_cvt_pk_f32_fp8(p8.y, false);
                    const v2f a3 = __builtin_amdgcn_cvt_pk_f32_fp8(p8.y, true);
                    l_i += ((a0.x + a0.y) + (a1.x + a1.y)) + ((a2.x + a2.y) + (a3.x + a3.y));
                }
""")
K.write_text(s)

P = pathlib.Path("r4d_attn_paged_h256_gqa6.hip")
t = P.read_text()
t = rep(t, """        if (fp8mode == 3)      P_LAUNCH(P_OPT | O_QK8 | O_PV8);""",
        """        if (fp8mode == 7)      P_LAUNCH(P_OPT | O_QK8 | O_PV8 | O_PVDEN);   // rx9y: 3 + PVDEN
        else if (fp8mode == 6) P_LAUNCH(P_OPT | O_PV8 | O_PVDEN);           // rx9y: 2 + PVDEN
        else if (fp8mode == 3) P_LAUNCH(P_OPT | O_QK8 | O_PV8);""")
t = rep(t, "    // R4D_ATTN_FP8: 0 = the shipped f16 legs; 1 = O_QK8, 2 = O_PV8, 3 = both. fp8 KV only.\n",
        "    // R4D_ATTN_FP8: 0 = the shipped f16 legs; 1 = O_QK8, 2 = O_PV8, 3 = both. fp8 KV only.\n"
        "    // rx9y: 6 = O_PV8 | O_PVDEN, 7 = O_QK8 | O_PV8 | O_PVDEN (consistent denominator for the fp8 P leg).\n")
P.write_text(t)
print("rx9y patched")
