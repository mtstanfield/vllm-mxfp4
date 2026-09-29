"""rx9y += PVSH: with the fp8 P leg, run P 4 octaves higher (PSHIFT 4) and let the reference max lag by at most 4 octaves
(GROW 4, was 8): the flush threshold drops from 2^-10 to 2^-14 of the reference max while p stays under e4m3's 448.
R4D_ATTN_FP8 14 = PV8|PVDEN|PVSH, 15 = QK8|PV8|PVDEN|PVSH. Other modes unchanged."""
import pathlib
K = pathlib.Path("r4d_attn_prefill_h256_gqa6.hip")
s = K.read_text()
def rep(src, old, new, n=1):
    assert src.count(old) == n, (src.count(old), old[:90]); return src.replace(old, new)
assert "O_PVSH" not in s
s = rep(s, "#define O_PVDEN 128\n", """#define O_PVDEN 128
// PVSH (rx9y): P runs 4 octaves higher and the reference max may lag by 4 octaves instead of 8, so a p is flushed below
// 2^-14 of the reference max instead of 2^-10 (e4m3's 448 = 2^8.8 still bounds shift + lag).
#define O_PVSH  256
""")
s = rep(s, "    constexpr float PSHIFT = D::SHIFT;\n    constexpr float PGROW  = PV8 ? 8.0f : D::GROW;   // e4m3 max is 448 = 2^8.8\n",
        "    constexpr int   PVSH   = (PV8 && (OPT & O_PVSH)) ? 1 : 0;\n"
        "    constexpr float PSHIFT = PVSH ? 4.0f : D::SHIFT;\n"
        "    constexpr float PGROW  = PV8 ? (PVSH ? 4.0f : 8.0f) : D::GROW;   // e4m3 max is 448 = 2^8.8\n")
K.write_text(s)
P = pathlib.Path("r4d_attn_paged_h256_gqa6.hip")
t = P.read_text()
t = rep(t, "        if (fp8mode == 7)      P_LAUNCH(P_OPT | O_QK8 | O_PV8 | O_PVDEN);   // rx9y: 3 + PVDEN\n",
        "        if (fp8mode == 15)     P_LAUNCH(P_OPT | O_QK8 | O_PV8 | O_PVDEN | O_PVSH);   // rx9y: 7 + PVSH\n"
        "        else if (fp8mode == 14) P_LAUNCH(P_OPT | O_PV8 | O_PVDEN | O_PVSH);         // rx9y: 6 + PVSH\n"
        "        else if (fp8mode == 7) P_LAUNCH(P_OPT | O_QK8 | O_PV8 | O_PVDEN);   // rx9y: 3 + PVDEN\n")
P.write_text(t)
print("PVSH patched")
