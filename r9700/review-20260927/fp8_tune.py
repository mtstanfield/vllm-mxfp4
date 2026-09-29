"""fp8_tune.py -- build the TunableOp table for the serve's skinny fp8 GEMMs (the MTP drafter's linears + the fp8 lm_head)
at every decode-relevant M the V2 runner uses (1..12, 16: capture sizes, padding, 1-2 requests x 5 verify rows). Run with
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 PYTORCH_TUNABLEOP_FILENAME=<table>; the table is written at exit."""
import torch
SHAPES = ((10240, 5120), (5120, 14336), (6144, 5120), (5120, 34816), (17408, 5120), (5120, 248320))
MS = list(range(1, 13)) + [16]
for K, N in SHAPES:
    w = (torch.randn(N, K, device="cuda") * 0.05).to(torch.float8_e4m3fn)
    sb = torch.rand(1, N, device="cuda") * 0.01 + 0.001
    for M in MS:
        a = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
        sa = torch.rand(M, 1, device="cuda") * 0.01 + 0.001
        torch._scaled_mm(a, w.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    print(f"tuned {K}x{N} M={MS}", flush=True)
print("results:", len(torch.cuda.tunable.get_results()))
