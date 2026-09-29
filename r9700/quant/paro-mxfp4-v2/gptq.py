"""gptq.py - GPTQ/LDLQ error-compensating rounding onto the MXFP4 grid, in the rotated basis the server uses.

Standard GPTQ (lazy 128-column batches, damped Cholesky of H^-1, no act-order: the per-32 scale groups are contiguous
columns). A group's exponent is chosen when its first column is reached, from the error-updated weights, weighting each
column's error by diag(H) in "search" mode.
"""
import torch
import mx


@torch.no_grad()
def gptq_mxfp4(w: torch.Tensor, h: torch.Tensor, mode: str = "search", blocksize: int = 128, percdamp: float = 0.01,
               actorder: bool = False):
    """w [N, K] rotated weight, h [K, K] rotated input Hessian -> (code, sign, e) like mx.quant_matrix.
    actorder: GPTQ "static" act-order - every block's exponent is fixed up front from the unmodified weights, then the
    columns are processed in descending diag(H) (most-used input channels first, so the rounding error lands on the
    least-used ones). The output format is unchanged."""
    w = w.float().clone()
    h = h.float().clone()
    n, k = w.shape
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    diag = torch.diag(h).clone()
    e_static = perm = None
    if actorder:
        cw = diag.view(1, k // mx.BLOCK, mx.BLOCK) if mode == "search" else None
        _, _, e_static = mx.quant_blocks(w.view(n, k // mx.BLOCK, mx.BLOCK), mode, cw)       # [N, K/32]
        perm = torch.argsort(diag, descending=True)
        w = w[:, perm]
        h = h[perm][:, perm]
    h += percdamp * torch.mean(diag) * torch.eye(k, device=h.device)
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    del h

    code = torch.zeros(n, k, dtype=torch.uint8, device=w.device)
    sign = torch.zeros(n, k, dtype=torch.bool, device=w.device)
    exps = e_static.clone() if actorder else torch.zeros(n, k // mx.BLOCK, device=w.device)
    group_of_col = (perm // mx.BLOCK).tolist() if actorder else None
    for i1 in range(0, k, blocksize):
        i2 = min(i1 + blocksize, k)
        w1 = w[:, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]
        e_cur = None
        for i in range(i2 - i1):
            col = i1 + i
            if actorder:
                e_cur = e_static[:, group_of_col[col]]
            elif col % mx.BLOCK == 0:
                g = col // mx.BLOCK
                wb = w1[:, i:i + mx.BLOCK].unsqueeze(1)                      # [N, 1, 32]
                cw = diag[col:col + mx.BLOCK].view(1, 1, -1) if mode == "search" else None
                _, _, e = mx.quant_blocks(wb, mode, cw)
                e_cur = e.view(-1)
                exps[:, g] = e_cur
            wc = w1[:, i]
            c, s = mx.round_e2m1(wc / torch.exp2(e_cur))
            q = mx.dequant(c.unsqueeze(-1), s.unsqueeze(-1), e_cur).view(-1)
            code[:, col] = c; sign[:, col] = s
            err = (wc - q) / hinv1[i, i]
            w1[:, i:] -= err.unsqueeze(1) * hinv1[i, i:].unsqueeze(0)
            err1[:, i] = err
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]
    if actorder:                                   # back to the original column order
        code_o = torch.empty_like(code); sign_o = torch.empty_like(sign)
        code_o[:, perm] = code; sign_o[:, perm] = sign
        code, sign = code_o, sign_o
    return code, sign, exps


def proxy_loss(w, q, h):
    """tr((W-Q) H (W-Q)^T) / N - the layer-output error GPTQ minimizes (for logging)."""
    d = (w.float() - q.float())
    return (torch.einsum("nk,kl,nl->", d, h.float(), d) / w.shape[0]).item()
