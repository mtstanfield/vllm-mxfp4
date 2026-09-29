"""radiance_drafthead.py (vllm-radiance-next tree): RADIANCE_DRAFT_VOCAB=<token-id file> -- the MTP draft head scores
only those rows (the module's own int2 coarse pass + exact rerank, run on the sub-matrix) and the rest of the vocabulary
row is -inf. Unset (production): nothing changes."""
import pathlib, shutil
p = pathlib.Path("/mnt/user/appdata/vllm-radiance-next/radiance_drafthead.py")
s = p.read_text()
if "RADIANCE_DRAFT_VOCAB" in s:
    print("already patched")
    raise SystemExit(0)
shutil.copy(p, str(p) + ".bak-20260927")


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''def install():
    if not FAST:''', '''# ---- local (2026-09-27): pruned DRAFT vocabulary --------------------------------------------------------------------
# RADIANCE_DRAFT_VOCAB=<file, one token id per line>: the MTP drafter scores only those rows -- the int2 coarse pass and
# exact rerank above, run on the sub-matrix (a 48k list reads ~1/5 of the full int2 head) -- and every other entry of the
# vocabulary row is -inf. Output cannot move: the target verifies with its own head; only acceptance can.
VOCAB_FILE = os.environ.get("RADIANCE_DRAFT_VOCAB", "")


class _SubHead:
    """The rows the draft may propose, in the shape _head_matrix reads: [n_sub, K] rows (+ [n_sub, 1] scale)."""

    def __init__(self, w, wsc):
        self.weight = w
        if wsc is not None:
            self.weight_scale = wsc


def _apply_head_vocab(self, lm_head, hidden_states, embedding_bias):
    sub = self._dv_sub
    if sub is None:
        rows, rsc = _head_matrix(lm_head)
        if rows is None or _head_is_empty(rows, rsc):
            return type(self)._apply_head(self, lm_head, hidden_states, embedding_bias)
        ids = self._dv_ids.to(rows.device)
        sub = _SubHead(rows.index_select(0, ids).contiguous(),
                       rsc.index_select(0, ids).reshape(-1, 1).contiguous() if rsc is not None else None)
        status = _quantize_head_now(self, sub)          # rebinds _apply_head to the int2 path; take it back
        self._apply_head = types.MethodType(_apply_head_vocab, self)
        self._dv_sub, self._dv_ids_dev, self._dv_nfull = sub, ids, rows.shape[0]
        sys.stderr.write(f"[radiance] DRAFT_VOCAB: {ids.numel()} of {rows.shape[0]} rows -> {status}\\n")
        sys.stderr.flush()
    y_sub = _apply_head_int2(self, sub, hidden_states, embedding_bias)
    y = torch.full((*y_sub.shape[:-1], self._dv_nfull), float("-inf"), dtype=y_sub.dtype, device=y_sub.device)
    y[..., self._dv_ids_dev] = y_sub
    return y


def _install_vocab():
    ids = sorted({int(t) for t in open(VOCAB_FILE).read().split()})
    ids_t = torch.tensor(ids, dtype=torch.long)
    for mod_name, cls_name in (("vllm.model_executor.models.qwen3_5_mtp", "Qwen3_5MTP"),
                               ("vllm.model_executor.models.qwen3_next_mtp", "Qwen3NextMTP")):
        try:
            cls = getattr(__import__(mod_name, fromlist=[cls_name]), cls_name)
        except Exception:
            continue
        if getattr(cls, "_radiance_vocab_wrapped", False):
            continue
        orig = cls.load_weights

        def wrapped(self, weights, _orig=orig):
            loaded = _orig(self, weights)
            lp = getattr(self, "logits_processor", None)
            if lp is None:
                sys.stderr.write("[radiance] DRAFT_VOCAB: drafter has no logits_processor, full head kept\\n")
            else:
                lp._dv_ids, lp._dv_sub = ids_t, None
                lp._apply_head = types.MethodType(_apply_head_vocab, lp)   # sub-head built on first real call
            return loaded

        cls.load_weights = wrapped
        cls._radiance_vocab_wrapped = True
    sys.stderr.write(f"[radiance] draft vocab armed: {len(ids)} ids from {VOCAB_FILE}\\n")
    sys.stderr.flush()


def install():
    if VOCAB_FILE and triton is not None:
        _install_vocab()
        return
    if not FAST:''')
p.write_text(s)
print("radiance_drafthead.py patched")
