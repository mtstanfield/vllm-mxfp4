# Spec-decode acceptance per bench from /metrics deltas: `snap` before, `delta` after (2026-09-15).
import urllib.request, re, json, os, sys
S=os.path.dirname(os.path.abspath(__file__)); F=os.path.join(S,"accept-prev.json")
m=urllib.request.urlopen(os.environ.get("VLLM_METRICS","http://192.168.88.89:1246/metrics"),timeout=10).read().decode()
cur={k:float(v) for k,v in re.findall(r"^vllm:(spec_decode_num_drafts_total|spec_decode_num_draft_tokens_total|spec_decode_num_accepted_tokens_total|generation_tokens_total)\{[^}]*\} (\S+)$",m,re.M)}
prev=json.load(open(F)) if os.path.exists(F) and sys.argv[1:]==["delta"] else None
json.dump(cur,open(F,"w"))
if prev:
    d={k:cur[k]-prev[k] for k in cur}
    print(f"DELTA gen={d['generation_tokens_total']:.0f} drafts={d['spec_decode_num_drafts_total']:.0f} tok/step={d['generation_tokens_total']/max(1,d['spec_decode_num_drafts_total']):.2f} accept={d['spec_decode_num_accepted_tokens_total']/max(1,d['spec_decode_num_draft_tokens_total']):.3f}")
else: print("snapshot", cur)
