import sys, torch
a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
bad = 0
for key in sorted(a):
    ta, tb = a[key], b[key]
    for i, (x, y) in enumerate(zip(ta, tb)):
        if not torch.equal(x, y):
            bad += 1
            print(f"DIFF {key}[{i}]: {(x != y).sum().item()} of {x.numel()} elements", flush=True)
print(f"{len(a)} entries compared, {bad} differing tensors")
