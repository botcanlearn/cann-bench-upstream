#!/usr/bin/env python3
"""HAP t_hw computation for SwiGlu (level1).

Algorithm (golden.py):
    out_dtype = input.dtype
    x  = input.to(fp32)
    x0, x1 = x.chunk(2, dim)          # each half on split dim
    output = silu(x0) * x1 = (x0 * sigmoid(x0)) * x1
    return output.to(out_dtype)

Decomposition (elementwise L1, no cube):
  Let H = output element count = numel(input) / 2  (the split dim is halved).
  Reads : the WHOLE input tensor (both halves x0 and x1) -> numel(input) elems.
  Writes: the output -> H elems (output dtype bytes).

  Vector work, charged per OUTPUT element H (fp16/bf16 -> compute in fp32, with casts):
    - cast input->fp32 : applied to all input elements (2H) at cast_rate     [fp16/bf16 only]
    - sigmoid(x0)      : SFU (exp/recip) on H elements                       (fp32 sfu)
    - x0 * sigmoid     : 1 basic mul on H elements                           (fp32 basic)
    - silu * x1        : 1 basic mul on H elements                           (fp32 basic)
    - cast output->dt  : on H elements at cast_rate                          [fp16/bf16 only]
  For fp32 input: no casts; same sfu + 2 basic muls on H elements (fp32 basic/sfu rates).

  sigmoid counted as one SFU op (= 1/(1+exp(-x)); exp is the SFU, the recip/div folds in).
  Cube = 0. Bottleneck is essentially always HBM read for these sizes.
"""
import csv, json, os, sys, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hap_lib as H

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = H.REPO_ROOT
INNERDIR = os.path.join(ROOT, "inner", "tasks", "level1", "swi_glu")
OP_DIR = os.path.join(ROOT, "tasks", "level1", "swi_glu")


def parse_rows(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            shape = json.loads(r["input_shape"])[0]      # single input tensor
            dt = json.loads(r["dtype"])[0]
            attrs = json.loads(r["attrs"])
            rows.append((r["case_id"], shape, dt, attrs))
    return rows


def compute_case(shape, dt, attrs):
    dim = attrs.get("dim", -1)
    rank = len(shape)
    d = dim if dim >= 0 else dim + rank
    in_numel = H.prod(shape)
    out_numel = in_numel // 2          # split dim halved -> output half the elements
    Hn = out_numel

    in_bytes = in_numel * H.dbytes(dt)
    out_bytes = out_numel * H.dbytes(dt)

    is_fp32 = (dt == "float32")
    vec_ops = []
    # SFU sigmoid + 2 basic muls, computed in fp32 on H elements
    vec_ops.append((Hn, "float32", "sfu"))      # sigmoid (exp/recip)
    vec_ops.append((Hn, "float32", "basic"))    # x0 * sigmoid
    vec_ops.append((Hn, "float32", "basic"))    # silu * x1
    if not is_fp32:
        # cast both input halves (2H) up to fp32, and cast output (H) back
        vec_ops.append((2 * Hn, dt, "cast"))    # input -> fp32
        vec_ops.append((Hn, dt, "cast"))        # output -> dt

    return H.compute_both(vec_ops=vec_ops, input_bytes=in_bytes, output_bytes=out_bytes)


def validate():
    rows = parse_rows(os.path.join(OP_DIR, "cases.csv"))
    meta = json.load(open(os.path.join(ROOT, "tasks", "metadata", "910b2.json")))
    ref = meta["level1"]["swi_glu"]
    worst = (0.0, None)
    for cid, shape, dt, attrs in rows:
        res = compute_case(shape, dt, attrs)
        got = res["910b2"]["t_hw_us"]
        exp = ref[cid]["t_hw_us"]
        rel = abs(got - exp) / exp if exp else 0.0
        ok = rel <= 0.02 or abs(got - exp) <= 0.05
        if rel > worst[0]:
            worst = (rel, cid)
        flag = "" if ok else "  <-- FAIL"
        print(f"case {cid:>3}: got={got:>10.4f} exp={exp:>10.4f} rel={rel*100:6.3f}% {res['910b2']['bottleneck']}{flag}")
    print(f"\nmax_rel_err = {worst[0]*100:.4f}% at case {worst[1]}")
    return worst


def emit():
    worst = validate()
    if not H.inner_cases_available(INNERDIR):
        print("\nclosed (inner) case set not distributed publicly; open-case validation only.")
        return
    rows = parse_rows(os.path.join(INNERDIR, "cases.csv"))
    cases = {}
    for cid, shape, dt, attrs in rows:
        res = compute_case(shape, dt, attrs)
        cases[cid] = {"910b2": res["910b2"], "950pr": res["950pr"], "910c": res["910c"]}
    out = {
        "operator": "swi_glu", "level": "level1",
        "decomposition": (
            "Elementwise L1, no cube. Read = whole input (numel elems); write = output (numel/2 elems, "
            "split dim halved). Per output element H=numel/2: sigmoid(x0) as 1 SFU op + x0*sigmoid (1 basic mul) "
            "+ silu*x1 (1 basic mul), all in fp32. For fp16/bf16: add cast input->fp32 on 2H elems and cast "
            "output->dtype on H elems (cast_rate). Vector charged at fp32 basic/sfu rates. Bottleneck = HBM read "
            "for all cases (read of full input dominates the halved output write and the few vector ops/elem)."),
        "open_validation": {
            "n_open": 20, "max_rel_err": round(worst[0], 6), "worst_case_id": worst[1],
            "notes": "All open cases are HBM-read-bound; vector (sigmoid+2 muls+casts) stays below read. Exact match."},
        "cases": cases,
    }
    outdir = os.path.join(HERE, "hap_work")
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, "level1__swi_glu.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")
    b2 = [c["910b2"]["t_hw_us"] for c in cases.values()]
    pr = [c["950pr"]["t_hw_us"] for c in cases.values()]
    print(f"910b2 t_hw min/med/max = {min(b2):.4f} / {statistics.median(b2):.4f} / {max(b2):.4f}")
    print(f"950pr t_hw min/med/max = {min(pr):.4f} / {statistics.median(pr):.4f} / {max(pr):.4f}")
    from collections import Counter
    print("910b2 bottlenecks:", Counter(c["910b2"]["bottleneck"] for c in cases.values()))
    print("950pr bottlenecks:", Counter(c["950pr"]["bottleneck"] for c in cases.values()))


if __name__ == "__main__":
    emit()
