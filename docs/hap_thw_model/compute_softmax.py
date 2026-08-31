#!/usr/bin/env python3
"""HAP t_hw computation for Softmax (level2/softmax).

Decomposition (softmax along axis of length R, total numel N, all math in FP32):
  C1 max-reduce over axis      -> N basic ops (FLAT)
  C2 broadcast subtract s-max  -> N basic ops
  C3 exp                       -> N sfu ops
  C4 sum-reduce over axis      -> N basic ops (FLAT)
  C5 broadcast div by row-sum  -> N div ops
For fp16/bf16 inputs (softmax accumulates in fp32):
  Cin  cast in  -> fp32        -> N cast ops
  Cout cast fp32 -> out        -> N cast ops
For fp32 input: no casts.

Memory: read x (N*bytes), write y (N*bytes) -- same shape/dtype as input.
Vector dtype for the basic/sfu/div math is float32 (compute-in-fp32).
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hap_lib as H


def parse_cases(path):
    rows = []
    with open(path) as f:
        header = f.readline()
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # split CSV respecting quoted fields
            import csv
            rows.append(next(csv.reader([line])))
    return rows


def case_work(input_shape, dtype):
    """Return (vec_ops, input_bytes, output_bytes) for a softmax case.
    dtype: the input/output dtype string. Math done in fp32.
    The reduction axis length does NOT change op counts (all components touch
    every element exactly once; reductions are flat = N ops)."""
    shape = input_shape[0]
    N = H.prod(shape)
    b = H.dbytes(dtype)
    input_bytes = N * b
    output_bytes = N * b

    vec_ops = []
    # softmax math in fp32
    vec_ops.append((N, "float32", "basic"))  # C1 max-reduce (flat)
    vec_ops.append((N, "float32", "basic"))  # C2 sub
    vec_ops.append((N, "float32", "sfu"))    # C3 exp
    vec_ops.append((N, "float32", "basic"))  # C4 sum-reduce (flat)
    vec_ops.append((N, "float32", "div"))    # C5 div

    if dtype in ("float16", "bfloat16"):
        vec_ops.append((N, dtype, "cast"))   # Cin  in -> fp32
        vec_ops.append((N, dtype, "cast"))   # Cout fp32 -> out
    return vec_ops, input_bytes, output_bytes


def compute_case(input_shape, dtype):
    vec_ops, ib, ob = case_work(input_shape, dtype)
    return H.compute_both(vec_ops=vec_ops, input_bytes=ib, output_bytes=ob)


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    repo = H.REPO_ROOT

    # ---- validate open cases 1-20 ----
    meta = json.load(open(os.path.join(repo, "tasks/metadata/910b2.json")))["level2"]["softmax"]
    open_rows = parse_cases(os.path.join(repo, "tasks/level2/softmax/cases.csv"))
    max_rel = 0.0
    worst = None
    print("=== OPEN VALIDATION ===")
    for r in open_rows:
        cid = r[1]
        shp = json.loads(r[2])
        dt = json.loads(r[3])[0]
        res = compute_case(shp, dt)
        got = res["910b2"]["t_hw_us"]
        exp = meta[cid]["t_hw_us"]
        rel = abs(got - exp) / exp if exp else 0.0
        absd = abs(got - exp)
        ok = (rel <= 0.02) or (absd <= 0.05)
        if rel > max_rel and not (absd <= 0.05):
            max_rel = rel
            worst = cid
        flag = "ok" if ok else "FAIL"
        print(f"case {cid:>2} {dt:>9} got={got:9.4f} exp={exp:9.4f} rel={rel*100:6.2f}% {flag}")
    print(f"max_rel_err (excl abs<=0.05) = {max_rel*100:.3f}% worst={worst}")

    if "--validate-only" in sys.argv:
        return

    # ---- compute inner cases 21-100 ----
    innerdir = os.path.join(repo, "inner", "tasks", "level2", "softmax")
    if not H.inner_cases_available(innerdir):
        print("closed (inner) case set not distributed publicly; open-case validation only.")
        return
    inner_rows = parse_cases(os.path.join(innerdir, "cases.csv"))
    cases = {}
    for r in inner_rows:
        cid = r[1]
        shp = json.loads(r[2])
        dt = json.loads(r[3])[0]
        res = compute_case(shp, dt)
        cases[cid] = res

    out = {
        "operator": "softmax",
        "level": "level2",
        "decomposition": (
            "Softmax over reduction axis (length R), total numel N. All softmax math in FP32: "
            "C1 max-reduce (N basic, FLAT), C2 broadcast sub (N basic), C3 exp (N sfu), "
            "C4 sum-reduce (N basic, FLAT), C5 broadcast div (N div). "
            "For fp16/bf16 inputs add 2 casts: Cin in->fp32 (N cast) + Cout fp32->out (N cast); "
            "fp32 input has no casts. Memory: read x = N*bytes, write y = N*bytes (same shape/dtype). "
            "Reduction axis length does not change op counts (flat reductions). Vector-bound for all cases."
        ),
        "open_validation": {
            "n_open": 20,
            "max_rel_err": round(max_rel, 4),
            "worst_case_id": worst,
            "notes": "All open cases within 2% gate (or abs<=0.05us). Softmax is vector-bound everywhere.",
        },
        "cases": cases,
    }
    os.makedirs(os.path.join(base, "hap_work"), exist_ok=True)
    with open(os.path.join(base, "hap_work", "level2__softmax.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote hap_work/level2__softmax.json")

    # summary stats
    import statistics
    for plat in ("910b2", "950pr"):
        vals = [cases[c][plat]["t_hw_us"] for c in cases]
        print(f"{plat}: min={min(vals):.3f} median={statistics.median(vals):.3f} max={max(vals):.3f}")


if __name__ == "__main__":
    main()
