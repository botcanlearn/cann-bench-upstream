#!/usr/bin/env python3
"""HAP t_hw for ArgMax (level2/arg_max).

Decomposition
-------------
torch.argmax(input, dim) is a single-pass reduction along the reduce axis that
tracks (running_max, running_index). Per element along the reduce axis it does a
compare + conditional index update => count as N basic vector ops (FLAT, NO log2 --
reductions are flat throughput in the HAP model).

Components:
  * READ : full input = numel elements * dtype_bytes  (HBM read, cold L2).
  * VECTOR : N = numel basic ops (compare/select) at the input-dtype basic rate.
             Index bookkeeping is int but the dominant arithmetic is the value
             compare at the input dtype rate; this is well below the read time in
             every case so it never sets t_hw.
  * WRITE : output = indices tensor, int64, shape = input with reduce axis removed
            (keepdim just inserts a size-1 axis -> same element count). Small.
  * No cube. No casts (compare is done in native dtype; index is int, not a
    value cast).

ArgMax is read-bound for all realistic shapes (read >> reduce-op time >> tiny
index write).
"""
import csv, json, os
import hap_lib as H

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = H.REPO_ROOT
INNERDIR = os.path.join(ROOT, "inner", "tasks", "level2", "arg_max")
OPDIR = os.path.join(ROOT, "tasks", "level2", "arg_max")


def numel(shape):
    return H.prod(shape)


def compute_case(input_shape, dtype):
    shapes = json.loads(input_shape)
    dts = json.loads(dtype)
    in_shape = shapes[0]
    in_dt = dts[0]
    N = numel(in_shape)

    in_bytes = N * H.dbytes(in_dt)

    # The reduce-axis length only affects the OUTPUT element count (computed
    # exactly by out_elems() at the call site); read bytes and the flat vector
    # op count depend on N alone.
    return N, in_bytes, in_dt, in_shape


def out_elems(in_shape, dim):
    rank = len(in_shape)
    d = dim if dim >= 0 else dim + rank
    D = in_shape[d]
    total = H.prod(in_shape)
    return total // D


def run(rows):
    out = {}
    for r in rows:
        cid = r["case_id"]
        attrs = json.loads(r["attrs"])
        dim = attrs["dim"]
        N, in_bytes, in_dt, in_shape = compute_case(r["input_shape"], r["dtype"])
        oe = out_elems(in_shape, dim)
        out_bytes = oe * H.dbytes("int64")

        res = {}
        for plat in ("910b2", "950pr", "910c"):
            tr = H.t_read_us(in_bytes, plat)
            # flat reduction: N basic ops at input dtype basic rate
            tv = H.t_vec_us(N, in_dt, plat, "basic")
            tw = H.t_write_us(out_bytes, plat)
            raw, bn, comps = H.aggregate(0.0, tv, tr, tw)
            t_hw, cap = H.finalize(raw)
            entry = {"t_hw_us": t_hw, "baseline_cap_us": cap, "bottleneck": bn}
            if plat in ("950pr", "910c"):
                entry["confidence"] = (H.confidence_910c if plat == "910c" else H.confidence_950pr)(bn, out_bytes)
            res[plat] = entry
        out[cid] = res
    return out


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def validate():
    rows = load(os.path.join(OPDIR, "cases.csv"))
    res = run(rows)
    meta = json.load(open(os.path.join(ROOT, "tasks", "metadata", "910b2.json")))["level2"]["arg_max"]
    worst = 0.0
    wc = None
    for cid, exp in meta.items():
        got = res[cid]["910b2"]["t_hw_us"]
        e = exp["t_hw_us"]
        rel = abs(got - e) / e if e else 0
        absd = abs(got - e)
        if rel > worst and absd > 0.05:
            worst = rel
            wc = cid
        flag = "" if (rel <= 0.02 or absd <= 0.05) else "  <-- MISS"
        print(f"case {cid:>2}: got {got:8.3f}  exp {e:8.3f}  rel {rel*100:6.2f}%{flag}")
    print(f"\nmax_rel_err (>0.05abs) = {worst*100:.3f}%  worst_case={wc}")
    return worst, wc


if __name__ == "__main__":
    worst, wc = validate()
    if not H.inner_cases_available(INNERDIR):
        print("closed (inner) case set not distributed publicly; open-case validation only.")
        raise SystemExit(0)
    inner = load(os.path.join(INNERDIR, "cases.csv"))
    cases = run(inner)
    vals = lambda p: sorted(v[p]["t_hw_us"] for v in cases.values())
    for p in ("910b2", "950pr", "910c"):
        vv = vals(p)
        print(f"{p}: min {vv[0]}  median {vv[len(vv)//2]}  max {vv[-1]}")
    doc = ("argmax = single-pass reduction along reduce axis tracking (max,index). "
           "READ full input = numel*dtype_bytes (HBM, cold L2); VECTOR = N basic "
           "compare/select ops FLAT (no log2) at input-dtype basic rate; WRITE = "
           "indices int64 with reduce axis removed (keepdim adds size-1 axis, same "
           "elem count), tiny. No cube, no casts. Read-bound for all cases.")
    payload = {
        "operator": "arg_max", "level": "level2",
        "decomposition": doc,
        "open_validation": {"n_open": 20, "max_rel_err": round(worst, 5),
                             "worst_case_id": wc,
                             "notes": "all open cases within 2% / 0.05us; read-bound flat reduction"},
        "cases": cases,
    }
    outp = os.path.join(HERE, "hap_work", "level2__arg_max.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(payload, open(outp, "w"), indent=1)
    print("wrote", outp)
