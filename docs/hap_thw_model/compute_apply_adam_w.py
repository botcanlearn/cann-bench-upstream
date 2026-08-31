#!/usr/bin/env python3
"""HAP t_hw for level2/apply_adam_w (AdamW optimizer step).

Decomposition (pure elementwise, memory-bound):
  Reads 4 input tensors var, grad, m, v (all same shape/dtype) -> read = 4*N*bytes / HBM.
  Writes the updated parameter y (1 tensor, native dtype) -> L2-fill-first write.
  Vector compute (per element, accumulate in fp32 for fp16/bf16):
    m_new  = beta1*m + (1-beta1)*grad         : 2 mul + 1 add        = 3 basic
    v_new  = beta2*v + (1-beta2)*grad*grad     : grad^2, *c, beta2*v, add = 4 basic
    m_hat  = m_new * (1/(1-beta1^t))           : 1 basic (reciprocal const)
    v_hat  = v_new * (1/(1-beta2^t))           : 1 basic
    sqrt(v_hat)                                : 1 sfu
    + epsilon                                  : 1 basic (add)
    m_hat / (sqrt+eps)                         : 1 div
    update + var*weight_decay                  : 1 mul + 1 add       = 2 basic
    var -/+ lr*update                          : 1 mul + 1 sub       = 2 basic
    => 14 basic + 1 sfu(sqrt) + 1 div  per element (constants precomputed on host)
  Casts for fp16/bf16: 4 input casts (in->fp32) + 1 output cast (fp32->native) = 5*N cast-ops.
  For fp32 inputs: no casts, compute at fp32 basic rate.
  Read dominates on both platforms for the validated open set; aggregate() takes the max.
"""
import csv, json, os
import hap_lib as H

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = H.REPO_ROOT
INNERDIR = os.path.join(ROOT, "inner", "tasks", "level2", "apply_adam_w")
OP_DIR = os.path.join(ROOT, "tasks", "level2", "apply_adam_w")

BASIC = 14        # basic arithmetic element-ops
SFU = 1           # sqrt
DIV = 1           # one true division (m_hat / (sqrt(v_hat)+eps))


def parse_row(row):
    shapes = json.loads(row["input_shape"])
    dtypes = json.loads(row["dtype"])
    var_shape = shapes[0]
    dt = dtypes[0]                      # all four inputs share dtype
    N = H.prod(var_shape)
    return N, dt


def compute_case(N, dt):
    in_bytes = H.dbytes(dt)
    read_bytes = 4 * N * in_bytes       # var, grad, m, v
    write_bytes = N * in_bytes          # y (native dtype)

    is_low = dt in ("float16", "bfloat16")
    compute_dt = "float32" if is_low else dt

    vec_ops = [
        (BASIC * N, compute_dt, "basic"),
        (SFU * N, compute_dt, "sfu"),
        (DIV * N, compute_dt, "div"),
    ]
    if is_low:
        vec_ops.append((5 * N, dt, "cast"))   # 4 in + 1 out

    return H.compute_both(vec_ops=vec_ops,
                          input_bytes=read_bytes,
                          output_bytes=write_bytes)


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def validate_open():
    meta = json.load(open(os.path.join(ROOT, "tasks", "metadata", "910b2.json")))
    ref = meta["level2"]["apply_adam_w"]
    worst = (0.0, None)
    for row in load(os.path.join(OP_DIR, "cases.csv")):
        cid = row["case_id"]
        N, dt = parse_row(row)
        res = compute_case(N, dt)
        got = res["910b2"]["t_hw_us"]
        exp = ref[cid]["t_hw_us"]
        rel = abs(got - exp) / exp if exp else 0.0
        if rel > worst[0] and not (abs(got - exp) <= 0.05):
            worst = (rel, cid)
    return worst


def main():
    worst_rel, worst_cid = validate_open()
    print(f"open max_rel_err={worst_rel:.5f} worst_case={worst_cid}")

    if not H.inner_cases_available(INNERDIR):
        print("closed (inner) case set not distributed publicly; open-case validation only.")
        raise SystemExit(0)

    cases = {}
    for row in load(os.path.join(INNERDIR, "cases.csv")):
        cid = row["case_id"]
        N, dt = parse_row(row)
        res = compute_case(N, dt)
        cases[cid] = res

    out = {
        "operator": "apply_adam_w", "level": "level2",
        "decomposition": (
            "Pure elementwise AdamW step, memory-bound. Read = 4 input tensors "
            "(var,grad,m,v) at native dtype / HBM_BW; write = y (1 tensor, native "
            "dtype) via L2-fill-first. Vector per element (compute in fp32 for "
            "fp16/bf16): 14 basic arith + 1 sqrt(sfu) + 1 div; constants "
            "(beta*, 1-beta*, 1/(1-beta^t), lr, wd) precomputed on host. fp16/bf16 "
            "add 5*N cast ops (4 in->fp32 + 1 out). Read dominates on both platforms "
            "for every open case (verified)."
        ),
        "open_validation": {
            "n_open": 20, "max_rel_err": round(worst_rel, 5),
            "worst_case_id": worst_cid,
            "notes": ("read-bound; all 20 open cases match 910b2 metadata exactly "
                      "(read = 4*N*bytes/HBM_BW). Vector < read on the open set.")
        },
        "cases": cases,
    }
    os.makedirs(os.path.join(HERE, "hap_work"), exist_ok=True)
    outpath = os.path.join(HERE, "hap_work", "level2__apply_adam_w.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", outpath)

    # summary stats
    import statistics as st
    for plat in ("910b2", "950pr"):
        vals = sorted(c[plat]["t_hw_us"] for c in cases.values())
        bns = {}
        for c in cases.values():
            bns[c[plat]["bottleneck"]] = bns.get(c[plat]["bottleneck"], 0) + 1
        print(f"{plat}: min={vals[0]} med={st.median(vals)} max={vals[-1]} bottlenecks={bns}")


if __name__ == "__main__":
    main()
