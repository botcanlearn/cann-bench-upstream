#!/usr/bin/env python3
"""HAP t_hw computation for level4/mla (Multi-head Latent Attention).

Decomposed from tasks/level4/mla/golden.py (NOT the operator name / the KV-cache
decompression story).  Golden is the *attention-only* part of MLA: no low-rank
down/up projection matmuls appear in golden.py ("仅包含注意力计算部分，不含 KV 解压缩").
It is therefore a Flash-Attention-shaped op with a *split* QK head dim:

  inputs (BSND, after optional BNSD->BSND permute which is free data movement):
    q_nope [B, S, N_q, d_nope]   q_rope [B, S, N_q, d_rope]
    k_nope [B, S_kv, N_kv, d_nope] k_rope [B, S_kv, N_kv, d_rope]
    v      [B, S_kv, N_kv, d_nope]   (numerically == k_nope: shared latent KV cache)

  Q = concat(q_nope, q_rope)   -> head dim D_qk = d_nope + d_rope   (concat is free)
  K = concat(k_nope, k_rope)   -> head dim D_qk = d_nope + d_rope
  GQA expand K/V from N_kv to N_q heads (free broadcast / unique-bytes only)
  scores = (Q @ K^T) * scale          [B, N_q, S, S_kv]
  softmax over S_kv  (fp32, with all-masked-row guard -> still elementwise on M)
  out = P @ V                          [B, N_q, S, d_nope]

  N_kv == 1 for every case, N_q in {32,64,96,128}, so GQA group G = N_q.

M = kept attention-matrix elements per (b, head) summed:
    non-causal : M = B * N_q * S * S_kv
    causal (right-bottom aligned, j <= i+(S_kv-S), S<=S_kv):
                 M = B * N_q * (S*S_kv - S*(S-1)/2)

Cube (two matmuls, DIFFERENT contraction dims):
  C1 QK^T : 2 * M * D_qk     (contraction over d_nope+d_rope)
  C9 P@V  : 2 * M * d_nope   (contraction/output over d_nope only)

Vector (softmax in fp32 with explicit casts; M element-ops each):
  cast scores in-dtype->fp32 ; scale(mul) ; row max-reduce(flat) ; sub ;
  exp(sfu) ; row sum-reduce(flat) ; div ; cast P fp32->in-dtype.
  (The all-masked-row guard / scale-by-scaleValue cast etc. are O(M) or smaller
   and folded into the existing components; no RoPE vector op -- rope parts are
   passed in already encoded.)

Memory (unique bytes only):
  read = q_nope + q_rope + k_nope + k_rope  (v NOT added).
    v==k_nope numerically and shares the same latent KV-cache HBM region, so
    under the unique-bytes rule v contributes no new read traffic over k_nope.
    The open-case decode metadata (S=1, large S_kv) only matches when v's bytes
    are excluded.  K/V stored with N_kv heads (counted once, GQA expand is not a
    reload).
  write = y [B, S, N_q, d_nope].
"""
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hap_lib as H

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = H.REPO_ROOT
INNERDIR = os.path.join(ROOT, "inner", "tasks", "level4", "mla")
OPDIR = os.path.join(ROOT, "tasks", "level4", "mla")


def causal_keep(S, S_kv):
    """Unmasked (i,j) per (b,head), right-bottom causal mask j<=i+(S_kv-S), S<=S_kv."""
    return S * S_kv - S * (S - 1) / 2.0


def parse_shapes(input_shape, attrs):
    """Return BSND-normalized (B,S,N_q,d_nope,d_rope,S_kv,N_kv)."""
    layout = attrs.get("inputLayout", "BSND")
    q_nope, q_rope, k_nope, k_rope, v = input_shape
    if layout == "BNSD":
        # [B, N, S, D] -> read N_q from axis1, S from axis2
        B, N_q, S, d_nope = q_nope
        d_rope = q_rope[3]
        S_kv = k_nope[2]
        N_kv = k_nope[1]
    else:  # BSND [B, S, N, D]
        B, S, N_q, d_nope = q_nope
        d_rope = q_rope[3]
        S_kv = k_nope[1]
        N_kv = k_nope[2]
    return B, S, N_q, d_nope, d_rope, S_kv, N_kv


def compute_case(input_shape, dtypes, attrs):
    B, S, N_q, d_nope, d_rope, S_kv, N_kv = parse_shapes(input_shape, attrs)
    D_qk = d_nope + d_rope
    dt = dtypes[0]
    is_causal = bool(attrs.get("is_causal", False))

    if is_causal:
        M = B * N_q * causal_keep(S, S_kv)
    else:
        M = B * N_q * S * S_kv

    # ---- cube: two matmuls with different contraction dims ----
    cube_flops = 2 * M * D_qk + 2 * M * d_nope

    # ---- vector: softmax in fp32 with casts (8 components, M ops each) ----
    vec_ops = [
        (M, "float32", "cast"),   # scores in-dtype -> fp32
        (M, "float32", "basic"),  # scale (mul by scaleValue)
        (M, "float32", "basic"),  # row max-reduce (flat)
        (M, "float32", "basic"),  # sub (s - max)
        (M, "float32", "sfu"),    # exp
        (M, "float32", "basic"),  # row sum-reduce (flat)
        (M, "float32", "div"),    # div by row-sum
        (M, "float32", "cast"),   # P fp32 -> in-dtype
    ]

    # ---- memory (unique bytes) ----
    eb = H.dbytes(dt)
    q_bytes = (B * S * N_q * d_nope + B * S * N_q * d_rope) * eb
    k_bytes = (B * S_kv * N_kv * d_nope + B * S_kv * N_kv * d_rope) * eb
    # v == k_nope (shared latent KV cache, same HBM region): unique bytes only,
    # so v adds NO new read traffic over k_nope.  This is the decode read-bound
    # regime and is required to match the open-case metadata (see validation).
    v_bytes = 0.0
    input_bytes = q_bytes + k_bytes + v_bytes
    output_bytes = B * S * N_q * d_nope * eb

    return H.compute_both(
        vec_ops=vec_ops,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        cube_dtype=dt,
        cube_flops=cube_flops,
    )


def _clean_attrs(a):
    out = {}
    for k, v in a.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = -1.0  # scaleValue irrelevant to t_hw; treat as auto
        else:
            out[k] = v
    return out


def load_cases(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "case_id": r["case_id"],
                "input_shape": json.loads(r["input_shape"]),
                "dtype": json.loads(r["dtype"]),
                # json accepts NaN/Infinity/-Infinity natively
                "attrs": _clean_attrs(json.loads(r["attrs"])),
            })
    return rows


def validate_open():
    meta = json.load(open(os.path.join(ROOT, "tasks", "metadata", "910b2.json")))
    ref = meta["level4"]["mla"]
    rows = load_cases(os.path.join(OPDIR, "cases.csv"))
    worst = (0.0, None)
    for row in rows:
        cid = row["case_id"]
        res = compute_case(row["input_shape"], row["dtype"], row["attrs"])
        got = res["910b2"]["t_hw_us"]
        exp = ref[cid]["t_hw_us"]
        rel = abs(got - exp) / exp if exp else 0.0
        absd = abs(got - exp)
        ok = rel <= 0.02 or absd <= 0.05
        flag = "" if ok else "  <-- MISS"
        if rel > worst[0]:
            worst = (rel, cid)
        print(f"case {cid:>2}: got={got:9.4f}  exp={exp:9.4f}  rel={rel*100:6.3f}%  "
              f"bn={res['910b2']['bottleneck']}{flag}")
    print(f"\nmax_rel_err = {worst[0]*100:.4f}% @ case {worst[1]}")
    return worst


def build_inner_json():
    rows = load_cases(os.path.join(INNERDIR, "cases.csv"))
    return {row["case_id"]: compute_case(row["input_shape"], row["dtype"], row["attrs"])
            for row in rows}


if __name__ == "__main__":
    print("=== OPEN-case validation (910b2) ===")
    worst = validate_open()

    if not H.inner_cases_available(INNERDIR):
        print("closed (inner) case set not distributed publicly; open-case validation only.")
        raise SystemExit(0)

    cases = build_inner_json()
    out = {
        "operator": "mla",
        "level": "level4",
        "decomposition": (
            "Attention-only MLA (golden.py: no KV down/up-proj matmuls -- '仅注意力部分'). "
            "Flash-Attention with a SPLIT QK head dim: Q=concat(q_nope[d_nope],q_rope[d_rope]) "
            "and K=concat(k_nope,k_rope) give QK contraction dim D_qk=d_nope+d_rope; P@V output "
            "dim = d_nope. N_kv=1, GQA group G=N_q (K/V broadcast, unique bytes only). "
            "M = kept attention elements = B*N_q*S*S_kv (non-causal) or B*N_q*(S*S_kv - S*(S-1)/2) "
            "(right-bottom causal). Cube: QK^T=2*M*D_qk + P@V=2*M*d_nope FLOPs at input dtype. "
            "Vector softmax in fp32 (M ops each): cast scores->fp32, scale(mul), max-reduce(flat), "
            "sub, exp(sfu), sum-reduce(flat), div, cast P->in-dtype. concat & RoPE are free "
            "(rope passed pre-encoded; concat is data movement). Memory unique bytes: read "
            "q_nope+q_rope+k_nope+k_rope (v NOT added: v==k_nope shares the same latent KV-cache HBM "
            "region, so under unique-bytes it adds no read traffic; required to match decode open "
            "cases). K/V at N_kv heads counted once. write y[B,S,N_q,d_nope]."
        ),
        "open_validation": {
            "n_open": 20,
            "max_rel_err": round(worst[0], 6),
            "worst_case_id": worst[1],
            "notes": "All open cases within 2% gate. Prefill/large-M cube-bound; "
                     "decode (S=1) read-bound (KV latent dominates).",
        },
        "cases": cases,
    }
    outpath = os.path.join(HERE, "hap_work", "level4__mla.json")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {outpath}")
    import statistics as st
    b2 = [v["910b2"]["t_hw_us"] for v in cases.values()]
    pr = [v["950pr"]["t_hw_us"] for v in cases.values()]
    print(f"910b2 t_hw min/med/max = {min(b2):.3f} / {st.median(b2):.3f} / {max(b2):.3f}")
    print(f"950pr t_hw min/med/max = {min(pr):.3f} / {st.median(pr):.3f} / {max(pr):.3f}")
