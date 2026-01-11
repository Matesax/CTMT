
import json, os
from pathlib import Path
import numpy as np

# ---------- IHS component scores ----------
def score_rank(rank, target=3):
    return 1.0 if rank>=target else max(0.0, rank/target)

def score_kappa(kappa, k_ref_good=1e2, k_ref_bad=1e6):
    if kappa<=0: return 0.0
    x = (np.log10(kappa) - np.log10(k_ref_good)) / (np.log10(k_ref_bad)-np.log10(k_ref_good))
    return float(1.0 - np.clip(x, 0.0, 1.0))

def score_monotonicity(violations, v_ref=1):
    return float(max(0.0, 1.0 - violations/(v_ref+1e-9)))

def score_composability(rel_spec, tau=1e-2):
    return float(np.exp(- rel_spec/max(tau,1e-12)))

def score_accept(coverage, target=0.95):
    return float(np.clip((coverage if coverage is not None else 0.0)/target, 0.0, 1.0))

def compute_ihs_for_window(rec, param_count=3):
    inv = rec["invariants"]
    s_rank  = score_rank(inv.get("rank",0), target=param_count)
    s_kappa = score_kappa(inv.get("kappa", np.inf))
    s_mono  = score_monotonicity(inv.get("violations",0))
    s_comp  = score_composability(inv.get("rel_spec",1.0))
    cov = rec.get("coverage95", rec.get("coverage95_GN", rec.get("coverage95_LS", 0.0)))
    s_acc   = score_accept(cov)

    # Weights
    w_acc, w_kap, w_rank, w_mono, w_comp = 0.35, 0.20, 0.20, 0.10, 0.15
    ihs = w_acc*s_acc + w_kap*s_kappa + w_rank*s_rank + w_mono*s_mono + w_comp*s_comp
    detail = {"s_rank": s_rank, "s_kappa": s_kappa, "s_mono": s_mono, "s_comp": s_comp, "s_acc": s_acc,
              "weights": {"accept":w_acc, "kappa":w_kap, "rank":w_rank, "mono":w_mono, "comp":w_comp}}
    return float(ihs), detail

def generate_report_from_json(in_json_path:str, out_md_path:str, title:str="# CTMT Invariant Health Report"):
    p = Path(in_json_path)
    lines = [title, ""]
    if not p.exists():
        lines += [f"**Input not found:** `{in_json_path}`.",
                  "Run your coherence pipeline to produce this JSON, then rerun the IHS report."]
        Path(out_md_path).write_text("\n".join(lines))
        return {"status":"missing_input","report":out_md_path}

    data = json.loads(p.read_text())
    results = data.get("results", [])
    if len(results)==0:
        lines += ["**No window results found in JSON.**"]
        Path(out_md_path).write_text("\n".join(lines))
        return {"status":"empty","report":out_md_path}

    ihs_vals = []
    for r in results:
        ihs, det = compute_ihs_for_window(r, param_count=len(data.get("config",{}).get("param_keys",[1,2,3])))
        ihs_vals.append(ihs)
        r["IHS"] = ihs
        r["IHS_detail"] = det

    ihs_arr = np.array(ihs_vals); q = np.quantile(ihs_arr, [0.1,0.25,0.5,0.75,0.9])
    lines += ["## Summary",
              f"Windows: {len(results)}",
              f"IHS median: {np.median(ihs_arr):.3f}  (IQR: {q[1]:.3f}–{q[3]:.3f})",
              f"IHS deciles: 10%={q[0]:.3f}, 50%={q[2]:.3f}, 90%={q[4]:.3f}", ""]

    order = np.argsort(-ihs_arr)
    topk = order[:min(5,len(order))]; botk = order[-min(5,len(order)):] if len(order)>0 else []
    lines += ["## Top windows (by IHS)"]
    for idx in topk:
        w = results[int(idx)]
        lines += [f"- w={w.get('w_index','?')} | IHS={w['IHS']:.3f} | coverage95={w.get('coverage95','-')}"]
    lines += ["", "## Bottom windows (by IHS)"]
    for idx in botk:
        w = results[int(idx)]
        lines += [f"- w={w.get('w_index','?')} | IHS={w['IHS']:.3f} | coverage95={w.get('coverage95','-')}"]

    lines += ["", "## Interpretation & Guidance",
              "- **IHS → 1.0**: strong coherence signature; Fisher invariants and acceptance aligned.",
              "- **0.6 ≤ IHS ≤ 0.8**: acceptable; check kernel/whitening if trending down.",
              "- **IHS < 0.6**: investigate: kappa inflation, monotonicity violations, composability drift, or acceptance loss (try CRSC, robust loss, GN, or medium-specific G).",
              "", "### Component Scores per window",
              "IHS combines: acceptance (35%), kappa (20%), rank (20%), monotonicity (10%), composability (15%).", ""]

    Path(out_md_path).write_text("\n".join(lines))

    aug_path = Path(in_json_path).with_name(Path(in_json_path).stem + "_with_ihs.json")
    data["results"] = results
    Path(aug_path).write_text(json.dumps(data, indent=2))
    return {"status":"ok","report":out_md_path, "aug_json":str(aug_path)}

# Example run (assumes you've already produced ctmt_coherence_results.json)
# res = generate_report_from_json('ctmt_coherence_results.json', 'ctmt_invariant_health_report.md')
