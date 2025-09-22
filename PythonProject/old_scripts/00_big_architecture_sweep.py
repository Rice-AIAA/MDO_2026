from dataclasses import dataclass
import itertools, pandas as pd
import numpy as np
from pathlib import Path

# ---- architecture definition ----
@dataclass(frozen=True)
class Arch:
    wing_place: str         # 'low','mid','high'
    tail_type: str          # 'conv','T','V','twinboom','canard'
    wingtip: str            # 'none','winglet','raked'
    prop_layout: str        # 'tractor','pusher','twinboom_pusher'
    engines: int            # 1 or 2
    gear: str               # 'skid','fixed','retract'

def enumerate_arches():
    return [
        Arch(wp, tt, wt, pl, eng, gr)
        for wp in ['low','mid','high']
        for tt in ['conv','T','V','twinboom','canard']
        for wt in ['none','winglet','raked']
        for pl in ['tractor','pusher']
        for eng in [1,2]
        for gr in ['fixed']  # expand later
    ]

# ---- cheap proxies/penalties ----
def oswald_e_base(wing_place):
    # baseline e before wingtip option
    return {'low':0.80,'mid':0.79,'high':0.78}[wing_place]

def e_with_tip(e0, wingtip):
    if wingtip == 'none':   return e0
    if wingtip == 'winglet':return e0 * 1.08   # ~8% better induced at CLcruise
    if wingtip == 'raked':  return e0 * 1.05
    return e0

def cd0_arch_pen(arch: Arch):
    p = 0.0
    if arch.wing_place == 'high': p += 0.003  # strut/interference allowance (tune)
    if arch.gear == 'fixed':      p += 0.004
    if arch.tail_type == 'twinboom': p += 0.0015
    if arch.prop_layout == 'pusher': p += 0.0007  # cooling/exhaust odds/ends
    return p

def sc_effectiveness(arch: Arch):
    # scale factor for tail pitch effectiveness
    m = 1.0
    if arch.tail_type == 'T':       m *= 1.05     # cleaner tail flow
    if arch.tail_type == 'V':       m *= 0.90     # coupling loss
    if arch.tail_type == 'canard':  m = 0.0       # tail-less; handled elsewhere
    return m

def risk_penalty(arch: Arch):
    r = 0.0
    if arch.tail_type == 'T':      r += 0.5  # deep-stall risk flag
    if arch.tail_type == 'V':      r += 0.3  # coupling/tuning
    if arch.prop_layout == 'pusher': r += 0.3  # ingestion/noise/tail strike
    if arch.engines == 2:          r += 0.2  # integration complexity
    return r  # (unitless; will be weighted)

def bending_penalty(span_m, arch: Arch):
    base = span_m**3
    if arch.wingtip == 'winglet': base *= 1.06
    if arch.tail_type == 'twinboom': base *= 1.03
    return base

# ---- quick evaluation at nominal operating point ----
def eval_arch(arch: Arch, span_m=1.5, S_ref=0.25, CL_cr=0.6):
    e0 = oswald_e_base(arch.wing_place)
    e  = e_with_tip(e0, arch.wingtip)
    k_ind = 1.0 / (np.pi * e * (span_m**2 / S_ref))  # ~ 1/(π e AR)

    # crude drag proxy at CL_cr: CD = CD0 + k_ind * CL^2
    CD0 = 0.012 + cd0_arch_pen(arch)  # 0.012 baseline airframe pad (tune to your data)
    CD  = CD0 + k_ind * (CL_cr**2)
    D_proxy = CD  # proportional; no qS here for ranking

    # S&C proxy: require effectiveness above threshold
    sc_eff = sc_effectiveness(arch)
    sc_pen = max(0.0, 1.0 - sc_eff)  # 0 if >=1, penalty if <1

    bend = bending_penalty(span_m, arch)

    risk = risk_penalty(arch)
    mfg  = 0.0
    if arch.gear == 'fixed': mfg -= 0.1  # simpler is better (negative = bonus)

    return dict(
        e=e, CD0=CD0, D_proxy=D_proxy,
        sc_pen=sc_pen, bend=bend, risk=risk, mfg=mfg
    )

# ---- overall score & screening ----
def score_arch(metrics, wD=1.0, wSC=0.5, wB=0.02, wR=0.1, wM=-0.05):
    # lower is better
    return (wD*metrics['D_proxy']
            + wSC*metrics['sc_pen']
            + wB*metrics['bend']
            + wR*metrics['risk']
            + wM*metrics['mfg'])

def run_script00(top_k=16):
    rows = []
    for a in enumerate_arches():
        m = eval_arch(a)
        feasible = (m['sc_pen'] <= 0.2)  # simple feasibility gate (tune)
        if not feasible:
            continue
        J = score_arch(m)
        rows.append({**a.__dict__, **m, 'score': J})

    df = pd.DataFrame(rows).sort_values('score').reset_index(drop=True)
    # keep top_k unique patterns; write CSV for Script 01
    return df.head(top_k)

if __name__ == "__main__":
    df_top = run_script00(top_k=16)
    print(df_top.to_string(index=False))
    df_top.to_csv("results/script00_architectures.csv", index=False)

# --- selection & export of architectures (end of file) ---

# Ensure ROOT exists in this script too
ROOT = Path(__file__).resolve().parent
try:
    ROOT
except NameError:
    ROOT = Path(__file__).resolve().parent

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ARCH_SELECTED_CSV = RESULTS_DIR / "arch_selected.csv"

def make_arch_id(r):
    # Works for either a pandas Series row or a dataclass/object
    def get(val):
        if isinstance(r, dict):
            return r[val]
        return getattr(r, val) if hasattr(r, val) else r[val]
    return f"arch-{get('wing_place')}-{get('tail_type')}-{get('wingtip')}-{get('prop_layout')}-{get('engines')}eng-{get('gear')}"

def select_architectures(df_in, K=6, diversify=True):
    df = df_in.copy()
    df["arch_id"] = df.apply(make_arch_id, axis=1)

    if not diversify:
        return df.sort_values("score").head(K).reset_index(drop=True)

    # One per (wing_place, tail_type) bucket, then fill by score
    buckets = (
        df.sort_values("score")
          .groupby(["wing_place", "tail_type"], as_index=False)
          .head(1)
    )
    remaining = (
        df[~df["arch_id"].isin(buckets["arch_id"])]
          .sort_values("score")
          .head(max(0, K - len(buckets)))
    )
    out = pd.concat([buckets, remaining], ignore_index=True)
    return out.sort_values("score").reset_index(drop=True)

if __name__ == "__main__":
    # Step 1: run the coarse sweep
    df_top = run_script00(top_k=16)
    print(df_top.to_string(index=False))
    df_top.to_csv(RESULTS_DIR / "script00_architectures.csv", index=False)

    # Step 2: select the set we’ll carry forward to Script-01
    TOP_K_ARCH = 10  # <- change this number as you like
    arch_sel = select_architectures(df_top, K=TOP_K_ARCH, diversify=True)

    # Step 3: save just the essentials for downstream scripts
    arch_cols = [
        "arch_id","wing_place","tail_type","wingtip","prop_layout",
        "engines","gear","score","e","CD0","D_proxy","bend","risk","mfg"
    ]
    arch_sel[arch_cols].to_csv(ARCH_SELECTED_CSV, index=False)
    print(f"Saved selected architectures -> {ARCH_SELECTED_CSV}")
    print(arch_sel[["arch_id","score"]].to_string(index=False))
