# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import itertools
import numpy as np
import pandas as pd

# local core helpers
from aeroopt.core.config_io import load_yaml

# --------- paths ---------
ROOT = Path(__file__).resolve().parents[3]  # project root
RESULTS_DIR = ROOT / "results" / "script00"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ARCH_SELECTED_CSV = RESULTS_DIR / "arch_selected.csv"

# --------- simple architecture struct ---------
class Arch(tuple):
    __slots__ = ()
    _fields = ("wing_place", "tail_type", "wingtip", "prop_layout", "engines", "gear")

    def __new__(cls, wing_place, tail_type, wingtip, prop_layout, engines, gear):
        return tuple.__new__(cls, (wing_place, tail_type, wingtip, prop_layout, engines, gear))

    @property
    def as_dict(self):
        return dict(zip(self._fields, tuple(self)))

# --------- proxies (kept here to keep script00 self-contained) ---------
def oswald_base(wing_place: str) -> float:
    return {"low": 0.80, "mid": 0.79, "high": 0.78}[wing_place]

def e_with_tip(e0: float, wingtip: str) -> float:
    if wingtip == "winglet":
        return e0 * 1.08
    if wingtip == "raked":
        return e0 * 1.05
    return e0

def cd0_arch_pen(arch: Arch) -> float:
    p = 0.0
    if arch[0] == "high":      p += 0.003   # high wing interference/strut allowance
    if arch[5] == "fixed":     p += 0.004   # fixed gear
    if arch[1] == "twinboom":  p += 0.0015
    if arch[3] == "pusher":    p += 0.0007
    return p

def sc_effectiveness(arch: Arch) -> float:
    m = 1.0
    if arch[1] == "T":      m *= 1.05
    if arch[1] == "V":      m *= 0.90
    if arch[1] == "canard": m = 0.0
    return m

def risk_penalty(arch: Arch) -> float:
    r = 0.0
    if arch[1] == "T":      r += 0.5
    if arch[1] == "V":      r += 0.3
    if arch[3] == "pusher": r += 0.3
    if arch[4] == 2:        r += 0.2
    return r

def bending_proxy(span_m: float, arch: Arch) -> float:
    base = span_m ** 3
    if arch[2] == "winglet":
        base *= 1.06
    if arch[1] == "twinboom":
        base *= 1.03
    return base

def make_arch_id(a: Arch) -> str:
    d = a.as_dict
    return f"arch-{d['wing_place']}-{d['tail_type']}-{d['wingtip']}-{d['prop_layout']}-{d['engines']}eng-{d['gear']}"

# --------- scoring ---------
def score_arch(m: dict, wD=1.0, wSC=0.5, wB=0.02, wR=0.1, wM=-0.05) -> float:
    return (wD*m["D_proxy"] + wSC*m["sc_pen"] + wB*m["bend"] + wR*m["risk"] + wM*m["mfg"])

# --------- main run ---------
def run(top_k: int = 10):
    cfg = load_yaml(ROOT / "configs" / "00_arch.yaml")  # weights, enumerations, thresholds
    # fallbacks if a field is missing in the YAML
    span_m   = float(cfg.get("span_m_nom", 1.5))
    S_ref    = float(cfg.get("S_ref_nom", 0.25))
    CL_cr    = float(cfg.get("CL_cruise_nom", 0.60))
    feasible_sc_min = float(cfg.get("sc_min", 1.0))
    wD, wSC, wB, wR, wM = [float(x) for x in cfg.get("weights", {}).values()] if cfg.get("weights") else (1.0, 0.5, 0.02, 0.1, -0.05)

    wing_places = cfg.get("wing_place", ["low","mid","high"])
    tail_types  = cfg.get("tail_type",  ["conv","T","V","twinboom","canard"])
    wingtips    = cfg.get("wingtip",    ["none","winglet","raked"])
    props       = cfg.get("prop_layout",["tractor","pusher"])
    engines     = cfg.get("engines",    [1,2])
    gears       = cfg.get("gear",       ["fixed"])

    rows = []
    for wp, tt, wt, pl, eng, gr in itertools.product(wing_places, tail_types, wingtips, props, engines, gears):
        a = Arch(wp, tt, wt, pl, int(eng), gr)

        e0 = oswald_base(wp)
        e  = e_with_tip(e0, wt)
        AR = (span_m**2) / S_ref
        k_ind = 1.0 / (np.pi * e * AR)

        CD0 = 0.012 + cd0_arch_pen(a)
        CD  = CD0 + k_ind * (CL_cr ** 2)

        sc_eff = sc_effectiveness(a)
        sc_pen = max(0.0, feasible_sc_min - sc_eff)

        bend = bending_proxy(span_m, a)
        risk = risk_penalty(a)
        mfg  = -0.1 if gr == "fixed" else 0.0

        m = dict(e=e, CD0=CD0, D_proxy=float(CD), sc_pen=float(sc_pen), bend=float(bend), risk=float(risk), mfg=float(mfg))
        J = score_arch(m, wD=wD, wSC=wSC, wB=wB, wR=wR, wM=wM)

        rows.append({**a.as_dict, **m, "score": float(J), "arch_id": make_arch_id(a)})

    df = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)

    # optional diversity gate
    K = int(cfg.get("top_k", top_k))
    diversify = bool(cfg.get("diversify", True))
    if diversify:
        buckets = df.groupby(["wing_place", "tail_type"], as_index=False).head(1)
        remaining = df[~df["arch_id"].isin(buckets["arch_id"])].head(max(0, K - len(buckets)))
        df_sel = pd.concat([buckets, remaining], ignore_index=True).sort_values("score").reset_index(drop=True)
    else:
        df_sel = df.head(K)

    cols_keep = ["arch_id","wing_place","tail_type","wingtip","prop_layout","engines","gear",
                 "score","e","CD0","D_proxy","bend","risk","mfg"]
    df_sel[cols_keep].to_csv(ARCH_SELECTED_CSV, index=False)

    # pretty print top list
    print(df.head(K).to_string(index=False))
    print(df_sel[["arch_id","score"]].to_string(index=False))
    print(f"Saved selected architectures -> {ARCH_SELECTED_CSV}")

if __name__ == "__main__":
    run()
