# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import itertools, json
import pandas as pd
import numpy as np

from aeroopt.core.config_io import load_yaml

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results"
# OLD:
# ARCH_SELECTED_CSV = RESULTS_DIR / "arch_selected.csv"
# NEW (matches Script 00):
ARCH_SELECTED_CSV = RESULTS_DIR / "script00" / "arch_selected.csv"
# You can also keep sweep outputs tidy in a subfolder:
SWEEP_SUMMARY_CSV = RESULTS_DIR / "script01" / "sweep_summary.csv"
SWEEP_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)

def make_config_id(arch_id: str, fam: str, AR: float, taper: float,
                   dih: float, twr: float, twt: float, airfoil: str) -> str:
    return (f"{arch_id}__{fam}_AR{AR:g}_tap{taper:g}_dih{int(dih)}_tw{int(twr)}_{int(twt)}_{airfoil}")

def _try_read_W_from_legacy_json(config_id: str) -> float | None:
    """
    Reads W from the old per-config JSONs, if they exist.
    Expected schema: {"mission": {"W": <number>}}
    """
    candidates = [
        RESULTS_DIR / f"config_{config_id}.json",
        RESULTS_DIR / "script01_configs" / f"config_{config_id}.json",  # optional legacy location
    ]
    for p in candidates:
        if p.exists():
            try:
                data = json.loads(p.read_text())
                W = data.get("mission", {}).get("W", None)
                if W is not None:
                    return float(W)
            except Exception:
                pass
    return None

def run():
    cfg = load_yaml(ROOT / "configs" / "01_sweep.yaml")
    env = load_yaml(ROOT / "configs" / "mission.yaml")  # for W_fallback
    W_fallback = float(env.get("W_fallback", 20.0))

    n_arch = int(cfg.get("n_arch_from_00", 10))

    # load selected architectures from Script 00
    if not ARCH_SELECTED_CSV.exists():
        raise FileNotFoundError(f"Missing {ARCH_SELECTED_CSV}. Run script 00 first.")
    arch_df = pd.read_csv(ARCH_SELECTED_CSV)
    arch_df = arch_df.sort_values("score").head(n_arch).reset_index(drop=True)

    # wing families & grids (from YAML, with defaults)
    families = cfg.get("families", {
        "wingA": {
            "AR":    [8.5, 9.5],
            "taper": [0.5, 0.7],
            "dihedral_deg": [5, 10],
            "twist_root_deg": [2, 3],
            "twist_tip_deg": [-1, -2],
            "airfoil": ["ag13"]
        },
        "wingB": {
            "AR":    [7.5, 8.5],
            "taper": [0.55, 0.6],
            "dihedral_deg": [5, 10],
            "twist_root_deg": [2, 3],
            "twist_tip_deg": [-1, -2],
            "airfoil": ["naca4412", "sg6043"]
        }
    })

    rows = []
    for _, arch in arch_df.iterrows():
        for fam_name, grid in families.items():
            ARs      = grid.get("AR", [])
            tapers   = grid.get("taper", [])
            dihs     = grid.get("dihedral_deg", [])
            twrs     = grid.get("twist_root_deg", [])
            twts     = grid.get("twist_tip_deg", [])
            airfoils = grid.get("airfoil", [])

            for AR, taper, dih, twr, twt, af in itertools.product(ARs, tapers, dihs, twrs, twts, airfoils):
                cfg_id = make_config_id(arch.arch_id, fam_name, AR, taper, dih, twr, twt, af)

                # NEW: pull W from legacy JSONs if available; else fallback
                W_json = _try_read_W_from_legacy_json(cfg_id)
                W_val = W_json if (W_json is not None) else W_fallback

                rows.append({
                    "config_id": cfg_id,
                    "arch_id": arch.arch_id,
                    "wing_place": arch.wing_place,
                    "tail_type": arch.tail_type,
                    "wingtip": arch.wingtip,
                    "prop_layout": arch.prop_layout,
                    "engines": arch.engines,
                    "gear": arch.gear,
                    "fam_name": fam_name,
                    "AR": AR,
                    "taper": taper,
                    "dihedral_deg": dih,
                    "twist_root_deg": twr,
                    "twist_tip_deg": twt,
                    "airfoil": af,
                    "arch_score": arch.score,
                    "W": float(W_val),            # <<<<<< carries W forward
                    # placeholders for Script 02 (optional quick filters)
                    "climb_ok": pd.NA,
                    "cruise_ok": pd.NA,
                    "LD_cruise": pd.NA,
                })

    df = pd.DataFrame(rows)
    df.to_csv(SWEEP_SUMMARY_CSV, index=False)
    print(f"Saved {len(df)} wing configurations to {SWEEP_SUMMARY_CSV}")
    print(df.head(12).to_string(index=False))

if __name__ == "__main__":
    run()
