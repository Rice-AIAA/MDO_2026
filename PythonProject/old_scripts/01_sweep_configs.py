from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import pandas as pd

# -------------------------
# Paths & setup
# -------------------------
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ARCH_SELECTED_CSV = RESULTS_DIR / "arch_selected.csv"   # from Script-00
SWEEP_SUMMARY_CSV = RESULTS_DIR / "sweep_summary.csv"   # output for Script-02

# -------------------------
# Wing-family seeds (edit freely)
# -------------------------
# name, AR, taper, dihedral_deg, twist_root_deg, twist_tip_deg, airfoil
FAMILY_SEEDS: List[Tuple[str, float, float, float, float, float, str]] = [
    ("wingA", 8.5, 0.50, 10, 2, -1, "ag13"),
    ("wingA", 8.5, 0.50,  5, 2, -1, "ag13"),
    ("wingA", 8.5, 0.70, 10, 2, -1, "ag13"),
    ("wingA", 8.5, 0.70,  5, 2, -1, "ag13"),
    ("wingA", 9.5, 0.50, 10, 3, -2, "ag13"),
    ("wingA", 9.5, 0.50,  5, 3, -2, "ag13"),
    ("wingA", 9.5, 0.70, 10, 3, -2, "ag13"),
    ("wingA", 9.5, 0.70,  5, 3, -2, "ag13"),
    ("wingB", 7.5, 0.55, 10, 2, -2, "naca4412"),
    ("wingB", 7.5, 0.55,  5, 2, -2, "naca4412"),
    ("wingB", 8.5, 0.60, 10, 3, -1, "sg6043"),
    ("wingB", 8.5, 0.60,  5, 3, -1, "sg6043"),
]

TOP_K_ARCH = 10  # take the best N architectures from Script-00

# -------------------------
# Family parser (Script-02-compatible)
# -------------------------
@dataclass
class Family:
    name: str
    AR: float
    taper: float
    dihedral_deg: float
    twist_root_deg: float
    twist_tip_deg: float
    airfoil_name: str

    @staticmethod
    def from_config_id(config_id: str) -> "Family":
        # Accept both "{arch_id}__{wing...}" and "{wing...}" (fallback)
        if "__" in config_id:
            _, tail = config_id.split("__", 1)
        else:
            tail = config_id
        parts = tail.split("_")
        return Family(
            name=parts[0],
            AR=float(parts[1].replace("AR", "")),
            taper=float(parts[2].replace("tap", "")),
            dihedral_deg=float(parts[3].replace("dih", "")),
            twist_root_deg=float(parts[4].replace("tw", "")),
            twist_tip_deg=float(parts[5]),
            airfoil_name=parts[6],
        )

# -------------------------
# Helpers
# -------------------------
def load_architectures(k: int) -> pd.DataFrame:
    """
    Read selected architectures produced by Script-00.
    Returns the top-k by score (ascending).
    """
    if not ARCH_SELECTED_CSV.exists():
        raise FileNotFoundError(
            f"Missing {ARCH_SELECTED_CSV}. Run Script-00 to generate best architectures."
        )
    df = pd.read_csv(ARCH_SELECTED_CSV)
    # safeguard: if user saved more than needed, trim here
    df = df.sort_values("score").head(k).reset_index(drop=True)
    return df

def make_config_id(arch_id: str, seed: Tuple[str, float, float, float, float, float, str]) -> str:
    name, AR, tap, dih, twr, twt, af = seed
    return f"{arch_id}__{name}_AR{AR}_tap{tap}_dih{dih}_tw{twr}_{twt}_{af}"

# -------------------------
# Main sweep
# -------------------------
def main():
    arch_df = load_architectures(TOP_K_ARCH)

    rows = []
    for _, a in arch_df.iterrows():
        arch_id = a["arch_id"]  # e.g. arch-low-conv-raked-tractor-1eng-fixed
        for seed in FAMILY_SEEDS:
            cid = make_config_id(arch_id, seed)
            name, AR, tap, dih, twr, twt, af = seed

            rows.append({
                "config_id": cid,
                # split fields (handy for filtering/plotting later)
                "arch_id": arch_id,
                "wing_place": a.get("wing_place", None),
                "tail_type": a.get("tail_type", None),
                "wingtip": a.get("wingtip", None),
                "prop_layout": a.get("prop_layout", None),
                "engines": a.get("engines", None),
                "gear": a.get("gear", None),
                # wing seed
                "fam_name": name,
                "AR": AR,
                "taper": tap,
                "dihedral_deg": dih,
                "twist_root_deg": twr,
                "twist_tip_deg": twt,
                "airfoil": af,
                # optional: pass through architecture score for info
                "arch_score": a.get("score", None),
            })

    if not rows:
        print("No configurations generated. Check arch_selected.csv and FAMILY_SEEDS.")
        return

    df = pd.DataFrame(rows)

    # IMPORTANT: Script-02 optionally sorts by these columns if present.
    # We don’t have real aero yet, so leave them blank; Script-02 will default.
    for col in ["climb_ok", "cruise_ok", "LD_cruise"]:
        if col not in df.columns:
            df[col] = pd.NA

    df.to_csv(SWEEP_SUMMARY_CSV, index=False)
    print(f"Saved {len(df)} wing configurations to {SWEEP_SUMMARY_CSV}")
    with pd.option_context("display.max_rows", 20, "display.max_colwidth", None):
        print(df.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
