from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import aerosandbox as asb
import aerosandbox.numpy as np
import pandas as pd
import re

# -------------------------
# Paths & setup
# -------------------------
ROOT = Path(__file__).resolve().parent
DIR_IN = ROOT / "results"              # Script 1 JSONs
DIR_OUT = ROOT / "results_02b_multi" #TESTING 1
DIR_OUT.mkdir(parents=True, exist_ok=True)
CSV_OUT = DIR_OUT / "multi_opt_summary.csv"

DIR_CACHE = ROOT / "cache"
DIR_CACHE.mkdir(parents=True, exist_ok=True)

TOP_CONFIGS_CSV = ROOT / "results" / "sweep_summary.csv"
NUM_CONFIGS_TO_TAKE = 12

# -------------------------
# Environment / mission
# -------------------------
RHO = 1.225
MU  = 1.81e-5
S_REF_NOM = 0.25      # keep nominal S_ref (same as Script 1)
V_CRUISE  = 18.0
V_CLIMB   = 14.0
V_STALL_MAX = 11.0    # crude stall speed requirement
W_FALLBACK = 20.0

TARGET_LW_CRUISE = 0.95
TARGET_LW_CLIMB  = 1.00

# Drag "pads"
CD0_PAD = 0.003       # generic pad (profile odds/ends)

# Tail volume proxy requirement (can be raised by you later)
TAIL_VOL_MIN = 0.35

# Trim tolerances & authority
TRIM_TOL_CM = 0.02
DELTA_E_CR_MAX = 15.0 * np.pi / 180.0
DELTA_E_CL_MAX = 20.0 * np.pi / 180.0

# Span limit (m)
B_MAX = 1.50

# Wing CLmax (crude)
CLMAX_WING = 1.30

# -------------------------
# Polars / cache
# -------------------------
ALPHAS_SWEEP = np.linspace(-6.0, 18.0, 13)
XFOIL_PATH: Optional[str] = None

def ensure_polars(name: str):
    af = asb.Airfoil(name)
    cache_file = str(DIR_CACHE / f"{af.name}.json")
    try:
        af.generate_polars(cache_filename=cache_file, alphas=ALPHAS_SWEEP,
                           xfoil_command=XFOIL_PATH if XFOIL_PATH else None)
    except TypeError:
        kw = {}
        if XFOIL_PATH:
            kw["xfoil_executable"] = XFOIL_PATH
        af.generate_polars(cache_filename=cache_file, alphas=ALPHAS_SWEEP, **kw)

for af_name in ["ag13", "naca4412", "sg6043", "naca0010", "naca0008"]:
    try:
        ensure_polars(af_name)
    except Exception as e:
        print(f"[warn] polar gen for '{af_name}' failed: {e}. Will try cached.")

# -------------------------
# Utilities
# -------------------------
def load_top_configs(n: int) -> List[str]:
    p = TOP_CONFIGS_CSV
    if p.exists():
        df = pd.read_csv(p)
        # Prefer lower arch_score first; tiebreak by family (optional)
        sort_keys = ["arch_score"]
        ascending = [True]
        if "fam_name" in df.columns:
            sort_keys.append("fam_name")
            ascending.append(True)
        df = df.sort_values(sort_keys, ascending=ascending, kind="mergesort")
        return df["config_id"].tolist()[:n]
    ids = [q.stem.replace("config_", "") for q in sorted(DIR_IN.glob("config_*.json"))]
    return ids[:n]

def read_W_from_config(config_id: str) -> float:
    p = DIR_IN / f"config_{config_id}.json"
    if p.exists():
        try:
            data = json.loads(p.read_text())
            W_json = data.get("mission", {}).get("W", None)
            if W_json is not None:
                return float(W_json)
        except Exception as e:
            print(f"[warn] read W from {p.name} failed: {e}")
    return float(W_FALLBACK)

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
        """
        Accepts either:
          - 'wingA_AR8.5_tap0.5_dih10_tw2_-1_ag13'
          - 'arch-...__wingA_AR8.5_tap0.5_dih10_tw2_-1_ag13'
        """
        # Keep only the wing part after '__' if present
        wing_part = config_id.split("__", 1)[-1]

        # Expected tokens:
        # [0]=wingA, [1]=AR8.5, [2]=tap0.5, [3]=dih10, [4]=tw2, [5]=-1, [6]=ag13
        parts = wing_part.split("_")
        if len(parts) < 7:
            raise ValueError(f"Config ID not in expected form: {config_id}")

        try:
            name = parts[0]
            AR = float(parts[1].replace("AR", ""))
            taper = float(parts[2].replace("tap", ""))
            dihedral_deg = float(parts[3].replace("dih", ""))
            twist_root_deg = float(parts[4].replace("tw", ""))
            twist_tip_deg = float(parts[5])  # already plain number, may be negative
            airfoil_name = parts[6]
        except Exception as e:
            raise ValueError(f"Failed to parse config_id '{config_id}': {e}")

        return Family(
            name=name,
            AR=AR,
            taper=taper,
            dihedral_deg=dihedral_deg,
            twist_root_deg=twist_root_deg,
            twist_tip_deg=twist_tip_deg,
            airfoil_name=airfoil_name,
        )

# -------------------------
# Geometry & helpers
# -------------------------
def airfoil_from_name(name: str) -> asb.Airfoil:
    return asb.Airfoil(name.lower())

def wing_dims_from_AR_taper(S: np.ndarray, AR: np.ndarray, taper: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = np.sqrt(AR * S)
    c_root = 2 * S / (b * (1 + taper))
    c_tip  = c_root * taper
    return b, c_root, c_tip

def c_mac_from_root_taper(c_root: np.ndarray, taper: np.ndarray) -> np.ndarray:
    lam = taper
    return (2/3) * c_root * (1 + lam + lam**2) / (1 + lam)

def tail_volume(geom: Dict, taper_w: np.ndarray) -> np.ndarray:
    lam = taper_w
    c_mac = (2/3) * geom["c_root_w"] * (1 + lam + lam**2) / (1 + lam)
    x_w_ac = 0.25 * c_mac
    x_t_ac = 0.25 + geom["boom_length"] + 0.25 * geom["c_root_t"]
    tail_arm = x_t_ac - x_w_ac
    return geom["S_tail"] * tail_arm / (geom["S_w"] * c_mac)

def fuselage_cd(Lf: np.ndarray, df: np.ndarray, V: float) -> np.ndarray:
    # simple turbulent flat-plate estimate with a convex-ish form factor
    Swet = np.pi * df * Lf
    ReL = np.maximum(RHO * V * Lf / MU, 1e3)
    Cf  = 0.455 / (np.log10(ReL)**2.58 + 1e-9)
    FF  = 1.2 + 0.15 * (Lf / df - 5.0)**2
    return (Cf * FF * Swet) / S_REF_NOM

# ---- Smooth penalty helper (replaces np.fmax) ----
# Replace smooth_relu with this:
def smooth_hinge(x, eps: float = 1e-6):
    # Smooth approx to max(0, x), stable for huge |x|
    # = 0.5 * (x + sqrt(x^2 + eps))
    return 0.5 * (x + np.sqrt(x * x + eps))

def build_airplane_symbolic(
    fam: Family,
    # wing
    AR_w: np.ndarray, taper_w: np.ndarray, dihedral_w_deg: np.ndarray,
    twist_root_w: np.ndarray, twist_tip_w: np.ndarray,
    # tail
    tail_span_frac: np.ndarray, tail_AR: np.ndarray, boom_length: np.ndarray,
    tail_incidence_deg: np.ndarray, delta_e_deg: np.ndarray,
    # fuselage
    Lf: np.ndarray, df: np.ndarray,
) -> Tuple[asb.Airplane, Dict[str, np.ndarray]]:
    S_w = S_REF_NOM
    b_w, c_root_w, c_tip_w = wing_dims_from_AR_taper(S_w, AR_w, taper_w)
    c_mac_w = c_mac_from_root_taper(c_root_w, taper_w)

    af_main = airfoil_from_name(fam.airfoil_name)

    wing = asb.Wing(
        name="Main Wing",
        symmetric=True,
        xsecs=[
            asb.WingXSec(
                xyz_le=[0.0, 0.0, 0.0],
                chord=c_root_w,
                twist=twist_root_w,
                airfoil=af_main,
            ),
            asb.WingXSec(
                xyz_le=[0.0, b_w / 2, np.tan(dihedral_w_deg * np.pi / 180) * (b_w / 2)],
                chord=c_tip_w,
                twist=twist_tip_w,
                airfoil=af_main,
            ),
        ],
    )

    # Tail
    b_tail = tail_span_frac * b_w
    S_tail = (b_tail**2) / tail_AR
    lam_t  = 0.8
    c_root_t = 2 * S_tail / (b_tail * (1 + lam_t))
    c_tip_t  = c_root_t * lam_t
    af_tail = asb.Airfoil("naca0010")

    tail_twist = tail_incidence_deg + delta_e_deg  # elevator modeled as added incidence

    htail = asb.Wing(
        name="H Tail",
        symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0.0, 0.0, 0.0], chord=c_root_t, twist=tail_twist, airfoil=af_tail),
            asb.WingXSec(xyz_le=[0.02, b_tail/2, 0.0], chord=c_tip_t,  twist=tail_twist, airfoil=af_tail),
        ],
    ).translate([0.25 + boom_length, 0.0, 0.0])

    fuse = asb.Fuselage(
        name="Fuse",
        xsecs=[
            asb.FuselageXSec(xyz_c=[-0.10, 0, 0], radius=df/2),
            asb.FuselageXSec(xyz_c=[ 0.00, 0, 0], radius=df/2),
            asb.FuselageXSec(xyz_c=[ 0.25 + boom_length, 0, 0], radius=df/2*0.6),
        ],
    )

    airplane = asb.Airplane(
        name="multi_opt",
        s_ref=S_w,
        wings=[wing, htail],
        fuselages=[fuse],
        c_ref=c_mac_w,
        b_ref=b_w
    )

    geom = dict(
        S_w=S_w, b_w=b_w, c_root_w=c_root_w, c_tip_w=c_tip_w, c_mac_w=c_mac_w,
        b_tail=b_tail, S_tail=S_tail, c_root_t=c_root_t, c_tip_t=c_tip_t,
        boom_length=boom_length
    )
    return airplane, geom

def aero_at(airplane: asb.Airplane, V: float, alpha_deg: np.ndarray) -> Dict[str, np.ndarray]:
    op = asb.OperatingPoint(velocity=V, alpha=alpha_deg, beta=0, p=0, q=0, r=0)
    ab = asb.AeroBuildup(airplane=airplane, op_point=op)
    out = ab.run()
    CL = out.get("CL", out.get("cl"))
    CD = out.get("CD", out.get("cd"))
    Cm = out.get("Cm", out.get("CM", out.get("cm")))
    q  = 0.5 * RHO * V**2
    L  = CL * q * airplane.s_ref
    return {"CL": CL, "CD": CD, "Cm": Cm, "L": L}

# -------------------------
# Optimization per config
# -------------------------
def optimize_one_config(config_id: str) -> Optional[Dict]:
    fam = Family.from_config_id(config_id)
    W = read_W_from_config(config_id)

    opti = asb.Opti()

    # ---------- decision variables ----------
    # Wing refinement around family values
    AR_w   = opti.variable(init_guess=fam.AR,             lower_bound=0.8*fam.AR,  upper_bound=1.2*fam.AR)
    taper_w= opti.variable(init_guess=fam.taper,          lower_bound=0.40,        upper_bound=0.80)
    dih_w  = opti.variable(init_guess=fam.dihedral_deg,   lower_bound=3.0,         upper_bound=12.0)
    twr_w  = opti.variable(init_guess=fam.twist_root_deg, lower_bound=0.0,         upper_bound=5.0)
    twt_w  = opti.variable(init_guess=fam.twist_tip_deg,  lower_bound=-5.0,        upper_bound=0.0)

    # Tail
    ts     = opti.variable(init_guess=0.30, lower_bound=0.22, upper_bound=0.42)  # span fraction
    ta     = opti.variable(init_guess=3.50, lower_bound=2.6,  upper_bound=6.0)   # tail AR
    Lb     = opti.variable(init_guess=0.40, lower_bound=0.28, upper_bound=0.60)  # boom length
    it_deg = opti.variable(init_guess=-2.0, lower_bound=-8.0, upper_bound=8.0)   # tail incidence

    # Elevator deflections (per condition), radians
    de_cr  = opti.variable(init_guess=(-3.0*np.pi/180), lower_bound=-DELTA_E_CR_MAX, upper_bound=DELTA_E_CR_MAX)
    de_cl  = opti.variable(init_guess=(-2.0*np.pi/180), lower_bound=-DELTA_E_CL_MAX, upper_bound=DELTA_E_CL_MAX)

    # Angles of attack (per condition)
    a_cr   = opti.variable(init_guess=2.0, lower_bound=-2.0, upper_bound=16.0)    # deg
    a_cl   = opti.variable(init_guess=3.0, lower_bound= 0.0, upper_bound=18.0)

    # Fuselage
    Lf     = opti.variable(init_guess=0.55, lower_bound=0.40, upper_bound=0.80)   # m
    df     = opti.variable(init_guess=0.09, lower_bound=0.05, upper_bound=0.14)   # m

    # ---------- build two airplanes (cruise/climb: different elevator only) ----------
    ap_cr, geom = build_airplane_symbolic(
        fam, AR_w, taper_w, dih_w, twr_w, twt_w,
        ts, ta, Lb, it_deg, de_cr * 180 / np.pi,
        Lf, df
    )
    ap_cl, _ = build_airplane_symbolic(
        fam, AR_w, taper_w, dih_w, twr_w, twt_w,
        ts, ta, Lb, it_deg, de_cl * 180 / np.pi,
        Lf, df
    )

    # ---------- constraints (existing) ----------
    # Tail volume proxy
    Vh = tail_volume(
        dict(S_w=geom["S_w"], c_root_w=geom["c_root_w"], S_tail=geom["S_tail"],
             c_root_t=geom["c_root_t"], boom_length=geom["boom_length"]),
        taper_w
    )
    opti.subject_to(Vh >= TAIL_VOL_MIN)

    # Span cap
    opti.subject_to( geom["b_w"] <= B_MAX )

    # Stall speed: W <= q(Vs) * S * CLmax
    qVs = 0.5 * RHO * (V_STALL_MAX**2)
    opti.subject_to( CLMAX_WING * qVs * ap_cr.s_ref >= W )

    # Cruise: lift target & trim (using elevator)
    acr = aero_at(ap_cr, V_CRUISE, a_cr)
    opti.subject_to( acr["L"] == TARGET_LW_CRUISE * W )
    opti.subject_to( np.abs(acr["Cm"]) <= TRIM_TOL_CM )

    # Climb: lift target
    acl = aero_at(ap_cl, V_CLIMB,  a_cl)
    opti.subject_to( acl["L"] == TARGET_LW_CLIMB * W )

    # ---------- NEW realism: static margin across CG band ----------
    SM_target = 0.10   # 10% MAC target
    x_cg_frac_forward = 0.22
    x_cg_frac_aft     = 0.28

    c_mac = geom["c_mac_w"]
    x_w_ac = 0.25 * c_mac
    x_t_ac = (0.25 + geom["boom_length"]) + 0.25 * geom["c_root_t"]
    tail_arm = x_t_ac - x_w_ac

    k_t = 0.9
    G_t = k_t * (geom["S_tail"] * tail_arm) / (geom["S_w"] * c_mac)
    x_np = x_w_ac + G_t * c_mac

    def SM_from_frac(frac):
        x_cg = frac * c_mac
        return (x_np - x_cg) / c_mac

    SM_fwd = SM_from_frac(x_cg_frac_forward)
    SM_aft = SM_from_frac(x_cg_frac_aft)
    opti.subject_to(SM_fwd >= SM_target)
    opti.subject_to(SM_aft >= SM_target)

    # ---------- NEW realism: elevator headroom + soft penalty ----------
    reserve_rad = 3.0 * np.pi / 180.0
    opti.subject_to(np.abs(de_cr) <= (DELTA_E_CR_MAX - reserve_rad))
    opti.subject_to(np.abs(de_cl) <= (DELTA_E_CL_MAX - reserve_rad))

    w_elev = 0.002
    elev_pen = w_elev * (de_cr**2 + de_cl**2)

    # ---------- NEW realism: tip-stall / tip-Re ----------
    # (a) gentle washout guard + limit AoA at tip in climb
    opti.subject_to(twt_w <= -1.5)            # at least ~1.5° washout
    opti.subject_to(a_cl + twt_w <= 15.0)     # crude cap (deg)

    # (b) Re_tip penalty + crude CL_tip cap
    nu = MU / RHO
    Re_tip = (V_CLIMB * geom["c_tip_w"]) / nu
    Re_min = 1.5e5
    w_Re = 0.5
    re_pen = w_Re * smooth_hinge(Re_min - Re_tip)  # smooth instead of fmax

    a0 = 2 * np.pi  # per rad; crude
    CL_tip_est = a0 * ( (a_cl + twt_w) * np.pi / 180.0 )
    CLmax_tip = 1.1
    opti.subject_to(CL_tip_est <= 0.9 * CLmax_tip)

    # ---------- NEW realism: fuselage usefulness + fineness soft range ----------
    V_bay_min = 0.0015  # m^3 (~1.5 L); tune to mission
    V_bay = 0.6 * (np.pi * (df/2)**2 * Lf)  # 60% packing efficiency
    opti.subject_to(V_bay >= V_bay_min)

    fineness = Lf / df
    w_fin = 0.02
    fin_pen = w_fin * (
        smooth_hinge(fineness - 8.0)**2 +      # too slender
        smooth_hinge(4.0 - fineness)**2        # too stubby
    )

    # ---------- base objective pieces ----------
    q_cr = 0.5 * RHO * V_CRUISE**2
    q_cl = 0.5 * RHO * V_CLIMB**2
    Sref = ap_cr.s_ref

    CDf_cr = fuselage_cd(Lf, df, V_CRUISE)
    CDf_cl = fuselage_cd(Lf, df, V_CLIMB)

    D_cr = (acr["CD"] + CDf_cr + CD0_PAD) * q_cr * Sref
    D_cl = (acl["CD"] + CDf_cl + CD0_PAD) * q_cl * Sref

    # ---------- NEW realism: structural span cost (bending surrogate) ----------
    w_bend = 0.08  # try 0.02–0.20
    bend_pen = w_bend * (geom["b_w"]**3)

    # ---------- Manufacturability nudges (soft, smoothed) ----------
    w_man = 0.003
    man_pen = w_man * (
        smooth_hinge(0.35 - taper_w)**2 +      # avoid ultra-low taper
        smooth_hinge(dih_w - 10.0)**2 +        # too much dihedral
        smooth_hinge(0.12 - geom["c_tip_w"])**2 +  # prevent tiny tip chord
        smooth_hinge(twr_w - 4.0)**2 +         # discourage extreme root incidence
        smooth_hinge(-1.0 - twt_w)**2          # encourage at least ~1° washout
    )

    # small regularizers (keep well-conditioned, non-binding)
    reg = (
        0.02 * (ts - 0.30)**2
        + 0.01 * (ta - 3.5 )**2
        + 0.01 * (Lb - 0.40)**2
    )
    # tiny Tikhonov L2 to aid stationarity / conditioning
    reg += 1e-5 * (
        AR_w**2 + taper_w**2 + dih_w**2 + twr_w**2 + twt_w**2 +
        ts**2 + ta**2 + Lb**2 + it_deg**2 +
        de_cr**2 + de_cl**2 + a_cr**2 + a_cl**2 + Lf**2 + df**2
    )

    # ---------- final objective ----------
    J = D_cr + 0.25 * D_cl + bend_pen + elev_pen + re_pen + fin_pen + man_pen + reg
    opti.minimize(J)

    # ---------- solver options (quiet file logs, L-BFGS, more iters) ----------
    log_file = str(DIR_OUT / f"ipopt_{config_id}.log")
    opti.solver('ipopt', {
        "print_time": False,
        "ipopt.max_iter": 4000,
        "ipopt.tol": 1e-6,
        "ipopt.acceptable_tol": 5e-4,
        "ipopt.acceptable_iter": 10,
        "ipopt.sb": "yes",
        "ipopt.print_level": 0,
        "ipopt.output_file": log_file,
        "ipopt.file_print_level": 5,
        "ipopt.print_frequency_iter": 50,
        "ipopt.hessian_approximation": "limited-memory",
        "ipopt.limited_memory_max_history": 20,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.nlp_scaling_method": "gradient-based",
        # Derivative check (enable once if needed):
        # "ipopt.derivative_test": "first-order",
        # "ipopt.derivative_test_tol": 1e-6,
    })

    # ---------- solve ----------
    try:
        sol = opti.solve()
    except Exception as e:
        print(f"[warn] opt failed for {config_id}: {e}")
        return None

    # brief status line (no iteration spam)
    st = sol.stats()
    print(f"  -> status: {st.get('return_status')}, iters: {st.get('iter_count')}, "
          f"dual_inf: {st.get('dual_inf', 'n/a')}, constr_viol: {st.get('pr_inf', 'n/a')}")

    # ---------- extract scalars ----------
    def sval(x): return float(sol.value(x))

    AR_w_v   = sval(AR_w); taper_w_v = sval(taper_w); dih_w_v = sval(dih_w)
    twr_w_v  = sval(twr_w); twt_w_v  = sval(twt_w)
    ts_v     = sval(ts);    ta_v     = sval(ta);      Lb_v     = sval(Lb)
    it_v     = sval(it_deg)
    de_cr_v  = sval(de_cr); de_cl_v  = sval(de_cl)
    a_cr_v   = sval(a_cr);  a_cl_v   = sval(a_cl)
    Lf_v     = sval(Lf);    df_v     = sval(df)

    # Rebuild numerically to report clean numbers
    ap_cr_v, geom_v = build_airplane_symbolic(
        fam, AR_w_v, taper_w_v, dih_w_v, twr_w_v, twt_w_v,
        ts_v, ta_v, Lb_v, it_v, np.rad2deg(de_cr_v),
        Lf_v, df_v
    )
    ap_cl_v, _ = build_airplane_symbolic(
        fam, AR_w_v, taper_w_v, dih_w_v, twr_w_v, twt_w_v,
        ts_v, ta_v, Lb_v, it_v, np.rad2deg(de_cl_v),
        Lf_v, df_v
    )

    def to_scalar(x):
        import numpy as _np
        a = _np.asarray(x)
        return float(a.reshape(-1)[0])

    acr_v = aero_at(ap_cr_v, V_CRUISE, a_cr_v)
    acl_v = aero_at(ap_cl_v, V_CLIMB,  a_cl_v)

    CDf_cr_v = fuselage_cd(Lf_v, df_v, V_CRUISE)
    CDf_cl_v = fuselage_cd(Lf_v, df_v, V_CLIMB)

    D_cr_v = (to_scalar(acr_v["CD"]) + float(CDf_cr_v) + CD0_PAD) * (0.5 * RHO * V_CRUISE**2) * float(ap_cr_v.s_ref)
    D_cl_v = (to_scalar(acl_v["CD"]) + float(CDf_cl_v) + CD0_PAD) * (0.5 * RHO * V_CLIMB**2 ) * float(ap_cl_v.s_ref)

    Vh_v = float(tail_volume(
        dict(S_w=geom_v["S_w"], c_root_w=geom_v["c_root_w"], S_tail=geom_v["S_tail"],
             c_root_t=geom_v["c_root_t"], boom_length=geom_v["boom_length"]),
        taper_w_v
    ))

    # quick static margin report (forward end)
    c_mac_v = float(geom_v["c_mac_w"])
    x_w_ac_v = 0.25 * c_mac_v
    x_t_ac_v = (0.25 + float(geom_v["boom_length"])) + 0.25 * float(geom_v["c_root_t"])
    tail_arm_v = x_t_ac_v - x_w_ac_v
    G_t_v = float(0.9 * (geom_v["S_tail"] * tail_arm_v) / (geom_v["S_w"] * c_mac_v))
    x_np_v = x_w_ac_v + G_t_v * c_mac_v
    x_cg_v = 0.22 * c_mac_v
    SM_fwd_v = (x_np_v - x_cg_v) / c_mac_v

    return dict(
        config_id=config_id,
        # wing
        AR_w=AR_w_v, taper_w=taper_w_v, dihedral_w_deg=dih_w_v,
        twist_root_w_deg=twr_w_v, twist_tip_w_deg=twt_w_v,
        b_w=float(geom_v["b_w"]),
        # tail
        tail_span_frac=ts_v, tail_AR=ta_v, boom_length_m=Lb_v,
        tail_incidence_deg=it_v,
        delta_e_cr_deg=float(np.rad2deg(de_cr_v)),
        delta_e_cl_deg=float(np.rad2deg(de_cl_v)),
        Vh=Vh_v,
        # fuselage
        Lf=Lf_v, df=df_v, fineness=float(Lf_v/df_v),
        # performance
        alpha_cruise_deg=a_cr_v, alpha_climb_deg=a_cl_v,
        L_over_W_cruise=to_scalar(acr_v["L"]) / float(W),
        L_over_W_climb=to_scalar(acl_v["L"]) / float(W),
        Cm_cruise=to_scalar(acr_v["Cm"]),
        SM_fwd=SM_fwd_v,
        D_cruise=D_cr_v, D_climb=D_cl_v,
        score=D_cr_v + 0.25 * D_cl_v
    )

# -------------------------
# Main
# -------------------------
def main():
    cfg_ids = load_top_configs(NUM_CONFIGS_TO_TAKE)
    if not cfg_ids:
        print("No configs found.")
        return

    rows = []
    for i, cfg in enumerate(cfg_ids, 1):
        print(f"[{i}/{len(cfg_ids)}] optimizing {cfg} ...")
        res = optimize_one_config(cfg)
        if res is not None:
            rows.append(res)

    if not rows:
        print("No optimization results.")
        pd.DataFrame(columns=[
            "config_id",
            "AR_w","taper_w","dihedral_w_deg","twist_root_w_deg","twist_tip_w_deg","b_w",
            "tail_span_frac","tail_AR","boom_length_m","tail_incidence_deg",
            "delta_e_cr_deg","delta_e_cl_deg","Vh",
            "Lf","df","fineness",
            "alpha_cruise_deg","alpha_climb_deg",
            "L_over_W_cruise","L_over_W_climb","Cm_cruise","SM_fwd",
            "D_cruise","D_climb","score"
        ]).to_csv(CSV_OUT, index=False)
        return

    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUT, index=False)
    print(f"Saved: {CSV_OUT}")

    best = df.sort_values("score").reset_index(drop=True)
    with pd.option_context("display.max_columns", None):
        print("\nTop results (multi-condition optimization):")
        print(best.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
