# src/aeroopt/script02_opt/run.py
from pathlib import Path
import math
import pandas as pd
import aerosandbox as asb
import aerosandbox.numpy as np

from aeroopt.core.config_io import load_yaml
from aeroopt.core.parsing import Family
from aeroopt.core.geometry import build_airplane
from aeroopt.core.aero import aero_at, configure_airfoils, ensure_polars_for_list
from aeroopt.core.constraints_obj import tail_volume, objective_terms, manuf_penalty

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
CACHE = ROOT / "cache"
OUTDIR = RESULTS / "script02"
OUTDIR.mkdir(parents=True, exist_ok=True)

TOP_CONFIGS_CSV = RESULTS / "script01" / "sweep_summary.csv"
CSV_OUT = OUTDIR / "multi_opt_summary.csv"
NUM_CONFIGS = 12


def main():
    mission = load_yaml("mission.yaml")
    bounds  = load_yaml("ranges.yaml")
    cons    = load_yaml("constraints.yaml")
    wts     = load_yaml("weights.yaml")
    solver  = load_yaml("solver.yaml")
    airf    = load_yaml("airfoils.yaml")

    configure_airfoils(airf, cache_dir=CACHE)

    df_cfg = pd.read_csv(TOP_CONFIGS_CSV)
    df_cfg.columns = df_cfg.columns.str.strip()
    if "W" not in df_cfg.columns:
        df_cfg["W"] = float(mission["W_fallback"])
    else:
        df_cfg["W"] = pd.to_numeric(df_cfg["W"], errors="coerce")

    need = sorted({str(a).strip() for a in df_cfg["airfoil"].dropna().tolist()}) if "airfoil" in df_cfg.columns else []
    ensure_polars_for_list(need)

    # take the first NUM_CONFIGS by arch_score, fam_name
    df_top = (df_cfg.sort_values(["arch_score", "fam_name"],
                                 ascending=[True, True],
                                 na_position="last", kind="mergesort")
                    .head(NUM_CONFIGS)
                    .reset_index(drop=True))

    rows = []
    for i, r in enumerate(df_top.itertuples(index=False), 1):
        cid, W_i = r.config_id, r.W
        print(f"[{i}/{len(df_top)}] optimizing {cid} ...")
        try:
            res = optimize_one(cid, mission, bounds, cons, wts, solver, airf, W_i)
        except Exception as e:
            print(f"[warn] {cid} failed: {e}")
            continue
        rows.append(res)

    if not rows:
        print("No results.")
        return

    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
    print(f"Saved: {CSV_OUT}")


def safe_weight(W_i, mission):
    """Return a finite float weight, falling back to mission['W_fallback'] if needed."""
    try:
        w = float(W_i)
        if not math.isfinite(w):
            raise ValueError
        return w
    except Exception:
        return float(mission["W_fallback"])


def optimize_one(config_id, mission, bounds, cons, wts, solver, airf, W_i):
    fam = Family.from_config_id(config_id)
    opti = asb.Opti()

    # --- decision variables ---
    AR_w    = opti.variable(init_guess=fam.AR,            lower_bound=bounds["wing"]["AR_scale"][0]*fam.AR,  upper_bound=bounds["wing"]["AR_scale"][1]*fam.AR)
    taper_w = opti.variable(init_guess=fam.taper,         lower_bound=bounds["wing"]["taper"][0],            upper_bound=bounds["wing"]["taper"][1])
    dih_w   = opti.variable(init_guess=fam.dihedral_deg,  lower_bound=bounds["wing"]["dihedral_deg"][0],     upper_bound=bounds["wing"]["dihedral_deg"][1])
    twr_w   = opti.variable(init_guess=fam.twist_root_deg,lower_bound=bounds["wing"]["twist_root_deg"][0],   upper_bound=bounds["wing"]["twist_root_deg"][1])
    twt_w   = opti.variable(init_guess=fam.twist_tip_deg, lower_bound=bounds["wing"]["twist_tip_deg"][0],    upper_bound=bounds["wing"]["twist_tip_deg"][1])

    ts      = opti.variable(init_guess=0.30,              lower_bound=bounds["tail"]["span_frac"][0],        upper_bound=bounds["tail"]["span_frac"][1])
    ta      = opti.variable(init_guess=3.5,               lower_bound=bounds["tail"]["AR"][0],               upper_bound=bounds["tail"]["AR"][1])
    Lb      = opti.variable(init_guess=0.40,              lower_bound=bounds["tail"]["boom_len_m"][0],       upper_bound=bounds["tail"]["boom_len_m"][1])
    it_deg  = opti.variable(init_guess=-2.0,              lower_bound=bounds["tail"]["incidence_deg"][0],    upper_bound=bounds["tail"]["incidence_deg"][1])

    de_cr   = opti.variable(init_guess=-3.0*np.pi/180,    lower_bound=bounds["controls"]["de_cr_deg"][0]*np.pi/180, upper_bound=bounds["controls"]["de_cr_deg"][1]*np.pi/180)
    de_cl   = opti.variable(init_guess=-2.0*np.pi/180,    lower_bound=bounds["controls"]["de_cl_deg"][0]*np.pi/180, upper_bound=bounds["controls"]["de_cl_deg"][1]*np.pi/180)
    a_cr    = opti.variable(init_guess=2.0,               lower_bound=bounds["controls"]["a_cr_deg"][0],     upper_bound=bounds["controls"]["a_cr_deg"][1])
    a_cl    = opti.variable(init_guess=3.0,               lower_bound=bounds["controls"]["a_cl_deg"][0],     upper_bound=bounds["controls"]["a_cl_deg"][1])

    Lf      = opti.variable(init_guess=0.55,              lower_bound=bounds["fuselage"]["Lf_m"][0],         upper_bound=bounds["fuselage"]["Lf_m"][1])
    df      = opti.variable(init_guess=0.09,              lower_bound=bounds["fuselage"]["df_m"][0],         upper_bound=bounds["fuselage"]["df_m"][1])

    # --- airplanes (cruise/climb differ in elevator only) ---
    ap_cr, geom = build_airplane(
        fam, AR_w, taper_w, dih_w, twr_w, twt_w,
        ts, ta, Lb, it_deg, de_cr*180/np.pi,
        Lf, df, mission["S_ref_nom"], fam.airfoil_name
    )
    ap_cl, _ = build_airplane(
        fam, AR_w, taper_w, dih_w, twr_w, twt_w,
        ts, ta, Lb, it_deg, de_cl*180/np.pi,
        Lf, df, mission["S_ref_nom"], fam.airfoil_name
    )

    # --- constraints ---
    Vh = tail_volume(geom, taper_w)
    opti.subject_to(Vh >= cons["tail_vol_min"])

    opti.subject_to(geom["b_w"] <= mission["b_max"])

    qVs = 0.5 * mission["rho"] * mission["V_stall_max"]**2
    W   = safe_weight(W_i, mission)
    opti.subject_to(mission["clmax_wing"] * qVs * ap_cr.s_ref >= W)

    acr = aero_at(ap_cr, mission["rho"], mission["V_cruise"], a_cr)
    acl = aero_at(ap_cl, mission["rho"], mission["V_climb"],  a_cl)
    opti.subject_to(acr["L"] == 0.95 * W)
    opti.subject_to(np.abs(acr["Cm"]) <= cons["trim_tol_cm"])
    opti.subject_to(acl["L"] == 1.00 * W)

    # Static margin band (simple analytic proxy)
    c_mac  = geom["c_mac_w"]
    x_w_ac = 0.25 * c_mac
    x_t_ac = (0.25 + geom["boom_length"]) + 0.25 * geom["c_root_t"]
    tail_arm = x_t_ac - x_w_ac
    G_t = 0.9 * (geom["S_tail"] * tail_arm) / (geom["S_w"] * c_mac)
    x_np = x_w_ac + G_t * c_mac

    def SM(cg_frac):
        return (x_np - cg_frac * c_mac) / c_mac

    opti.subject_to(SM(cons["cg_frac_forward"]) >= cons["sm_target"])
    opti.subject_to(SM(cons["cg_frac_aft"])     >= cons["sm_target"])

    # Headroom + tip stall/Re
    reserve = 3.0*np.pi/180
    opti.subject_to(np.abs(de_cr) <= cons["delta_e_cr_max_deg"]*np.pi/180 - reserve)
    opti.subject_to(np.abs(de_cl) <= cons["delta_e_cl_max_deg"]*np.pi/180 - reserve)
    opti.subject_to(twt_w <= cons["washout_min_deg"])
    opti.subject_to(a_cl + twt_w <= cons["tip_a_climb_cap_deg"])

    nu = mission["mu"] / mission["rho"]
    Re_tip = (mission["V_climb"] * geom["c_tip_w"]) / nu
    re_pen = wts["w_Re"] * 0.5 * ((cons["re_tip_min"] - Re_tip) + np.sqrt((cons["re_tip_min"] - Re_tip)**2 + 1e-6))

    a0 = 2*np.pi
    CL_tip_est = a0 * (a_cl + twt_w) * np.pi/180
    opti.subject_to(CL_tip_est <= 0.9 * cons["cl_tip_max"])

    # Fuselage usefulness + fineness
    V_bay = 0.6 * (np.pi * (df/2)**2 * Lf)
    opti.subject_to(V_bay >= cons["bay_volume_min_m3"])

    fineness = Lf / df
    fin_pen = wts["w_fin"] * (
        0.5*(fineness - cons["fineness_max"] + np.sqrt((fineness - cons["fineness_max"])**2 + 1e-6))**2 +
        0.5*(cons["fineness_min"] - fineness + np.sqrt((cons["fineness_min"] - fineness)**2 + 1e-6))**2
    )

    # --- objective ---
    D_cr, D_cl = objective_terms(mission, wts, geom, acr, acl, Lf, df)
    bend_pen = wts["w_bend"] * (geom["b_w"]**3)
    elev_pen = wts["w_elev"] * (de_cr**2 + de_cl**2)
    man_pen  = manuf_penalty(wts["w_man"], taper_w, dih_w, geom["c_tip_w"], twr_w, twt_w)
    reg = (wts["regularizers"]["ts"]*(ts-0.30)**2 +
           wts["regularizers"]["ta"]*(ta-3.5)**2 +
           wts["regularizers"]["Lb"]*(Lb-0.40)**2)
    reg += wts["regularizers"]["tikhonov"] * (AR_w**2 + taper_w**2 + dih_w**2 + twr_w**2 + twt_w**2 +
                                              ts**2 + ta**2 + Lb**2 + it_deg**2 + de_cr**2 + de_cl**2 +
                                              a_cr**2 + a_cl**2 + Lf**2 + df**2)

    J = D_cr + 0.25*D_cl + bend_pen + elev_pen + re_pen + fin_pen + man_pen + reg
    opti.minimize(J)

    # --- solver ---
    OUTDIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTDIR / f"ipopt_{config_id}.log"
    ip = solver["ipopt"]
    opti.solver("ipopt", {
        "print_time": False,
        "ipopt.max_iter": ip["max_iter"],
        "ipopt.tol": ip["tol"],
        "ipopt.acceptable_tol": ip["acceptable_tol"],
        "ipopt.acceptable_iter": ip["acceptable_iter"],
        "ipopt.sb": ip["sb"],
        "ipopt.print_level": ip["print_level"],
        "ipopt.output_file": str(log_path),
        "ipopt.file_print_level": ip["file_print_level"],
        "ipopt.print_frequency_iter": ip["print_frequency_iter"],
        "ipopt.hessian_approximation": ip["hessian_approximation"],
        "ipopt.limited_memory_max_history": ip["limited_memory_max_history"],
        "ipopt.mu_strategy": ip["mu_strategy"],
        "ipopt.nlp_scaling_method": ip["nlp_scaling_method"],
    })

    sol = opti.solve()

    def to_scalar(x):
        import numpy as _np
        a = _np.asarray(x)
        return float(a.reshape(-1)[0])

    def sval(x):
        return float(sol.value(x))

    # recompute for reporting
    apc, g = build_airplane(
        fam, sval(AR_w), sval(taper_w), sval(dih_w), sval(twr_w), sval(twt_w),
        sval(ts), sval(ta), sval(Lb), sval(it_deg), sval(de_cr)*180/np.pi,
        sval(Lf), sval(df), mission["S_ref_nom"], fam.airfoil_name
    )
    acr_v = aero_at(apc, mission["rho"], mission["V_cruise"], sval(a_cr))
    acl_v = aero_at(apc, mission["rho"], mission["V_climb"],  sval(a_cl))
    D_cr_v, D_cl_v = objective_terms(mission, wts, g, acr_v, acl_v, sval(Lf), sval(df))

    return dict(
        config_id=config_id,
        W=W_i,  # keep for traceability
        AR_w=sval(AR_w), taper_w=sval(taper_w), dihedral_w_deg=sval(dih_w),
        twist_root_w_deg=sval(twr_w), twist_tip_w_deg=sval(twt_w), b_w=float(g["b_w"]),
        tail_span_frac=sval(ts), tail_AR=sval(ta), boom_length_m=sval(Lb), tail_incidence_deg=sval(it_deg),
        delta_e_cr_deg=sval(de_cr)*180/np.pi, delta_e_cl_deg=sval(de_cl)*180/np.pi,
        Lf=sval(Lf), df=sval(df), fineness=float(sval(Lf)/sval(df)),
        alpha_cruise_deg=sval(a_cr), alpha_climb_deg=sval(a_cl),
        D_cruise=to_scalar(D_cr_v),
        D_climb=to_scalar(D_cl_v),
        score=to_scalar(D_cr_v + 0.25 * D_cl_v),
    )


if __name__ == "__main__":
    main()
