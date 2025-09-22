import aerosandbox.numpy as np
from .geometry import c_mac_from_root_taper
from .physics import fuselage_cd, smooth_hinge

def tail_volume(geom, taper_w):
    lam = taper_w
    c_mac = c_mac_from_root_taper(geom["c_root_w"], lam)
    x_w_ac = 0.25 * c_mac
    x_t_ac = 0.25 + geom["boom_length"] + 0.25 * geom["c_root_t"]
    tail_arm = x_t_ac - x_w_ac
    return geom["S_tail"] * tail_arm / (geom["S_w"] * c_mac)

def objective_terms(cfg_mission, cfg_weights, geom, acr, acl, Lf, df):
    RHO = cfg_mission["rho"]; Vc = cfg_mission["V_cruise"]; Vl = cfg_mission["V_climb"]
    CD0_PAD = cfg_mission["cd0_pad"]; Sref = geom["S_w"]
    CDf_cr = fuselage_cd(RHO, cfg_mission["mu"], cfg_mission["S_ref_nom"], Lf, df, Vc)
    CDf_cl = fuselage_cd(RHO, cfg_mission["mu"], cfg_mission["S_ref_nom"], Lf, df, Vl)
    q_cr = 0.5 * RHO * Vc**2; q_cl = 0.5 * RHO * Vl**2
    D_cr = (acr["CD"] + CDf_cr + CD0_PAD) * q_cr * Sref
    D_cl = (acl["CD"] + CDf_cl + CD0_PAD) * q_cl * Sref
    return D_cr, D_cl

def manuf_penalty(w_man, taper_w, dih_w, c_tip_w, twr_w, twt_w):
    return w_man * (
        smooth_hinge(0.35 - taper_w)**2 +
        smooth_hinge(dih_w - 10.0)**2 +
        smooth_hinge(0.12 - c_tip_w)**2 +
        smooth_hinge(twr_w - 4.0)**2 +
        smooth_hinge(-1.0 - twt_w)**2
    )
