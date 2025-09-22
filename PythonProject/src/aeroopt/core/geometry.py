import aerosandbox as asb
import aerosandbox.numpy as np

def wing_dims_from_AR_taper(S, AR, taper):
    b = np.sqrt(AR * S)
    c_root = 2 * S / (b * (1 + taper))
    c_tip = c_root * taper
    return b, c_root, c_tip

def c_mac_from_root_taper(c_root, taper):
    lam = taper
    return (2/3) * c_root * (1 + lam + lam**2) / (1 + lam)

def build_airplane(fam, AR_w, taper_w, dihedral_deg, twist_root, twist_tip,
                   tail_span_frac, tail_AR, boom_len, tail_inc_deg, de_deg,
                   Lf, df, S_ref_nom, airfoil_name="ag13"):
    b_w, c_root_w, c_tip_w = wing_dims_from_AR_taper(S_ref_nom, AR_w, taper_w)
    c_mac_w = c_mac_from_root_taper(c_root_w, taper_w)
    af_main = asb.Airfoil(airfoil_name.lower())
    wing = asb.Wing(
        name="Main Wing", symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0,0,0], chord=c_root_w, twist=twist_root, airfoil=af_main),
            asb.WingXSec(xyz_le=[0, b_w/2, np.tan(dihedral_deg*np.pi/180)*(b_w/2)],
                         chord=c_tip_w, twist=twist_tip, airfoil=af_main),
        ],
    )
    b_tail = tail_span_frac * b_w
    S_tail = (b_tail**2) / tail_AR
    lam_t = 0.8
    c_root_t = 2 * S_tail / (b_tail * (1 + lam_t))
    c_tip_t = c_root_t * lam_t
    af_tail = asb.Airfoil("naca0010")
    tail_twist = tail_inc_deg + de_deg
    htail = asb.Wing(
        name="H Tail", symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0,0,0], chord=c_root_t, twist=tail_twist, airfoil=af_tail),
            asb.WingXSec(xyz_le=[0.02, b_tail/2, 0.0], chord=c_tip_t, twist=tail_twist, airfoil=af_tail),
        ],
    ).translate([0.25 + boom_len, 0, 0])
    fuse = asb.Fuselage(
        name="Fuse",
        xsecs=[
            asb.FuselageXSec(xyz_c=[-0.10, 0, 0], radius=df/2),
            asb.FuselageXSec(xyz_c=[ 0.00, 0, 0], radius=df/2),
            asb.FuselageXSec(xyz_c=[ 0.25 + boom_len, 0, 0], radius=(df/2)*0.6),
        ],
    )
    airplane = asb.Airplane(
        name="multi_opt",
        s_ref=S_ref_nom,
        wings=[wing, htail],
        fuselages=[fuse],
        c_ref=c_mac_w,
        b_ref=b_w,
    )
    geom = dict(
        S_w=S_ref_nom, b_w=b_w, c_root_w=c_root_w, c_tip_w=c_tip_w, c_mac_w=c_mac_w,
        b_tail=b_tail, S_tail=S_tail, c_root_t=c_root_t, c_tip_t=c_tip_t, boom_length=boom_len
    )
    return airplane, geom
