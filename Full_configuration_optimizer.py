import aerosandbox as asb
import aerosandbox.numpy as np
import aerosandbox.tools.units as u
from pathlib import Path
import itertools, copy
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
from aerosandbox.tools.string_formatting import eng_string

print("Performing common setup...")

# --- Dirs
Path("cache").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

# --- Global switches
MAKE_PLOTS = True

# =============================
# Airfoil catalogs (enumerated)
# =============================
# Keep these short & low-Re friendly. You can tweak at will.
WING_AIRFOILS = ["ag35", "sd7037", "mh32", "e205", "naca4412"]
HTAIL_AIRFOILS = ["naca0008", "naca0010"]
VTAIL_AIRFOILS = ["naca0008", "naca0010"]

# Generate cached polars over relevant Re grids to avoid extrapolation noise.
# Choose a modest, representative grid; ASB interpolates between them.
def ensure_polars(af_name: str):
    af = asb.Airfoil(name=af_name)
    alphas = np.linspace(-10, 20, 31)
    try:
        # Try "Re" first (some builds support it)
        af.generate_polars(cache_filename=f"cache/{af_name}.json", alphas=alphas, Re=1.5e5)
    except TypeError:
        # Fall back to no Reynolds kwarg
        try:
            af.generate_polars(cache_filename=f"cache/{af_name}.json", alphas=alphas)
        except Exception as e:
            print(f"[warn] Could not generate polars for {af_name}: {e}")
    except Exception as e:
        print(f"[warn] Could not generate polars for {af_name}: {e}")
    return af

AIRFOIL_DB = {
    name: ensure_polars(name)
    for name in set(WING_AIRFOILS + HTAIL_AIRFOILS + VTAIL_AIRFOILS)
}

# =============================
# Last-year constants / ranges
# =============================
# Weight, density, etc. (same as MATLAB globals)
W_N = 15.5688                 # N (total weight)
rho = 1.293                   # kg/m^3
wing_area_density = 1.409     # kg/m^2 (weight/area)
WING_WEIGHT_LIMIT_LBF = 0.9   # lbf
WING_WEIGHT_LIMIT_N = WING_WEIGHT_LIMIT_LBF * 4.448
WING_WEIGHT_LIMIT_KG = WING_WEIGHT_LIMIT_N / 9.81

# Target flight condition from last year:
Vt_target = 15.0              # m/s
alpha_Vt_max_deg = 5.0

# Stability derivative target bands (from last year’s cost function)
Cm_alpha_band = (-0.05, -0.02)
Cn_beta_band  = ( 0.04,  0.10)
Cl_beta_band  = (-0.10, -0.02)

# Static margin requirement
SM_min = 0.08  # 8% MAC

# Fuselage length cap from your current script
Lf_total_max = 0.826  # m

# =============================
# Helper: build a 3-segment main wing
# Variables follow last-year bounds
# =============================
def build_parametric_main_wing(opti,
                               wing_span,               # variable
                               wing_root_chord,         # variable
                               wing_taper_start,        # variable [0.3, 0.7] span fraction to break
                               wing_taper_ratio,        # variable [0.3, 0.8]
                               wing_dihedral_deg,       # variable [0, 10]
                               wing_tip_offset,         # variable [0, 0.25] m (y-sweep lever)
                               wing_airfoil_name):
    """
    Three x-sections: root, break (at taper_start), tip.
    Uses wing_tip_offset to create a modest outer-panel sweep/LE offset.
    """
    af = AIRFOIL_DB[wing_airfoil_name]
    span_half = wing_span / 2
    y_break = span_half * wing_taper_start
    y_tip   = span_half
    chord_root = wing_root_chord
    chord_tip  = wing_root_chord * wing_taper_ratio

    # Use wing_tip_offset in x and a simple dihedral on the outer panel
    xyz_le_root  = np.array([-0.25 * chord_root, 0.0, 0.0])
    xyz_le_break = np.array([-0.25 * chord_root, y_break, 0.0])
    xyz_le_tip   = np.array([
        xyz_le_break[0] + wing_tip_offset,
        y_tip,
        (y_tip - y_break) * np.tand(wing_dihedral_deg)
    ])

    wing = asb.Wing(
        name="Main Wing",
        symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=xyz_le_root,  chord=chord_root,            airfoil=af),
            asb.WingXSec(xyz_le=xyz_le_break, chord=chord_root,            airfoil=af),
            asb.WingXSec(xyz_le=xyz_le_tip,   chord=chord_tip,  airfoil=af),
        ]
    )
    return wing

# =============================
# Common optimization core
# =============================
def optimize_architecture(config_name: str,
                          make_airplane_fn,            # fn(opti, vars, airfoils) -> (airplane, mass_props, x_nose, x_tail, wing_ref)
                          wing_airfoil_name: str,
                          htail_airfoil_name: str | None,
                          vtail_airfoil_name: str | None,
                          sm_min: float = SM_min,
                          sm_for_flying_wing: float = 0.05  # your script used 5% for flying wing
                          ):
    """
    Runs AeroSandbox optimization for a single architecture with a given airfoil combo.
    Returns (sol, report_dict) on success; returns None on infeasible.
    """
    opti = asb.Opti()

    # ---------- Decision variables (last-year ranges) ----------
    # Wing
    wing_span         = opti.variable(init_guess=0.85, lower_bound=0.762, upper_bound=0.900)    # m
    wing_root_chord   = opti.variable(init_guess=0.22, lower_bound=0.150, upper_bound=0.350)    # m
    wing_taper_start  = opti.variable(init_guess=0.50, lower_bound=0.300, upper_bound=0.700)    # y/span break
    wing_taper_ratio  = opti.variable(init_guess=0.55, lower_bound=0.300, upper_bound=0.800)    # ct/croot
    wing_dihedral_deg = opti.variable(init_guess=5.0,  lower_bound=0.0,   upper_bound=10.0)     # deg
    wing_tip_offset   = opti.variable(init_guess=0.10, lower_bound=0.0,   upper_bound=0.25)     # m

    # Fuselage key stations (nose & tail), and we respect total length cap
    x_nose = opti.variable(init_guess=-0.10, upper_bound=0.0)
    x_tail = opti.variable(init_guess= 0.55, lower_bound=0.10)
    opti.subject_to(x_tail - x_nose <= Lf_total_max)

    # Tail layout knobs (map last-year ratios into positions/size)
    # Horizontal tail
    htail_x_ratio   = opti.variable(init_guess=0.75, lower_bound=0.50, upper_bound=0.90)
    htail_span      = opti.variable(init_guess=0.20, lower_bound=0.10, upper_bound=0.25)  # m (absolute, matches last-year "spans")
    htail_aoa_deg   = opti.variable(init_guess=-1.0, lower_bound=-5.0, upper_bound=5.0)
    htail_chord     = opti.variable(init_guess=0.12, lower_bound=0.10, upper_bound=0.15)
    htail_y_ratio   = opti.variable(init_guess=0.70, lower_bound=0.40, upper_bound=0.90)  # spanwise placement fraction for single-fin case
    # Vertical tail
    vtail_chord     = opti.variable(init_guess=0.12, lower_bound=0.10, upper_bound=0.15)
    vtail_height    = opti.variable(init_guess=0.12, lower_bound=0.10, upper_bound=0.15)
    vtail_offset    = opti.variable(init_guess=0.05, lower_bound=0.00, upper_bound=0.10)  # used as a lateral/longitudinal tweak
    vtail_x_ratio   = opti.variable(init_guess=0.75, lower_bound=0.50, upper_bound=0.90)

    # Ballast location & mass (your script had these)
    ballast_mass = opti.variable(init_guess=0.0, lower_bound=0.0)
    ballast_xcg  = opti.variable(init_guess=0.0, lower_bound=x_nose, upper_bound=x_tail)

    # Operating points
    op_cruise = asb.OperatingPoint(
        velocity=opti.variable(init_guess=14.0, lower_bound=1.0, log_transform=True),
        alpha=opti.variable(init_guess=2.0, lower_bound=-10.0, upper_bound=20.0)
    )
    op_Vt = asb.OperatingPoint(
        velocity=Vt_target,
        alpha=opti.variable(init_guess=2.0, lower_bound=-10.0, upper_bound=alpha_Vt_max_deg)
    )

    # ---------- Build geometry ----------
    wing = build_parametric_main_wing(opti,
                                      wing_span, wing_root_chord, wing_taper_start,
                                      wing_taper_ratio, wing_dihedral_deg, wing_tip_offset,
                                      wing_airfoil_name)

    # Delegate to architecture-specific builder to create: airplane, mass_props dict, reference wing pointer
    airplane, mass_props, wing_ref = make_airplane_fn(
        opti, wing, x_nose, x_tail,
        htail_x_ratio, htail_span, htail_aoa_deg, htail_chord, htail_y_ratio, htail_airfoil_name,
        vtail_chord, vtail_height, vtail_offset, vtail_x_ratio, vtail_airfoil_name
    )

    # Mass bookkeeping (add ballast)
    mass_props["ballast"] = asb.MassProperties(mass=ballast_mass, x_cg=ballast_xcg)
    # Glue weight approximation (same pattern you used)
    mass_sum_pre_glue = sum(v for k, v in mass_props.items() if k != "ballast")
    mass_props["glue_weight"] = asb.MassProperties(mass=mass_sum_pre_glue.mass * 0.08)
    mass_props_TOGW = sum(mass_props.values())

    # ---------- Aero & stability at two op points ----------
    ab_cr = asb.AeroBuildup(airplane=airplane, op_point=op_cruise, xyz_ref=mass_props_TOGW.xyz_cg)
    aero_cr = ab_cr.run_with_stability_derivatives(alpha=True, beta=True)
    ab_vt = asb.AeroBuildup(airplane=airplane, op_point=op_Vt, xyz_ref=mass_props_TOGW.xyz_cg)
    aero_vt = ab_vt.run_with_stability_derivatives(alpha=True, beta=False)

    # Static margin at cruise
    SM = (aero_cr["x_np"] - mass_props_TOGW.x_cg) / wing_ref.mean_aerodynamic_chord()

    # Wing weight constraint (area-density model, same as last year)
    wing_mass_from_area = wing_ref.area() * wing_area_density  # kg
    opti.subject_to(wing_mass_from_area <= WING_WEIGHT_LIMIT_KG)

    # Lift = Weight (cruise & Vt), trim & SM
    opti.subject_to(aero_cr["L"] >= 9.81 * mass_props_TOGW.mass)    # allow small reserve
    opti.subject_to(aero_cr["Cm"] == 0.0)
    # SM requirement (flying wing can use smaller requirement)
    sm_req = sm_for_flying_wing if ("Flying" in config_name or "Wing" in config_name) else sm_min
    opti.subject_to(SM >= sm_req)

    opti.subject_to(aero_vt["L"] >= 9.81 * mass_props_TOGW.mass)
    # alpha_Vt already upper-bounded by alpha_Vt_max_deg via variable upper_bound

    # Stability derivative bands at cruise
    # Note: ASB names: "Cma", "Cnb", "Clb" (derivatives w.r.t alpha or beta)
    opti.subject_to(aero_cr["Cma"] <= Cm_alpha_band[1])
    opti.subject_to(aero_cr["Cma"] >= Cm_alpha_band[0])
    opti.subject_to(aero_cr["Cnb"] >= Cn_beta_band[0])
    opti.subject_to(aero_cr["Cnb"] <= Cn_beta_band[1])
    opti.subject_to(aero_cr["Clb"] >= Cl_beta_band[0])
    opti.subject_to(aero_cr["Clb"] <= Cl_beta_band[1])

    # Fuselage length limit already enforced via x_tail - x_nose

    # ---------- Objective ----------
    # Keep it close to last year’s “performance” part but with clean terms:
    #   - Minimize sink rate at cruise (proxy for power required / endurance)
    #   - Small mass regularization (discourages overweight solutions, but hard wing-weight limit already applies)
    #   - Keep alpha at Vt small (already constrained, but add margin in objective)
    sink_rate_cr = (aero_cr["D"] * op_cruise.velocity) / (9.81 * mass_props_TOGW.mass)
    # Penalty on alpha at Vt to favor margin below the 5 deg cap
    alpha_margin = (op_Vt.alpha / alpha_Vt_max_deg) ** 2
    # Light L/D encouragement (equivalent to drag minimization)
    LD_cr = aero_cr["L"] / aero_cr["D"]
    J = sink_rate_cr / 0.30 + 0.1 * (mass_props_TOGW.mass / 0.100) + 0.1 * alpha_margin - 0.02 * LD_cr

    opti.minimize(J)

    # ---------- Solve ----------
    try:
        sol = opti.solve()
    except Exception as e:
        print(f"[{config_name}] infeasible or failed: {e}")
        return None

    # ---------- Reporting ----------
    s = lambda x: sol.value(x)

    # Optional AVL cross-check (kept from your script)
    avl_aero = {}
    try:
        avl_airplane = copy.deepcopy(sol(airplane))
        if avl_airplane.wings:
            w = avl_airplane.wings[0]
            if len(w.xsecs) > 2:
                keep = np.arange(len(w.xsecs)) % 2 == 0
                keep[0] = True; keep[-1] = True
                w.xsecs = np.array(w.xsecs)[keep]
        avl_aero = asb.AVL(airplane=avl_airplane, op_point=sol(op_cruise), xyz_ref=sol(mass_props_TOGW.xyz_cg)).run()
    except Exception as e:
        avl_aero = {"note": f"AVL failed: {e}"}

    report = dict(
        config=config_name,
        wing_af=wing_airfoil_name,
        htail_af=htail_airfoil_name,
        vtail_af=vtail_airfoil_name,
        span=s(wing_span),
        c_root=s(wing_root_chord),
        taper_ratio=s(wing_taper_ratio),
        dihedral_deg=s(wing_dihedral_deg),
        weight_kg=float(sol(mass_props_TOGW.mass)),
        SM=float(sol(SM)),
        LD_cr=float(sol(LD_cr)),
        sink_rate=float(sol(sink_rate_cr)),
        alpha_cr=float(sol(op_cruise.alpha)),
        alpha_Vt=float(sol(op_Vt.alpha)),
        Cma=float(sol(aero_cr["Cma"])),
        Cnb=float(sol(aero_cr["Cnb"])),
        Clb=float(sol(aero_cr["Clb"])),
        J=float(sol(J)),
        avl=avl_aero
    )

    if MAKE_PLOTS:
        plot_full_report(config_name, sol, airplane, op_cruise, mass_props, mass_props_TOGW)

    return sol, report


def plot_full_report(config_name, sol, airplane, op_point, mass_props, mass_props_TOGW):
    airplane = sol(airplane)
    op_point = sol(op_point)
    mass_props = sol(mass_props)
    mass_props_TOGW = sol(mass_props_TOGW)

    alpha_range = np.linspace(-15, 15, 100)
    aero_polars = asb.AeroBuildup(
        airplane=airplane,
        op_point=asb.OperatingPoint(velocity=op_point.velocity, alpha=alpha_range)
    ).run()

    fig = plt.figure(figsize=(24, 16))
    fig.suptitle(f"{config_name.replace('_',' ')}: Full Report", fontsize=24, y=0.98)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.2])

    gs_left = gs[:, 0].subgridspec(2, 2)
    ax_top = fig.add_subplot(gs_left[0, 0], projection='3d');   ax_top.set_title("Top View")
    ax_front = fig.add_subplot(gs_left[1, 0], projection='3d'); ax_front.set_title("Front View")
    ax_side = fig.add_subplot(gs_left[0, 1], projection='3d');  ax_side.set_title("Side View")
    ax_3d = fig.add_subplot(gs_left[1, 1], projection='3d');    ax_3d.set_title("3D Perspective")
    airplane.draw(ax=ax_top, show=False);   ax_top.view_init(90,-90);   ax_top.set_axis_off()
    airplane.draw(ax=ax_front, show=False); ax_front.view_init(0,-90);  ax_front.set_axis_off()
    airplane.draw(ax=ax_side, show=False);  ax_side.view_init(0,-180);  ax_side.set_axis_off()
    airplane.draw(ax=ax_3d, show=False)

    ax_pie = fig.add_subplot(gs[0, 1])
    name_remaps = {k: k.replace("_", " ").title() for k in mass_props.keys()}
    mplot = {k: v for k, v in mass_props.items() if v.mass > 1e-9}
    if "ballast" in mplot and mplot["ballast"].mass < 1e-6: del mplot["ballast"]
    plt.sca(ax_pie)
    p.pie(
        values=[v.mass for v in mplot.values()],
        names=[name_remaps.get(n, n) for n in mplot.keys()],
        center_text=f"$\\bf{{{config_name.replace('_',' ')}}}$\nTOGW: {mass_props_TOGW.mass * 1e3:.1f} g",
        startangle=110, arm_length=30, arm_radius=20, y_max_labels=1.1
    )
    ax_pie.set_title("Mass Budget", fontsize=18, pad=20)

    gs_aero = gs[1, 1].subgridspec(2, 2, wspace=0.25, hspace=0.4)
    ax_cl = fig.add_subplot(gs_aero[0, 0]); ax_cd = fig.add_subplot(gs_aero[0, 1])
    ax_cm = fig.add_subplot(gs_aero[1, 0]); ax_ld = fig.add_subplot(gs_aero[1, 1])
    ax_cl.plot(alpha_range, aero_polars["CL"]); ax_cl.set_xlabel("alpha [deg]"); ax_cl.set_ylabel("CL"); ax_cl.grid(True)
    ax_cd.plot(alpha_range, aero_polars["CD"]); ax_cd.set_xlabel("alpha [deg]"); ax_cd.set_ylabel("CD"); ax_cd.set_ylim(bottom=0); ax_cd.grid(True)
    ax_cm.plot(alpha_range, aero_polars["Cm"]); ax_cm.set_xlabel("alpha [deg]"); ax_cm.set_ylabel("Cm"); ax_cm.grid(True)
    ax_ld.plot(alpha_range, aero_polars["CL"]/aero_polars["CD"]); ax_ld.set_xlabel("alpha [deg]"); ax_ld.set_ylabel("L/D"); ax_ld.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    p.show_plot(show=True, savefig=f"figures/{config_name}_full_report.png")

# =============================
# Architecture builders
# =============================
def _mass_from_volume(vol, density_kg_m3):
    return vol * density_kg_m3

def build_common_fuselage(x_nose, x_tail):
    # Simple “pod + boom” like your original
    return asb.Fuselage(
        name="Fuse",
        xsecs=[
            asb.FuselageXSec(xyz_c=[x_nose, 0, 0], height=1.5 * u.inch, width=2.0 * u.inch),
            asb.FuselageXSec(xyz_c=[x_nose + 8 * u.inch, 0, 0], height=1.5 * u.inch, width=2.0 * u.inch),
            asb.FuselageXSec(xyz_c=[x_nose + 8 * u.inch, 0, 0], radius=7e-3 / 2),
            asb.FuselageXSec(xyz_c=[x_tail, 0, 0], radius=7e-3 / 2),
        ]
    )

def make_airplane_conventional(opti, wing, x_nose, x_tail,
                               htail_x_ratio, htail_span, htail_aoa_deg, htail_chord, htail_y_ratio, htail_af_name,
                               vtail_chord, vtail_height, vtail_offset, vtail_x_ratio, vtail_af_name):
    # Conventional: H-tail + single vertical fin (this builder is used only for the "Conventional_Tail" sweep)
    h_af = AIRFOIL_DB[htail_af_name]; v_af = AIRFOIL_DB[vtail_af_name]
    x_ht = x_nose + htail_x_ratio * (x_tail - x_nose)

    # Horizontal tail with rigid incidence via per-section twist
    htail = asb.Wing(
        name="Horizontal Stabilizer",
        symmetric=True,
        xsecs=[
            asb.WingXSec(
                xyz_le=[0.00, 0.0, 0.0],
                chord=htail_chord,
                airfoil=h_af,
                twist=htail_aoa_deg,  # incidence
            ),
            asb.WingXSec(
                xyz_le=[0.02, htail_span / 2, 0.0],
                chord=htail_chord * 0.8,
                airfoil=h_af,
                twist=htail_aoa_deg,  # same incidence
            ),
        ],
    ).translate([x_ht, 0, 0])

    # Single vertical fin — give it a tiny local span to avoid singular frame
    x_vt = x_nose + vtail_x_ratio * (x_tail - x_nose)
    eps_y = 1e-5  # tiny span to keep y2 != y1
    vtail = asb.Wing(
        name="Vertical Stabilizer",
        symmetric=False,
        xsecs=[
            asb.WingXSec(xyz_le=[0.00, 0.0,   0.0],            chord=vtail_chord,       airfoil=v_af),
            asb.WingXSec(xyz_le=[0.03, eps_y, vtail_height],  chord=vtail_chord * 0.7, airfoil=v_af),
        ],
    ).translate([x_vt, 0, 0])

    fuselage = build_common_fuselage(x_nose, x_tail)

    airplane = asb.Airplane(
        name="Conventional",
        wings=[wing, htail, vtail],
        fuselages=[fuselage]
    )

    # Mass properties (same style as before)
    density_wing_solid = 2 * u.lbm / u.foot ** 3
    mass_props = {}
    mass_props["wing"] = asb.mass_properties_from_radius_of_gyration(
        mass=_mass_from_volume(wing.volume(), density_wing_solid), x_cg=0
    )
    tail_total_vol = htail.volume() + vtail.volume()
    mass_props["tail_surfaces"] = asb.mass_properties_from_radius_of_gyration(
        mass=_mass_from_volume(tail_total_vol, 80), x_cg=x_ht + 0.5 * htail_chord
    )
    mass_props["motor"]          = asb.mass_properties_from_radius_of_gyration(mass=4.49e-3, x_cg=x_nose - 0.3 * u.inch)
    mass_props["motor_bolts"]    = asb.mass_properties_from_radius_of_gyration(mass=4 * 0.075e-3, x_cg=x_nose)
    mass_props["propeller"]      = asb.mass_properties_from_radius_of_gyration(mass=1.54e-3, x_cg=x_nose - 0.7 * u.inch)
    mass_props["propeller_band"] = asb.mass_properties_from_radius_of_gyration(mass=0.06e-3, x_cg=x_nose - 0.7 * u.inch)
    mass_props["flight_computer"]= asb.mass_properties_from_radius_of_gyration(mass=4.30e-3, x_cg=x_nose + 2 * u.inch + 1.3 * u.inch / 2)
    mass_props["battery"]        = asb.mass_properties_from_radius_of_gyration(mass=4.61e-3, x_cg=x_nose + 2 * u.inch)

    return airplane, mass_props, wing

def make_airplane_htail_twinfins(opti, wing, x_nose, x_tail,
                                 htail_x_ratio, htail_span, htail_aoa_deg, htail_chord, htail_y_ratio, htail_af_name,
                                 vtail_chord, vtail_height, vtail_offset, vtail_x_ratio, vtail_af_name):
    h_af = AIRFOIL_DB[htail_af_name]; v_af = AIRFOIL_DB[vtail_af_name]

    x_ht = x_nose + htail_x_ratio * (x_tail - x_nose)

    # Horizontal tail with incidence via per-section twist
    htail = asb.Wing(
        name="Horizontal Stabilizer", symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0.00, 0.0,            0.0], chord=htail_chord,      airfoil=h_af, twist=htail_aoa_deg),
            asb.WingXSec(xyz_le=[0.02, htail_span/2,   0.0], chord=htail_chord * 0.8, airfoil=h_af, twist=htail_aoa_deg),
        ]
    ).translate([x_ht, 0, 0])

    # Two vertical fins at the H-tail tips (± span/2)
    fin_x_le_offset = 0.01
    eps_y = 1e-5  # tiny local span to avoid singular local frame

    vt_right = asb.Wing(
        name="Right Vertical Stabilizer", symmetric=False,
        xsecs=[
            asb.WingXSec(xyz_le=[0.00,   0.0,        0.0],           chord=vtail_chord,       airfoil=v_af),
            asb.WingXSec(xyz_le=[0.03, eps_y, vtail_height],         chord=vtail_chord * 0.7, airfoil=v_af),
        ]
    ).translate([x_ht + fin_x_le_offset, +htail_span/2, 0.0])

    vt_left = vt_right.translate([0.0, -htail_span, 0.0]); vt_left.name = "Left Vertical Stabilizer"

    fuselage = build_common_fuselage(x_nose, x_tail)

    airplane = asb.Airplane(
        name="H-Tail (Twin Fins)",
        wings=[wing, htail, vt_right, vt_left],
        fuselages=[fuselage]
    )

    density_wing_solid = 2 * u.lbm / u.foot ** 3
    mass_props = {}
    mass_props["wing"] = asb.mass_properties_from_radius_of_gyration(
        mass=_mass_from_volume(wing.volume(), density_wing_solid), x_cg=0
    )
    tail_total_vol = htail.volume() + vt_right.volume() + vt_left.volume()
    mass_props["tail_surfaces"] = asb.mass_properties_from_radius_of_gyration(
        mass=_mass_from_volume(tail_total_vol, 80), x_cg=x_ht + 0.5 * htail_chord
    )
    mass_props["motor"] = asb.mass_properties_from_radius_of_gyration(mass=4.49e-3, x_cg=x_nose - 0.3 * u.inch)
    mass_props["motor_bolts"] = asb.mass_properties_from_radius_of_gyration(mass=4*0.075e-3, x_cg=x_nose)
    mass_props["propeller"] = asb.mass_properties_from_radius_of_gyration(mass=1.54e-3, x_cg=x_nose - 0.7 * u.inch)
    mass_props["propeller_band"] = asb.mass_properties_from_radius_of_gyration(mass=0.06e-3, x_cg=x_nose - 0.7 * u.inch)
    mass_props["flight_computer"] = asb.mass_properties_from_radius_of_gyration(mass=4.30e-3, x_cg=x_nose + 2*u.inch + 1.3*u.inch/2)
    mass_props["battery"] = asb.mass_properties_from_radius_of_gyration(mass=4.61e-3, x_cg=x_nose + 2*u.inch)

    return airplane, mass_props, wing

def make_airplane_vtail(opti, wing, x_nose, x_tail,
                        htail_x_ratio, htail_span, htail_aoa_deg, htail_chord, htail_y_ratio, htail_af_name,
                        vtail_chord, vtail_height, vtail_offset, vtail_x_ratio, vtail_af_name):
    v_af = AIRFOIL_DB[vtail_af_name]
    x_t = x_nose + htail_x_ratio * (x_tail - x_nose)

    vtail_dihedral_deg = 37.0
    vtail = asb.Wing(
        name="V-Tail", symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0, 0, 0], chord=htail_chord,      airfoil=v_af, twist=htail_aoa_deg),
            asb.WingXSec(xyz_le=[0.02, htail_span/2, (htail_span/2)*np.tand(vtail_dihedral_deg)],
                         chord=htail_chord*0.8, airfoil=v_af, twist=htail_aoa_deg),
        ]
    ).translate([x_t, 0, 0])

    fuselage = build_common_fuselage(x_nose, x_tail)

    airplane = asb.Airplane(
        name="V-Tail",
        wings=[wing, vtail],
        fuselages=[fuselage]
    )

    density_wing_solid = 2 * u.lbm / u.foot ** 3
    mass_props = {}
    mass_props["wing"] = asb.mass_properties_from_radius_of_gyration(
        mass=_mass_from_volume(wing.volume(), density_wing_solid), x_cg=0
    )
    mass_props["tail_surfaces"] = asb.mass_properties_from_radius_of_gyration(
        mass=_mass_from_volume(vtail.volume(), 80), x_cg=x_t + 0.5 * htail_chord
    )
    mass_props["motor"] = asb.mass_properties_from_radius_of_gyration(mass=4.49e-3, x_cg=x_nose - 0.3 * u.inch)
    mass_props["motor_bolts"] = asb.mass_properties_from_radius_of_gyration(mass=4*0.075e-3, x_cg=x_nose)
    mass_props["propeller"] = asb.mass_properties_from_radius_of_gyration(mass=1.54e-3, x_cg=x_nose - 0.7 * u.inch)
    mass_props["propeller_band"] = asb.mass_properties_from_radius_of_gyration(mass=0.06e-3, x_cg=x_nose - 0.7 * u.inch)
    mass_props["flight_computer"] = asb.mass_properties_from_radius_of_gyration(mass=4.30e-3, x_cg=x_nose + 2*u.inch + 1.3*u.inch/2)
    mass_props["battery"] = asb.mass_properties_from_radius_of_gyration(mass=4.61e-3, x_cg=x_nose + 2*u.inch)

    return airplane, mass_props, wing

def make_airplane_reverse_v(opti, wing, x_nose, x_tail,
                            htail_x_ratio, htail_span, htail_aoa_deg, htail_chord, htail_y_ratio, htail_af_name,
                            vtail_chord, vtail_height, vtail_offset, vtail_x_ratio, vtail_af_name):
    v_af = AIRFOIL_DB[vtail_af_name]
    x_t = x_nose + htail_x_ratio * (x_tail - x_nose)

    v_tail_anhedral_deg = 45.0
    v_tail_sweep_deg = 20.0
    v_tail_taper = 0.7

    reverse_v_tail = asb.Wing(
        name="Reverse V-Tail", symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0, 0, 0], chord=htail_chord, airfoil=v_af, twist=htail_aoa_deg),
            asb.WingXSec(
                xyz_le=[(htail_span/2)*np.tand(v_tail_sweep_deg),  htail_span/2, -(htail_span/2)*np.tand(v_tail_anhedral_deg)],
                chord=htail_chord * v_tail_taper, airfoil=v_af, twist=htail_aoa_deg
            )
        ]
    ).translate([x_t, 0, (htail_span/2)*np.tand(v_tail_anhedral_deg)])

    # Twin booms from wing rear to tail root
    boom_radius = 6e-3 / 2
    wing_root_chord = wing.xsecs[0].chord
    boom_end_x = 0.75 * wing_root_chord
    boom_end_y = (wing.span()/2) * 0.4
    boom_start_x = x_t + (htail_span/2)*np.tand(v_tail_sweep_deg) + (htail_chord*v_tail_taper)
    boom_start_y = htail_span/2

    boom_right = asb.Fuselage(
        name="Right Boom",
        xsecs=[
            asb.FuselageXSec(xyz_c=[boom_end_x,  boom_end_y, 0], radius=boom_radius),
            asb.FuselageXSec(xyz_c=[boom_start_x, boom_start_y, 0], radius=boom_radius),
        ]
    )
    boom_left  = asb.Fuselage(
        name="Left Boom",
        xsecs=[
            asb.FuselageXSec(xyz_c=[boom_end_x, -boom_end_y, 0], radius=boom_radius),
            asb.FuselageXSec(xyz_c=[boom_start_x, -boom_start_y, 0], radius=boom_radius),
        ]
    )

    fuselage = build_common_fuselage(x_nose, x_tail)

    airplane = asb.Airplane(
        name="Reverse V-Tail Twin-Boom",
        wings=[wing, reverse_v_tail],
        fuselages=[fuselage, boom_right, boom_left]
    )

    density_wing_solid = 2 * u.lbm / u.foot ** 3
    mass_props = {}
    mass_props["wing"] = asb.mass_properties_from_radius_of_gyration(
        mass=_mass_from_volume(wing.volume(), density_wing_solid), x_cg=0
    )
    mass_props["tail_surfaces"] = asb.mass_properties_from_radius_of_gyration(
        mass=_mass_from_volume(reverse_v_tail.volume(), 80), x_cg=x_t + 0.5 * htail_chord
    )

    boom_len = np.sqrt((boom_start_x - boom_end_x)**2 + (boom_start_y - boom_end_y)**2)
    mass_props["twin_booms"] = asb.mass_properties_from_radius_of_gyration(
        mass=2 * 7.0e-3 * (boom_len / 826e-3), x_cg=(boom_start_x + boom_end_x)/2
    )
    mass_props["motor"] = asb.mass_properties_from_radius_of_gyration(mass=4.49e-3, x_cg=x_nose - 0.3 * u.inch)
    mass_props["motor_bolts"] = asb.mass_properties_from_radius_of_gyration(mass=4*0.075e-3, x_cg=x_nose)
    mass_props["propeller"] = asb.mass_properties_from_radius_of_gyration(mass=1.54e-3, x_cg=x_nose - 0.7 * u.inch)
    mass_props["propeller_band"] = asb.mass_properties_from_radius_of_gyration(mass=0.06e-3, x_cg=x_nose - 0.7 * u.inch)
    mass_props["flight_computer"] = asb.mass_properties_from_radius_of_gyration(mass=4.30e-3, x_cg=x_nose + 2*u.inch + 1.3*u.inch/2)
    mass_props["battery"] = asb.mass_properties_from_radius_of_gyration(mass=4.61e-3, x_cg=x_nose + 2*u.inch)

    return airplane, mass_props, wing

def make_airplane_flying_wing(opti, wing, x_nose, x_tail,
                              htail_x_ratio, htail_span, htail_aoa_deg, htail_chord, htail_y_ratio, htail_af_name,
                              vtail_chord, vtail_height, vtail_offset, vtail_x_ratio, vtail_af_name):
    # Reuse the main parametric wing as provided; add simple winglets as “vertical surfaces”
    winglet_height = 0.08
    winglet_chord  = 0.07
    winglet_sweep  = 15.0
    winglet_cant   = 10.0
    v_af = AIRFOIL_DB[vtail_af_name]  # symmetric small-thickness foil

    # Tip data
    tip = wing.xsecs[-1]
    chord_tip = tip.chord
    xyz_le_tip = tip.xyz_le

    winglets = asb.Wing(
        name="Winglets", symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=xyz_le_tip + np.array([chord_tip*0.75, 0, 0]), chord=winglet_chord, airfoil=v_af),
            asb.WingXSec(xyz_le=xyz_le_tip + np.array([chord_tip*0.75 + winglet_height*np.tand(winglet_sweep),
                                                       winglet_height*np.sind(winglet_cant),
                                                       winglet_height*np.cosd(winglet_cant)]),
                         chord=winglet_chord*0.7, airfoil=v_af),
        ]
    )

    airplane = asb.Airplane(
        name="Flying Wing",
        wings=[wing, winglets],
        fuselages=[]
    )

    density_wing_solid = 2 * u.lbm / u.foot ** 3
    mass_props = {}
    mass_props["wing"] = asb.mass_properties_from_radius_of_gyration(
        mass=_mass_from_volume(wing.volume(), density_wing_solid),
        x_cg=wing.aerodynamic_center()[0], z_cg=wing.aerodynamic_center()[2]
    )
    mass_props["winglets"] = asb.mass_properties_from_radius_of_gyration(
        mass=_mass_from_volume(winglets.volume(), 80),
        x_cg=winglets.aerodynamic_center()[0], z_cg=winglets.aerodynamic_center()[2]
    )
    # Light pusher-motor-ish masses onto the wing (kept compatible with your mass model spirit)
    x_pusher_mount = 0.6 * wing.xsecs[0].chord
    mass_props["motor"] = asb.mass_properties_from_radius_of_gyration(mass=4.49e-3, x_cg=x_pusher_mount - 0.3*u.inch)
    mass_props["motor_bolts"] = asb.mass_properties_from_radius_of_gyration(mass=4*0.075e-3, x_cg=x_pusher_mount)
    mass_props["propeller"] = asb.mass_properties_from_radius_of_gyration(mass=1.54e-3, x_cg=x_pusher_mount + 0.4*u.inch)
    mass_props["propeller_band"] = asb.mass_properties_from_radius_of_gyration(mass=0.06e-3, x_cg=x_pusher_mount + 0.4*u.inch)
    x_avionics = -0.1 * wing.xsecs[0].chord
    mass_props["flight_computer"] = asb.mass_properties_from_radius_of_gyration(mass=4.30e-3, x_cg=x_avionics)
    mass_props["battery"] = asb.mass_properties_from_radius_of_gyration(mass=4.61e-3, x_cg=x_avionics - 1.5 * u.inch)

    return airplane, mass_props, wing


# ================
# Driver
# ================
def sweep_architecture(arch_name: str, make_airplane_fn, sm_override=None):
    """
    Enumerates airfoil catalogs, optimizes geometry, prints the best outcome.
    """
    results = []
    for wing_af, ht_af, vt_af in itertools.product(WING_AIRFOILS, HTAIL_AIRFOILS, VTAIL_AIRFOILS):
        print(f"\n--- {arch_name}: wing={wing_af}, htail={ht_af}, vtail={vt_af} ---")
        out = optimize_architecture(
            config_name=arch_name,
            make_airplane_fn=make_airplane_fn,
            wing_airfoil_name=wing_af,
            htail_airfoil_name=ht_af,
            vtail_airfoil_name=vt_af,
            sm_min=SM_min if sm_override is None else sm_override
        )
        if out is not None:
            _, rep = out
            results.append(rep)

    if not results:
        print(f"[{arch_name}] No feasible solutions found.")
        return

    # Sort by objective J ascending (lower is better)
    results.sort(key=lambda r: r["J"])
    best = results[0]

    print("\n" + "=" * 60)
    print(f"Best {arch_name} (by objective J):")
    for k in ["wing_af", "htail_af", "vtail_af", "span", "c_root", "taper_ratio",
              "dihedral_deg", "weight_kg", "SM", "LD_cr", "sink_rate", "alpha_cr", "alpha_Vt", "Cma", "Cnb", "Clb", "J"]:
        print(f"{k:>14}: {best[k]}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Conventional single-fin
    sweep_architecture("Conventional_Tail",        make_airplane_conventional)
    # True H-tail with twin fins
    sweep_architecture("H_Tail",                    make_airplane_htail_twinfins)
    # V-tail
    sweep_architecture("V_Tail",                    make_airplane_vtail)
    # Reverse V-tail twin boom
    sweep_architecture("Reverse_V_Tail_Twin_Boom",  make_airplane_reverse_v)
    # Flying wing (lower SM requirement, as in your earlier script)
    sweep_architecture("Flying_Wing",               make_airplane_flying_wing, sm_override=0.05)
