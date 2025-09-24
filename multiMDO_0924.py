# multiMDO_graphs_0923.py
# For each configuration, opens ONE composite figure window:
# [ airplane three-view image (captured in-memory) ] + [ mass pie ] + [ advanced aero graphs ]
# Nothing is saved to disk.

import aerosandbox as asb
import aerosandbox.numpy as np
import aerosandbox.tools.units as u
import copy
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import aerosandbox.tools.pretty_plots as p
from aerosandbox.tools.string_formatting import eng_string

# ##############################################################################
# # COMMON SETUP
# ##############################################################################
print("Performing common setup...")

# --- Global list to store results for final comparison
results_summary = []

# --- Create directory only for polar cache
Path("cache").mkdir(exist_ok=True)

# --- Global Parameters
make_plots = True
wing_method = 'foam'
wing_span = 1
wing_dihedral_angle_deg = 20

# --- Airfoil Loading and Polar Generation
airfoils = {
    name: asb.Airfoil(name=name) for name in [
        "ag04", "ag09", "ag13", "naca0008"
    ]
}

for name, af in airfoils.items():
    try:
        af.generate_polars(
            cache_filename=f"cache/{name}.json",
            alphas=np.linspace(-10, 20, 21)
        )
        print(f"Generated polars for {name}")
    except Exception as e:
        print(f"Could not generate polars for {name}: {e}")

# --- General printout helper
print_title = lambda s: print(s.upper().join(["\n" + "*" * 25] * 2))


def _three_view_to_array(airplane, dpi: int = 200):
    """
    Render airplane three-view to an offscreen Matplotlib figure and return as a NumPy RGB image.
    No files are written.
    """
    # Render three-view without showing
    plt.ioff()
    airplane.draw_three_view(show=False)
    fig = plt.gcf()

    # Draw to an Agg canvas and grab the RGBA buffer
    canvas = FigureCanvas(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    # Updated to use buffer_rgba() to resolve deprecation warning
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(h, w, 4)  # Reshape to 4 channels (RGBA)

    plt.close(fig)
    return img


def _composite_report_show(
        config_name: str,
        airplane,
        op_point,
        mass_props: dict,
        mass_props_TOGW,
        aero: dict,
        dpi: int = 200,
):
    """
    Builds one composite figure containing a comprehensive report, then shows it.
    """
    # --- Generate aerodynamic polars across alpha for plotting
    alpha_range = np.linspace(-15, 15, 180)
    aero_polars = asb.AeroBuildup(
        airplane=airplane,
        op_point=asb.OperatingPoint(
            velocity=op_point.velocity,
            alpha=alpha_range
        ),
        xyz_ref=mass_props_TOGW.xyz_cg
    ).run()

    # --- Create the composite figure
    fig = plt.figure(figsize=(24, 18), dpi=dpi, constrained_layout=True)
    fig.suptitle(f"{config_name.replace('_', ' ')}: Summary Report", fontsize=24, y=0.98)

    # --- GridSpec layout: 3 rows x 3 cols
    gs = fig.add_gridspec(3, 3)

    # === Row 0: Airplane View and Mass Budget ===
    ax_plane_big = fig.add_subplot(gs[0, 0:2])
    try:
        tv_img = _three_view_to_array(airplane, dpi=max(150, dpi))
        ax_plane_big.imshow(tv_img)
        ax_plane_big.set_title("Airplane (Three-View Rendering)", fontsize=16)
        ax_plane_big.axis("off")
    except Exception as e:
        ax_plane_big.text(0.5, 0.5, f"three_view render failed:\n{e}", ha="center", va="center")
        ax_plane_big.axis("off")

    ax_pie = fig.add_subplot(gs[0, 2])
    name_remaps = {k: k.replace("_", " ").title() for k in mass_props.keys()}
    mass_props_to_plot = {k: v for k, v in mass_props.items() if float(v.mass) > 1e-9}
    if "ballast" in mass_props_to_plot and float(mass_props_to_plot["ballast"].mass) < 1e-6:
        del mass_props_to_plot["ballast"]

    plt.sca(ax_pie)
    p.pie(
        values=[v.mass for v in mass_props_to_plot.values()],
        names=[name_remaps.get(n, n) for n in mass_props_to_plot.keys()],
        center_text=(
            f"$\\bf{{Mass\\ Budget}}$\n"
            f"TOGW: {mass_props_TOGW.mass * 1e3:.2f} g"
        ),
        label_format=lambda name, value, percentage: f"{name}, {value * 1e3:.2f} g, {percentage:.1f}%",
        startangle=110, arm_length=28, arm_radius=18, y_max_labels=1.10
    )
    ax_pie.set_title("Mass Budget", fontsize=16, pad=16)

    # === Row 1: Core Aerodynamic Polars ===
    ax_cl = fig.add_subplot(gs[1, 0])
    ax_cd = fig.add_subplot(gs[1, 1])
    ax_ld = fig.add_subplot(gs[1, 2])

    ax_cl.plot(alpha_range, aero_polars["CL"])
    ax_cl.set_xlabel(r"$\alpha$ [deg]"); ax_cl.set_ylabel(r"$C_L$"); ax_cl.set_title(r"$C_L$ vs. $\alpha$"); ax_cl.grid(True)

    ax_cd.plot(alpha_range, aero_polars["CD"])
    ax_cd.set_xlabel(r"$\alpha$ [deg]"); ax_cd.set_ylabel(r"$C_D$"); ax_cd.set_title(r"$C_D$ vs. $\alpha$"); ax_cd.grid(True)
    ax_cd.set_ylim(bottom=0)

    ax_ld.plot(alpha_range, aero_polars["CL"] / aero_polars["CD"])
    ax_ld.set_xlabel(r"$\alpha$ [deg]"); ax_ld.set_ylabel(r"$L/D$"); ax_ld.set_title(r"$L/D$ vs. $\alpha$"); ax_ld.grid(True)

    # === Row 2: Stability and Advanced Aero ===
    ax_lift_dist = fig.add_subplot(gs[2, 0])
    ax_sm = fig.add_subplot(gs[2, 1])
    gs_bottom_right = gs[2, 2].subgridspec(2, 1, hspace=0.1)
    ax_cm = fig.add_subplot(gs_bottom_right[0, 0])
    ax_coupling = fig.add_subplot(gs_bottom_right[1, 0])

    # --- FIX: Create a new, clean op_point with definite float values for the VLM.
    vlm_op_point = asb.OperatingPoint(
        velocity=float(op_point.velocity),
        alpha=float(op_point.alpha),
    )
    # Spanwise Lift Distribution
    try:
        vlm_aero = asb.VortexLatticeMethod(airplane=airplane, op_point=vlm_op_point).run()
        span_data = vlm_aero["spanwise_data_wing_1"]
        ax_lift_dist.plot(span_data["y"], span_data["Cl"], "-", label="Actual Lift", color="k")
        ax_lift_dist.plot(vlm_aero['yl_elliptical_dist'], vlm_aero['Cl_elliptical_dist'], "--",
                          label="Elliptical Ideal", color="C2")
        ax_lift_dist.set_xlabel("Spanwise location $y$ [m]"); ax_lift_dist.set_ylabel("Sectional Lift Coeff. $C_l$")
        ax_lift_dist.set_title("Spanwise Lift Distribution"); ax_lift_dist.grid(True); ax_lift_dist.legend()
    except Exception as e:
        ax_lift_dist.text(0.5, 0.5, f"VLM for lift dist failed:\n{e}", ha="center", va="center")

    # Static Margin vs. CG
    x_np = aero['x_np']
    mac = airplane.wings[0].mean_aerodynamic_chord()
    static_margin_actual = (x_np - mass_props_TOGW.x_cg) / mac
    x_cg_range = np.linspace(x_np - 0.5 * mac, x_np + 0.5 * mac, 200)
    sm_range = (x_np - x_cg_range) / mac
    ax_sm.plot(x_cg_range, sm_range * 100)
    ax_sm.axvline(mass_props_TOGW.x_cg, color='r', linestyle='--', label=f"Actual CG\nSM = {static_margin_actual * 100:.1f}%")
    ax_sm.axhline(0, color='k', linewidth=0.8); ax_sm.set_xlabel("CG Location $x_{cg}$ [m]")
    ax_sm.set_ylabel("Static Margin [% MAC]"); ax_sm.set_title("Static Margin vs. CG Location"); ax_sm.grid(True); ax_sm.legend()

    # Cm vs. Alpha
    ax_cm.plot(alpha_range, aero_polars["Cm"])
    ax_cm.set_xlabel(r"$\alpha$ [deg]"); ax_cm.set_ylabel(r"$C_m$"); ax_cm.set_title(r"$C_m$ vs. $\alpha$"); ax_cm.grid(True)

    # Directional-Roll Coupling
    clb = aero.get('Clb', np.nan); cnr = aero.get('Cnr', np.nan)
    clr = aero.get('Clr', np.nan); cnb = aero.get('Cnb', np.nan)
    if not any(np.isnan([clb, cnr, clr, cnb])) and clr != 0 and cnb != 0:
        coupling_metric = (clb * cnr) / (clr * cnb)
        metric_text = f"{coupling_metric:.3f}"
        stability_text, color = ("Spiral Stability", "green") if coupling_metric > 0 else ("Spiral Divergence", "red")
    else:
        metric_text = "N/A"
        stability_text, color = ("Analysis Indeterminate", "gray")

    ax_coupling.set_title("Directional-Roll Coupling Proxy"); ax_coupling.axis('off')
    ax_coupling.text(0.5, 0.7, r"$\frac{C_{l_\beta} C_{n_r}}{C_{l_r} C_{n_\beta}}$", ha='center', fontsize=24)
    ax_coupling.text(0.5, 0.4, f"= {metric_text}", ha='center', fontsize=20)
    ax_coupling.text(0.5, 0.1, stability_text, ha='center', fontsize=16, color=color,
                     bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))

    plt.show()


def report_solution(sol, config_name, airplane, op_point, mass_props, mass_props_TOGW, LD_cruise, sink_rate, aero):
    """Print outputs, open the composite figure, and log data for final comparison."""
    s = lambda x: sol.value(x)

    # Substitute numerical values into all of the objects
    airplane = sol(airplane)
    op_point = sol(op_point)
    mass_props = sol(mass_props)
    mass_props_TOGW = sol(mass_props_TOGW)
    aero = sol(aero)

    # --- Run AVL Analysis (optional)
    avl_aero = {}
    try:
        avl_airplane = copy.deepcopy(airplane)
        if avl_airplane.wings and len(avl_airplane.wings[0].xsecs) > 2:
            wing_lowres = avl_airplane.wings[0]
            xsecs_to_keep = np.arange(len(wing_lowres.xsecs)) % 2 == 0
            xsecs_to_keep[0], xsecs_to_keep[-1] = True, True
            wing_lowres.xsecs = np.array(wing_lowres.xsecs)[xsecs_to_keep]
        avl_aero = asb.AVL(airplane=avl_airplane, op_point=op_point, xyz_ref=mass_props_TOGW.xyz_cg).run()
    except Exception as e:
        print(f"AVL analysis failed for {config_name}: {e}")
        class EmptyDict(dict):
            def __getitem__(self, item): return "AVL Run Failed"
            def get(self, key, default=None): return "AVL Run Failed"
        avl_aero = EmptyDict()

    def fmt(x):
        try: return f"{s(x):.6g}"
        except (TypeError, ValueError): return "N/A"

    print_title(f"{config_name} Outputs")
    output_data = {
        "mass_TOGW": f"{fmt(mass_props_TOGW.mass)} kg ({fmt(mass_props_TOGW.mass / u.lbm)} lbm)",
        "L/D (actual)": fmt(LD_cruise), "Cruise Airspeed": f"{fmt(op_point.velocity)} m/s",
        "Cruise AoA": f"{fmt(op_point.alpha)} deg", "Cruise CL": fmt(aero.get('CL')), "Sink Rate": fmt(sink_rate),
        "Cma": fmt(aero.get('Cma')), "Cnb": fmt(aero.get('Cnb')), "Cm": fmt(aero.get('Cm')),
        "Wing Reynolds Number": eng_string(op_point.reynolds(sol(airplane.wings[0].mean_aerodynamic_chord()))),
        "AVL: Cma": avl_aero.get('Cma'), "AVL: Cnb": avl_aero.get('Cnb'), "AVL: Cm": avl_aero.get('Cm'),
        "AVL: Clb Cnr / Clr Cnb": avl_aero.get('Clb Cnr / Clr Cnb'),
        "CG location": f"({fmt(mass_props_TOGW.xyz_cg[0])}, {fmt(mass_props_TOGW.xyz_cg[1])}, {fmt(mass_props_TOGW.xyz_cg[2])}) m",
        "Wing Span": f"{fmt(wing_span)} m ({fmt(wing_span / u.foot)} ft)",
    }
    for k, v in output_data.items():
        print(f"{k.rjust(25)} = {v}")

    print_title(f"{config_name} Mass Properties")
    for k, v in mass_props.items():
        print(f"{k.rjust(25)} = {s(v.mass) * 1e3:.2f} g ({s(v.mass) / u.oz:.2f} oz)")

    # Data collection for final table
    avl_coupling = avl_aero.get('Clb Cnr / Clr Cnb', np.nan)
    summary_data = {
        "Configuration": config_name.replace("_", " "),
        "TOGW [g]": s(mass_props_TOGW.mass) * 1e3,
        "L/D": s(LD_cruise), "Sink Rate [m/s]": s(sink_rate), "V_cruise [m/s]": s(op_point.velocity),
        "AoA [deg]": s(op_point.alpha), "Cma": s(aero.get('Cma')), "Cnb": s(aero.get('Cnb')),
        "AVL Coupling": avl_coupling if isinstance(avl_coupling, (float, int)) else np.nan,
        "CG_x [m]": s(mass_props_TOGW.xyz_cg)[0]
    }
    results_summary.append(summary_data)

    if make_plots:
        _composite_report_show(
            config_name=config_name, airplane=airplane, op_point=op_point,
            mass_props=mass_props, mass_props_TOGW=mass_props_TOGW,
            aero=aero, dpi=200,
        )


# ##############################################################################
# # CONFIGURATION 1: H-TAIL
# ##############################################################################
def optimize_h_tail():
    config_name = "H_Tail"
    print_title(f"Optimizing: {config_name}")
    opti = asb.Opti()

    # --- Specs
    op_point = asb.OperatingPoint(velocity=opti.variable(init_guess=14, lower_bound=1, log_transform=True),
                                  alpha=opti.variable(init_guess=0, lower_bound=-10, upper_bound=20))
    design_mass_TOGW = opti.variable(init_guess=0.1, lower_bound=1e-3)
    design_mass_TOGW = np.maximum(design_mass_TOGW, 1e-3)
    LD_cruise = opti.variable(init_guess=15, lower_bound=0.1, log_transform=True)

    # --- Geometry
    x_nose = opti.variable(init_guess=-0.1, upper_bound=0)
    x_tail = opti.variable(init_guess=0.5, lower_bound=0.1)
    wing_root_chord = opti.variable(init_guess=0.15, lower_bound=1e-3)

    wing_center_span_fraction = 0.4
    wing = asb.Wing(name="Main Wing", symmetric=True, xsecs=[
        asb.WingXSec(xyz_le=[-0.25 * wing_root_chord, 0, 0], chord=wing_root_chord, airfoil=airfoils["ag13"]),
        asb.WingXSec(xyz_le=[-0.25 * wing_root_chord, wing_span / 2 * wing_center_span_fraction, 0],
                     chord=wing_root_chord, airfoil=airfoils["ag13"]),
        asb.WingXSec(
            xyz_le=[-0.25 * wing_root_chord * 0.7, wing_span / 2,
                    wing_span / 2 * (1 - wing_center_span_fraction) * np.tand(wing_dihedral_angle_deg)],
            chord=wing_root_chord * 0.7, airfoil=airfoils["ag13"])
    ])

    h_tail_span = 0.35 * wing_span
    h_tail_root_chord = 0.08
    h_tail = asb.Wing(
        name="Horizontal Stabilizer", symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0, 0, 0], chord=h_tail_root_chord, airfoil=airfoils["naca0008"]),
            asb.WingXSec(xyz_le=[0.02, h_tail_span / 2, 0], chord=h_tail_root_chord * 0.8, airfoil=airfoils["naca0008"])
        ]
    ).translate([x_tail, 0, 0])

    v_tail_height = 0.12
    v_tail_root_chord = 0.09
    v_tail = asb.Wing(
        name="Vertical Stabilizer", symmetric=False,
        xsecs=[
            asb.WingXSec(xyz_le=[0, 0, 0], chord=v_tail_root_chord, airfoil=airfoils["naca0008"]),
            asb.WingXSec(xyz_le=[0.03, 0, v_tail_height], chord=v_tail_root_chord * 0.7, airfoil=airfoils["naca0008"])
        ]
    )
    v_tail_right = v_tail.translate([x_tail + 0.01, h_tail_span / 2, 0]); v_tail_right.name = "Right Vertical Stabilizer"
    v_tail_left = v_tail.translate([x_tail + 0.01, -h_tail_span / 2, 0]); v_tail_left.name = "Left Vertical Stabilizer"

    x_pod_end = x_nose + 8 * u.inch
    fuselage = asb.Fuselage(
        name="Fuse",
        xsecs=[
            asb.FuselageXSec(xyz_c=[x_nose, 0, 0], height=1.5 * u.inch, width=2.0 * u.inch),
            asb.FuselageXSec(xyz_c=[x_pod_end, 0, 0], height=1.5 * u.inch, width=2.0 * u.inch),
            asb.FuselageXSec(xyz_c=[x_pod_end, 0, 0], radius=7e-3 / 2),
            asb.FuselageXSec(xyz_c=[x_tail, 0, 0], radius=7e-3 / 2)
        ]
    )

    airplane = asb.Airplane(name="Feather H-Tail", wings=[wing, h_tail, v_tail_right, v_tail_left], fuselages=[fuselage])

    # --- Mass
    mass_props = {}
    density = 2 * u.lbm / u.foot ** 3
    mass_props['wing'] = asb.mass_properties_from_radius_of_gyration(
        mass=wing.volume() * density, x_cg=0,
        z_cg=(0.03591) * (np.sind(wing_dihedral_angle_deg) / np.sind(11)) * (wing_span / 1)
    )
    total_tail_volume = h_tail.volume() + v_tail_right.volume() + v_tail_left.volume()
    mass_props["tail_surfaces"] = asb.mass_properties_from_radius_of_gyration(
        mass=total_tail_volume * 80, x_cg=x_tail + 0.50 * h_tail_root_chord
    )
    mass_props["linkages"] = asb.MassProperties(mass=1e-3, x_cg=(x_nose + x_tail) / 2)
    mass_props["motor"] = asb.mass_properties_from_radius_of_gyration(mass=4.49e-3, x_cg=x_nose - 0.3 * u.inch)
    mass_props["motor_bolts"] = asb.mass_properties_from_radius_of_gyration(mass=4 * 0.075e-3, x_cg=x_nose)
    mass_props["propeller"] = asb.mass_properties_from_radius_of_gyration(mass=1.54e-3, x_cg=x_nose - 0.7 * u.inch)
    mass_props["propeller_band"] = asb.mass_properties_from_radius_of_gyration(
        mass=0.06e-3, x_cg=mass_props["propeller"].x_cg
    )
    mass_props["flight_computer"] = asb.mass_properties_from_radius_of_gyration(
        mass=4.30e-3, x_cg=x_nose + 2 * u.inch + (1.3 * u.inch) / 2
    )
    mass_props["battery"] = asb.mass_properties_from_radius_of_gyration(mass=4.61e-3, x_cg=x_nose + 2 * u.inch)
    mass_props["pod_structure"] = asb.MassProperties(mass=10e-3, x_cg=(x_nose + x_pod_end) / 2)
    boom_length = np.maximum(0, x_tail - x_pod_end)
    mass_props["boom"] = asb.mass_properties_from_radius_of_gyration(
        mass=7.0e-3 * (boom_length / 826e-3), x_cg=(x_pod_end + x_tail) / 2
    )
    mass_props["ballast"] = asb.MassProperties(
        mass=opti.variable(init_guess=0, lower_bound=0),
        x_cg=opti.variable(init_guess=0, lower_bound=x_nose, upper_bound=x_tail)
    )

    mass_sum_pre_glue = sum(v for k, v in mass_props.items() if k != "ballast")
    mass_props['glue_weight'] = asb.MassProperties(mass=mass_sum_pre_glue.mass * 0.08)
    mass_props_TOGW = sum(mass_props.values())

    # --- Aero & Stability
    ab = asb.AeroBuildup(airplane=airplane, op_point=op_point, xyz_ref=mass_props_TOGW.xyz_cg)
    aero = ab.run_with_stability_derivatives(alpha=True, beta=True)
    static_margin = (aero["x_np"] - mass_props_TOGW.x_cg) / wing.mean_aerodynamic_chord()
    LD = aero["L"] / aero["D"]
    sink_rate = (aero["D"] * op_point.velocity) / (9.81 * mass_props_TOGW.mass)

    # --- Objective and Constraints
    opti.minimize(sink_rate / 0.3 + mass_props_TOGW.mass / 0.100 * 0.1 + (mass_props["ballast"].x_cg / 1e3) ** 2)
    opti.subject_to([
        aero["L"] >= 9.81 * mass_props_TOGW.mass,
        aero["Cm"] == 0,
        static_margin == 0.08,
        LD_cruise == LD,
        design_mass_TOGW == mass_props_TOGW.mass
    ])
    tail_moment_arm = h_tail.aerodynamic_center(chord_fraction=0.25)[0] - mass_props_TOGW.xyz_cg[0]
    opti.subject_to(
        h_tail.area() * tail_moment_arm / (wing.area() * wing.mean_aerodynamic_chord()) > 0.4
    )
    total_vtail_area = v_tail_right.area() + v_tail_left.area()
    opti.subject_to(
        total_vtail_area * tail_moment_arm / (wing.area() * wing.span()) > 0.02
    )
    opti.subject_to([
        x_nose < -0.25 * wing_root_chord - 0.5 * u.inch,
        x_tail - x_nose < 0.826
    ])

    # --- Solve and Report
    sol = opti.solve(verbose=False)
    report_solution(sol, config_name, airplane, op_point, mass_props, mass_props_TOGW, LD_cruise, sink_rate, aero)


# ##############################################################################
# # CONFIGURATION 2: CONVENTIONAL TAIL
# ##############################################################################
def optimize_conventional_tail():
    config_name = "Conventional_Tail"
    print_title(f"Optimizing: {config_name}")
    opti = asb.Opti()

    # --- Specs
    op_point = asb.OperatingPoint(velocity=opti.variable(init_guess=14, lower_bound=1, log_transform=True),
                                  alpha=opti.variable(init_guess=0, lower_bound=-10, upper_bound=20))
    design_mass_TOGW = opti.variable(init_guess=0.1, lower_bound=1e-3)
    design_mass_TOGW = np.maximum(design_mass_TOGW, 1e-3)
    LD_cruise = opti.variable(init_guess=15, lower_bound=0.1, log_transform=True)

    # --- Geometry
    x_nose = opti.variable(init_guess=-0.1, upper_bound=0)
    x_tail = opti.variable(init_guess=0.5, lower_bound=0.1)
    wing_root_chord = opti.variable(init_guess=0.15, lower_bound=1e-3)

    wing_center_span_fraction = 0.4
    wing = asb.Wing(name="Main Wing", symmetric=True, xsecs=[
        asb.WingXSec(xyz_le=[-0.25 * wing_root_chord, 0, 0], chord=wing_root_chord, airfoil=airfoils["ag13"]),
        asb.WingXSec(xyz_le=[-0.25 * wing_root_chord, wing_span / 2 * wing_center_span_fraction, 0],
                     chord=wing_root_chord, airfoil=airfoils["ag13"]),
        asb.WingXSec(
            xyz_le=[-0.25 * wing_root_chord * 0.7, wing_span / 2,
                    wing_span / 2 * (1 - wing_center_span_fraction) * np.tand(wing_dihedral_angle_deg)],
            chord=wing_root_chord * 0.7, airfoil=airfoils["ag13"])
    ])

    h_tail_span = 0.35 * wing_span
    h_tail_root_chord = 0.08
    h_tail = asb.Wing(
        name="Horizontal Stabilizer", symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0, 0, 0], chord=h_tail_root_chord, airfoil=airfoils["naca0008"]),
            asb.WingXSec(xyz_le=[0.02, h_tail_span / 2, 0], chord=h_tail_root_chord * 0.8, airfoil=airfoils["naca0008"])
        ]
    ).translate([x_tail, 0, 0])

    v_tail_height = 0.12
    v_tail_root_chord = 0.09
    v_tail = asb.Wing(
        name="Vertical Stabilizer", symmetric=False,
        xsecs=[
            asb.WingXSec(xyz_le=[0, 0, 0], chord=v_tail_root_chord, airfoil=airfoils["naca0008"]),
            asb.WingXSec(xyz_le=[0.03, 0, v_tail_height], chord=v_tail_root_chord * 0.7, airfoil=airfoils["naca0008"])
        ]
    ).translate([x_tail + 0.01, 0, 0])

    x_pod_end = x_nose + 8 * u.inch
    fuselage = asb.Fuselage(
        name="Fuse",
        xsecs=[
            asb.FuselageXSec(xyz_c=[x_nose, 0, 0], height=1.5 * u.inch, width=2.0 * u.inch),
            asb.FuselageXSec(xyz_c=[x_pod_end, 0, 0], height=1.5 * u.inch, width=2.0 * u.inch),
            asb.FuselageXSec(xyz_c=[x_pod_end, 0, 0], radius=7e-3 / 2),
            asb.FuselageXSec(xyz_c=[x_tail, 0, 0], radius=7e-3 / 2)
        ]
    )

    airplane = asb.Airplane(name="Feather Conventional Tail", wings=[wing, h_tail, v_tail], fuselages=[fuselage])

    # --- Mass
    mass_props = {}
    density = 2 * u.lbm / u.foot ** 3
    mass_props['wing'] = asb.mass_properties_from_radius_of_gyration(
        mass=wing.volume() * density, x_cg=0,
        z_cg=(0.03591) * (np.sind(wing_dihedral_angle_deg) / np.sind(11)) * (wing_span / 1)
    )
    total_tail_volume = h_tail.volume() + v_tail.volume()
    mass_props["tail_surfaces"] = asb.mass_properties_from_radius_of_gyration(
        mass=total_tail_volume * 80, x_cg=x_tail + 0.50 * h_tail_root_chord
    )
    mass_props["linkages"] = asb.MassProperties(mass=1e-3, x_cg=(x_nose + x_tail) / 2)
    mass_props["motor"] = asb.mass_properties_from_radius_of_gyration(mass=4.49e-3, x_cg=x_nose - 0.3 * u.inch)
    mass_props["motor_bolts"] = asb.mass_properties_from_radius_of_gyration(mass=4 * 0.075e-3, x_cg=x_nose)
    mass_props["propeller"] = asb.mass_properties_from_radius_of_gyration(mass=1.54e-3, x_cg=x_nose - 0.7 * u.inch)
    mass_props["propeller_band"] = asb.mass_properties_from_radius_of_gyration(
        mass=0.06e-3, x_cg=mass_props["propeller"].x_cg
    )
    mass_props["flight_computer"] = asb.mass_properties_from_radius_of_gyration(
        mass=4.30e-3, x_cg=x_nose + 2 * u.inch + (1.3 * u.inch) / 2
    )
    mass_props["battery"] = asb.mass_properties_from_radius_of_gyration(mass=4.61e-3, x_cg=x_nose + 2 * u.inch)
    mass_props["pod_structure"] = asb.MassProperties(mass=10e-3, x_cg=(x_nose + x_pod_end) / 2)
    boom_length = np.maximum(0, x_tail - x_pod_end)
    mass_props["boom"] = asb.mass_properties_from_radius_of_gyration(
        mass=7.0e-3 * (boom_length / 826e-3), x_cg=(x_pod_end + x_tail) / 2
    )
    mass_props["ballast"] = asb.MassProperties(
        mass=opti.variable(init_guess=0, lower_bound=0),
        x_cg=opti.variable(init_guess=0, lower_bound=x_nose, upper_bound=x_tail)
    )

    mass_sum_pre_glue = sum(v for k, v in mass_props.items() if k != "ballast")
    mass_props['glue_weight'] = asb.MassProperties(mass=mass_sum_pre_glue.mass * 0.08)
    mass_props_TOGW = sum(mass_props.values())

    # --- Aero & Stability
    ab = asb.AeroBuildup(airplane=airplane, op_point=op_point, xyz_ref=mass_props_TOGW.xyz_cg)
    aero = ab.run_with_stability_derivatives(alpha=True, beta=True)
    static_margin = (aero["x_np"] - mass_props_TOGW.x_cg) / wing.mean_aerodynamic_chord()
    LD = aero["L"] / aero["D"]
    sink_rate = (aero["D"] * op_point.velocity) / (9.81 * mass_props_TOGW.mass)

    # --- Objective and Constraints
    opti.minimize(sink_rate / 0.3 + mass_props_TOGW.mass / 0.100 * 0.1 + (mass_props["ballast"].x_cg / 1e3) ** 2)
    opti.subject_to([
        aero["L"] >= 9.81 * mass_props_TOGW.mass,
        aero["Cm"] == 0,
        static_margin == 0.08,
        LD_cruise == LD,
        design_mass_TOGW == mass_props_TOGW.mass
    ])
    tail_moment_arm = h_tail.aerodynamic_center(chord_fraction=0.25)[0] - mass_props_TOGW.xyz_cg[0]
    opti.subject_to(
        h_tail.area() * tail_moment_arm / (wing.area() * wing.mean_aerodynamic_chord()) > 0.4
    )
    opti.subject_to(
        v_tail.area() * tail_moment_arm / (wing.area() * wing.span()) > 0.02
    )
    opti.subject_to([
        x_nose < -0.25 * wing_root_chord - 0.5 * u.inch,
        x_tail - x_nose < 0.826
    ])

    # --- Solve and Report
    sol = opti.solve(verbose=False)
    report_solution(sol, config_name, airplane, op_point, mass_props, mass_props_TOGW, LD_cruise, sink_rate, aero)


# ##############################################################################
# # CONFIGURATION 3: V-TAIL
# ##############################################################################
def optimize_v_tail():
    config_name = "V_Tail"
    print_title(f"Optimizing: {config_name}")
    opti = asb.Opti()

    # --- Specs
    op_point = asb.OperatingPoint(velocity=opti.variable(init_guess=14, lower_bound=1, log_transform=True),
                                  alpha=opti.variable(init_guess=0, lower_bound=-10, upper_bound=20))
    design_mass_TOGW = opti.variable(init_guess=0.1, lower_bound=1e-3)
    design_mass_TOGW = np.maximum(design_mass_TOGW, 1e-3)
    LD_cruise = opti.variable(init_guess=15, lower_bound=0.1, log_transform=True)

    # --- Geometry
    x_nose = opti.variable(init_guess=-0.1, upper_bound=0)
    x_tail = opti.variable(init_guess=0.5, lower_bound=0.1)
    wing_root_chord = opti.variable(init_guess=0.15, lower_bound=1e-3)

    wing_center_span_fraction = 0.4
    wing = asb.Wing(name="Main Wing", symmetric=True, xsecs=[
        asb.WingXSec(xyz_le=[-0.25 * wing_root_chord, 0, 0], chord=wing_root_chord, airfoil=airfoils["ag13"]),
        asb.WingXSec(xyz_le=[-0.25 * wing_root_chord, wing_span / 2 * wing_center_span_fraction, 0],
                     chord=wing_root_chord, airfoil=airfoils["ag13"]),
        asb.WingXSec(
            xyz_le=[-0.25 * wing_root_chord * 0.7, wing_span / 2,
                    wing_span / 2 * (1 - wing_center_span_fraction) * np.tand(wing_dihedral_angle_deg)],
            chord=wing_root_chord * 0.7, airfoil=airfoils["ag13"])
    ])

    vtail_dihedral_deg = 37
    v_tail_span = 0.35 * wing_span
    v_tail_root_chord = 0.08
    vtail = asb.Wing(
        name="V-Tail", symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0, 0, 0], chord=v_tail_root_chord, airfoil=airfoils["naca0008"]),
            asb.WingXSec(
                xyz_le=[0.02, v_tail_span / 2, (v_tail_span / 2) * np.tand(vtail_dihedral_deg)],
                chord=v_tail_root_chord * 0.8, airfoil=airfoils["naca0008"])
        ]
    ).translate([x_tail, 0, 0])

    x_pod_end = x_nose + 8 * u.inch
    fuselage = asb.Fuselage(
        name="Fuse",
        xsecs=[
            asb.FuselageXSec(xyz_c=[x_nose, 0, 0], height=1.5 * u.inch, width=2.0 * u.inch),
            asb.FuselageXSec(xyz_c=[x_pod_end, 0, 0], height=1.5 * u.inch, width=2.0 * u.inch),
            asb.FuselageXSec(xyz_c=[x_pod_end, 0, 0], radius=7e-3 / 2),
            asb.FuselageXSec(xyz_c=[x_tail, 0, 0], radius=7e-3 / 2)
        ]
    )

    airplane = asb.Airplane(name="Feather V-Tail", wings=[wing, vtail], fuselages=[fuselage])

    # --- Mass
    mass_props = {}
    density = 2 * u.lbm / u.foot ** 3
    mass_props['wing'] = asb.mass_properties_from_radius_of_gyration(
        mass=wing.volume() * density, x_cg=0,
        z_cg=(0.03591) * (np.sind(wing_dihedral_angle_deg) / np.sind(11)) * (wing_span / 1)
    )
    mass_props["tail_surfaces"] = asb.mass_properties_from_radius_of_gyration(
        mass=vtail.volume() * 80, x_cg=x_tail + 0.50 * v_tail_root_chord
    )
    mass_props["linkages"] = asb.MassProperties(mass=1e-3, x_cg=(x_nose + x_tail) / 2)
    mass_props["motor"] = asb.mass_properties_from_radius_of_gyration(mass=4.49e-3, x_cg=x_nose - 0.3 * u.inch)
    mass_props["motor_bolts"] = asb.mass_properties_from_radius_of_gyration(mass=4 * 0.075e-3, x_cg=x_nose)
    mass_props["propeller"] = asb.mass_properties_from_radius_of_gyration(mass=1.54e-3, x_cg=x_nose - 0.7 * u.inch)
    mass_props["propeller_band"] = asb.mass_properties_from_radius_of_gyration(
        mass=0.06e-3, x_cg=mass_props["propeller"].x_cg
    )
    mass_props["flight_computer"] = asb.mass_properties_from_radius_of_gyration(
        mass=4.30e-3, x_cg=x_nose + 2 * u.inch + (1.3 * u.inch) / 2
    )
    mass_props["battery"] = asb.mass_properties_from_radius_of_gyration(mass=4.61e-3, x_cg=x_nose + 2 * u.inch)
    mass_props["pod_structure"] = asb.MassProperties(mass=10e-3, x_cg=(x_nose + x_pod_end) / 2)
    boom_length = np.maximum(0, x_tail - x_pod_end)
    mass_props["boom"] = asb.mass_properties_from_radius_of_gyration(
        mass=7.0e-3 * (boom_length / 826e-3), x_cg=(x_pod_end + x_tail) / 2
    )
    mass_props["ballast"] = asb.MassProperties(
        mass=opti.variable(init_guess=0, lower_bound=0),
        x_cg=opti.variable(init_guess=0, lower_bound=x_nose, upper_bound=x_tail)
    )

    mass_sum_pre_glue = sum(v for k, v in mass_props.items() if k != "ballast")
    mass_props['glue_weight'] = asb.MassProperties(mass=mass_sum_pre_glue.mass * 0.08)
    mass_props_TOGW = sum(mass_props.values())

    # --- Aero & Stability
    ab = asb.AeroBuildup(airplane=airplane, op_point=op_point, xyz_ref=mass_props_TOGW.xyz_cg)
    aero = ab.run_with_stability_derivatives(alpha=True, beta=True)
    static_margin = (aero["x_np"] - mass_props_TOGW.x_cg) / wing.mean_aerodynamic_chord()
    LD = aero["L"] / aero["D"]
    sink_rate = (aero["D"] * op_point.velocity) / (9.81 * mass_props_TOGW.mass)

    # --- Objective and Constraints
    opti.minimize(sink_rate / 0.3 + mass_props_TOGW.mass / 0.100 * 0.1 + (mass_props["ballast"].x_cg / 1e3) ** 2)
    opti.subject_to([
        aero["L"] >= 9.81 * mass_props_TOGW.mass,
        aero["Cm"] == 0,
        static_margin == 0.08,
        LD_cruise == LD,
        design_mass_TOGW == mass_props_TOGW.mass
    ])
    tail_moment_arm = vtail.aerodynamic_center(chord_fraction=0.25)[0] - mass_props_TOGW.xyz_cg[0]
    projected_h_tail_area = vtail.area() * np.sind(vtail_dihedral_deg)
    opti.subject_to(
        projected_h_tail_area * tail_moment_arm / (wing.area() * wing.mean_aerodynamic_chord()) > 0.4
    )
    projected_v_tail_area = vtail.area() * np.cosd(vtail_dihedral_deg)
    opti.subject_to(
        projected_v_tail_area * tail_moment_arm / (wing.area() * wing.span()) > 0.02
    )
    opti.subject_to([
        x_nose < -0.25 * wing_root_chord - 0.5 * u.inch,
        x_tail - x_nose < 0.826
    ])

    # --- Solve and Report
    sol = opti.solve(verbose=False)
    report_solution(sol, config_name, airplane, op_point, mass_props, mass_props_TOGW, LD_cruise, sink_rate, aero)


# ##############################################################################
# # CONFIGURATION 4: REVERSE V-TAIL (TWIN BOOM)
# ##############################################################################
def optimize_reverse_v_tail_twin_boom():
    config_name = "Reverse_V_Tail_Twin_Boom"
    print_title(f"Optimizing: {config_name}")
    opti = asb.Opti()

    # --- Specs
    op_point = asb.OperatingPoint(velocity=opti.variable(init_guess=14, lower_bound=1, log_transform=True),
                                  alpha=opti.variable(init_guess=0, lower_bound=-10, upper_bound=20))
    design_mass_TOGW = opti.variable(init_guess=0.1, lower_bound=1e-3)
    design_mass_TOGW = np.maximum(design_mass_TOGW, 1e-3)
    LD_cruise = opti.variable(init_guess=15, lower_bound=0.1, log_transform=True)

    # --- Geometry
    x_nose = opti.variable(init_guess=-0.1, upper_bound=0)
    x_tail = opti.variable(init_guess=0.5, lower_bound=0.1)
    wing_root_chord = opti.variable(init_guess=0.15, lower_bound=1e-3)

    wing_center_span_fraction = 0.4
    wing = asb.Wing(name="Main Wing", symmetric=True, xsecs=[
        asb.WingXSec(xyz_le=[-0.25 * wing_root_chord, 0, 0], chord=wing_root_chord, airfoil=airfoils["ag13"]),
        asb.WingXSec(xyz_le=[-0.25 * wing_root_chord, wing_span / 2 * wing_center_span_fraction, 0],
                     chord=wing_root_chord, airfoil=airfoils["ag13"]),
        asb.WingXSec(
            xyz_le=[-0.25 * wing_root_chord * 0.7, wing_span / 2,
                    wing_span / 2 * (1 - wing_center_span_fraction) * np.tand(wing_dihedral_angle_deg)],
            chord=wing_root_chord * 0.7, airfoil=airfoils["ag13"])
    ])

    v_tail_anhedral_deg = 45
    v_tail_span = 0.40 * wing_span
    v_tail_root_chord = 0.09
    v_tail_taper = 0.7
    v_tail_sweep = 20
    tail_root_height = (v_tail_span / 2) * np.tand(v_tail_anhedral_deg)
    reverse_v_tail = asb.Wing(
        name="Reverse V-Tail", symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0, 0, 0], chord=v_tail_root_chord, airfoil=airfoils["naca0008"]),
            asb.WingXSec(
                xyz_le=[(v_tail_span / 2) * np.tand(v_tail_sweep), v_tail_span / 2,
                        -(v_tail_span / 2) * np.tand(v_tail_anhedral_deg)],
                chord=v_tail_root_chord * v_tail_taper, airfoil=airfoils["naca0008"])
        ]
    ).translate([x_tail, 0, tail_root_height])

    x_pod_end = x_nose + 8 * u.inch
    fuselage = asb.Fuselage(
        name="Fuse",
        xsecs=[
            asb.FuselageXSec(xyz_c=[x_nose, 0, 0], height=1.5 * u.inch, width=2.0 * u.inch),
            asb.FuselageXSec(xyz_c=[x_pod_end, 0, 0], height=1.5 * u.inch, width=2.0 * u.inch),
            asb.FuselageXSec(xyz_c=[x_pod_end, 0, 0], radius=7e-3 / 2),
            asb.FuselageXSec(xyz_c=[x_tail, 0, 0], radius=7e-3 / 2)
        ]
    )

    v_tail_tip_chord = v_tail_root_chord * v_tail_taper
    boom_start_x = x_tail + (v_tail_span / 2) * np.tand(v_tail_sweep) + v_tail_tip_chord
    boom_start_y = v_tail_span / 2
    boom_end_x = 0.75 * wing_root_chord
    boom_end_y = wing_span / 2 * wing_center_span_fraction
    boom_radius = 6e-3 / 2
    boom_right = asb.Fuselage(
        name="Right Boom",
        xsecs=[
            asb.FuselageXSec(xyz_c=[boom_end_x, boom_end_y, 0], radius=boom_radius),
            asb.FuselageXSec(xyz_c=[boom_start_x, boom_start_y, 0], radius=boom_radius)
        ]
    )
    boom_left = asb.Fuselage(
        name="Left Boom",
        xsecs=[
            asb.FuselageXSec(xyz_c=[boom_end_x, -boom_end_y, 0], radius=boom_radius),
            asb.FuselageXSec(xyz_c=[boom_start_x, -boom_start_y, 0], radius=boom_radius)
        ]
    )

    airplane = asb.Airplane(name="Feather Twin-Boom V-Tail", wings=[wing, reverse_v_tail],
                            fuselages=[fuselage, boom_right, boom_left])

    # --- Mass
    mass_props = {}
    density = 2 * u.lbm / u.foot ** 3
    mass_props['wing'] = asb.mass_properties_from_radius_of_gyration(
        mass=wing.volume() * density, x_cg=0,
        z_cg=(0.03591) * (np.sind(wing_dihedral_angle_deg) / np.sind(11)) * (wing_span / 1)
    )
    mass_props["tail_surfaces"] = asb.mass_properties_from_radius_of_gyration(
        mass=reverse_v_tail.volume() * 80, x_cg=x_tail + 0.50 * v_tail_root_chord
    )
    mass_props["linkages"] = asb.MassProperties(mass=1e-3, x_cg=(x_nose + x_tail) / 2)
    mass_props["motor"] = asb.mass_properties_from_radius_of_gyration(mass=4.49e-3, x_cg=x_nose - 0.3 * u.inch)
    mass_props["motor_bolts"] = asb.mass_properties_from_radius_of_gyration(mass=4 * 0.075e-3, x_cg=x_nose)
    mass_props["propeller"] = asb.mass_properties_from_radius_of_gyration(mass=1.54e-3, x_cg=x_nose - 0.7 * u.inch)
    mass_props["propeller_band"] = asb.mass_properties_from_radius_of_gyration(
        mass=0.06e-3, x_cg=mass_props["propeller"].x_cg
    )
    mass_props["flight_computer"] = asb.mass_properties_from_radius_of_gyration(
        mass=4.30e-3, x_cg=x_nose + 2 * u.inch + (1.3 * u.inch) / 2
    )
    mass_props["battery"] = asb.mass_properties_from_radius_of_gyration(mass=4.61e-3, x_cg=x_nose + 2 * u.inch)
    mass_props["pod_structure"] = asb.MassProperties(mass=10e-3, x_cg=(x_nose + x_pod_end) / 2)
    center_boom_length = np.maximum(0, x_tail - x_pod_end)
    mass_props["center_boom"] = asb.MassProperties(mass=7.0e-3 * (center_boom_length / 826e-3),
                                                   x_cg=(x_pod_end + x_tail) / 2)
    boom_length_twin = np.sqrt((boom_start_x - boom_end_x) ** 2 + (boom_start_y - boom_end_y) ** 2)
    mass_props["twin_booms"] = asb.mass_properties_from_radius_of_gyration(
        mass=2 * 7.0e-3 * (boom_length_twin / 826e-3), x_cg=(boom_start_x + boom_end_x) / 2
    )
    mass_props["ballast"] = asb.MassProperties(
        mass=opti.variable(init_guess=0, lower_bound=0),
        x_cg=opti.variable(init_guess=0, lower_bound=x_nose, upper_bound=x_tail)
    )

    mass_sum_pre_glue = sum(v for k, v in mass_props.items() if k != "ballast")
    mass_props['glue_weight'] = asb.MassProperties(mass=mass_sum_pre_glue.mass * 0.08)
    mass_props_TOGW = sum(mass_props.values())

    # --- Aero & Stability
    ab = asb.AeroBuildup(airplane=airplane, op_point=op_point, xyz_ref=mass_props_TOGW.xyz_cg)
    aero = ab.run_with_stability_derivatives(alpha=True, beta=True)
    static_margin = (aero["x_np"] - mass_props_TOGW.x_cg) / wing.mean_aerodynamic_chord()
    LD = aero["L"] / aero["D"]
    sink_rate = (aero["D"] * op_point.velocity) / (9.81 * mass_props_TOGW.mass)

    # --- Objective and Constraints
    opti.minimize(sink_rate / 0.3 + mass_props_TOGW.mass / 0.100 * 0.1 + (mass_props["ballast"].x_cg / 1e3) ** 2)
    opti.subject_to([
        aero["L"] >= 9.81 * mass_props_TOGW.mass,
        aero["Cm"] == 0,
        static_margin == 0.08,
        LD_cruise == LD,
        design_mass_TOGW == mass_props_TOGW.mass
    ])
    tail_moment_arm = reverse_v_tail.aerodynamic_center(chord_fraction=0.25)[0] - mass_props_TOGW.xyz_cg[0]
    projected_h_tail_area = reverse_v_tail.area() * np.cosd(v_tail_anhedral_deg)
    opti.subject_to(
        projected_h_tail_area * tail_moment_arm / (wing.area() * wing.mean_aerodynamic_chord()) > 0.4
    )
    projected_v_tail_area = reverse_v_tail.area() * np.sind(v_tail_anhedral_deg)
    opti.subject_to(
        projected_v_tail_area * tail_moment_arm / (wing.area() * wing.span()) > 0.02
    )
    opti.subject_to([
        x_nose < -0.25 * wing_root_chord - 0.5 * u.inch,
        x_tail - x_nose < 0.826
    ])

    # --- Solve and Report
    sol = opti.solve(verbose=False)
    report_solution(sol, config_name, airplane, op_point, mass_props, mass_props_TOGW, LD_cruise, sink_rate, aero)


# ##############################################################################
# # CONFIGURATION 5: FLYING WING (UPDATED)
# ##############################################################################
def optimize_flying_wing():
    config_name = "Flying_Wing"
    print_title(f"Optimizing: {config_name}")
    opti = asb.Opti()

    # --- Specs
    op_point = asb.OperatingPoint(
        velocity=opti.variable(init_guess=14, lower_bound=1, log_transform=True),
        alpha=opti.variable(init_guess=2.0, lower_bound=-10, upper_bound=20)
    )
    design_mass_TOGW = opti.variable(init_guess=0.09, lower_bound=1e-3)
    design_mass_TOGW = np.maximum(design_mass_TOGW, 1e-3)
    LD_cruise = opti.variable(init_guess=14, lower_bound=0.1, log_transform=True)

    # --- Geometry
    wing_root_chord = opti.variable(init_guess=0.25, lower_bound=1e-3)
    wing_tip_twist_deg = opti.variable(init_guess=-3, lower_bound=-10, upper_bound=0)  # Washout

    wing_dihedral_angle_deg_local = 10  # Reduced dihedral for a more classic flying wing look
    wing_sweep_deg = 30
    wing_taper_ratio = 0.5
    wing_center_span_fraction = 0.20

    span_break = wing_span / 2 * wing_center_span_fraction
    span_tip = wing_span / 2
    span_outer_panel = span_tip - span_break
    chord_tip = wing_root_chord * wing_taper_ratio

    xyz_le_break = np.array([-0.25 * wing_root_chord, span_break, 0])
    xyz_le_tip = np.array([
        xyz_le_break[0] + span_outer_panel * np.tand(wing_sweep_deg),
        span_tip,
        xyz_le_break[2] + span_outer_panel * np.tand(wing_dihedral_angle_deg_local)
    ])

    wing = asb.Wing(
        name="Main Wing", symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[-0.25 * wing_root_chord, 0, 0], chord=wing_root_chord, twist=0, airfoil=airfoils["ag13"]),
            asb.WingXSec(xyz_le=xyz_le_break, chord=wing_root_chord, twist=0, airfoil=airfoils["ag13"]),
            asb.WingXSec(xyz_le=xyz_le_tip, chord=chord_tip, twist=wing_tip_twist_deg, airfoil=airfoils["ag13"])
        ]
    )

    winglet_height = 0.08; winglet_chord = 0.07; winglet_sweep = 15; winglet_cant = 10
    winglets = asb.Wing(
        name="Winglets", symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=xyz_le_tip + np.array([chord_tip * 0.75, 0, 0]), chord=winglet_chord, airfoil=airfoils["naca0008"]),
            asb.WingXSec(xyz_le=xyz_le_tip + np.array([
                chord_tip * 0.75 + winglet_height * np.tand(winglet_sweep),
                winglet_height * np.sind(winglet_cant),
                winglet_height * np.cosd(winglet_cant)
            ]), chord=winglet_chord * 0.7, airfoil=airfoils["naca0008"])
        ]
    )
    airplane = asb.Airplane(name="Feather Flying Wing", wings=[wing, winglets], fuselages=[])

    # --- Mass
    mass_props = {}
    density = 2 * u.lbm / u.foot ** 3
    mass_props['wing'] = asb.mass_properties_from_radius_of_gyration(
        mass=wing.volume() * density, x_cg=wing.aerodynamic_center()[0], z_cg=wing.aerodynamic_center()[2])
    mass_props["winglets"] = asb.mass_properties_from_radius_of_gyration(
        mass=winglets.volume() * 80, x_cg=winglets.aerodynamic_center()[0], z_cg=winglets.aerodynamic_center()[2])
    x_pusher_prop_mount = 0.6 * wing_root_chord
    mass_props["motor"] = asb.mass_properties_from_radius_of_gyration(mass=4.49e-3, x_cg=x_pusher_prop_mount - 0.3 * u.inch)
    mass_props["motor_bolts"] = asb.mass_properties_from_radius_of_gyration(mass=4 * 0.075e-3, x_cg=x_pusher_prop_mount)
    mass_props["propeller"] = asb.mass_properties_from_radius_of_gyration(mass=1.54e-3, x_cg=x_pusher_prop_mount + 0.4 * u.inch)
    mass_props["propeller_band"] = asb.mass_properties_from_radius_of_gyration(mass=0.06e-3, x_cg=mass_props["propeller"].x_cg)
    x_avionics_bay = -0.1 * wing_root_chord
    mass_props["flight_computer"] = asb.mass_properties_from_radius_of_gyration(mass=4.30e-3, x_cg=x_avionics_bay)
    mass_props["battery"] = asb.mass_properties_from_radius_of_gyration(mass=4.61e-3, x_cg=x_avionics_bay - 1.5 * u.inch)
    mass_props["ballast"] = asb.MassProperties(
        mass=opti.variable(init_guess=0, lower_bound=0),
        x_cg=opti.variable(init_guess=0, lower_bound=-0.25 * wing_root_chord, upper_bound=0.75 * wing_root_chord)
    )

    mass_sum_pre_glue = sum(v for k, v in mass_props.items() if k != "ballast")
    mass_props['glue_weight'] = asb.MassProperties(mass=mass_sum_pre_glue.mass * 0.08)
    mass_props_TOGW = sum(mass_props.values())

    # --- Aero & Stability
    ab = asb.AeroBuildup(airplane=airplane, op_point=op_point, xyz_ref=mass_props_TOGW.xyz_cg)
    aero = ab.run_with_stability_derivatives(alpha=True, beta=True)
    static_margin = (aero["x_np"] - mass_props_TOGW.x_cg) / wing.mean_aerodynamic_chord()
    LD = aero["L"] / aero["D"]
    sink_rate = (aero["D"] * op_point.velocity) / (9.81 * mass_props_TOGW.mass)

    # --- Objective and Constraints
    opti.minimize(sink_rate / 0.3 + mass_props_TOGW.mass / 0.100 * 0.1 + (mass_props["ballast"].x_cg / 1e3) ** 2)
    opti.subject_to([
        aero["L"] >= 9.81 * mass_props_TOGW.mass,
        aero["Cm"] == 0,
        static_margin == 0.05,
        LD_cruise == LD,
        design_mass_TOGW == mass_props_TOGW.mass,
    ])

    # --- Solve and Report
    sol = opti.solve(verbose=False)
    report_solution(sol, config_name, airplane, op_point, mass_props, mass_props_TOGW, LD_cruise, sink_rate, aero)


if __name__ == "__main__":
    # Each call will open one composite figure window (no saving).
    optimize_h_tail()
    optimize_conventional_tail()
    optimize_v_tail()
    optimize_reverse_v_tail_twin_boom()
    optimize_flying_wing()

    # --- Create and print the final comparison table
    print("\n\n" + "=" * 80)
    print("      OVERALL COMPARISON OF OPTIMIZED CONFIGURATIONS")
    print("=" * 80)

    if results_summary:
        df = pd.DataFrame(results_summary)
        df.set_index("Configuration", inplace=True)
        # Format columns for better readability
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].map('{:.3f}'.format)
        print(df.to_string())
    else:
        print("No results were generated to compare.")
