# src/aeroopt/core/aero.py
from __future__ import annotations
import aerosandbox as asb
import aerosandbox.numpy as np
from pathlib import Path

_AIRFOIL_CFG = {
    "xfoil_path": None,
    "alphas_deg": list(np.linspace(-6.0, 18.0, 13)),
    "cache_dir": None,
}

def configure_airfoils(cfg: dict | None = None, cache_dir: str | Path | None = None):
    global _AIRFOIL_CFG
    if cfg:
        for k in ("xfoil_path", "alphas_deg"):
            if k in cfg:
                _AIRFOIL_CFG[k] = cfg[k]
    if cache_dir is not None:
        _AIRFOIL_CFG["cache_dir"] = str(cache_dir)

def ensure_polar(name: str):
    af = asb.Airfoil(name.lower())
    # ✅ ensure NumPy array (AeroSandbox expects array-like with .min())
    alphas = np.array(_AIRFOIL_CFG["alphas_deg"], dtype=float)

    cache_dir = _AIRFOIL_CFG["cache_dir"]
    cache_file = None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache_file = str(Path(cache_dir) / f"{af.name}.json")

    xfoil_path = _AIRFOIL_CFG["xfoil_path"]
    if xfoil_path:
        # ✅ correct kw for modern AeroSandbox
        af.generate_polars(cache_filename=cache_file,
                           alphas=alphas,
                           xfoil_executable=xfoil_path)
    else:
        af.generate_polars(cache_filename=cache_file, alphas=alphas)

def ensure_polars_for_list(names: list[str]):
    seen = set()
    for n in names:
        if n and n.lower() not in seen:
            ensure_polar(n)
            seen.add(n.lower())

def aero_at(airplane: asb.Airplane, rho: float, V: float, alpha_deg) -> dict:
    op = asb.OperatingPoint(velocity=V, alpha=alpha_deg, beta=0, p=0, q=0, r=0)
    ab = asb.AeroBuildup(airplane=airplane, op_point=op)
    out = ab.run()
    CL = out.get("CL", out.get("cl"))
    CD = out.get("CD", out.get("cd"))
    Cm = out.get("Cm", out.get("CM", out.get("cm")))
    q = 0.5 * rho * V**2
    L = CL * q * airplane.s_ref
    return {"CL": CL, "CD": CD, "Cm": Cm, "L": L}
