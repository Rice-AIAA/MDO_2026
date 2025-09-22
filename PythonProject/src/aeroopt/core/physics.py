import aerosandbox.numpy as np

def fuselage_cd(rho, mu, S_ref_nom, Lf, df, V):
    Swet = np.pi * df * Lf
    ReL = np.maximum(rho * V * Lf / mu, 1e3)
    Cf  = 0.455 / (np.log10(ReL)**2.58 + 1e-9)
    FF  = 1.2 + 0.15 * (Lf/df - 5.0)**2
    return (Cf * FF * Swet) / S_ref_nom

def smooth_hinge(x, eps=1e-6):
    return 0.5 * (x + np.sqrt(x*x + eps))
