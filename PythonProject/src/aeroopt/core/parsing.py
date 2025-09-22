from dataclasses import dataclass

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
        # accepts "...__wingA_AR8.5_tap0.5_dih10_tw2_-1_ag13" OR "wingA_AR8.5_..."
        wing = config_id.split("__", 1)[-1]
        parts = wing.split("_")
        return Family(
            name=parts[0],
            AR=float(parts[1].replace("AR","")),
            taper=float(parts[2].replace("tap","")),
            dihedral_deg=float(parts[3].replace("dih","")),
            twist_root_deg=float(parts[4].replace("tw","")),
            twist_tip_deg=float(parts[5]),
            airfoil_name=parts[6],
        )
