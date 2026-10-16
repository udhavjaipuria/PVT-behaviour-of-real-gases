import json
from importlib.resources import files
import joblib
import numpy as np
from scipy.optimize import brentq

R = 8.31446261815324

def gases():
    return tuple(json.loads(files("realgas_ml").joinpath("data/metadata.json").read_text())["fluids"])

class GasModel:
    """Predict v, Z, molar h and molar s from temperature and pressure.

    Predictions are independent statistical surrogates, not a consistent EOS.
    Out-of-domain predictions are rejected. Names use CoolProp spelling.
    """
    def __init__(self, gas):
        metadata = json.loads(files("realgas_ml").joinpath("data/metadata.json").read_text())
        if gas not in metadata["fluids"]:
            raise ValueError(f"Unknown gas {gas!r}; choose from {gases()}")
        self.gas = gas
        self.info = metadata["fluids"][gas]
        self.model = joblib.load(files("realgas_ml").joinpath(f"data/{gas}.joblib"))

    def predict(self, T, P, n=1.0):
        """Scalar inputs: T [K], P [Pa], n [mol]. H [J], S [J/K].

        h/s follow CoolProp DEF per-fluid reference conventions, not formation
        enthalpies or a shared absolute-entropy convention.
        """
        T, P, n = float(T), float(P), float(n)
        if not np.all(np.isfinite([T, P, n])) or n <= 0:
            raise ValueError("Inputs must be finite and n must be positive")
        lo, hi = self.info["T_range_K"]
        plo, phi = self.info["P_range_Pa"]
        if not lo <= T <= hi or not plo <= P <= phi:
            raise ValueError(f"Outside domain: T in [{lo}, {hi}] K, P in [{plo}, {phi}] Pa")
        x = np.array([[(T-lo)/(hi-lo), np.log(P/plo)/np.log(phi/plo)]])
        values = self.model.predict(x)[0].tolist()
        z = float(np.exp(values[0]))
        v = z * self.info["gas_constant_J_mol_K"] * T / P
        return dict(gas=self.gas, T_K=T, P_Pa=P, Z=z, v_m3_mol=v,
                    V_m3=n*v, rho_kg_m3=self.info["molar_mass_kg_mol"]/v,
                    h_J_mol=values[1], s_J_mol_K=values[2], H_J=n*values[1],
                    S_J_K=n*values[2])

    def pressure(self, T, v):
        """Invert predicted molar volume [m³/mol] at T [K] to P [Pa]."""
        if not np.isfinite(v) or v <= 0:
            raise ValueError("Molar volume must be finite and positive")
        lo, hi = self.info["P_range_Pa"]
        f = lambda p: self.predict(T, p)["v_m3_mol"] - v
        if f(lo)*f(hi) > 0:
            raise ValueError("Volume outside invertible pressure domain")
        return float(brentq(f, lo, hi))

