import numpy as np
import pytest
from realgas_ml import GasModel, gases

@pytest.mark.parametrize("gas", gases())
def test_volume_and_inverse(gas):
    m = GasModel(gas)
    T = np.mean(m.info["T_range_K"])
    r = m.predict(T, 1e6, n=2)
    assert r["v_m3_mol"] > 0
    assert r["V_m3"] == 2*r["v_m3_mol"]
    assert m.pressure(T,r["v_m3_mol"]) == pytest.approx(1e6,rel=1e-7)
    assert r["H_J"] == 2*r["h_J_mol"]

def test_reject_invalid():
    m=GasModel("Nitrogen")
    for T,P,n in [(100,1e6,1),(400,0,1),(400,1e6,0),(float("nan"),1e6,1)]:
        with pytest.raises(ValueError): m.predict(T,P,n)
    with pytest.raises(ValueError): GasModel("unknown")
