from realgas_ml import GasModel

gas = GasModel("CarbonDioxide")
properties = gas.predict(T=400, P=2e6, n=2)
for name, value in properties.items():
    print(f"{name}: {value}")
print("Recovered pressure [Pa]:", gas.pressure(400, properties["v_m3_mol"]))
