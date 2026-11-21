import pandas as pd, numpy as np

RUN = "runs/regression/rdkit_mlp/2026-11-16_17-36-35"   # <- your run folder
g = "CYP1A2_pIC50_direct_inhibition"

val  = pd.read_csv("data/regression/val.csv")
pred = pd.read_csv(f"{RUN}/validation_predictions.csv")

y = val[g].to_numpy(float)
m = np.isfinite(y)                      # only labelled molecules
y, p = y[m], pred[g].to_numpy(float)[m]
low  = val[g + "_conf_low"].to_numpy(float)[m]
high = val[g + "_conf_high"].to_numpy(float)[m]

error = lambda q: np.maximum(low - q, 0) + np.maximum(q - high, 0)

print(f"{m.sum()} labelled molecules")
print(f"{(error(p) == 0).sum()} landed inside their interval (zero error)")
print(f"numerator   {error(p).sum():.3f}")
print(f"denominator {error(y.mean()).sum():.3f}")
print(f"ST-RAE      {error(p).sum() / error(y.mean()).sum():.4f}")