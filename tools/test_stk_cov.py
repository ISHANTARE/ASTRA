import numpy as np

def build_6x6_from_lower_triangular(values) -> np.ndarray:
    assert len(values) == 21
    cov = np.zeros((6, 6))
    idx = 0
    for i in range(6):
        for j in range(i + 1):
            cov[i, j] = values[idx]
            cov[j, i] = values[idx]
            idx += 1
    return cov

line = "0.000000000000000 0.003436819735943 -0.002165947283578 0.001366481567153 0.003338255835269 -0.002103125776542 0.003247739819309 0.000000694687274 -0.000000436650434 0.000000677203572 0.000000000143064 0.000005182656689 -0.000003267062474 0.000005035164665 0.000000001047041 0.000000007818310 0.000002789469760 -0.000001756901153 0.000002713154090 0.000000000564606 0.000000004208545 0.000000002269302"
vals = [float(x) for x in line.split()]
print("Time:", vals[0])
cov = build_6x6_from_lower_triangular(vals[1:])
print("Cov matrix shape:", cov.shape)
print("Cond:", np.linalg.cond(cov))
print("Eigenvalues:", np.linalg.eigvalsh(cov))
