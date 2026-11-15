"""Estimates and bounds vs volume fraction for a two-phase isotropic material (spheres)."""
import numpy as np
import matplotlib.pyplot as plt
from pyhom import (isotropic_stiffness, bulk_shear, voigt, reuss, hashin_shtrikman,
                   mori_tanaka, selfconsistent)

soft, stiff = (100.0, 40.0), (200.0, 100.0)
Lp = [isotropic_stiffness(*soft), isotropic_stiffness(*stiff)]
fs = np.linspace(0.0, 1.0, 21)

curves = {k: [] for k in ("Voigt", "Reuss", "HS upper", "HS lower", "MT (soft matrix)",
                          "MT (stiff matrix)", "Self-consistent")}
for f in fs:
    fv = [1 - f, f]
    curves["Voigt"].append(bulk_shear(voigt(fv, Lp)))
    curves["Reuss"].append(bulk_shear(reuss(fv, Lp)))
    curves["HS upper"].append(hashin_shtrikman(1 - f, stiff, soft) if 0 < f < 1 else (soft if f == 0 else stiff))
    curves["HS lower"].append(hashin_shtrikman(f, soft, stiff) if 0 < f < 1 else (soft if f == 0 else stiff))
    curves["MT (soft matrix)"].append(bulk_shear(mori_tanaka(fv, Lp, 0, n=24)))
    curves["MT (stiff matrix)"].append(bulk_shear(mori_tanaka(fv, Lp, 1, n=24)))
    curves["Self-consistent"].append(bulk_shear(selfconsistent(fv, Lp, n=24)[0]))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for j, name in enumerate(["Bulk modulus k", "Shear modulus mu"]):
    for key, v in curves.items():
        ax[j].plot(fs, np.array(v)[:, j], label=key)
    ax[j].set_xlabel("volume fraction of the stiff phase"); ax[j].set_ylabel(name); ax[j].grid(alpha=.3)
ax[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig("biphasic_estimates.png", dpi=150); plt.show()
