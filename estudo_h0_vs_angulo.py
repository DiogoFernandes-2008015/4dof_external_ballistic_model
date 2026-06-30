"""
Estudo paramétrico: altura de lançamento (h0) x ângulo de impacto
==================================================================
Lançamento horizontal (QE = 0°) — granada solta de drone em voo reto,
velocidade de lançamento = velocidade de voo do drone.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mortar_120mm_mpm import solve_trajectory

MV_LIST = [15, 25, 35]   # velocidades de voo do drone [m/s]
H0_LIST = np.arange(20, 521, 20)  # altitudes de lançamento [m]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

resultados = {}
for mv in MV_LIST:
    angs, alcances = [], []
    for h0 in H0_LIST:
        res = solve_trajectory(muzzle_velocity=mv, elevation_deg=0.0,
                                launch_height=float(h0), rtol=1e-6, atol=1e-6)
        angs.append(res['impact_angle_deg'])
        alcances.append(res['x_impact'])
    resultados[mv] = (angs, alcances)
    ax1.plot(H0_LIST, angs, marker='o', ms=3.5, lw=1.8, label=f'V$_{{drone}}$ = {mv} m/s')
    ax2.plot(H0_LIST, alcances, marker='o', ms=3.5, lw=1.8, label=f'V$_{{drone}}$ = {mv} m/s')

ax1.set_xlabel('Altura de lançamento h₀ [m]', fontsize=11)
ax1.set_ylabel('Ângulo de impacto [°] (0°=raso, 90°=vertical)', fontsize=11)
ax1.set_title('Ângulo de Impacto vs Altura de Lançamento\n(QE = 0°, lançamento horizontal)', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3); ax1.legend(fontsize=9)

ax2.set_xlabel('Altura de lançamento h₀ [m]', fontsize=11)
ax2.set_ylabel('Alcance horizontal de impacto [m]', fontsize=11)
ax2.set_title('Alcance vs Altura de Lançamento\n(QE = 0°, lançamento horizontal)', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3); ax2.legend(fontsize=9)

fig.tight_layout()
fig.savefig('h0_vs_angulo_impacto.png', dpi=150, bbox_inches='tight')
print("→ h0_vs_angulo_impacto.png")

print()
print(f"{'h0 [m]':>8} | " + " | ".join(f"V={mv}m/s ang[°]" for mv in MV_LIST))
for i, h0 in enumerate(H0_LIST):
    if h0 % 100 == 0:
        linha = f"{h0:>8} | " + " | ".join(f"{resultados[mv][0][i]:>13.2f}" for mv in MV_LIST)
        print(linha)
