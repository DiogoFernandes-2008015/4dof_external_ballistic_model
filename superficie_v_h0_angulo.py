"""
Superfície: Velocidade de Lançamento x Altura x Ângulo de Impacto
====================================================================
Lançamento horizontal (QE = 0°) — granada solta de drone em voo reto.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mortar_120mm_mpm import solve_trajectory

V_LIST  = np.arange(10, 41, 2.5)     # velocidade de voo do drone [m/s]
H0_LIST = np.arange(20, 521, 20)     # altura de lançamento [m]

ANG = np.zeros((len(H0_LIST), len(V_LIST)))
RNG = np.zeros((len(H0_LIST), len(V_LIST)))

print("Calculando grade (pode levar alguns minutos)...")
for i, h0 in enumerate(H0_LIST):
    for j, v in enumerate(V_LIST):
        res = solve_trajectory(muzzle_velocity=float(v), elevation_deg=0.0,
                                launch_height=float(h0), rtol=1e-6, atol=1e-6)
        ANG[i, j] = res['impact_angle_deg']
        RNG[i, j] = res['x_impact']
    print(f"  h0 = {h0:>4.0f} m  ...ok")

VV, HH = np.meshgrid(V_LIST, H0_LIST)

np.savez('superficie_dados.npz', V=V_LIST, H0=H0_LIST, ANG=ANG, RNG=RNG)

# ── Figura 1: Superfície 3D ──────────────────────────────────────
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(VV, HH, ANG, cmap='viridis', edgecolor='k',
                        linewidth=0.15, antialiased=True, alpha=0.95)
ax.set_xlabel('Velocidade de voo do drone [m/s]', fontsize=10, labelpad=10)
ax.set_ylabel('Altura de lançamento h₀ [m]', fontsize=10, labelpad=10)
ax.set_zlabel('Ângulo de impacto [°]', fontsize=10, labelpad=8)
ax.set_title('Ângulo de Impacto — Superfície V x h₀\n(QE = 0°, lançamento horizontal, Morteiro 120mm)',
             fontsize=12, fontweight='bold')
fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08, label='Ângulo de impacto [°]')
ax.view_init(elev=25, azim=-60)
fig.tight_layout()
fig.savefig('superficie_3d_angulo.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("→ superficie_3d_angulo.png")

# ── Figura 2: Mapa de contorno (mais fácil de ler valores) ───────
fig2, ax2 = plt.subplots(figsize=(9, 7))
cf = ax2.contourf(VV, HH, ANG, levels=20, cmap='viridis')
cs = ax2.contour(VV, HH, ANG, levels=10, colors='k', linewidths=0.6, alpha=0.6)
ax2.clabel(cs, inline=True, fontsize=8, fmt='%.0f°')
ax2.set_xlabel('Velocidade de voo do drone [m/s]', fontsize=11)
ax2.set_ylabel('Altura de lançamento h₀ [m]', fontsize=11)
ax2.set_title('Mapa de Ângulo de Impacto [°]\n(QE = 0°, lançamento horizontal, Morteiro 120mm)',
              fontsize=12, fontweight='bold')
fig2.colorbar(cf, ax=ax2, label='Ângulo de impacto [°]')
fig2.tight_layout()
fig2.savefig('mapa_contorno_angulo.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print("→ mapa_contorno_angulo.png")

print()
print("Resumo (linhas = h0, colunas = V):")
header = "h0\\V " + " ".join(f"{v:>5.1f}" for v in V_LIST)
print(header)
for i, h0 in enumerate(H0_LIST):
    if h0 % 100 == 0:
        print(f"{h0:>4.0f} " + " ".join(f"{ANG[i,j]:>5.1f}" for j in range(len(V_LIST))))
