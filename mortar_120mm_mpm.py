"""
Trajetória de Morteiro 120mm — Modelo Massa Ponto Modificado (MPM)
==================================================================
Referência: McCoy, R. L., "Modern Exterior Ballistics", 2ª ed.
            Capítulo 9, §9.7–9.8 (equações 9.58–9.62)

USO
---
    python mortar_120mm_mpm.py --mv 318 --qe 65 --h0 0
    python mortar_120mm_mpm.py --mv 102 --qe 45 --h0 200 --saida resultados/

Parâmetros obrigatórios:
  --mv   velocidade de lançamento [m/s]
  --qe   ângulo de elevação (QE)   [graus]

Parâmetros opcionais:
  --h0      altura de lançamento acima do solo [m]   (padrão: 0)
  --saida   diretório de saída                       (padrão: ./)
  --tmax    tempo máximo de integração [s]            (padrão: automático)
  --rtol    tolerância relativa do integrador         (padrão: 1e-7)
  --atol    tolerância absoluta do integrador         (padrão: 1e-7)

Modelo MPM para morteiro de aletas (não girante, p = 0):
─────────────────────────────────────────────────────────
  dV/dt = -(ρS·CD)/(2m)·v·V + (ρS·CL_α)/(2m)·v²·α_R + g   (eq. 9.59)

  Yaw de repouso — eq. 9.58 (Bradley, forma simplificada para aletas):
    α_R = (d / v⁴) · [v × (v × g)]

  CD   = CD₀  + CD_δ²·sin²αt
  CL_α = CL_α₀ + CL_α₂·sin²αt
  CM_α = CM_α₀ + CM_α₂·sin²αt
  (todos do Appendix C, McCoy p. 218)

Implementação:
  • Diffrax Dopri5 (RK4/5 adaptativo) via jax.pure_callback
  • Interpolação de Lagrange 4 pontos para todos os coeficientes
  • Atmosfera ICAO (McCoy Cap. 8, convertida para SI)
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax

# ══════════════════════════════════════════════════════════════════
# 1. CONSTANTES FÍSICAS — MORTEIRO 120mm (McCoy Tab. 9.5 / App. C)
# ══════════════════════════════════════════════════════════════════
D_REF   = 0.11956                    # diâmetro de referência [m]
MASS    = 13.585                     # massa [kg]
S_REF   = np.pi * D_REF**2 / 4.0    # área de referência [m²]
G0      = 9.80665                    # gravidade padrão [m/s²]
R_EARTH = 6.371e6                    # raio terrestre [m]

# ══════════════════════════════════════════════════════════════════
# 2. TABELAS DE COEFICIENTES AERODINÂMICOS (App. C, McCoy p. 218)
#    Nomenclatura BRL Aeroballistic
# ══════════════════════════════════════════════════════════════════
_CD0_M   = np.array([0.0,  0.70,  0.85,  0.87,  0.90,  0.93,  0.95])
_CD0_V   = np.array([0.119, 0.119, 0.120, 0.122, 0.126, 0.148, 0.182])

_CD2_M   = np.array([0.0,  0.40,  0.60,  0.70,  0.75,  0.85,  0.90,  0.95])
_CD2_V   = np.array([2.32, 2.44,  2.66,  2.87,  3.01,  3.55,  4.03,  5.20])

_CLA0_M  = np.array([0.0,  0.60,  0.80,  0.90,  0.95])
_CLA0_V  = np.array([1.75, 1.95,  2.02,  2.06,  2.08])

_CLA2_M  = np.array([0.0,  0.50,  0.60,  0.63,  0.70,  0.80,  0.90,  0.95])
_CLA2_V  = np.array([14.8, 14.8,   4.5,   1.4,   0.4,   8.8,  28.3,  40.0])

_CMA0_M  = np.array([0.0,  0.40,  0.60,  0.80,  0.90,  0.92,  0.95])
_CMA0_V  = np.array([-0.02,-1.02,-1.62, -2.41, -2.72, -2.75, -2.71])

_CMA2_M  = np.array([0.0,  0.45,  0.60,  0.70,  0.75,  0.80,  0.85,  0.90,  0.95])
_CMA2_V  = np.array([-15.1,-15.1,-12.7,  -8.5,  -4.5,   1.5,  13.9,  30.2,  59.9])

_CMQD0_M = np.array([0.0,  0.80,  0.85,  0.90,  0.92,  0.95])
_CMQD0_V = np.array([-22.0,-21.1,-21.9, -24.2, -26.8, -31.5])

_CMQD2_M = np.array([0.0,  0.50,  0.60,  0.70,  0.80,  0.85,  0.90,  0.95])
_CMQD2_V = np.array([ 48.0,-46.0,-86.0,-144.0,-259.0,-357.0,-468.0,-745.0])

# ══════════════════════════════════════════════════════════════════
# 3. INTERPOLAÇÃO DE LAGRANGE 4 PONTOS
# ══════════════════════════════════════════════════════════════════
def lagrange4(x_tab, y_tab, x):
    """
    Interpolação polinomial de Lagrange com 4 pontos vizinhos.
    Extrapolação constante fora do domínio tabelado.
    """
    n = len(x_tab)
    if x <= x_tab[0]:  return float(y_tab[0])
    if x >= x_tab[-1]: return float(y_tab[-1])
    idx = int(np.searchsorted(x_tab, x, side='right')) - 1
    i0  = int(np.clip(idx - 1, 0, n - 4))
    pts = range(i0, i0 + 4)
    res = 0.0
    for j in pts:
        L = 1.0
        for k in pts:
            if k != j:
                L *= (x - x_tab[k]) / (x_tab[j] - x_tab[k])
        res += float(y_tab[j]) * L
    return res

def get_aero(mach, alpha_t_rad):
    """
    Retorna (CD, CL_alpha, CM_alpha, CMqd) para dado Mach e αt [rad].
    Variação com sin²(αt) conforme App. C de McCoy.
    """
    s2   = float(np.sin(alpha_t_rad))**2
    CD   = lagrange4(_CD0_M,  _CD0_V,  mach) + lagrange4(_CD2_M,  _CD2_V,  mach) * s2
    CLa  = lagrange4(_CLA0_M, _CLA0_V, mach) + lagrange4(_CLA2_M, _CLA2_V, mach) * s2
    CMa  = lagrange4(_CMA0_M, _CMA0_V, mach) + lagrange4(_CMA2_M, _CMA2_V, mach) * s2
    CMqd = lagrange4(_CMQD0_M,_CMQD0_V,mach) + lagrange4(_CMQD2_M,_CMQD2_V,mach) * s2
    return CD, CLa, CMa, CMqd

# ══════════════════════════════════════════════════════════════════
# 4. ATMOSFERA ICAO (McCoy Cap. 8 — convertida para SI)
# ══════════════════════════════════════════════════════════════════
RHO0_SI    = 1.22500    # kg/m³  (ICAO nível do mar)
H_DECAY_SI = 9.6e-5     # 1/m    (McCoy eq. 8.19, convertida ft→m)

def atmosphere(z_m):
    """(rho [kg/m³], speed_of_sound [m/s]) para altitude z_m [m]."""
    z   = max(float(z_m), 0.0)
    T_K = max(288.15 - 0.0065 * z, 216.65)   # ISA troposfera
    a   = 20.0468 * np.sqrt(T_K)
    rho = RHO0_SI * np.exp(-H_DECAY_SI * z)
    return rho, a

def gravity(z_m):
    """Aceleração gravitacional com correção de altitude [m/s²]."""
    return G0 * (R_EARTH / (R_EARTH + max(float(z_m), 0.0)))**2

# ══════════════════════════════════════════════════════════════════
# 5. YAW DE REPOUSO — eq. 9.58 (McCoy), morteiro de aletas (p = 0)
#    α_R = (d / v⁴) · [v × (v × g)]
#    2-D (plano vertical):  v=(Vx,Vz), g=(0,-g)
#      v×g = (0, 0, -Vx·g)
#      v×(v×g) = (-Vz·Vx·g,  Vx²·g)
# ══════════════════════════════════════════════════════════════════
def yaw_of_repose_2d(Vx, Vz, speed, g):
    """Retorna (aRx, aRz) [rad] — pitch of repose no plano vertical."""
    if speed < 1e-3:
        return 0.0, 0.0
    fac = D_REF / speed**4
    return fac * (-Vz * Vx * g), fac * (Vx**2 * g)

# ══════════════════════════════════════════════════════════════════
# 6. RHS DO MPM — NUMPY PURO (via jax.pure_callback)
#    Estado: y = [x, z, Vx, Vz]
#    x  = alcance horizontal [m]   (origem = ponto de lançamento)
#    z  = altitude absoluta  [m]   (origem = nível do solo no local de disparo)
# ══════════════════════════════════════════════════════════════════
def _rhs_numpy(y_np):
    x, z, Vx, Vz = y_np
    speed = max(float(np.hypot(Vx, Vz)), 1e-3)

    rho, a_s = atmosphere(z)
    mach     = speed / a_s
    g        = gravity(z)

    # Passo 1: coefs a αt≈0, yaw de repouso inicial
    _, _, CMa, _ = get_aero(mach, 0.0)
    aRx, aRz     = yaw_of_repose_2d(Vx, Vz, speed, g)
    alpha_t      = float(np.hypot(aRx, aRz))

    # Passo 2: refina com αt calculado
    CD, CLa, CMa, _ = get_aero(mach, alpha_t)
    aRx, aRz         = yaw_of_repose_2d(Vx, Vz, speed, g)

    q = 0.5 * rho * S_REF / MASS   # ρS/(2m)

    # Arrasto (eq. 9.59, p=0, sem Magnus)
    drag = -q * CD * speed
    # Sustentação via yaw de repouso
    lift = q * CLa * speed**2

    return np.array([Vx, Vz,
                     drag * Vx + lift * aRx,
                     drag * Vz + lift * aRz - g],
                    dtype=np.float64)

_cb_shape = jax.ShapeDtypeStruct((4,), jnp.float64)

def _vector_field(t, y, args):
    return jax.pure_callback(_rhs_numpy, _cb_shape, y, vmap_method='sequential')

# ══════════════════════════════════════════════════════════════════
# 7. INTEGRAÇÃO COM DIFFRAX
# ══════════════════════════════════════════════════════════════════
def solve_trajectory(muzzle_velocity, elevation_deg, launch_height=0.0,
                     t_max=None, n_save=4000, rtol=1e-7, atol=1e-7):
    """
    Integra a trajetória MPM do morteiro 120mm.

    Parâmetros
    ----------
    muzzle_velocity : float  — velocidade de lançamento [m/s]
    elevation_deg   : float  — ângulo de elevação (QE) [°]
    launch_height   : float  — altura de lançamento acima do solo [m]
    t_max           : float  — tempo máximo [s] (None = estimativa automática)
    n_save          : int    — pontos de saída
    rtol, atol      : float  — tolerâncias do integrador

    Retorna
    -------
    dict: t, x, z, Vx, Vz, speed, mach, alpha_R_deg,
          CD_arr, CLa_arr, CMa_arr, CMqd_arr,
          t_impact, x_impact, z_max, v_impact
    """
    theta0 = np.radians(elevation_deg)

    # Estimativa automática de t_max (parabólico no vácuo × 2)
    if t_max is None:
        V0z  = muzzle_velocity * np.sin(theta0)
        t_max = max(2.0 * (V0z + np.sqrt(V0z**2 + 2 * G0 * launch_height)) / G0 * 1.5,
                    5.0)

    y0 = jnp.array([0.0, float(launch_height),
                     muzzle_velocity * np.cos(theta0),
                     muzzle_velocity * np.sin(theta0)])

    term   = diffrax.ODETerm(_vector_field)
    solver = diffrax.Dopri5()
    ctrl   = diffrax.PIDController(rtol=rtol, atol=atol)
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, t_max, n_save))

    sol = diffrax.diffeqsolve(
        term, solver,
        t0=0.0, t1=t_max, dt0=0.05,
        y0=y0, saveat=saveat,
        stepsize_controller=ctrl,
        max_steps=2_000_000, args=None,
    )

    t_arr  = np.array(sol.ts,  dtype=np.float64)
    y_arr  = np.array(sol.ys,  dtype=np.float64)
    x_arr  = y_arr[:, 0]
    z_arr  = y_arr[:, 1]
    Vx_arr = y_arr[:, 2]
    Vz_arr = y_arr[:, 3]

    # ── Detecção de impacto (z ≤ 0, após t > 0.5 s) ──────────────
    mask = (z_arr <= 0.0) & (t_arr > 0.5)
    idx  = np.where(mask)[0]
    if len(idx) > 0:
        i    = idx[0]
        frac = z_arr[i-1] / (z_arr[i-1] - z_arr[i])
        t_arr  = np.append(t_arr[:i],  t_arr[i-1]  + frac * (t_arr[i]  - t_arr[i-1]))
        x_arr  = np.append(x_arr[:i],  x_arr[i-1]  + frac * (x_arr[i]  - x_arr[i-1]))
        Vx_arr = np.append(Vx_arr[:i], Vx_arr[i-1] + frac * (Vx_arr[i] - Vx_arr[i-1]))
        Vz_arr = np.append(Vz_arr[:i], Vz_arr[i-1] + frac * (Vz_arr[i] - Vz_arr[i-1]))
        z_arr  = np.append(z_arr[:i],  0.0)

    speed_arr = np.hypot(Vx_arr, Vz_arr)

    # ── Pós-processamento: coefs e α_R ao longo do voo ───────────
    N = len(t_arr)
    mach_arr  = np.zeros(N)
    alpha_arr = np.zeros(N)
    CD_arr    = np.zeros(N)
    CLa_arr   = np.zeros(N)
    CMa_arr   = np.zeros(N)
    CMqd_arr  = np.zeros(N)

    for k in range(N):
        z_k = max(float(z_arr[k]), 0.0)
        rho, a_s = atmosphere(z_k)
        M_k = speed_arr[k] / a_s
        g_k = gravity(z_k)
        mach_arr[k] = M_k

        # αt inicial (zero) → CMa → α_R → refina
        _, _, CMa0, _ = get_aero(M_k, 0.0)
        aRx, aRz = yaw_of_repose_2d(float(Vx_arr[k]), float(Vz_arr[k]),
                                     max(speed_arr[k], 1e-3), g_k)
        at = float(np.hypot(aRx, aRz))
        CD, CLa, CMa, CMqd = get_aero(M_k, at)

        alpha_arr[k] = np.degrees(at)
        CD_arr[k]    = CD
        CLa_arr[k]   = CLa
        CMa_arr[k]   = CMa
        CMqd_arr[k]  = CMqd

    # Ângulo de impacto: ângulo entre o vetor velocidade final e a horizontal.
    # 0° = impacto raso (rasante); 90° = impacto vertical (mergulho puro).
    impact_angle_deg = float(np.degrees(
        np.arctan2(-Vz_arr[-1], Vx_arr[-1])
    ))

    return dict(
        t=t_arr, x=x_arr, z=z_arr, Vx=Vx_arr, Vz=Vz_arr,
        speed=speed_arr, mach=mach_arr, alpha_R_deg=alpha_arr,
        CD=CD_arr, CLa=CLa_arr, CMa=CMa_arr, CMqd=CMqd_arr,
        t_impact=float(t_arr[-1]),
        x_impact=float(x_arr[-1]),
        z_max=float(np.max(z_arr)),
        v_impact=float(speed_arr[-1]),
        impact_angle_deg=impact_angle_deg,
        launch_height=launch_height,
    )

# ══════════════════════════════════════════════════════════════════
# 8. GRÁFICOS
# ══════════════════════════════════════════════════════════════════
def plot_trajectory(res, mv, qe, h0, out_dir):
    """
    Gráfico 1 — Trajetória no plano vertical.
    Inclui marcadores de apogeu e ponto de impacto.
    """
    t, x, z = res['t'], res['x'], res['z']
    speed    = res['speed']
    z_max    = res['z_max']
    idx_max  = int(np.argmax(z))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x / 1e3, z, 'steelblue', lw=2.2, label='Trajetória MPM')
    ax.axhline(h0, color='saddlebrown', lw=1.0, ls='--', alpha=0.6, label=f'Solo (h₀ = {h0:.0f} m)')

    # Apogeu
    ax.plot(x[idx_max] / 1e3, z[idx_max], 'r^', ms=9, zorder=5,
            label=f'Apogeu  z = {z_max:.0f} m  (t = {t[idx_max]:.1f} s)')
    # Impacto
    ax.plot(x[-1] / 1e3, 0.0, 'kx', ms=11, mew=2.5, zorder=5,
            label=f'Impacto  x = {x[-1]:.0f} m  (t = {res["t_impact"]:.1f} s, v = {res["v_impact"]:.0f} m/s)')

    # Colormap de velocidade sobre a trajetória
    from matplotlib.collections import LineCollection
    points  = np.array([x / 1e3, z]).T.reshape(-1, 1, 2)
    segs    = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, cmap='plasma', alpha=0.55, lw=3)
    lc.set_array(speed[:-1])
    ax.add_collection(lc)
    plt.colorbar(lc, ax=ax, label='Velocidade [m/s]', pad=0.01)

    ax.set_xlabel('Alcance [km]', fontsize=11)
    ax.set_ylabel('Altitude [m]', fontsize=11)
    ax.set_title(
        f'Trajetória MPM — Morteiro 120mm\n'
        f'V₀ = {mv} m/s | QE = {qe}° | h₀ = {h0} m',
        fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    fname = os.path.join(out_dir, 'trajetoria.png')
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {fname}")


def plot_angle_of_attack(res, mv, qe, h0, out_dir):
    """
    Gráfico 2 — Ângulo de ataque (pitch of repose |α_R|) ao longo do tempo.
    """
    t, alpha = res['t'], res['alpha_R_deg']
    mach     = res['mach']
    idx_max  = int(np.argmax(res['z']))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # α_R vs tempo
    ax1.plot(t, alpha, color='darkorange', lw=2.0)
    ax1.axvline(t[idx_max], color='r', lw=1.2, ls='--', alpha=0.7,
                label=f'Apogeu (t = {t[idx_max]:.1f} s)')
    ax1.set_ylabel('|α_R| [graus]', fontsize=11)
    ax1.set_title(
        f'Ângulo de Repouso (Pitch of Repose) — Morteiro 120mm\n'
        f'V₀ = {mv} m/s | QE = {qe}° | h₀ = {h0} m',
        fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    # Mach vs tempo (eixo secundário)
    ax2.plot(t, mach, color='steelblue', lw=2.0, label='Mach')
    ax2.axvline(t[idx_max], color='r', lw=1.2, ls='--', alpha=0.7)
    ax2.set_xlabel('Tempo [s]', fontsize=11)
    ax2.set_ylabel('Número de Mach', fontsize=11)
    ax2.set_title('Número de Mach vs Tempo', fontsize=11)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = os.path.join(out_dir, 'angulo_ataque.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {fname}")


def plot_aero_coefficients(res, mv, qe, h0, out_dir):
    """
    Gráfico 3 — Coeficientes aerodinâmicos ao longo do tempo.
    Painel superior: CD, CL_alpha  (escala independente)
    Painel inferior: CM_alpha, (CMq + CMdot_alpha)
    """
    t = res['t']
    idx_max = int(np.argmax(res['z']))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        f'Coeficientes Aerodinâmicos ao Longo do Voo — Morteiro 120mm\n'
        f'V₀ = {mv} m/s | QE = {qe}° | h₀ = {h0} m',
        fontsize=12, fontweight='bold')

    plots = [
        (axes[0, 0], res['CD'],   r'$C_D$',                           'steelblue'),
        (axes[0, 1], res['CLa'],  r'$C_{L_\alpha}$',                  'forestgreen'),
        (axes[1, 0], res['CMa'],  r'$C_{M_\alpha}$',                  'firebrick'),
        (axes[1, 1], res['CMqd'], r'$C_{M_q}+C_{M_{\dot{\alpha}}}$',  'purple'),
    ]

    for ax, data, label, col in plots:
        ax.plot(t, data, color=col, lw=2.0, label=label)
        ax.axvline(t[idx_max], color='k', lw=1.0, ls='--', alpha=0.6,
                   label=f'Apogeu t={t[idx_max]:.1f} s')
        ax.set_xlabel('Tempo [s]', fontsize=10)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = os.path.join(out_dir, 'coeficientes_aero.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {fname}")


def plot_aero_tables(out_dir):
    """
    Gráfico 4 — Tabelas de coeficientes (interpolação Lagrange vs dados tabelados).
    """
    mach_r = np.linspace(0.0, 0.95, 300)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(
        "Coeficientes Aerodinâmicos — Morteiro 120mm\n"
        "(Appendix C, McCoy — Interpolação de Lagrange 4 pontos)",
        fontsize=12, fontweight='bold')

    datasets = [
        (axes[0,0], _CD0_M, _CD0_V,   r'$C_{D_0}$',                       'steelblue'),
        (axes[0,1], _CD2_M, _CD2_V,   r'$C_{D_{\delta^2}}$',               'darkorange'),
        (axes[0,2], _CLA0_M,_CLA0_V,  r'$C_{L_{\alpha_0}}$',               'forestgreen'),
        (axes[0,3], _CLA2_M,_CLA2_V,  r'$C_{L_{\alpha_2}}$',               'limegreen'),
        (axes[1,0], _CMA0_M,_CMA0_V,  r'$C_{M_{\alpha_0}}$',               'firebrick'),
        (axes[1,1], _CMA2_M,_CMA2_V,  r'$C_{M_{\alpha_2}}$',               'tomato'),
        (axes[1,2], _CMQD0_M,_CMQD0_V,r'$(C_{M_q}+C_{M_{\dot\alpha}})_0$','purple'),
        (axes[1,3], _CMQD2_M,_CMQD2_V,r'$(C_{M_q}+C_{M_{\dot\alpha}})_2$','orchid'),
    ]

    for ax, xd, yd, lbl, col in datasets:
        curve = [lagrange4(xd, yd, m) for m in mach_r]
        ax.plot(mach_r, curve, color=col, lw=2.0)
        ax.scatter(xd, yd, c='k', zorder=5, s=35, label='Tabela McCoy')
        ax.set_xlabel('Mach', fontsize=9)
        ax.set_title(lbl, fontsize=11)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = os.path.join(out_dir, 'tabelas_coeficientes.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {fname}")

# ══════════════════════════════════════════════════════════════════
# 9. RELATÓRIO TEXTO
# ══════════════════════════════════════════════════════════════════
def print_report(res, mv, qe, h0):
    """Imprime sumário numérico do disparo."""
    t  = res['t']
    x  = res['x']
    z  = res['z']
    M  = res['mach']
    s  = res['speed']
    idx_max = int(np.argmax(z))

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        RESULTADO — TRAJETÓRIA MPM — MORTEIRO 120mm      ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Velocidade de lançamento : {mv:>8.2f} m/s               ║")
    print(f"║  Ângulo de elevação (QE)  : {qe:>8.2f} °                 ║")
    print(f"║  Altura de lançamento (h₀): {h0:>8.2f} m                 ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Tempo de voo (ToF)       : {res['t_impact']:>8.2f} s                 ║")
    print(f"║  Alcance de impacto       : {res['x_impact']:>8.1f} m                 ║")
    print(f"║  Altitude máxima (Zmax)   : {res['z_max']:>8.1f} m                 ║")
    print(f"║  Mach no apogeu           : {M[idx_max]:>8.4f}                   ║")
    print(f"║  Velocidade de impacto    : {res['v_impact']:>8.2f} m/s               ║")
    print(f"║  Ângulo de impacto        : {res['impact_angle_deg']:>8.2f} °                 ║")
    print(f"║  Mach de impacto          : {M[-1]:>8.4f}                   ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  α_R máximo (apogeu)      : {res['alpha_R_deg'][idx_max]:>8.3f} °                 ║")
    print(f"║  CD no lançamento         : {res['CD'][0]:>8.4f}                   ║")
    print(f"║  CD no impacto            : {res['CD'][-1]:>8.4f}                   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

# ══════════════════════════════════════════════════════════════════
# 10. INTERFACE DE LINHA DE COMANDO
# ══════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Trajetória MPM — Morteiro 120mm (McCoy, Cap. 9)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--mv',    type=float, required=True,
                   help='Velocidade de lançamento [m/s]')
    p.add_argument('--qe',    type=float, required=True,
                   help='Ângulo de elevação (QE) [graus]')
    p.add_argument('--h0',    type=float, default=0.0,
                   help='Altura de lançamento acima do solo [m]')
    p.add_argument('--saida', type=str,   default='./',
                   help='Diretório de saída para os gráficos')
    p.add_argument('--tmax',  type=float, default=None,
                   help='Tempo máximo de integração [s] (padrão: automático)')
    p.add_argument('--rtol',  type=float, default=1e-7,
                   help='Tolerância relativa do integrador')
    p.add_argument('--atol',  type=float, default=1e-7,
                   help='Tolerância absoluta do integrador')
    return p.parse_args()


def main():
    args = parse_args()

    # Validações básicas
    if args.mv <= 0:
        print("ERRO: velocidade de lançamento deve ser positiva.", file=sys.stderr)
        sys.exit(1)
    if not (0 <= args.qe < 90):
        print("ERRO: ângulo de elevação deve estar em [0°, 90°).", file=sys.stderr)
        sys.exit(1)
    if args.h0 < 0:
        print("ERRO: altura de lançamento não pode ser negativa.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.saida, exist_ok=True)

    print()
    print("═" * 62)
    print("  TRAJETÓRIA MASSA PONTO MODIFICADO (MPM) — MORTEIRO 120mm")
    print("  McCoy, Modern Exterior Ballistics, 2ª Ed., Cap. 9")
    print("  Integrador  : Diffrax Dopri5 (RK4/5 adaptativo)")
    print("  Interpolação: Lagrange 4 pontos")
    print("═" * 62)
    print(f"  V₀  = {args.mv} m/s")
    print(f"  QE  = {args.qe}°")
    print(f"  h₀  = {args.h0} m")
    print(f"  Saída: {os.path.abspath(args.saida)}")
    print("─" * 62)
    print("  Integrando trajetória...", end="", flush=True)

    res = solve_trajectory(
        muzzle_velocity=args.mv,
        elevation_deg=args.qe,
        launch_height=args.h0,
        t_max=args.tmax,
        rtol=args.rtol,
        atol=args.atol,
    )
    print(" concluído.")

    print_report(res, args.mv, args.qe, args.h0)

    print("  Salvando gráficos...")
    plot_trajectory(res, args.mv, args.qe, args.h0, args.saida)
    plot_angle_of_attack(res, args.mv, args.qe, args.h0, args.saida)
    plot_aero_coefficients(res, args.mv, args.qe, args.h0, args.saida)
    plot_aero_tables(args.saida)

    print()
    print("  Gráficos salvos:")
    print("    • trajetoria.png          — trajetória no plano vertical")
    print("    • angulo_ataque.png       — |α_R| e Mach vs tempo")
    print("    • coeficientes_aero.png   — CD, CL_α, CM_α, CMqd vs tempo")
    print("    • tabelas_coeficientes.png— curvas tabeladas (interpolação)")
    print("═" * 62)


if __name__ == "__main__":
    main()


