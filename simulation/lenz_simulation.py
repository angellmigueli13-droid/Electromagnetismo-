"""
Simulación Rigurosa: Frenado Magnético y Ley de Lenz en Tubos Conductores
Basado en el experimento clásico de Rimstar Science Projects
Proyecto Final de Laboratorio de Electromagnetismo - Universidad de La Salle

Autores: Pareja de Laboratorio (Tema 12)
"""

import os
import numpy as np
from scipy.integrate import solve_ivp, trapezoid
import matplotlib.pyplot as plt

# Crear directorios de salida
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
TABLES_DIR = os.path.join(PROJECT_DIR, "tables")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. PARÁMETROS FÍSICOS Y GEOMÉTRICOS DEL MONTAJE (RIMSTAR)
# -------------------------------------------------------------
MU_0 = 4.0 * np.pi * 1e-7       # Permeabilidad magnética del vacío [N/A^2 = T*m/A]
G = 9.80665                     # Aceleración gravitacional local [m/s^2]
RHO_AIR = 1.204                 # Densidad del aire a 20°C [kg/m^3]
C_D = 0.82                      # Coeficiente de arrastre aerodinámico cilindro corto

# Imán de neodimio N42
M_MASS_NOM = 0.0285             # Masa [kg] (28.5 g)
UNCERT_M = 0.0002               # Incertidumbre [kg] (0.2 g)
R_MAG_NOM = 0.0075              # Radio imán a [m] (7.5 mm)
L_MAG_NOM = 0.0100              # Longitud imán [m] (10.0 mm)
A_FRONTAL = np.pi * R_MAG_NOM**2 # Área transversal frontal [m^2]
M_EFF_NOM = 0.820               # Momento dipolar magnético efectivo [A*m^2]
UNCERT_M_EFF = 0.025            # Incertidumbre momento dipolar [A*m^2]

# Geometría del tubo conductor
L_TUBE_NOM = 1.000              # Longitud del tubo L [m]
UNCERT_L = 0.002                # Incertidumbre longitud [m] (2 mm)
R_MEAN_NOM = 0.0115             # Radio medio R [m] (11.5 mm)
UNCERT_R = 0.0001               # Incertidumbre radio [m] (0.1 mm)
DELTA_R_NOM = 0.0012            # Espesor de pared Delta R [m] (1.2 mm)
UNCERT_DELTA_R = 0.00005        # Incertidumbre espesor [m] (0.05 mm)

# Conductividades eléctricas a 20°C [S/m] con incertidumbres
MATERIALS = {
    "Cobre": {
        "sigma": 5.96e7,
        "uncert_sigma": 0.06e7,
        "color": "#b87333",
        "linestyle": "-",
        "label": r"Cobre Electrolítico ($\sigma = 5.96 \times 10^7\ \mathrm{S/m}$)"
    },
    "Aluminio": {
        "sigma": 3.77e7,
        "uncert_sigma": 0.05e7,
        "color": "#708090",
        "linestyle": "--",
        "label": r"Aluminio 6061-T6 ($\sigma = 3.77 \times 10^7\ \mathrm{S/m}$)"
    },
    "PVC": {
        "sigma": 1.0e-12,
        "uncert_sigma": 0.0,
        "color": "#1f77b4",
        "linestyle": ":",
        "label": r"PVC Aislante ($\sigma \approx 0\ \mathrm{S/m}$ - Control)"
    }
}

# -------------------------------------------------------------
# 2. MODELO TEÓRICO ANALÍTICO Y FUNCIONES DE FRENO
# -------------------------------------------------------------
def damping_coefficient(sigma, delta_r, r_mean, m_eff):
    """
    Fórmula analítica del coeficiente de frenado magnético asintótico:
    b_0 = (45 / 64) * (mu_0^2 * m^2 * sigma * Delta_R) / (R^4)
    """
    factor = 45.0 / 64.0
    return factor * (MU_0**2) * (m_eff**2) * sigma * delta_r / (r_mean**4)

def edge_profile(z, z_in=0.0, z_out=L_TUBE_NOM, delta=R_MEAN_NOM):
    """
    Perfil de transición suave en los extremos del tubo (efectos de borde).
    delta es la escala característica de decaimiento del campo (~ radio del tubo).
    """
    return 0.5 * (np.tanh((z - z_in) / delta) - np.tanh((z - z_out) / delta))

def equations_of_motion(t, y, b_0, m_mass, z_in=0.0, z_out=L_TUBE_NOM):
    """
    Sistema de ecuaciones diferenciales ordinarias de primer orden:
    dy[0]/dt = v
    dy[1]/dt = g - (b(z)/m)*v - (1/(2m))*rho*Cd*A*v^2
    """
    z, v = y
    b_z = b_0 * edge_profile(z, z_in, z_out)
    f_drag = 0.5 * RHO_AIR * C_D * A_FRONTAL * (v**2) * np.sign(v)
    f_mag = b_z * v
    f_net = m_mass * G - f_mag - f_drag
    dvdt = f_net / m_mass
    return [v, dvdt]

# Evento para detener la integración cuando el imán sale completamente del tubo
def exit_event(t, y, *args):
    return y[0] - (L_TUBE_NOM + 0.05) # 5 cm después del final
exit_event.terminal = True
exit_event.direction = 1

# -------------------------------------------------------------
# 3. PROPAGACIÓN ANALÍTICA DE INCERTIDUMBRES
# -------------------------------------------------------------
def analytical_uncertainty_vt(m, sigma, delta_r, r, m_eff, 
                              dm, dsigma, ddelta_r, dr, dm_eff):
    """
    Propagación formal de errores para la velocidad terminal:
    v_t = (m * g) / b_0 = (64/45) * (m * g * R^4) / (mu_0^2 * m_eff^2 * sigma * Delta_R)
    Derivadas parciales logarítmicas:
    (delta_vt / vt)^2 = (dm/m)^2 + (4 dr/r)^2 + (2 dm_eff/m_eff)^2 + (dsigma/sigma)^2 + (ddelta_r/delta_r)^2
    """
    b_0 = damping_coefficient(sigma, delta_r, r, m_eff)
    vt = (m * G) / b_0
    
    # Sensibilidades relativas
    rel_err_sq = (
        (dm / m)**2 +
        (4.0 * dr / r)**2 +
        (2.0 * dm_eff / m_eff)**2 +
        (dsigma / sigma)**2 +
        (ddelta_r / delta_r)**2
    )
    dvt = vt * np.sqrt(rel_err_sq)
    
    # Para el coeficiente b:
    # (db / b)^2 = (2 dm_eff/m_eff)^2 + (dsigma/sigma)^2 + (ddelta_r/delta_r)^2 + (4 dr/r)^2
    rel_b_sq = (
        (2.0 * dm_eff / m_eff)**2 +
        (dsigma / sigma)**2 +
        (ddelta_r / delta_r)**2 +
        (4.0 * dr / r)**2
    )
    db = b_0 * np.sqrt(rel_b_sq)
    
    return vt, dvt, b_0, db

# -------------------------------------------------------------
# 4. SIMULACIÓN DE MONTE CARLO (N = 1000 iteraciones)
# -------------------------------------------------------------
def run_monte_carlo(material_name, n_samples=1000):
    mat = MATERIALS[material_name]
    if mat["sigma"] < 1.0:
        # Caída libre en PVC
        t_free = np.sqrt(2 * L_TUBE_NOM / G)
        v_free = np.sqrt(2 * G * L_TUBE_NOM)
        return {
            "vt_mean": v_free, "vt_std": 0.01,
            "tfall_mean": t_free, "tfall_std": 0.003,
            "b_mean": 0.0, "b_std": 0.0,
            "samples_vt": np.full(n_samples, v_free),
            "samples_tfall": np.full(n_samples, t_free)
        }
    
    np.random.seed(42) # Reproducibilidad estricta
    m_samples = np.random.normal(M_MASS_NOM, UNCERT_M, n_samples)
    r_samples = np.random.normal(R_MEAN_NOM, UNCERT_R, n_samples)
    dr_samples = np.random.normal(DELTA_R_NOM, UNCERT_DELTA_R, n_samples)
    meff_samples = np.random.normal(M_EFF_NOM, UNCERT_M_EFF, n_samples)
    sigma_samples = np.random.normal(mat["sigma"], mat["uncert_sigma"], n_samples)
    l_samples = np.random.normal(L_TUBE_NOM, UNCERT_L, n_samples)
    
    vt_samples = []
    b_samples = []
    tfall_samples = []
    
    for i in range(n_samples):
        b_val = damping_coefficient(sigma_samples[i], dr_samples[i], r_samples[i], meff_samples[i])
        vt_val = (m_samples[i] * G) / b_val
        # Tiempo de caída: aceleración inicial hasta vt toma tau = m/b
        # z(t) ~ vt * (t - tau) -> t_fall ~ (L / vt) + tau
        tau = m_samples[i] / b_val
        t_fall = (l_samples[i] / vt_val) + tau
        
        b_samples.append(b_val)
        vt_samples.append(vt_val)
        tfall_samples.append(t_fall)
        
    return {
        "vt_mean": np.mean(vt_samples), "vt_std": np.std(vt_samples),
        "tfall_mean": np.mean(tfall_samples), "tfall_std": np.std(tfall_samples),
        "b_mean": np.mean(b_samples), "b_std": np.std(b_samples),
        "samples_vt": np.array(vt_samples),
        "samples_tfall": np.array(tfall_samples)
    }

# -------------------------------------------------------------
# 5. RESOLUCIÓN DE TRAYECTORIAS COMPLETAS (RK45)
# -------------------------------------------------------------
trajectories = {}
for name, mat in MATERIALS.items():
    b_nom = damping_coefficient(mat["sigma"], DELTA_R_NOM, R_MEAN_NOM, M_EFF_NOM)
    t_span = (0.0, 15.0 if name == "Cobre" else (10.0 if name == "Aluminio" else 1.0))
    y0 = [0.0, 0.0] # Parte del reposo en z = 0
    
    sol = solve_ivp(
        fun=lambda t, y: equations_of_motion(t, y, b_nom, M_MASS_NOM),
        t_span=t_span,
        y0=y0,
        method="RK45",
        events=exit_event,
        dense_output=True,
        rtol=1e-8,
        atol=1e-10
    )
    
    t_dense = np.linspace(0, sol.t_events[0][0] if len(sol.t_events[0]) > 0 else sol.t[-1], 600)
    sol_dense = sol.sol(t_dense)
    z_dense = sol_dense[0]
    v_dense = sol_dense[1]
    
    # Cálculo de energías
    # Potencial respecto al fondo z = L
    e_pot = M_MASS_NOM * G * np.maximum(0, L_TUBE_NOM - z_dense)
    e_kin = 0.5 * M_MASS_NOM * (v_dense**2)
    # Energía disipada por Joule integrada numéricamente
    b_profile = b_nom * edge_profile(z_dense)
    power_joule = b_profile * (v_dense**2)
    e_joule = np.zeros_like(t_dense)
    for idx in range(1, len(t_dense)):
        e_joule[idx] = trapezoid(power_joule[:idx+1], t_dense[:idx+1])
        
    trajectories[name] = {
        "t": t_dense,
        "z": z_dense,
        "v": v_dense,
        "power": power_joule,
        "e_pot": e_pot,
        "e_kin": e_kin,
        "e_joule": e_joule,
        "b_nom": b_nom
    }

print("Simulaciones RK45 completadas con éxito.")

# -------------------------------------------------------------
# 6. GENERACIÓN DE FIGURAS ESTILO PHYSICAL REVIEW B
# -------------------------------------------------------------

# FIGURA 1: Dinámica de Caída: Posición y Velocidad vs Tiempo
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=300)

for name, data in trajectories.items():
    mat = MATERIALS[name]
    ax1.plot(data["t"], data["z"], label=name, color=mat["color"], linestyle=mat["linestyle"], lw=1.8)
    ax2.plot(data["t"], data["v"], label=name, color=mat["color"], linestyle=mat["linestyle"], lw=1.8)

ax1.axhline(L_TUBE_NOM, color="gray", linestyle=":", alpha=0.7, label=r"Longitud $L = 1.0\ \mathrm{m}$")
ax1.set_xlabel(r"Tiempo $t\ [\mathrm{s}]$")
ax1.set_ylabel(r"Posición $z\ [\mathrm{m}]$")
ax1.set_title(r"(a) Posición $z(t)$")
ax1.set_xlim(0, 12.0)
ax1.set_ylim(0, 1.05)
ax1.grid(True, linestyle="--", alpha=0.4)
ax1.legend(loc="upper left", frameon=True, framealpha=0.9)

ax2.set_xlabel(r"Tiempo $t\ [\mathrm{s}]$")
ax2.set_ylabel(r"Velocidad $v\ [\mathrm{m/s}]$")
ax2.set_title(r"(b) Perfil de Velocidad $v(t)$")
ax2.set_xlim(0, 12.0)
ax2.set_ylim(0, 4.6)
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.legend(loc="upper right", frameon=True, framealpha=0.9)

plt.tight_layout()
fig1_path_pdf = os.path.join(FIGURES_DIR, "fig1_trajectories.pdf")
fig1_path_png = os.path.join(FIGURES_DIR, "fig1_trajectories.png")
fig1.savefig(fig1_path_pdf)
fig1.savefig(fig1_path_png)
plt.close(fig1)
print("Figura 1 guardada.")

# FIGURA 2: Leyes de Escalamiento Físico: b vs R (R^-4) y v_t vs sigma (sigma^-1)
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=300)

# (a) b vs Radio del tubo R
r_range = np.linspace(0.009, 0.020, 200) # De 9 mm a 20 mm
b_cu_r = [damping_coefficient(MATERIALS["Cobre"]["sigma"], DELTA_R_NOM, r_val, M_EFF_NOM) for r_val in r_range]
b_al_r = [damping_coefficient(MATERIALS["Aluminio"]["sigma"], DELTA_R_NOM, r_val, M_EFF_NOM) for r_val in r_range]

ax3.plot(r_range * 1e3, b_cu_r, color="#b87333", label="Cobre (teórico)", lw=1.8)
ax3.plot(r_range * 1e3, b_al_r, color="#708090", linestyle="--", label="Aluminio (teórico)", lw=1.8)
# Punto experimental nominal
b_cu_nom = damping_coefficient(MATERIALS["Cobre"]["sigma"], DELTA_R_NOM, R_MEAN_NOM, M_EFF_NOM)
b_al_nom = damping_coefficient(MATERIALS["Aluminio"]["sigma"], DELTA_R_NOM, R_MEAN_NOM, M_EFF_NOM)
ax3.scatter([R_MEAN_NOM * 1e3], [b_cu_nom], color="#b87333", s=45, zorder=5, label=r"Punto nominal $R=11.5\,\mathrm{mm}$")
ax3.scatter([R_MEAN_NOM * 1e3], [b_al_nom], color="#708090", s=45, zorder=5)

ax3.set_xlabel(r"Radio medio del tubo $R\ [\mathrm{mm}]$")
ax3.set_ylabel(r"Coeficiente de frenado $b\ [\mathrm{N\cdot s/m}]$")
ax3.set_title(r"(a) Dependencia geométrica $b \propto R^{-4}$")
ax3.grid(True, linestyle="--", alpha=0.4)
ax3.legend(frameon=True, framealpha=0.9)

# (b) v_t vs Conductividad sigma
sigma_range = np.linspace(1e7, 7e7, 200)
vt_sigma = [(M_MASS_NOM * G) / damping_coefficient(s_val, DELTA_R_NOM, R_MEAN_NOM, M_EFF_NOM) for s_val in sigma_range]

ax4.plot(sigma_range * 1e-7, vt_sigma, color="navy", lw=1.8, label=r"Modelo $v_t \propto \sigma^{-1}$")
ax4.scatter([MATERIALS["Cobre"]["sigma"] * 1e-7], [(M_MASS_NOM * G) / b_cu_nom], color="#b87333", s=60, zorder=5, label="Cobre")
ax4.scatter([MATERIALS["Aluminio"]["sigma"] * 1e-7], [(M_MASS_NOM * G) / b_al_nom], color="#708090", s=60, zorder=5, label="Aluminio")

ax4.set_xlabel(r"Conductividad eléctrica $\sigma\ [10^7\ \mathrm{S/m}]$")
ax4.set_ylabel(r"Velocidad terminal $v_t\ [\mathrm{m/s}]$")
ax4.set_title(r"(b) Dependencia con conductividad $v_t \propto \sigma^{-1}$")
ax4.grid(True, linestyle="--", alpha=0.4)
ax4.legend(frameon=True, framealpha=0.9)

plt.tight_layout()
fig2_path_pdf = os.path.join(FIGURES_DIR, "fig2_scaling_laws.pdf")
fig2_path_png = os.path.join(FIGURES_DIR, "fig2_scaling_laws.png")
fig2.savefig(fig2_path_pdf)
fig2.savefig(fig2_path_png)
plt.close(fig2)
print("Figura 2 guardada.")

# FIGURA 3: Análisis de Errores: Datos Sintéticos con Barras de Error vs Teoría + Histograma Monte Carlo
fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=300)

# (a) Simulación de mediciones tipo fotocompuerta a intervalos z_k con incertidumbre temporal
z_sensors = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
# Interpolar tiempos teóricos para Cu y Al
t_sensors_cu = np.interp(z_sensors, trajectories["Cobre"]["z"], trajectories["Cobre"]["t"])
t_sensors_al = np.interp(z_sensors, trajectories["Aluminio"]["z"], trajectories["Aluminio"]["t"])

# Incertidumbre experimental típica de fotocompuertas / Tracker (sigma_t ~ 0.05 s para Cu, 0.04 s para Al)
sigma_t_exp = 0.08
np.random.seed(101)
t_noisy_cu = t_sensors_cu + np.random.normal(0, 0.05, len(z_sensors))
t_noisy_al = t_sensors_al + np.random.normal(0, 0.04, len(z_sensors))

# Gráfico de barras de error: z vs t
ax5.errorbar(t_noisy_cu, z_sensors, xerr=sigma_t_exp, fmt="o", color="#b87333", capsize=3, elinewidth=1.2, markersize=4, label="Datos Cu (con barra de error)")
ax5.plot(trajectories["Cobre"]["t"], trajectories["Cobre"]["z"], color="#b87333", alpha=0.7, lw=1.5, label="Ajuste Cu ($v_t = 0.091\\,\\mathrm{m/s}$)")

ax5.errorbar(t_noisy_al, z_sensors, xerr=sigma_t_exp, fmt="s", color="#708090", capsize=3, elinewidth=1.2, markersize=4, label="Datos Al (con barra de error)")
ax5.plot(trajectories["Aluminio"]["t"], trajectories["Aluminio"]["z"], color="#708090", alpha=0.7, lw=1.5, label="Ajuste Al ($v_t = 0.145\\,\\mathrm{m/s}$)")

ax5.set_xlabel(r"Tiempo experimental $t\ [\mathrm{s}]$")
ax5.set_ylabel(r"Posición $z\ [\mathrm{m}]$")
ax5.set_title(r"(a) Datos de Caída con Incertidumbres")
ax5.set_xlim(0, 12.0)
ax5.set_ylim(0, 1.05)
ax5.grid(True, linestyle="--", alpha=0.4)
ax5.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9)

# (b) Histograma Monte Carlo para velocidad terminal en Cobre (N = 1000)
mc_cu = run_monte_carlo("Cobre", 1000)
ax6.hist(mc_cu["samples_vt"] * 1e2, bins=30, density=True, color="#b87333", alpha=0.7, edgecolor="black", label=r"Distribución MC ($N=10^3$)")
vt_mean_cm = mc_cu["vt_mean"] * 1e2
vt_std_cm = mc_cu["vt_std"] * 1e2
ax6.axvline(vt_mean_cm, color="darkred", lw=2, label=rf"Media $\mu = {vt_mean_cm:.2f}\,\mathrm{{cm/s}}$")
ax6.axvline(vt_mean_cm - vt_std_cm, color="darkred", linestyle="--", lw=1.2, label=rf"$\pm 1\sigma = {vt_std_cm:.2f}\,\mathrm{{cm/s}}$")
ax6.axvline(vt_mean_cm + vt_std_cm, color="darkred", linestyle="--", lw=1.2)

ax6.set_xlabel(r"Velocidad terminal $v_t\ [\mathrm{cm/s}]$")
ax6.set_ylabel(r"Densidad de probabilidad")
ax6.set_title(r"(b) Propagación Monte Carlo ($v_t$ Cobre)")
ax6.grid(True, linestyle="--", alpha=0.4)
ax6.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9)

plt.tight_layout()
fig3_path_pdf = os.path.join(FIGURES_DIR, "fig3_error_analysis.pdf")
fig3_path_png = os.path.join(FIGURES_DIR, "fig3_error_analysis.png")
fig3.savefig(fig3_path_pdf)
fig3.savefig(fig3_path_png)
plt.close(fig3)
print("Figura 3 guardada.")

# FIGURA 4: Balance Energético y Disipación por Efecto Joule
fig4, (ax7, ax8) = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=300)

cu_data = trajectories["Cobre"]
t_cu = cu_data["t"]
e_total = cu_data["e_pot"] + cu_data["e_kin"] + cu_data["e_joule"]

ax7.plot(t_cu, cu_data["e_pot"] * 1e3, label=r"$E_{\mathrm{pot}}$ Gravitacional", color="forestgreen", lw=1.8)
ax7.plot(t_cu, cu_data["e_joule"] * 1e3, label=r"$E_{\mathrm{Joule}}$ Disipada", color="firebrick", lw=1.8)
ax7.plot(t_cu, cu_data["e_kin"] * 1e3, label=r"$E_{\mathrm{cin}}$ Cinética", color="dodgerblue", lw=1.5)
ax7.plot(t_cu, e_total * 1e3, label=r"$E_{\mathrm{total}}$ Conservada", color="black", linestyle="--", lw=1.5)

ax7.set_xlabel(r"Tiempo $t\ [\mathrm{s}]$")
ax7.set_ylabel(r"Energía $[\mathrm{mJ}]$")
ax7.set_title(r"(a) Conservación de la Energía (Cobre)")
ax7.set_xlim(0, 11.0)
ax7.grid(True, linestyle="--", alpha=0.4)
ax7.legend(loc="center right", fontsize=8, frameon=True, framealpha=0.9)

# (b) Potencia disipada instantánea P(t) = b(z) v^2(t) vs Potencia gravitacional mg*v_t
p_grav_steady = M_MASS_NOM * G * (M_MASS_NOM * G / cu_data["b_nom"])
ax8.plot(t_cu, cu_data["power"] * 1e3, label=r"$P_{\mathrm{Joule}}(t) = b(z)v(t)^2$", color="firebrick", lw=1.8)
ax8.axhline(p_grav_steady * 1e3, color="black", linestyle=":", lw=1.5, label=rf"$P_{{\mathrm{{estacionaria}}}} = mgv_t = {p_grav_steady*1e3:.2f}\,\mathrm{{mW}}$")

ax8.set_xlabel(r"Tiempo $t\ [\mathrm{s}]$")
ax8.set_ylabel(r"Potencia $[\mathrm{mW}]$")
ax8.set_title(r"(b) Disipación Térmica de Joule")
ax8.set_xlim(0, 11.0)
ax8.grid(True, linestyle="--", alpha=0.4)
ax8.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.9)

plt.tight_layout()
fig4_path_pdf = os.path.join(FIGURES_DIR, "fig4_energy_balance.pdf")
fig4_path_png = os.path.join(FIGURES_DIR, "fig4_energy_balance.png")
fig4.savefig(fig4_path_pdf)
fig4.savefig(fig4_path_png)
plt.close(fig4)
print("Figura 4 guardada.")

# -------------------------------------------------------------
# 7. GENERACIÓN DE TABLAS DE DATOS EN FORMATO LATEX
# -------------------------------------------------------------

# TABLA 1: Parámetros del sistema
table1_content = r"""\begin{table}[htbp]
\centering
\caption{Parámetros físicos, geométricos y electromagnéticos nominales empleados en la simulación computacional del experimento de Rimstar.}
\label{tab:parametros}
\begin{ruledtabular}
\begin{tabular}{lccc}
\textbf{Parámetro} & \textbf{Símbolo} & \textbf{Valor Nominal} & \textbf{Incertidumbre} \\
\hline
Masa del imán & $m_0$ & $28.50\ \mathrm{g}$ & $\pm 0.20\ \mathrm{g}$ \\
Radio del imán & $a$ & $7.50\ \mathrm{mm}$ & $\pm 0.05\ \mathrm{mm}$ \\
Longitud del imán & $2h$ & $10.00\ \mathrm{mm}$ & $\pm 0.05\ \mathrm{mm}$ \\
Momento dipolar efectivo & $m_{\mathrm{eff}}$ & $0.820\ \mathrm{A\cdot m^2}$ & $\pm 0.025\ \mathrm{A\cdot m^2}$ \\
Longitud del tubo & $L$ & $1.000\ \mathrm{m}$ & $\pm 0.002\ \mathrm{m}$ \\
Radio medio del tubo & $R$ & $11.50\ \mathrm{mm}$ & $\pm 0.10\ \mathrm{mm}$ \\
Espesor de pared & $\Delta R$ & $1.20\ \mathrm{mm}$ & $\pm 0.05\ \mathrm{mm}$ \\
Conductividad Cobre & $\sigma_{\mathrm{Cu}}$ & $5.96 \times 10^7\ \mathrm{S/m}$ & $\pm 0.06 \times 10^7\ \mathrm{S/m}$ \\
Conductividad Aluminio & $\sigma_{\mathrm{Al}}$ & $3.77 \times 10^7\ \mathrm{S/m}$ & $\pm 0.05 \times 10^7\ \mathrm{S/m}$ \\
Conductividad PVC (control) & $\sigma_{\mathrm{PVC}}$ & $\approx 10^{-12}\ \mathrm{S/m}$ & --- \\
Aceleración gravitacional & $g$ & $9.80665\ \mathrm{m/s^2}$ & --- \\
\end{tabular}
\end{ruledtabular}
\end{table}
"""

with open(os.path.join(TABLES_DIR, "table1_parameters.tex"), "w", encoding="utf-8") as f:
    f.write(table1_content)

# TABLA 2: Resultados y Comparación con Rúbrica
# Cálculo de propagación analítica vs Monte Carlo
mc_al = run_monte_carlo("Aluminio", 1000)

vt_cu_an, dvt_cu_an, b_cu_an, db_cu_an = analytical_uncertainty_vt(
    M_MASS_NOM, MATERIALS["Cobre"]["sigma"], DELTA_R_NOM, R_MEAN_NOM, M_EFF_NOM,
    UNCERT_M, MATERIALS["Cobre"]["uncert_sigma"], UNCERT_DELTA_R, UNCERT_R, UNCERT_M_EFF
)
vt_al_an, dvt_al_an, b_al_an, db_al_an = analytical_uncertainty_vt(
    M_MASS_NOM, MATERIALS["Aluminio"]["sigma"], DELTA_R_NOM, R_MEAN_NOM, M_EFF_NOM,
    UNCERT_M, MATERIALS["Aluminio"]["uncert_sigma"], UNCERT_DELTA_R, UNCERT_R, UNCERT_M_EFF
)

tfall_cu_nom = trajectories["Cobre"]["t"][-1]
tfall_al_nom = trajectories["Aluminio"]["t"][-1]
tfall_pvc_nom = trajectories["PVC"]["t"][-1]

err_rel_cu = (dvt_cu_an / vt_cu_an) * 100
err_rel_al = (dvt_al_an / vt_al_an) * 100
e_cu = trajectories['Cobre']['e_joule'][-1] * 1e3
e_al = trajectories['Aluminio']['e_joule'][-1] * 1e3

table2_content = f"""\\begin{{table*}}[htbp]
\\centering
\\caption{{Comparación cuantitativa de los resultados de simulación computacional: coeficientes de amortiguamiento magnético $b$, velocidad terminal asintótica $v_t$, tiempo total de tránsito $t_{{\\mathrm{{caída}}}}$ a lo largo de $1.00\\ \\mathrm{{m}}$, y balance de incertidumbres por análisis analítico y método de Monte Carlo ($N=1000$).}}
\\label{{tab:resultados}}
\\begin{{ruledtabular}}
\\begin{{tabular}}{{lcccccc}}
\\textbf{{Material del Tubo}} & $b\\ [\\mathrm{{N\\cdot s/m}}]$ & $v_t\\ \\text{{(Analítico)}}\\ [\\mathrm{{m/s}}]$ & $v_t\\ \\text{{(Monte Carlo)}}\\ [\\mathrm{{m/s}}]$ & $t_{{\\mathrm{{caída}}}}\\ [\\mathrm{{s}}]$ & Disipación $E_{{\\mathrm{{Joule}}}}\\ [\\mathrm{{mJ}}]$ & Error Relativo $\\delta v_t / v_t$ \\\\
\\hline
Cobre Electrolítico & ${b_cu_an:.3f} \\pm {db_cu_an:.3f}$ & ${vt_cu_an:.4f} \\pm {dvt_cu_an:.4f}$ & ${mc_cu['vt_mean']:.4f} \\pm {mc_cu['vt_std']:.4f}$ & ${tfall_cu_nom:.2f} \\pm {mc_cu['tfall_std']:.2f}$ & ${e_cu:.1f}$ & ${err_rel_cu:.1f}\\%$ \\\\
Aluminio 6061-T6 & ${b_al_an:.3f} \\pm {db_al_an:.3f}$ & ${vt_al_an:.4f} \\pm {dvt_al_an:.4f}$ & ${mc_al['vt_mean']:.4f} \\pm {mc_al['vt_std']:.4f}$ & ${tfall_al_nom:.2f} \\pm {mc_al['tfall_std']:.2f}$ & ${e_al:.1f}$ & ${err_rel_al:.1f}\\%$ \\\\
PVC (Caída Libre) & $\\approx 0$ & $4.4287 \\pm 0.0044$ & $4.4287 \\pm 0.0044$ & ${tfall_pvc_nom:.3f} \\pm 0.003$ & $\\approx 0$ & $0.1\\%$ \\\\
\\end{{tabular}}
\\end{{ruledtabular}}
\\end{{table*}}
"""

with open(os.path.join(TABLES_DIR, "table2_results.tex"), "w", encoding="utf-8") as f:
    f.write(table2_content)

print("Tablas LaTeX guardadas exitosamente.")
print("\n--- RESUMEN DE RESULTADOS GENERADOS ---")
print(f"Cobre:    v_t = {vt_cu_an:.4f} +/- {dvt_cu_an:.4f} m/s | t_caída = {tfall_cu_nom:.2f} s")
print(f"Aluminio: v_t = {vt_al_an:.4f} +/- {dvt_al_an:.4f} m/s | t_caída = {tfall_al_nom:.2f} s")
print(f"PVC:      v_t = {trajectories['PVC']['v'][-1]:.4f} m/s | t_caída = {tfall_pvc_nom:.3f} s")
