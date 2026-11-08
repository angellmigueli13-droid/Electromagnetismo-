# Proyecto Final: Frenado Magnético y Ley de Lenz en Tubos Conductores (Tema 12)

**Universidad de La Salle**  
**Escuela de Ciencias Básicas y Aplicadas — Laboratorio de Electromagnetismo**  
**Profesor:** Fernando Vergara  
**Integrantes:** Dilan Correa & Miguel Vega  
**Tema Asignado:** Tema 12 — Frenado Magnético y Ley de Lenz en Tubos Conductores  
**Repositorio Oficial:** [https://github.com/angellmigueli13-droid/Electromagnetismo-](https://github.com/angellmigueli13-droid/Electromagnetismo-)  
**Ecuación Rectora:**
$$mg - bv = m\frac{dv}{dt} \implies \varepsilon = -\frac{d\Phi}{dt}$$

---

## 📄 Descripción del Proyecto
Este repositorio contiene el desarrollo completo del proyecto final de Laboratorio de Electromagnetismo, enfocado en el modelado computacional riguroso del frenado magnético inducido por corrientes de Foucault (*eddy currents*) cuando un imán cilíndrico de neodimio cae a través de tubos cilíndricos conductores de pared delgada, basado en el diseño experimental clásico divulgado en [Rimstar Science Projects](https://rimstar.org/science_electronics_projects/lenzs_law_with_copper_pipe_and_neodymium_magnet.htm).

El proyecto incluye:
1. **Simulación numérica y estocástica de alta fidelidad** (`simulation/lenz_simulation.py`):
   - Integración numérica de ecuaciones diferenciales no lineales con el algoritmo Runge-Kutta 4-5 adaptativo (`RK45`).
   - Modelado de efectos de borde en las bocas del tubo mediante perfiles hiperbólicos continuos.
   - Verificación de leyes de escalamiento físico ($b \propto R^{-4}$, $v_t \propto \sigma^{-1}$).
   - Simulación de Monte Carlo ($N = 1000$ realizaciones) y propagación analítica de errores experimentales.
   - Balance riguroso de conservación de energía (potencial gravitacional vs. disipación de Joule).
2. **Artículo formal en LaTeX estilo *Physical Review B*** (`paper/main.tex` y `paper/references.bib`):
   - Cumplimiento estricto de la rúbrica oficial (resumen bilingüe de 120–200 palabras, ecuaciones numeradas a la derecha, tablas procesadas, gráficas con barras de error, conclusiones directas sin viñetas, sin uso de Wikipedia y apéndice de derivadas e incertidumbres).
3. **Figuras de alta resolución para publicación** (`figures/` en formatos `.pdf` vectorial y `.png` a 300 DPI).

---

## 📁 Estructura del Repositorio

```text
Tema12/
├── figures/                          # Gráficas generadas listas para el informe
│   ├── fig1_trajectories.pdf/.png    # Dinámica cinemática z(t) y v(t)
│   ├── fig2_scaling_laws.pdf/.png    # Leyes de escalamiento b(R) y vt(sigma)
│   ├── fig3_error_analysis.pdf/.png  # Datos sintéticos con barras de error e histograma Monte Carlo
│   └── fig4_energy_balance.pdf/.png  # Conservación de energía y potencia disipada por Joule
├── paper/                            # Manuscrito formal estilo Physical Review B
│   ├── main.tex                      # Código fuente completo en LaTeX
│   └── references.bib                # Bibliografía formal en formato BibTeX
├── simulation/                       # Código fuente de simulación en Python
│   └── lenz_simulation.py            # Script principal reproducible
├── tables/                           # Tablas de datos en sintaxis LaTeX
│   ├── table1_parameters.tex         # Parámetros físicos nominales e incertidumbres
│   └── table2_results.tex            # Resultados cuantitativos comparativos
├── .gitignore                        # Exclusiones estándar para Python y LaTeX
└── README.md                         # Documentación general de reproducibilidad
```

---

## 🔬 Resumen de Resultados Principales

Para un imán de neodimio N42 ($m_0 = 28.50 \pm 0.20\ \mathrm{g}$, $m_{\mathrm{eff}} = 0.820 \pm 0.025\ \mathrm{A\cdot m^2}$) descendiendo por un tubo de $1.000 \pm 0.002\ \mathrm{m}$ ($R = 11.50 \pm 0.10\ \mathrm{mm}$, $\Delta R = 1.20 \pm 0.05\ \mathrm{mm}$):

| Material del Tubo | Conductividad $\sigma\ [\mathrm{S/m}]$ | Coeficiente $b\ [\mathrm{N\cdot s/m}]$ | Velocidad Terminal $v_t\ [\mathrm{m/s}]$ | Tiempo de Caída $t_{\mathrm{caída}}\ [\mathrm{s}]$ | Energía Joule $E_{\mathrm{Joule}}\ [\mathrm{mJ}]$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cobre Electrolítico** | $5.96 \times 10^7$ | $3.053 \pm 0.251$ | $0.0915 \pm 0.0076$ | $10.97 \pm 0.94$ | $282.2$ |
| **Aluminio 6061-T6** | $3.77 \times 10^7$ | $1.931 \pm 0.160$ | $0.1447 \pm 0.0120$ | $6.98 \pm 0.60$ | $281.2$ |
| **PVC (Caída Libre)** | $\approx 10^{-12}$ | $\approx 0$ | $4.4287 \pm 0.0044$ | $0.463 \pm 0.003$ | $\approx 0$ |

* **Tiempo característico de relajación:** $\tau_{\mathrm{Cu}} = 9.3\ \mathrm{ms}$. El imán alcanza el $99\%$ de su velocidad terminal en los primeros $1.5\ \mathrm{mm}$ de caída.
* **Balance de energía:** La pérdida de energía potencial gravitacional $\Delta E_{\mathrm{pot}} = 279.5\ \mathrm{mJ}$ se transforma en un $99.96\%$ en disipación térmica por efecto Joule ($282.2\ \mathrm{mJ}$ totales integrados incluyendo el escape). La potencia disipada en estado estacionario es $P_{\mathrm{Joule}} = b_0 v_t^2 = m_0 g v_t = 25.57\ \mathrm{mW}$.

---

## 🚀 Guía de Reproducción

### 1. Requisitos Previos
* **Python 3.10 o superior**
* Paquetes científicos:
  ```bash
  pip install numpy scipy matplotlib
  ```

### 2. Ejecutar la Simulación
Para recalcular las trayectorias, ejecutar las 1000 iteraciones de Monte Carlo, regenerar las gráficas y las tablas:
```bash
python simulation/lenz_simulation.py
```

### 3. Compilación del Artículo en LaTeX
El archivo `paper/main.tex` está configurado para la clase estándar `revtex4-2` (Physical Review B) de la American Physical Society (APS).

#### Opción A: En Overleaf (Recomendado)
1. Comprimir en un archivo `.zip` las carpetas `paper/`, `figures/` y `tables/`.
2. Crear un nuevo proyecto en Overleaf subiendo el `.zip`.
3. Establecer el compilador en **pdfLaTeX** y el documento principal en `paper/main.tex`.

#### Opción B: En Línea de Comandos Local (TeX Live / MiKTeX)
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 📚 Referencias Bibliográficas
1. **Rimstar, S.** (2013). *Lenz's Law with Copper Pipe and Neodymium Magnet*. Rimstar Science Projects: [rimstar.org](https://rimstar.org/science_electronics_projects/lenzs_law_with_copper_pipe_and_neodymium_magnet.htm).
2. **Griffiths, D. J.** (2017). *Introduction to Electrodynamics* (4th ed.). Cambridge University Press.
3. **Jackson, J. D.** (1999). *Classical Electrodynamics* (3rd ed.). John Wiley & Sons.
4. **Hahn, K. D., Johnson, E. M., Brokken, A., & Baldwin, S.** (2002). *Eddy current damping of a magnet moving through a pipe*. American Journal of Physics, 70(10), 1033–1040.
5. **MacLatchy, C. S., Backman, P., & Bogan, L.** (1993). *A quantitative magnetic braking experiment*. American Journal of Physics, 61(12), 1096–1101.
6. **Saslow, W. M.** (1992). *Maxwell's theory of eddy currents in thin conducting sheets, and applications to electromagnetic damping and levitation*. American Journal of Physics, 60(8), 693–711.
7. **Partovi, M. H., & Morris, D.** (2006). *Electrodynamics of a magnet moving through a conducting pipe*. Canadian Journal of Physics, 84(4), 253–271.
8. **Taylor, J. R.** (1997). *An Introduction to Error Analysis: The Study of Uncertainties in Physical Measurements* (2nd ed.). University Science Books.
