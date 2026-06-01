# ╔══════════════════════════════════════════════════════════════════════════╗
# ║   AG SIMPLE — Cualquier función de n variables reales / enteras        ║
# ║                incluye dominios con valores negativos                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ── DISEÑO EN PAPEL ──────────────────────────────────────────────────────────
#
#  REPRESENTACIÓN  (codificación binaria por variable)
#      Cada variable xᵢ ∈ [minᵢ, maxᵢ] se codifica con BITS_VAR bits.
#      Cromosoma = [ seg₁ | seg₂ | … | segₙ ]   longitud = N × BITS_VAR bits
#
#  DECODIFICACIÓN  (por cada segmento de B = BITS_VAR bits)
#      Paso 1 — entero sin signo:
#          v'ᵢ = int(crom[i·B : (i+1)·B], 2)            ∈ [0, 2^B − 1]
#      Paso 2 — cambio de escala al dominio:
#          xᵢ  = minᵢ + v'ᵢ · (rangoᵢ / (2^B − 1))     ∈ [minᵢ, maxᵢ]
#      Paso 3 — si variable entera:
#          xᵢ  = round(xᵢ)
#
#  SELECCIÓN — Ruleta con desplazamiento lineal  (admite fitness < 0)
#      Para cada individuo i:
#          f'ᵢ = fᵢ − f_min + ε      →  siempre f'ᵢ > 0
#      La probabilidad de selección es proporcional a f'ᵢ.
#
#  CRUCE    — 1 punto aleatorio,  probabilidad Pc
#      Punto de corte en [1, LONG_CROM − 1] a nivel de bit.
#      Hijo1 = c1[:punto] + c2[punto:]
#      Hijo2 = c2[:punto] + c1[punto:]
#
#  MUTACIÓN — Flip bit a bit,  probabilidad Pm por bit
#      Cada bit del cromosoma se invierte de forma independiente.
#
# ── FUNCIÓN DE DEMOSTRACIÓN ──────────────────────────────────────────────────
#
#  (Extensión multivariable de Michalewicz 1992 — misma estructura que
#   la función 1D del fichero optimización.py)
#
#  Máx  f(x₁, x₂, x₃) = x₁·sin(10π·x₁) + x₂·sin(10π·x₂) + x₃
#
#      x₁ ∈ [-2.0,  2.0]   →  real
#      x₂ ∈ [-2.0,  2.0]   →  real
#      x₃ ∈ [-3,    3   ]  →  entero
#
#  Óptimo aprox.:
#      x₁ ≈ ±1.850,  x₂ ≈ ±1.850,  x₃ = 3   →   f ≈ 6.70
#
# ── PARÁMETROS ───────────────────────────────────────────────────────────────
#
#  Bits / variable : 20       (precisión ≈ 1.9 × 10⁻⁵ en [-2, 2])
#  Cromosoma total : 60 bits
#  Población       : 80 individuos
#  Pc              : 0.80
#  Pm              : 1 / LONG_CROM  (≈ 0.0167)
#  Generaciones    : 200
#
# ─────────────────────────────────────────────────────────────────────────────

import random
import math

# ── Definición de variables ───────────────────────────────────────────────────
#  ▸ Para cambiar el problema: editar VARIABLES y funcion_objetivo()
#
#  Cada entrada tiene:
#      "nombre" : etiqueta para mostrar en pantalla
#      "min"    : límite inferior del dominio  (puede ser negativo)
#      "max"    : límite superior del dominio
#      "tipo"   : "real" → valor continuo   |   "entero" → se redondea al entero más cercano

VARIABLES = [
    {"nombre": "x₁", "min": -2.0, "max":  2.0, "tipo": "real"   },
    {"nombre": "x₂", "min": -2.0, "max":  2.0, "tipo": "real"   },
    {"nombre": "x₃", "min": -3,   "max":  3,   "tipo": "entero" },
]

# ── Parámetros del AG ─────────────────────────────────────────────────────────

N_VARS    = len(VARIABLES)           # número de variables  (se calcula automáticamente)
BITS_VAR  = 20                       # bits por variable
LONG_CROM = N_VARS * BITS_VAR        # longitud total del cromosoma = 60
DIVISOR   = 2**BITS_VAR - 1         # 1 048 575

PC        = 0.80                     # probabilidad de cruce
PM        = 1.0 / LONG_CROM         # probabilidad de mutación por bit (≈ 0.0167)
POB_SIZE  = 80                       # tamaño de la población
N_GEN     = 200                      # número de generaciones
EPSILON   = 1e-6                     # desplazamiento mínimo para ruleta (evita div/0)

# ─── Función objetivo ─────────────────────────────────────────────────────────
# ▸ Recibe una lista de valores ya decodificados: vals = [x₁, x₂, …, xₙ]
# ▸ Devuelve el escalar a MAXIMIZAR.
# ▸ Para MINIMIZAR f(x): retornar  -f(vals)
# ▸ Para cambiar el problema sólo hay que editar este bloque.

def funcion_objetivo(vals):
    """
    f(x₁, x₂, x₃) = x₁·sin(10π·x₁) + x₂·sin(10π·x₂) + x₃

    Extensión multivariable de la función 1D de Michalewicz (1992).
    El tercer sumando x₃ es entero ∈ {-3, …, 3}: el AG debe descubrir
    que el óptimo global requiere x₃ = 3 (valor entero máximo).
    """
    x1, x2, x3 = vals
    return (x1 * math.sin(10 * math.pi * x1)
            + x2 * math.sin(10 * math.pi * x2)
            + x3)

# ─── Codificación y decodificación ───────────────────────────────────────────

def generar_cromosoma():
    """Genera un cromosoma binario aleatorio de LONG_CROM bits."""
    return ''.join(random.choice('01') for _ in range(LONG_CROM))

def decodificar(crom):
    """
    Decodifica el cromosoma en la lista de fenotipos [x₁, x₂, …, xₙ].

    Por cada variable i:
        seg   = crom[i·BITS_VAR : (i+1)·BITS_VAR]
        v'ᵢ   = int(seg, 2)                          entero sin signo
        xᵢ    = minᵢ + v'ᵢ · (rangoᵢ / DIVISOR)     escalar al dominio
    Si la variable es entera → xᵢ = round(xᵢ)
    """
    vals = []
    for i, var in enumerate(VARIABLES):
        seg   = crom[i * BITS_VAR : (i + 1) * BITS_VAR]
        v_int = int(seg, 2)
        rango = var["max"] - var["min"]
        x     = var["min"] + v_int * (rango / DIVISOR)
        if var["tipo"] == "entero":
            x = round(x)
        vals.append(x)
    return vals

# ─── Evaluación ───────────────────────────────────────────────────────────────

def fitness(crom):
    """Evalúa el cromosoma: decodifica y calcula la función objetivo."""
    return funcion_objetivo(decodificar(crom))

def calcular_fits(poblacion):
    """Devuelve la lista de fitness para cada cromosoma de la población."""
    return [fitness(c) for c in poblacion]

# ─── Selección por ruleta con desplazamiento lineal ───────────────────────────

def ajustar_fits(fits_raw):
    """
    Desplazamiento lineal para garantizar valores positivos en la ruleta.
        f'ᵢ = fᵢ − min(f) + ε      →  siempre f'ᵢ > 0

    Devuelve (fits_ajustados, suma_total).
    Esto permite usar la ruleta incluso cuando todos los fitness son negativos
    o cuando hay variación nula (todos iguales → distribución uniforme).
    """
    f_min  = min(fits_raw)
    ajuste = [f - f_min + EPSILON for f in fits_raw]
    return ajuste, sum(ajuste)

def seleccion_ruleta(poblacion, fits_aj, total_aj):
    """
    Selecciona UN individuo por ruleta proporcional a fits_aj.
    Requiere que todos los valores de fits_aj sean > 0 y total_aj > 0.
    """
    r    = random.uniform(0, total_aj)
    acum = 0.0
    for crom, f in zip(poblacion, fits_aj):
        acum += f
        if acum >= r:
            return crom
    return poblacion[-1]   # salvaguarda numérica

# ─── Cruce y mutación ─────────────────────────────────────────────────────────

def cruce(c1, c2):
    """
    Cruce de 1 punto aleatorio con probabilidad PC.
    El punto de corte se elige a nivel de bit en [1, LONG_CROM − 1]
    para mantener la generalidad independientemente del número de variables.
    Devuelve (hijo1, hijo2).
    """
    if random.random() < PC:
        punto = random.randint(1, LONG_CROM - 1)
        h1 = c1[:punto] + c2[punto:]
        h2 = c2[:punto] + c1[punto:]
        return h1, h2
    return c1, c2

def mutar(crom):
    """
    Mutación bit a bit: cada bit se invierte (flip) con probabilidad PM.
    Opera sobre toda la longitud del cromosoma.
    """
    bits = list(crom)
    for i in range(LONG_CROM):
        if random.random() < PM:
            bits[i] = '1' if bits[i] == '0' else '0'
    return ''.join(bits)

# ─── Visualización ────────────────────────────────────────────────────────────

def _vars_str(vals):
    """Formatea los valores del mejor individuo para mostrarlo en pantalla."""
    partes = []
    for j, var in enumerate(VARIABLES):
        if var["tipo"] == "real":
            partes.append(f"{var['nombre']} = {vals[j]:+.5f}")
        else:
            partes.append(f"{var['nombre']} = {int(vals[j]):+d}")
    return "   ".join(partes)

def mostrar_resumen(gen, poblacion):
    """Imprime estadísticas de la generación: f_max, f_avg y variables del mejor."""
    fits   = calcular_fits(poblacion)
    mejor  = max(fits)
    medio  = sum(fits) / len(fits)
    idx    = fits.index(mejor)
    vals   = decodificar(poblacion[idx])
    print(f"  Gen {gen:>4}  |  f_max = {mejor:+10.5f}  |"
          f"  f_avg = {medio:+10.5f}  |  {_vars_str(vals)}")

# ─── Algoritmo Principal ──────────────────────────────────────────────────────

SEP = "=" * 84

# Encabezado
vars_desc = ",  ".join(
    f"{v['nombre']} ∈ [{v['min']}, {v['max']}] ({v['tipo']})"
    for v in VARIABLES
)
print(SEP)
print("  AG SIMPLE — Maximizar f(x₁, …, xₙ)  |  n variables reales / enteras")
print(f"  Variables : {vars_desc}")
print(f"  Bits/var = {BITS_VAR}  |  Crom = {LONG_CROM} bits  |"
      f"  Pc = {PC}  |  Pm = {PM:.5f}  |  Pob = {POB_SIZE}  |  Gens = {N_GEN}")
print(SEP)
print(f"  {'Generación':>10}  |  {'f_max':^14}  |  {'f_avg':^14}  |  Variables del mejor")
print("  " + "─" * 80)

# ── Generación 0: población inicial aleatoria ─────────────────────────────────
poblacion = [generar_cromosoma() for _ in range(POB_SIZE)]
mostrar_resumen(0, poblacion)

# Historial (útil para graficar la convergencia si se desea)
historial_mejor = []
historial_avg   = []

# ── Bucle evolutivo ───────────────────────────────────────────────────────────
for gen in range(1, N_GEN + 1):

    # 1. Calcular fitness y ajustar para la ruleta
    fits_raw          = calcular_fits(poblacion)
    fits_aj, total_aj = ajustar_fits(fits_raw)

    # 2. Generar nueva población
    nueva_pob = []
    while len(nueva_pob) < POB_SIZE:

        # Selección por ruleta
        p1 = seleccion_ruleta(poblacion, fits_aj, total_aj)
        p2 = seleccion_ruleta(poblacion, fits_aj, total_aj)

        # Cruce de 1 punto
        h1, h2 = cruce(p1, p2)

        # Mutación bit a bit
        h1 = mutar(h1)
        h2 = mutar(h2)

        nueva_pob.append(h1)
        if len(nueva_pob) < POB_SIZE:
            nueva_pob.append(h2)

    poblacion = nueva_pob

    # 3. Registrar historial
    fits_gen = calcular_fits(poblacion)
    historial_mejor.append(max(fits_gen))
    historial_avg.append(sum(fits_gen) / len(fits_gen))

    # 4. Mostrar cada 20 generaciones y la última
    if gen % 20 == 0 or gen == N_GEN:
        mostrar_resumen(gen, poblacion)

# ── Resultado final ───────────────────────────────────────────────────────────
fits_final = calcular_fits(poblacion)
mejor_idx  = fits_final.index(max(fits_final))
mejor_crom = poblacion[mejor_idx]
mejor_vals = decodificar(mejor_crom)
mejor_fit  = fits_final[mejor_idx]

print()
print(SEP)
print("  SOLUCIÓN ENCONTRADA")
print(SEP)
for j, var in enumerate(VARIABLES):
    if var["tipo"] == "real":
        print(f"    {var['nombre']} = {mejor_vals[j]:+.6f}"
              f"   dominio [{var['min']}, {var['max']}]  (real)")
    else:
        print(f"    {var['nombre']} = {int(mejor_vals[j]):+d}"
              f"        dominio [{var['min']}, {var['max']}]  (entero)")
print(f"\n    f(x₁, …, xₙ) = {mejor_fit:+.6f}")
print(f"    Óptimo aprox.:  ≈ +6.70  (x₁ ≈ ±1.850,  x₂ ≈ ±1.850,  x₃ = 3)")
print(SEP)
