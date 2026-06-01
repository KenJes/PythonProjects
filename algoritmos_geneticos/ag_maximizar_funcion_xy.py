# ╔══════════════════════════════════════════════════════════════════╗
# ║       ALGORITMO GENÉTICO — Maximizar z = (4x² - 4y²) / 3       ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# ── DISEÑO EN PAPEL ──────────────────────────────────────────────────
#
#  FUNCIÓN OBJETIVO
#      Máx  z = (4x² - 4y²) / 3       x, y ∈ [-32, 31],  x, y ∈ ℝ
#
#  REPRESENTACIÓN (codificación binaria)
#      Cromosoma : 8 bits = [ b7 b6 b5 b4 | b3 b2 b1 b0 ]
#                              ─── x ────   ─── y ────
#      Cada variable usa 4 bits → valor entero v ∈ [0, 15]
#      Decodificación:
#          valor_real = LIM_INF + v × (LIM_SUP - LIM_INF) / (2^BITS - 1)
#                     = -32     + v × (63 / 15)
#                     = -32     + v × 4.2
#
#  ESPACIO DE BÚSQUEDA
#      v=0  → -32.0  (mínimo del dominio)
#      v=15 →  31.0  (máximo del dominio)
#      Paso entre valores posibles: 63/15 ≈ 4.2
#
#  ÓPTIMO TEÓRICO ALCANZABLE CON 4 BITS
#      Maximizar z requiere |x| máximo e |y| mínimo.
#      x = -32.0 (v=0),  y = 1.6 (v=8, el más cercano a 0)
#      z_max = (4 × 1024 − 4 × 2.56) / 3 ≈ 1361.92
#
#  PARÁMETROS
#      Población    :  6 individuos
#      Pc           :  0.7   (probabilidad de cruce)
#      Pm           :  1/8   (prob. de mutación — flip de 1 bit aleatorio)
#      Punto cruce  :  posición 4 (frontera x | y)
#      Selección    :  Elitista — top 3 por fitness
#      Generaciones :  5
#
#  OPERADORES
#      Cruce    : 1 punto fijo (pos. 4).
#                 Hijo1 = x_pad1 | y_pad2
#                 Hijo2 = x_pad2 | y_pad1
#      Mutación : se invierte 1 bit elegido al azar dentro del cromosoma.
#
# ─────────────────────────────────────────────────────────────────────

import random

# ─── Parámetros ───────────────────────────────────────────────────────────────
BITS_VAR    = 4                           # bits por variable (x o y)
LONG_CROM   = BITS_VAR * 2               # longitud total del cromosoma (8 bits)
LIM_INF     = -32                        # límite inferior del dominio
LIM_SUP     = 31                         # límite superior del dominio
PASO        = (LIM_SUP - LIM_INF) / (2**BITS_VAR - 1)  # ≈ 4.2
PC          = 0.7                        # probabilidad de cruce
PM          = 1 / LONG_CROM             # probabilidad de mutación (1/8 ≈ 0.125)
N_ELITISTAS = 3                          # padres seleccionados por elitismo
PUNTO_CRUCE = BITS_VAR                   # posición de corte (= 4, frontera x|y)
N_GEN       = 5                          # número de generaciones
POB_SIZE    = 6                          # tamaño de la población

# ─── Funciones auxiliares ─────────────────────────────────────────────────────

def generar_cromosoma():
    """Genera un cromosoma aleatorio de LONG_CROM bits."""
    return ''.join(random.choice('01') for _ in range(LONG_CROM))

def decodificar(crom):
    """
    Decodifica un cromosoma de 8 bits en (x, y) reales.
        Bits 0-3 (mitad izquierda) → x
        Bits 4-7 (mitad derecha)   → y
    Fórmula: real = LIM_INF + entero_sin_signo × PASO
    """
    vx = int(crom[:BITS_VAR], 2)
    vy = int(crom[BITS_VAR:], 2)
    x  = LIM_INF + vx * PASO
    y  = LIM_INF + vy * PASO
    return x, y

def fitness(crom):
    """Calcula z = (4x² - 4y²) / 3 para el cromosoma dado."""
    x, y = decodificar(crom)
    return (4 * x**2 - 4 * y**2) / 3

def mutar(crom):
    """Invierte (flip) un bit elegido al azar dentro del cromosoma."""
    pos  = random.randint(0, LONG_CROM - 1)
    bits = list(crom)
    bits[pos] = '1' if bits[pos] == '0' else '0'
    return ''.join(bits)

def cruce(c1, c2):
    """
    Cruce de 1 punto fijo en PUNTO_CRUCE (posición 4).
        Hijo1 = x_de_c1 | y_de_c2
        Hijo2 = x_de_c2 | y_de_c1
    Se aplica con probabilidad PC; sin cruce devuelve copias de los padres.
    """
    if random.random() < PC:
        h1 = c1[:PUNTO_CRUCE] + c2[PUNTO_CRUCE:]
        h2 = c2[:PUNTO_CRUCE] + c1[PUNTO_CRUCE:]
    else:
        h1, h2 = c1, c2
    return h1, h2

def seleccion_elitista(poblacion):
    """Devuelve los N_ELITISTAS cromosomas con mayor fitness."""
    return sorted(poblacion, key=fitness, reverse=True)[:N_ELITISTAS]

def mostrar_tabla(num_gen, poblacion, etiquetas=None):
    """Imprime la tabla de la generación con cromosoma, x, y y z."""
    SEP = "─" * 62
    print(f"\n{SEP}")
    print(f"  Generación {num_gen}")
    print(SEP)
    print(f"  {'Cromosoma':<12}  {'x':>7}  {'y':>7}  {'z = f(x,y)':>12}  {'Origen'}")
    print(f"  {'─' * 58}")
    for idx, crom in enumerate(poblacion):
        x, y = decodificar(crom)
        z    = fitness(crom)
        orig = etiquetas[idx] if etiquetas else "—"
        print(f"  {crom:<12}  {x:>7.2f}  {y:>7.2f}  {z:>12.4f}  {orig}")

# ─── Algoritmo Principal ──────────────────────────────────────────────────────

print("=" * 62)
print("  ALGORITMO GENÉTICO — Máx z = (4x² − 4y²) / 3")
print(f"  Dominio: x, y ∈ [{LIM_INF}, {LIM_SUP}]  |  Paso = {PASO:.4f}")
print(f"  Bits/var = {BITS_VAR}  |  Pc = {PC}  |  Pm = {PM:.4f}  |  Élite = {N_ELITISTAS}")
print("=" * 62)

# Generación 1 — población inicial aleatoria
poblacion = [generar_cromosoma() for _ in range(POB_SIZE)]
mostrar_tabla(1, poblacion)

# Seleccionar padres iniciales
padres = seleccion_elitista(poblacion)
print(f"\n  >> Padres seleccionados: {padres}")
print(f"     Fitness: {[round(fitness(p), 4) for p in padres]}")

# ─── Bucle evolutivo (generaciones 2 … N_GEN) ────────────────────────────────
for gen in range(2, N_GEN + 1):
    hijos      = []
    etiquetas  = []

    for i in range(len(padres)):
        for j in range(i + 1, len(padres)):
            p1, p2 = padres[i], padres[j]

            # Cruce
            h1, h2 = cruce(p1, p2)

            # Mutación independiente para cada hijo
            if random.random() < PM:
                h1 = mutar(h1)
            if random.random() < PM:
                h2 = mutar(h2)

            hijos.extend([h1, h2])
            etiquetas.extend([f"P{i+1}×P{j+1}→H1", f"P{i+1}×P{j+1}→H2"])

    mostrar_tabla(gen, hijos, etiquetas)

    if gen < N_GEN:
        padres = seleccion_elitista(hijos)
        print(f"\n  >> Padres seleccionados: {padres}")
        print(f"     Fitness: {[round(fitness(p), 4) for p in padres]}")

# ─── Resultado final ─────────────────────────────────────────────────────────
todos  = hijos if N_GEN > 1 else poblacion
mejor  = max(todos, key=fitness)
xb, yb = decodificar(mejor)
print()
print("=" * 62)
print("  MEJOR INDIVIDUO ENCONTRADO")
print("=" * 62)
print(f"  Cromosoma : {mejor}")
print(f"  x         : {xb:.4f}  (bits: {mejor[:BITS_VAR]}  → valor int {int(mejor[:BITS_VAR], 2)})")
print(f"  y         : {yb:.4f}  (bits: {mejor[BITS_VAR:]}  → valor int {int(mejor[BITS_VAR:], 2)})")
print(f"  z = f(x,y): {fitness(mejor):.4f}")
print(f"  Óptimo teórico alcanzable (4 bits): ≈ 1361.9200")
print("=" * 62)
