
import random
import math

# ─── Parámetros ──────────────────────────────────────────────────────────────
LONG_CROM   = 22                        # bits del cromosoma
LIM_INF     = -1.0                      # límite inferior del dominio
RANGO       = 3.0                       # longitud del dominio (2 - (-1))
DIVISOR     = 2**LONG_CROM - 1         # 4 194 303

PC          = 0.25                      # probabilidad de cruce
PM          = 0.01                      # probabilidad de mutación por bit
POB_SIZE    = 50                        # tamaño de la población
N_GEN       = 150                       # número de generaciones

# ─── Funciones auxiliares ─────────────────────────────────────────────────────

def generar_cromosoma():
    """Genera un cromosoma aleatorio de LONG_CROM bits."""
    return ''.join(random.choice('01') for _ in range(LONG_CROM))

def decodificar(crom):
    """
    Decodifica un cromosoma binario de 22 bits al fenotipo x ∈ [-1, 2].
        Paso 1: x' = valor entero sin signo del cromosoma en base 2
        Paso 2: x  = LIM_INF + x' × (RANGO / (2^22 - 1))
    """
    x_prima = int(crom, 2)
    return LIM_INF + x_prima * (RANGO / DIVISOR)

def fitness(crom):
    """Calcula f(x) = x · sin(10π · x) + 1.0  (siempre ≥ 0 en este dominio)."""
    x = decodificar(crom)
    return x * math.sin(10 * math.pi * x) + 1.0

def seleccion_ruleta(poblacion):
    """
    Selección proporcional al fitness (ruleta).
    Devuelve UN individuo elegido aleatoriamente con peso = fitness.
    """
    fits  = [fitness(c) for c in poblacion]
    total = sum(fits)
    # Si todos los fitness son 0 seleccionamos al azar
    if total == 0:
        return random.choice(poblacion)
    r     = random.uniform(0, total)
    acum  = 0.0
    for crom, f in zip(poblacion, fits):
        acum += f
        if acum >= r:
            return crom
    return poblacion[-1]

def cruce(c1, c2):
    """
    Cruce de 1 punto aleatorio con probabilidad PC.
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
    Mutación bit a bit: cada bit se invierte con probabilidad PM.
    """
    bits = list(crom)
    for i in range(LONG_CROM):
        if random.random() < PM:
            bits[i] = '1' if bits[i] == '0' else '0'
    return ''.join(bits)

def mostrar_resumen(gen, poblacion):
    """Imprime estadísticas de la generación actual."""
    fits   = [fitness(c) for c in poblacion]
    mejor  = max(fits)
    medio  = sum(fits) / len(fits)
    idx    = fits.index(mejor)
    x_m    = decodificar(poblacion[idx])
    print(f"  Gen {gen:>4}  |  f_max = {mejor:8.6f}  |  f_avg = {medio:8.6f}"
          f"  |  x_mejor = {x_m:+.6f}")

# ─── Algoritmo Principal ─────────────────────────────────────────────────────

print("=" * 70)
print("  AG SIMPLE — Máx f(x) = x · sin(10π·x) + 1.0   x ∈ [-1, 2]")
print(f"  Cromosoma: {LONG_CROM} bits  |  Pc = {PC}  |  Pm = {PM}"
      f"  |  Pob = {POB_SIZE}  |  Gens = {N_GEN}")
print("=" * 70)
print(f"  {'Generación':>10}  |  {'f_max':^12}  |  {'f_avg':^12}  |  {'x_mejor':^14}")
print("  " + "─" * 66)

# Generación 0 — población inicial aleatoria
poblacion = [generar_cromosoma() for _ in range(POB_SIZE)]
mostrar_resumen(0, poblacion)

# Historial del mejor para graficar si se desea
historial_mejor = []
historial_avg   = []

# ─── Bucle evolutivo ─────────────────────────────────────────────────────────
for gen in range(1, N_GEN + 1):
    nueva_pob = []

    while len(nueva_pob) < POB_SIZE:
        # Selección por ruleta
        p1 = seleccion_ruleta(poblacion)
        p2 = seleccion_ruleta(poblacion)

        # Cruce
        h1, h2 = cruce(p1, p2)

        # Mutación
        h1 = mutar(h1)
        h2 = mutar(h2)

        nueva_pob.append(h1)
        if len(nueva_pob) < POB_SIZE:
            nueva_pob.append(h2)

    poblacion = nueva_pob

    fits = [fitness(c) for c in poblacion]
    historial_mejor.append(max(fits))
    historial_avg.append(sum(fits) / len(fits))

    # Mostrar cada 10 generaciones y la última
    if gen % 10 == 0 or gen == N_GEN:
        mostrar_resumen(gen, poblacion)

# ─── Resultado final ─────────────────────────────────────────────────────────
fits  = [fitness(c) for c in poblacion]
idx   = fits.index(max(fits))
mejor = poblacion[idx]
x_opt = decodificar(mejor)

print()
print("=" * 70)
print("  MEJOR INDIVIDUO ENCONTRADO")
print("=" * 70)
print(f"  Cromosoma  : {mejor}")
print(f"  x' (int)   : {int(mejor, 2)}")
print(f"  x (fenotipo): {x_opt:+.6f}")
print(f"  f(x)        : {fitness(mejor):.6f}")
print(f"  Óptimo conocido ≈ 1.850773  en x ≈ 1.850773")
print("=" * 70)
