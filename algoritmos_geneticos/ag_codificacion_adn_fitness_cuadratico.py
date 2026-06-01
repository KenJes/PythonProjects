# Algoritmo Genético con Codificación ADN (4 bits)
# Pc = 0.5  |  Pm = 0.16 (inversión de todos los dígitos)
# Fitness: f(x) = x²  |  Selección: Elitista (3)
# Cruce: punto fijo en la mitad (posición 2)
#   Hijo1(P1xP2) = P1[:2] + P2[2:]
#   Hijo2(P2xP1) = P2[2:] + P1[:2]
# Condición de paro: 4 generaciones

import random

# ─── Parámetros ──────────────────────────────────────────────────────────────
POBLACION_INICIAL = [0, 14, 12, 8, 10, 2]
PC             = 0.5    # Probabilidad de cruce
PM             = 0.16   # Probabilidad de mutación
N_ELITISTAS    = 3      # Padres seleccionados por elitismo
PUNTO_CRUCE    = 2      # Posición fija de cruce (mitad de 4 bits)
N_GENERACIONES = 4

# ─── Funciones auxiliares ─────────────────────────────────────────────────────
def codificar(x):
    return format(x, '04b')

def decodificar(adn):
    return int(adn, 2)

def fitness(x):
    return x ** 2

def mutar(adn):
    """Inversión completa de todos los bits."""
    return ''.join('1' if b == '0' else '0' for b in adn)

def seleccion_elitista(poblacion):
    return sorted(poblacion, key=fitness, reverse=True)[:N_ELITISTAS]

def mostrar_tabla(num_gen, individuos, etiquetas=None):
    labels = etiquetas if etiquetas else [str(x) for x in individuos]
    ancho = max(len(str(l)) for l in labels)
    ancho = max(ancho, 13)
    print(f"\n{'─' * 54}")
    print(f"  Generación {num_gen}")
    print(f"{'─' * 54}")
    print(f"  {'Individuo (x)':<{ancho}}  {'ADN':^10}  {'f(x)':>8}")
    print(f"  {'─' * 50}")
    for idx, x in enumerate(individuos):
        print(f"  {labels[idx]:<{ancho}}  {codificar(x):^10}  {fitness(x):>8}")

# ─── Algoritmo Principal ──────────────────────────────────────────────────────
print("=" * 54)
print("  ALGORITMO GENÉTICO — Codificación ADN 4 bits")
print(f"  f(x) = x²  |  Elitista({N_ELITISTAS})  |  Probabilidad de cruce = {PC}  |  Probabilidad de mutación = {PM}")
print("=" * 54)

poblacion = POBLACION_INICIAL[:]
mostrar_tabla(1, poblacion)

padres = seleccion_elitista(poblacion)
print(f"\n  >> Padres seleccionados: {padres}")
print(f"     ADN: {[codificar(p) for p in padres]}")

for gen in range(2, N_GENERACIONES + 1):
    hijos = []
    etiquetas = []

    for i in range(len(padres)):
        for j in range(i + 1, len(padres)):
            p1, p2 = padres[i], padres[j]
            adn1, adn2 = codificar(p1), codificar(p2)

            # Cruce con probabilidad PC
            if random.random() < PC:
                h1_adn = adn1[:PUNTO_CRUCE] + adn2[PUNTO_CRUCE:]
                h2_adn = adn2[PUNTO_CRUCE:] + adn1[:PUNTO_CRUCE]
            else:
                h1_adn, h2_adn = adn1, adn2

            # Mutación con probabilidad PM (inversión total de bits)
            if random.random() < PM:
                h1_adn = mutar(h1_adn)
            if random.random() < PM:
                h2_adn = mutar(h2_adn)

            h1, h2 = decodificar(h1_adn), decodificar(h2_adn)
            hijos.extend([h1, h2])
            etiquetas.extend([f"{p1}X{p2}={h1}", f"{p2}X{p1}={h2}"])

    mostrar_tabla(gen, hijos, etiquetas)

    if gen < N_GENERACIONES:
        padres = seleccion_elitista(hijos)
        print(f"\n  >> Padres seleccionados: {padres}")
        print(f"     ADN: {[codificar(p) for p in padres]}")
    else:
        mejor = max(hijos, key=fitness)
        print(f"\n  >> Mejor individuo final: x={mejor},  ADN={codificar(mejor)},  f(x)={fitness(mejor)}")