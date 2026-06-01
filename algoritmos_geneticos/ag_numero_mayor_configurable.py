import random

# ─── Operaciones de crossover disponibles (se usan las primeras n_hijos_par) ───
OPERACIONES = [
    ("suma",          lambda a, b: a + b),
    ("producto",      lambda a, b: a * b),
    ("prod+suma",     lambda a, b: a * b + (a + b)),
    ("max_cuadrado",  lambda a, b: max(a, b) * max(a, b)),
    ("prod_x_suma",   lambda a, b: (a * b) * (a + b)),
    ("diferencia+1",  lambda a, b: abs(a - b) + 1),
    ("inv_digit",     lambda a, b: int(str(a)[::-1]) + int(str(b)[::-1])),
]
MAX_HIJOS_PAR = len(OPERACIONES)

# ─── Configuración inicial (se pregunta UNA sola vez) ──────────────────────────
print("=== Algoritmo Genético — Número Natural Más Grande ===\n")

objetivo       = int(input("Número objetivo a alcanzar              : "))
pob_inicial    = int(input("Tamaño de la población inicial          : "))
n_padres       = int(input("Cuántos padres seleccionar por gen.     : "))
n_hijos_par    = int(input(f"Hijos por par de padres (máx {MAX_HIJOS_PAR})      : "))
n_mutaciones   = int(input("Cuántos individuos mutar por gen.       : "))

print("\nMecanismo de selección de padres:")
print("  1 - Élite        (los N de mayor valor)")
print("  2 - Ruleta       (probabilístico, peso = valor)")
print("  3 - Torneo       (enfrentamientos aleatorios de 3)")
print("  4 - Posicional   (primeros N en la lista)")
print("  5 - Aleatorio    (N al azar)")
mecanismo = int(input("Elige mecanismo [1-5]                   : "))

print("\nTipo de mutación a aplicar:")
print("  1 - Invertir dígitos")
print("  2 - Incrementar en 5")
print("  3 - Decrementar en 5")
print("  4 - Agregar un dígito '0' al final")
print("  5 - Eliminar el último dígito")
print("  6 - Duplicar el número")
print("  7 - Dividir el número entre 2 (redondeando hacia abajo)")
print("  8 - Reemplazar el número por la suma de sus dígitos")
tipo_mutacion = int(input("Elige tipo de mutación [1-8]            : "))

# Clamp para no pedir más operaciones de las que existen
n_hijos_par = min(n_hijos_par, MAX_HIJOS_PAR)
ops_activas = OPERACIONES[:n_hijos_par]

print(f"\nOperaciones de crossover activas: {[nombre for nombre, _ in ops_activas]}")

# ─── Población inicial ─────────────────────────────────────────────────────────
individuos = [random.randint(1, 10) for _ in range(pob_inicial)]
print(f"Generación 0 — Población inicial: {individuos}\n")


# ─── Función de selección de padres ───────────────────────────────────────────
def seleccionar_padres(individuos, n, mecanismo):
    n = min(n, len(individuos))
    match mecanismo:
        case 1:  # Élite
            return sorted(individuos, reverse=True)[:n]
        case 2:  # Ruleta ponderada
            total = sum(individuos)
            if total == 0:
                return random.sample(individuos, n)
            pesos = [v / total for v in individuos]
            return random.choices(individuos, weights=pesos, k=n)
        case 3:  # Torneo
            padres = []
            for _ in range(n):
                rivales = random.sample(individuos, min(3, len(individuos)))
                padres.append(max(rivales))
            return padres
        case 4:  # Posicional
            return individuos[:n]
        case 5:  # Aleatorio
            return random.sample(individuos, n)
        case _:
            return sorted(individuos, reverse=True)[:n]


# ─── Bucle evolutivo ───────────────────────────────────────────────────────────
generacion = 0
maximo = max(individuos)

while maximo < objetivo:
    generacion += 1

    # Selección
    padres = seleccionar_padres(individuos, n_padres, mecanismo)
    print(f"--- Generación {generacion} ---")
    print(f"Padres: {padres}")

    # Crossover dinámico: por cada par, aplicar cada operación activa
    hijos = []
    for i in range(len(padres)):
        for j in range(i + 1, len(padres)):
            for nombre, op in ops_activas:
                resultado = op(padres[i], padres[j])
                hijos.append(resultado)
    print(f"Hijos (crossover): {hijos}")

    # Mutación dinámica: aplicar tipo_mutacion a n_mutaciones individuos aleatorios
    indices_a_mutar = random.sample(range(len(hijos)), min(n_mutaciones, len(hijos)))
    for idx in indices_a_mutar:
        antes = hijos[idx]
        if tipo_mutacion == 1:
            hijos[idx] = int(str(hijos[idx])[::-1])
        elif tipo_mutacion == 2:
            hijos[idx] += 5
        elif tipo_mutacion == 3:
            hijos[idx] -= 5
        elif tipo_mutacion == 4:
            hijos[idx] = int(str(hijos[idx]) + "0")
        elif tipo_mutacion == 5:
            hijos[idx] = int(str(hijos[idx])[:-1]) if len(str(hijos[idx])) > 1 else 0
        elif tipo_mutacion == 6:
            hijos[idx] *= 2
        elif tipo_mutacion == 7:
            hijos[idx] //= 2
        elif tipo_mutacion == 8:
            hijos[idx] = sum(int(d) for d in str(hijos[idx]))
        print(f"  Mutación índice {idx}: {antes} → {hijos[idx]}")

    print(f"Hijos (tras mutación): {hijos}")

    individuos = hijos
    maximo = max(individuos)
    print(f"Máximo: {maximo}\n")

print(f"¡Objetivo alcanzado en {generacion} generaciones!")
print(f"Valor máximo encontrado: {maximo}")
