#Este es un algoritmo genético simple para encontrar el googol que es 1 con 100 ceros (10^100)
import random

googol = 10 ** 100
poblacion_size = 6
individuos = [random.randint(1, 10) for _ in range(poblacion_size)]
print("Población inicial:", individuos)

generacion = 0

while True:
    generacion += 1

    # Selección: ordenar de mayor a menor y tomar los 3 mejores
    padres = sorted(individuos, reverse=True)[:3]
    print(f"\n--- Generación {generacion} ---")
    print("Padres seleccionados:", padres)

    # Crossover: cada par genera 2 hijos (suma y producto)
    hijos = []
    for i in range(len(padres)):
        for j in range(i + 1, len(padres)):
            hijos.append(padres[i] + padres[j])   # suma
            hijos.append(padres[i] * padres[j])   # producto
    print("Hijos (crossover):", hijos)

    # Mutación: 1 individuo aleatorio invierte sus dígitos
    idx_mutado = random.randint(0, len(hijos) - 1)
    antes = hijos[idx_mutado]
    hijos[idx_mutado] = int(str(hijos[idx_mutado])[::-1])
    print(f"Mutación: índice {idx_mutado} | {antes} → {hijos[idx_mutado]}")
    print("Hijos (tras mutación):", hijos)

    individuos = hijos
    maximo = max(individuos)
    print(f"Máximo de la generación: {maximo}")

    # Si el máximo supera o iguala al googol, se detiene el ciclo
    if maximo >= googol:
        print(f"\n¡Objetivo alcanzado en {generacion} generaciones!")
        print(f"Valor máximo: {maximo}")
        print(f"Distancia al googol: {maximo - googol}")
        break