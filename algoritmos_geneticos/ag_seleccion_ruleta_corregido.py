#Algoritmo Genético Simple Con Selección por Ranking forma 1:  (amax − (amax − amin) * (ranking -1) / (m - 1)) * 1 / m 
# a max = 1.2, a min = 2 - a_max = 0.8, m = 6
# Condición necesaria para que Σ pᵢ = 1:  a_max + a_min = 2  →  a_min = 2 - a_max
 
import random

individuos = [1,2,3,4,5,6] # Individuos 'X'
print("Individuos:", individuos)

def fitness(individuo):          # Función de Fitness Y = X^2
    return individuo ** 2

#Calcular el raking de cada individuo
fitness_values = [fitness(ind) for ind in individuos]

#Ordenar los individuos por su valor de fitness y asignarles un ranking
ranking = sorted(range(len(fitness_values)), key=lambda k: fitness_values[k], reverse=True)

# Probabilidad de selección por ranking
a_max = 1.2
a_min = 2 - a_max   # = 0.8  (correcto: a_max + a_min = 2  →  Σ pᵢ = 1)
# a_min = a_max - 1 es incorrecto: daría a_min=0.2 y Σ pᵢ = 0.7
m = len(individuos) # m = 6
# ranking.index(i) devuelve la posición 0-based (0=mejor) = rank-1 del libro
# NO se resta 1 extra: sería rank-2 y desplazaría toda la distribución
probabilidades_seleccion = [(a_max - (a_max - a_min) * ranking.index(i) / (m - 1)) * 1 / m for i in range(m)]
probablidad_acumulada = [sum(probabilidades_seleccion[:i+1]) for i in range(m)]


#Tabla de selección por ranking y probabilidades de selección
print("\nRanking de Individuos:")
print("-" * 35)
print("Individuo | Fitness | Ranking | Probabilidad | Probabilidad Acumulada")
for i in range(len(individuos)):
    print(f"{individuos[i]:>9} | {fitness_values[i]:>7} | {ranking.index(i)+1:>7} | {probabilidades_seleccion[i]:>12.6f} | {probablidad_acumulada[i]:>20.6f}")
print("-" * 35)


