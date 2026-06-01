# Algoritmo Genético Simple para maximizar una funcion de 2 variables
# La función a maximizar es z = (4x^2 - 4y^2)/3, con x e y en el rango de [-32, 31]
# Método de seleccion por ranking, método de cruce uniforme y método de mutación intercambio

import random
print("\nAlgoritmo Genético Simple para maximizar la función z = (4x^2 - 4y^2)/3:")

# Parámetros del algoritmo genético 
tamaño_poblacion = 10
num_generaciones = 20
rango_min = -32
rango_max = 31

# Función de fitness a maximizar
def fitness(individuo):
    x, y = individuo
    return (4 * x ** 2 - 4 * y ** 2) / 3 

# Generar la población inicial de individuos aleatorios dentro del rango especificado
poblacion = [(random.randint(rango_min, rango_max), random.randint(rango_min, rango_max)) for _ in range(tamaño_poblacion)]
print("\nPoblación Inicial:")
for i, individuo in enumerate(poblacion):
    print(f"Individuo {i+1}: {individuo}, Fitness: {fitness(individuo)}")

for generacion in range(num_generaciones):
    # Calcular el fitness de cada individuo en la población
    fitness_values = [fitness(ind) for ind in poblacion]
    
    # Selección por ranking
    ranking = sorted(range(len(fitness_values)), key=lambda k: fitness_values[k], reverse=True) 
    a_max = 1.2
    a_min = 0.2
    m = len(poblacion)
    probabilidades_seleccion = [(a_max - (a_max - a_min) * (ranking.index(i)-1) / (m - 1)) * 1 / m for i in range(m)]
    
    # Crear la nueva población mediante selección, cruce y mutación
    nueva_poblacion = []
    for _ in range(tamaño_poblacion):
        # Selección de padres basada en las probabilidades de selección por ranking
        padre1 = random.choices(poblacion, weights=probabilidades_seleccion, k=1)[0]
        padre2 = random.choices(poblacion, weights=probabilidades_seleccion, k=1)[0]
        
        # Cruce uniforme para generar un hijo
        mascara_cruce = [random.randint(0, 1) for _ in range(2)] # Máscara de cruce para dos genes (x e y)
        hijo = [padre1[i] if mascara_cruce[i] == 1 else padre2[i] for i in range(2)]
        
        # Mutación por intercambio con una probabilidad del 10%
        if random.random() < 0.1:
            gen_mutar = random.randint(0, 1) # Seleccionar aleatoriamente un gen para mutar (x o y)
            hijo[gen_mutar] = random.randint(rango_min, rango_max) # Mutar el gen seleccionado con un nuevo valor aleatorio dentro del rango
        
        nueva_poblacion.append(tuple(hijo)) # Agregar el nuevo individuo a la nueva población
    
    poblacion = nueva_poblacion # Actualizar la población para la siguiente generación
    print(f"\nGeneración {generacion + 1}:")
    for i, individuo in enumerate(poblacion):
        print(f"Individuo {i+1}: {individuo}, Fitness: {fitness(individuo)}")

# Al finalizar las generaciones, encontrar el mejor individuo y su fitness
fitness_values_final = [fitness(ind) for ind in poblacion]
mejor_individuo = poblacion[fitness_values_final.index(max(fitness_values_final))]
print(f"\nMejor Individuo: {mejor_individuo}, Fitness: {fitness(mejor_individuo)}")

