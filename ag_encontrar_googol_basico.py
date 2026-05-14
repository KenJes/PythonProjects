import random
googol = 10 ** 100
poblacion=[8, 4, 7, 1, 7, 6]
print("Generación 0:", poblacion)
padres = poblacion[:3]
print("Padres seleccionados:", padres)

hijos = []
for i in range(0, len(padres) - 1):
    for j in range(i + 1, len(padres)):
        padre1 = padres[i]
        padre2 = padres[j]
        hijo1 = padre1 + padre2
        hijo2 = padre1 * padre2
        hijos.append(hijo1)
        hijos.append(hijo2)


mutado = random.randint(0, len(hijos) - 1)
hijos[mutado] = int(str(hijos[mutado])[::-1])
print ("Generación 1 - Hijos después de la mutación: ", hijos)
individuos = hijos
maximo = max(individuos)
print("Número natural más grande encontrado:", maximo)
generacion = 1
while maximo <= googol:
    generacion += 1
    Padres = (individuos[:3])
    hijos = []
    for i in range(0, len(Padres) - 1):
        for j in range(i + 1, len(Padres)):
            padre1 = Padres[i]
            padre2 = Padres[j]
            hijo1 = padre1 + padre2
            hijo2 = padre1 * padre2
            hijos.append(hijo1)
            hijos.append(hijo2)
    mutado = random.randint(0, len(hijos) - 1)
    hijos[mutado] = int(str(hijos[mutado])[::-1]) 
    print (f"Generación {generacion} - Hijos después de la mutación: ", hijos)
    individuos = hijos 
    print (f"Generación {generacion} - Población final: ", individuos)
    maximo = max(individuos)
    print (f"Generación {generacion} - El número natural más grande encontrado es: ", maximo)