#Este es un algporítmo genético simple para encontrar el numero natural más grande
import random
#Iniciarlizar población de 6 individuos aleatoriamente
poblacion = 6
individuos = [random.randint(1, 10) for _ in range(poblacion)]
print ("Generación 0 - Población inicial: ", individuos)
#Seleccionar padres mediante el mecanismo de selección de los 3 primeros individuos de la población
Padres = (individuos[:3])

#Crossover Cruzar a los padres para crear 6 hijos, el primer hijo se crea sumando el primer y segundo padre, el segundo hijo se crea multiplicando el primer y el segundo padre, el tercer hijo se crea sumando el segundo y el tercer padre, el cuarto hijo se crea multiplicando el segundo y el tercer padre, el quinto hijo se crea sumando el primer y el tercer padre, y el sexto hijo se crea multiplicando el primer y el tercer padre.
hijos = []
for i in range(0, len(Padres) - 1):
    for j in range(i + 1, len(Padres)):
        padre1 = Padres[i]
        padre2 = Padres[j]
        hijo1 = padre1 + padre2
        hijo2 = padre1 * padre2
        hijos.append(hijo1)
        hijos.append(hijo2)

#Mutación: Mutar a 1 individuo aleatorio de la nueva generación cambiando sus digitos al revés ejemplo: si el hijo es 12, mutarlo a 21
mutado = random.randint(0, len(hijos) - 1)
hijos[mutado] = int(str(hijos[mutado])[::-1]) 
print ("Hijos después de la mutación: ", hijos)

#Reemplazar a la población original con la nueva generación de hijos
individuos = hijos 
print ("Generación 1 - Población final: ", individuos)

#Evaluar la población final para encontrar el número natural más grande
maximo = max(individuos)
print ("El número natural más grande encontrado es: ", maximo)  

#Simular generaciones hasta encontrar un número natural mayor a 10000
generacion = 1
while maximo <= 10000:
    generacion += 1
    #Seleccionar padres mediante el mecanismo de selección de los 3 primeros individuos de la población
    Padres = (individuos[:3])
    #Crossover Cruzar a los padres para crear 6 hijos, el primer hijo se crea sumando el primer y segundo padre, el segundo hijo se crea multiplicando el primer y el segundo padre, el tercer hijo se crea sumando el segundo y el tercer padre, el cuarto hijo se crea multiplicando el segundo y el tercer padre, el quinto hijo se crea sumando el primer y el tercer padre, y el sexto hijo se crea multiplicando el primer y el tercer padre.
    hijos = []
    for i in range(0, len(Padres) - 1):
        for j in range(i + 1, len(Padres)):
            padre1 = Padres[i]
            padre2 = Padres[j]
            hijo1 = padre1 + padre2
            hijo2 = padre1 * padre2
            hijos.append(hijo1)
            hijos.append(hijo2)
    #Mutación: Mutar a 1 individuo aleatorio de la nueva generación cambiando sus digitos al revés ejemplo: si el hijo es 12, mutarlo a 21
    mutado = random.randint(0, len(hijos) - 1)
    hijos[mutado] = int(str(hijos[mutado])[::-1]) 
    print (f"Generación {generacion} - Hijos después de la mutación: ", hijos)
    #Reemplazar a la población original con la nueva generación de hijos
    individuos = hijos 
    print (f"Generación {generacion} - Población final: ", individuos)
    #Evaluar la población final para encontrar el número natural más grande
    maximo = max(individuos)
    print (f"Generación {generacion} - El número natural más grande encontrado es: ", maximo)