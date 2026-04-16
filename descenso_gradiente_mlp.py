import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# RED NEURONAL ARTIFICIAL - PERCEPTRON MULTICAPA (MLP)
# Algoritmo de Descenso de Gradiente (Batch Gradient Descent)
# =============================================================================
# Este programa implementa una red neuronal multicapa que aprende mediante
# el algoritmo de descenso de gradiente por lotes (batch). A diferencia de
# la regla delta generalizada (que actualiza patron por patron), aqui se
# acumulan los gradientes de TODOS los patrones del conjunto de entrenamiento
# y se actualizan los pesos una sola vez al final de cada epoca.
#
# El descenso de gradiente minimiza la funcion de costo E (error cuadratico
# medio) moviendo los pesos en la direccion opuesta al gradiente:
#   W = W - lr * dE/dW
# donde dE/dW es la derivada parcial del error total respecto a cada peso.
#
# Se utiliza el problema XOR como ejemplo de prueba.
# =============================================================================

# --- Semilla para reproducibilidad de resultados ---
np.random.seed(42)

# --- Hiperparametros de la red ---
tasa_aprendizaje = 0.5      # Controla el tamano del paso en la direccion del gradiente
epocas_maximas = 10000       # Numero maximo de ciclos de aprendizaje
error_minimo = 1e-5          # Umbral de error para detener el entrenamiento

# --- Arquitectura de la red ---
# Se define como una lista donde cada elemento indica el numero de neuronas por capa.
# [2, 4, 1] significa: 2 neuronas de entrada, 4 en la capa oculta, 1 de salida.
arquitectura = [2, 4, 1]

# --- Conjunto de entrenamiento ---
# Se utiliza el problema XOR (OR exclusivo) como ejemplo.
# El perceptron simple no puede resolver XOR porque no es linealmente separable,
# por lo que se necesita una capa oculta para aprender esta funcion.
# X(n): patrones de entrada
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
# S(n): salidas deseadas para cada patron
S = np.array([
    [0],
    [1],
    [1],
    [0]
])

N = len(X)  # Numero total de patrones de entrenamiento

# --- Funcion de activacion: Sigmoide ---
# Se utiliza la funcion sigmoide porque es diferenciable en todo su dominio,
# lo cual es requisito para calcular los gradientes en el descenso de gradiente.
# f(x) = 1 / (1 + exp(-x)), con rango de salida en (0, 1).
def sigmoide(x):
    return 1.0 / (1.0 + np.exp(-x))

# --- Derivada de la funcion sigmoide ---
# f'(x) = f(x) * (1 - f(x))
# Se necesita para calcular las derivadas parciales del error respecto a los pesos.
def derivada_sigmoide(x):
    fx = sigmoide(x)
    return fx * (1.0 - fx)

# =============================================================================
# PASO 1: Inicializacion de pesos y umbrales de la red
# =============================================================================
# Se inicializan los pesos y umbrales de forma aleatoria con valores pequenos
# alrededor de 0 (entre -0.5 y 0.5). Esto evita la saturacion de la sigmoide
# y permite que la red comience el aprendizaje en una zona sensible de la funcion.

num_capas = len(arquitectura)

# Lista de matrices de pesos: W[l] tiene dimension (neuronas_capa_l, neuronas_capa_l+1)
# W[l][i][j] conecta la neurona i de la capa l con la neurona j de la capa l+1
pesos = []
for l in range(num_capas - 1):
    W = np.random.uniform(-0.5, 0.5, size=(arquitectura[l], arquitectura[l + 1]))
    pesos.append(W)

# Lista de vectores de umbrales (bias): b[l] tiene dimension (1, neuronas_capa_l+1)
# Cada neurona tiene su propio umbral que se ajusta durante el entrenamiento
umbrales = []
for l in range(num_capas - 1):
    b = np.random.uniform(-0.5, 0.5, size=(1, arquitectura[l + 1]))
    umbrales.append(b)

print("=" * 60)
print("MLP - DESCENSO DE GRADIENTE POR LOTES (BATCH)")
print("=" * 60)
print("Arquitectura de la red:", arquitectura)
print("Tasa de aprendizaje:", tasa_aprendizaje)
print("Epocas maximas:", epocas_maximas)
print("Error minimo objetivo:", error_minimo)
print("-" * 60)

# --- Historial de errores para graficar la convergencia ---
historial_errores = []

# =============================================================================
# CICLO PRINCIPAL DE APRENDIZAJE - DESCENSO DE GRADIENTE POR LOTES
# =============================================================================
# En el descenso de gradiente por lotes, a diferencia de la actualizacion
# patron por patron (estocastica), se siguen estos pasos en cada epoca:
#   1. Se propagan TODOS los patrones hacia adelante
#   2. Se calcula el error total sobre todos los patrones
#   3. Se calculan los gradientes acumulados de todos los patrones
#   4. Se actualizan los pesos UNA SOLA VEZ con el gradiente promedio
# Esto produce actualizaciones mas estables pero puede ser mas lento.

for epoca in range(epocas_maximas):

    # =========================================================================
    # Inicializacion de acumuladores de gradientes
    # =========================================================================
    # Se crean matrices de ceros con las mismas dimensiones que los pesos y
    # umbrales. Aqui se acumularan los gradientes de todos los patrones antes
    # de aplicar la actualizacion al final de la epoca.
    grad_pesos = []
    grad_umbrales = []
    for l in range(num_capas - 1):
        grad_pesos.append(np.zeros_like(pesos[l]))
        grad_umbrales.append(np.zeros_like(umbrales[l]))

    error_total = 0.0  # Acumulador del error total de la epoca

    # =========================================================================
    # PASO 2: Propagacion hacia adelante de TODOS los patrones
    # =========================================================================
    # Se recorre cada patron del conjunto de entrenamiento para calcular
    # la salida de la red y los gradientes correspondientes, pero SIN
    # modificar los pesos todavia.

    for n in range(N):

        # ---------------------------------------------------------------------
        # Propagacion hacia adelante del patron n
        # ---------------------------------------------------------------------
        # Se toma el patron n (X(n), S(n)) y se propaga a traves de la red
        # capa por capa para obtener la respuesta Y(n).
        entrada = X[n:n+1, :]          # Dimension: (1, num_entradas)
        salida_deseada = S[n:n+1, :]   # Dimension: (1, num_salidas)

        # Listas para almacenar activaciones y sumas ponderadas de cada capa.
        # Se necesitan para calcular los gradientes en la retropropagacion.
        activaciones = [entrada]  # a[0] = entrada del patron
        sumas_ponderadas = []     # net[l] = suma ponderada antes de la activacion

        # Se propaga la senal capa por capa desde la entrada hasta la salida
        a = entrada
        for l in range(num_capas - 1):
            # Suma ponderada: net_l = a_{l-1} * W[l] + b[l]
            net = a @ pesos[l] + umbrales[l]
            sumas_ponderadas.append(net)

            # Se aplica la funcion de activacion sigmoide: a_l = f(net_l)
            a = sigmoide(net)
            activaciones.append(a)

        # La salida de la red Y(n) es la activacion de la ultima capa
        Y = activaciones[-1]

        # ---------------------------------------------------------------------
        # PASO 3: Evaluacion del error cuadratico del patron n
        # ---------------------------------------------------------------------
        # error_n = 0.5 * sum((S(n) - Y(n))^2)
        # El factor 0.5 facilita el calculo de la derivada.
        # Se acumula para obtener el error total de la epoca.
        error_n = 0.5 * np.sum((salida_deseada - Y) ** 2)
        error_total += error_n

        # ---------------------------------------------------------------------
        # Calculo de gradientes por retropropagacion (sin actualizar pesos)
        # ---------------------------------------------------------------------
        # Se calculan los gradientes del error respecto a cada peso y umbral,
        # pero en lugar de actualizar inmediatamente, se ACUMULAN los gradientes
        # de todos los patrones para hacer una unica actualizacion al final.

        # Lista para almacenar los valores delta de cada capa
        deltas = [None] * (num_capas - 1)

        # Calculo de delta para la capa de salida:
        # delta_salida = -(S(n) - Y(n)) * f'(net_salida)
        # El signo negativo se debe a que el gradiente del error E = 0.5*(S-Y)^2
        # respecto a los pesos apunta en la direccion de MAYOR error,
        # y queremos ir en la direccion contraria (descenso).
        # Nota: se usa (S - Y) directamente y se suma al gradiente con signo
        # negativo para mantener la convencion dE/dW.
        indice_salida = num_capas - 2
        deltas[indice_salida] = (salida_deseada - Y) * derivada_sigmoide(sumas_ponderadas[indice_salida])

        # Calculo de delta para las capas ocultas (retropropagacion):
        # Se recorre desde la ultima capa oculta hacia la primera.
        # delta[l] = (delta[l+1] * W[l+1]^T) * f'(net[l])
        # El error se retropropaga ponderado por los pesos de la capa siguiente.
        for l in range(num_capas - 3, -1, -1):
            deltas[l] = (deltas[l + 1] @ pesos[l + 1].T) * derivada_sigmoide(sumas_ponderadas[l])

        # Acumulacion de gradientes:
        # En lugar de actualizar los pesos, se suman los gradientes de este patron
        # a los acumuladores. El gradiente para cada peso es: dE/dW[l] = a[l]^T * delta[l]
        # Se acumula con signo positivo porque delta ya contiene (S - Y), y la
        # actualizacion final sera: W += lr * gradiente_promedio (ascenso en direccion de reduccion del error).
        for l in range(num_capas - 1):
            # Gradiente de pesos: contribucion del patron n
            grad_pesos[l] += activaciones[l].T @ deltas[l]
            # Gradiente de umbrales: contribucion del patron n
            grad_umbrales[l] += deltas[l]

    # =========================================================================
    # PASO 4: Actualizacion de pesos y umbrales (una vez por epoca)
    # =========================================================================
    # Esta es la diferencia clave con la actualizacion estocastica:
    # Se calcula el GRADIENTE PROMEDIO dividiendo entre N (numero de patrones)
    # y se actualizan los pesos una sola vez al final de la epoca.
    #   W[l] = W[l] + lr * (1/N) * sum(gradientes de todos los patrones)
    #   b[l] = b[l] + lr * (1/N) * sum(gradientes de todos los patrones)
    # Esto equivale a moverse en la direccion del gradiente promedio del error
    # total, lo que produce actualizaciones mas suaves y estables.

    for l in range(num_capas - 1):
        # Se divide entre N para obtener el gradiente promedio
        pesos[l] += tasa_aprendizaje * (grad_pesos[l] / N)
        umbrales[l] += tasa_aprendizaje * (grad_umbrales[l] / N)

    # =========================================================================
    # PASO 5: Evaluacion del error total de entrenamiento
    # =========================================================================
    # Se calcula el error cuadratico medio (ECM) sobre todos los patrones:
    # E = (1/N) * sum(error_n) para n = 1, 2, ..., N
    error_medio = error_total / N
    historial_errores.append(error_medio)

    # Imprimir progreso cada 1000 epocas
    if (epoca + 1) % 1000 == 0:
        print(f"Epoca {epoca + 1:>6d} | Error medio: {error_medio:.8f}")

    # =========================================================================
    # PASO 6: Criterio de parada
    # =========================================================================
    # Se verifica si el error ha alcanzado el minimo deseado.
    # Si E < error_minimo, se detiene el entrenamiento.
    # De lo contrario, se repite el ciclo completo en la siguiente epoca.
    if error_medio < error_minimo:
        print(f"\nConvergencia alcanzada en la epoca {epoca + 1}")
        print(f"Error medio final: {error_medio:.10f}")
        break
else:
    # Se ejecuta si se completaron todas las epocas sin alcanzar el error minimo
    print(f"\nSe completaron las {epocas_maximas} epocas sin convergencia")
    print(f"Error medio final: {error_medio:.10f}")

# =============================================================================
# RESULTADOS FINALES
# =============================================================================
# Se propaga cada patron por la red entrenada para verificar las predicciones.

print("\n" + "=" * 60)
print("RESULTADOS DEL ENTRENAMIENTO (DESCENSO DE GRADIENTE)")
print("=" * 60)
print(f"{'Patron':<10} {'Entrada':<15} {'Deseada':<10} {'Obtenida':<12} {'Resultado'}")
print("-" * 60)

for n in range(N):
    # Propagacion hacia adelante del patron n con los pesos ya entrenados
    a = X[n:n+1, :]
    for l in range(num_capas - 1):
        net = a @ pesos[l] + umbrales[l]
        a = sigmoide(net)

    salida_obtenida = a[0, 0]
    salida_deseada_val = S[n, 0]
    # Se redondea la salida para comparar con el valor deseado (0 o 1)
    prediccion = round(salida_obtenida)
    correcto = "OK" if prediccion == salida_deseada_val else "FALLO"

    print(f"  {n+1:<8} {str(X[n]):<15} {salida_deseada_val:<10.0f} {salida_obtenida:<12.6f} {correcto}")

print("-" * 60)

# =============================================================================
# GRAFICA DE CONVERGENCIA DEL ERROR
# =============================================================================
# Se muestra como el error de entrenamiento disminuye a lo largo de las epocas.
# En el descenso de gradiente por lotes, la curva de error es tipicamente
# mas suave que en la actualizacion estocastica, ya que cada actualizacion
# considera la informacion de todos los patrones.

plt.figure(figsize=(10, 6))
plt.plot(historial_errores, color='green', linewidth=1.5)
plt.title('Convergencia del Error - MLP con Descenso de Gradiente por Lotes')
plt.xlabel('Epoca')
plt.ylabel('Error Cuadratico Medio')
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Escala logaritmica para apreciar mejor la convergencia
plt.tight_layout()
plt.show()
