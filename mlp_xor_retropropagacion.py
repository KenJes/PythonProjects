import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# RED NEURONAL ARTIFICIAL - PERCEPTRON MULTICAPA (MLP)
# Algoritmo de Retropropagacion con Regla Delta Generalizada
# =============================================================================
# Este programa implementa una red neuronal multicapa que aprende mediante
# el algoritmo de retropropagacion (backpropagation) usando la regla delta
# generalizada. Se utiliza el problema XOR como ejemplo, ya que no es
# linealmente separable y requiere al menos una capa oculta para resolverse.
# =============================================================================

# --- Semilla para reproducibilidad de resultados ---
np.random.seed(42)

# --- Hiperparametros de la red ---
tasa_aprendizaje = 0.5      # Controla la magnitud de los ajustes en cada iteracion
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
# lo cual es requisito para aplicar la regla delta generalizada.
# f(x) = 1 / (1 + exp(-x)), con rango de salida en (0, 1).
def sigmoide(x):
    return 1.0 / (1.0 + np.exp(-x))

# --- Derivada de la funcion sigmoide ---
# f'(x) = f(x) * (1 - f(x))
# Se necesita para calcular los valores delta en la retropropagacion.
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

print("Arquitectura de la red:", arquitectura)
print("Tasa de aprendizaje:", tasa_aprendizaje)
print("Epocas maximas:", epocas_maximas)
print("Error minimo objetivo:", error_minimo)
print("-" * 60)

# --- Historial de errores para graficar la convergencia ---
historial_errores = []

# =============================================================================
# PASOS 5, 6 y 7: Ciclo principal de aprendizaje
# =============================================================================
# Se repiten los pasos 2, 3 y 4 para todos los patrones (paso 5),
# se evalua el error total (paso 6), y se repite todo hasta alcanzar
# un minimo del error de entrenamiento o completar m ciclos (paso 7).

for epoca in range(epocas_maximas):

    error_total = 0.0  # Acumulador del error total de la epoca

    # =========================================================================
    # PASO 5: Recorrer todos los patrones del conjunto de entrenamiento
    # =========================================================================
    for n in range(N):

        # =====================================================================
        # PASO 2: Propagacion hacia adelante (Forward Pass)
        # =====================================================================
        # Se toma un patron n del conjunto de entrenamiento (X(n), S(n))
        # y se propaga hacia la salida de la red para obtener la respuesta Y(n).

        # Entrada del patron n (se redimensiona a vector fila)
        entrada = X[n:n+1, :]  # Dimension: (1, num_entradas)
        salida_deseada = S[n:n+1, :]  # Dimension: (1, num_salidas)

        # Listas para almacenar las activaciones y sumas ponderadas de cada capa.
        # Se necesitan durante la retropropagacion para calcular los deltas.
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

        # =====================================================================
        # PASO 3: Evaluacion del error cuadratico del patron n
        # =====================================================================
        # Se calcula el error cuadratico cometido por la red para el patron n:
        # error_n = 0.5 * sum((S(n) - Y(n))^2)
        # El factor 0.5 facilita el calculo de la derivada en la retropropagacion.
        error_n = 0.5 * np.sum((salida_deseada - Y) ** 2)
        error_total += error_n

        # =====================================================================
        # PASO 4: Regla Delta Generalizada (Backpropagation)
        # =====================================================================
        # Se modifican los pesos y umbrales de la red propagando el error
        # desde la capa de salida hacia las capas ocultas.

        # Lista para almacenar los valores delta de cada capa
        deltas = [None] * (num_capas - 1)

        # -----------------------------------------------------------------
        # PASO 4a: Calcular delta para las neuronas de la capa de salida
        # -----------------------------------------------------------------
        # delta_salida = (S(n) - Y(n)) * f'(net_salida)
        # Donde f'(net) es la derivada de la sigmoide evaluada en la suma ponderada.
        # Este valor indica cuanto debe ajustarse cada neurona de salida.
        indice_salida = num_capas - 2  # Indice de la ultima capa en la lista
        deltas[indice_salida] = (salida_deseada - Y) * derivada_sigmoide(sumas_ponderadas[indice_salida])

        # -----------------------------------------------------------------
        # PASO 4b: Calcular delta para las capas ocultas (retropropagacion)
        # -----------------------------------------------------------------
        # Se recorre desde la ultima capa oculta hacia la primera.
        # delta_oculta[l] = (delta[l+1] * W[l+1]^T) * f'(net[l])
        # El error se retropropaga ponderado por los pesos de la capa siguiente,
        # y se escala por la derivada de la funcion de activacion.
        for l in range(num_capas - 3, -1, -1):
            deltas[l] = (deltas[l + 1] @ pesos[l + 1].T) * derivada_sigmoide(sumas_ponderadas[l])

        # -----------------------------------------------------------------
        # PASO 4c: Modificar pesos y umbrales de la red
        # -----------------------------------------------------------------
        # Se actualizan los pesos y umbrales de todas las capas.
        # La actualizacion es proporcional a la tasa de aprendizaje, al delta
        # de la neurona destino, y a la activacion de la neurona origen.
        for l in range(num_capas - 1):
            # Actualizacion de pesos: W[l] += lr * a[l]^T * delta[l]
            # a[l]^T * delta[l] es el producto exterior que genera la matriz de ajustes
            pesos[l] += tasa_aprendizaje * activaciones[l].T @ deltas[l]

            # Actualizacion de umbrales: b[l] += lr * delta[l]
            # Los umbrales se ajustan directamente con el delta (su entrada es siempre 1)
            umbrales[l] += tasa_aprendizaje * deltas[l]

    # =========================================================================
    # PASO 6: Evaluacion del error total de entrenamiento
    # =========================================================================
    # Se calcula el error cuadratico medio (ECM) sobre todos los patrones:
    # E = (1/N) * sum(error_n) para n = 1, 2, ..., N
    # Este valor indica que tan bien la red aproxima las salidas deseadas.
    error_medio = error_total / N
    historial_errores.append(error_medio)

    # Imprimir progreso cada 1000 epocas
    if (epoca + 1) % 1000 == 0:
        print(f"Epoca {epoca + 1:>6d} | Error medio: {error_medio:.8f}")

    # =========================================================================
    # PASO 7: Criterio de parada
    # =========================================================================
    # Se verifica si el error de entrenamiento ha alcanzado el minimo deseado.
    # Si es asi, se detiene el entrenamiento. De lo contrario, se repiten
    # los pasos 2-6 en la siguiente epoca.
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
print("RESULTADOS DEL ENTRENAMIENTO")
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
# Se muestra como el error de entrenamiento disminuye a lo largo de las epocas,
# lo que permite verificar visualmente que la red esta aprendiendo.

plt.figure(figsize=(10, 6))
plt.plot(historial_errores, color='blue', linewidth=1.5)
plt.title('Convergencia del Error de Entrenamiento - MLP con Retropropagacion')
plt.xlabel('Epoca')
plt.ylabel('Error Cuadratico Medio')
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Escala logaritmica para apreciar mejor la convergencia
plt.tight_layout()
plt.show()
