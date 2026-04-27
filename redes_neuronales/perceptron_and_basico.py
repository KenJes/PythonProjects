import numpy as np
# 1.Definir conjunto de ejemplos X y Etiquetas y

X = np.array([
    [0, 0],
    [0, 1], 
    [1, 0], 
    [1, 1]])
y = np.array([0, 0, 0, 1])

# 2. Se definen las epocas y la tasa de aprendizaje
epochs = 10
lr = 0.1
umbral = 0.5
# 3. Se agrega el cesgo 

Xb = np.hstack([X, np.ones((X.shape[0], 1))])                  #El cesgo se agrega como una columna adicional de unos al conjunto de características X, resultando en Xb con forma (4, 3) donde la última columna representa el sesgo.

# 4. Se inicializan los pesos

np.random.seed(42)
w = np.random.uniform(-0.5, 0.5, size=(Xb.shape[1],))          #Los pesos se inicializan aleatoriamente entre -0.5 y 0.5, con una dimensión igual al número de características más el sesgo (3 en este caso)

# 5. Se define la función de activación escalón

def step(z):
    return 1 if z >= umbral else 0

def sign(z):
    return 1 if z >= 0 else -1

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def linear(z):
    return z

# 6. Se entrena el perceptrón
for epoch in range(epochs):                                     # Para cada época se itera sobre cada ejemplo
    errors = 0                                                  # Se inicializa el contador de errores
    for xi, yi in zip(Xb, y):                                   # Para cada ejemplo i = 1, 2, 3, 4 se obtiene el vector de características xi y la etiqueta yi
        z = np.dot(xi, w)                                       # Calcular potencial de activación Z = w1*x1 + w2*x2 + b
        yout = step(z)                                          # Calcular salida del perceptrón yout = f(z)
        delta = yi - yout                                       # Calcular error delta = yi - yout
        if delta != 0:                                          # Si el error es diferente de cero, se actualizan los pesos
            w += lr * delta * xi                                # Se actualizan los pesos w = w + lr * delta * xi
            errors += 1                                         # Se incrementa el contador de errores                                                           
    print(f"Epoch {epoch+1}/{epochs}, Pesos: {w}, Errores: {errors}")   # Se muestra el número de época, los pesos actualizados y el número de errores cometidos en esa época
    if errors == 0:                                              # Si no hubo errores, se detiene el entrenamiento
        print(f"\nConvergencia alcanzada en epoch {epoch+1}")
        break
