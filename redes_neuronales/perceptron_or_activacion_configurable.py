import numpy as np

# ── Datos: compuerta AND ─────────────────────────────────────────────────────
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0, 1, 1, 1])

# ── Parámetros ajustables ────────────────────────────────────────────────────
epocas             = 100
taza_aprendizaje   = -0.1       # Tasa de aprendizaje (0.1 es común, pero puede ajustarse)
umbral             = 0.5       # Umbral de clasificación (0.5 para sigmoid/linear, 0 para step)
funcion_activacion = 'sigmoid'  # 'step' | 'sign' | 'sigmoid' | 'linear'

# ── Sesgo y pesos iniciales ──────────────────────────────────────────────────
xb = np.hstack([X, np.ones((len(X), 1))])  # Columna de sesgo (bias)
np.random.seed(42)
w  = np.random.uniform(-0.5, 0.5, size=xb.shape[1])

# ── Funciones de activación ──────────────────────────────────────────────────
# step:    binaria  0/1   → 1 si z >= umbral, 0 si no
# sign:    bipolar -1/+1  → 1 si z >= 0,      -1 si no  (distinta a step)
# sigmoid: continua (0,1) → se binariza con umbral al clasificar
# linear:  continua z     → se binariza con umbral al clasificar
def step(z):    return 1 if z >= umbral else 0
def sign(z):    return 1 if z >= 0 else -1
def sigmoid(z): return 1 / (1 + np.exp(-z))
def linear(z):  return z

_fn = {'step': step, 'sign': sign, 'sigmoid': sigmoid, 'linear': linear}

def activar(z):
    return _fn[funcion_activacion](z)

def clasificar(yout):
    """Convierte cualquier salida a 0/1 para comparar con las etiquetas."""
    if funcion_activacion == 'sign':
        return 1 if yout > 0 else 0   # +1 → 1,  -1 → 0
    return 1 if yout >= umbral else 0  # step / sigmoid / linear

# ── Métricas ─────────────────────────────────────────────────────────────────
def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else (1.0 if ss_res == 0 else 0.0)

# ── Entrenamiento ─────────────────────────────────────────────────────────────
print(f"Función: '{funcion_activacion}'  |  Umbral: {umbral} | Tasa de aprendizaje: {taza_aprendizaje}\n")

for epoch in range(epocas):
    errores = 0
    for xi, yi in zip(xb, y):
        yout = activar(np.dot(xi, w))
        pred = clasificar(yout)
        if pred != yi:
            w += taza_aprendizaje * (yi - yout) * xi
            errores += 1
    print(f"Época {epoch+1}/{epocas} | Pesos: {w} | Errores: {errores}")
    if errores == 0:
        print(f"\nConvergencia en época {epoch +1}")
        break

# ── Evaluación final ──────────────────────────────────────────────────────────
y_raw = np.array([activar(np.dot(xi, w)) for xi in xb])
y_bin = np.array([clasificar(v) for v in y_raw])

print(f"\n── Resultados [función='{funcion_activacion}', umbral={umbral}] ──")
print(f"MSE: {mse(y, y_bin):.4f}  |  R²: {r2_score(y, y_bin):.4f}\n")
for xi, yi, raw, pred in zip(X, y, y_raw, y_bin):
    print(f"  {xi} → raw={raw:.4f}  pred={pred}  esperado={yi}")
