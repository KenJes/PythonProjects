import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
#  PERCEPTRÓN SIMPLE CON SESGO - DEMOSTRACIÓN VISUAL
#  Compara AND, OR y XOR con animación del entrenamiento
# ============================================================

# 1. Definir los tres problemas lógicos
problemas = {
    "AND": np.array([0, 0, 0, 1]),
    "OR":  np.array([0, 1, 1, 1]),
    "XOR": np.array([0, 1, 1, 0]),
}

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Xb = np.hstack([X, np.ones((X.shape[0], 1))])  # Agregar sesgo

# 2. Parámetros
epochs = 20
lr = 0.1

# Función de activación escalón
def step(z):
    return 1 if z >= 0 else 0

# 3. Entrenar perceptrón y guardar historial de pesos y errores
def entrenar(Xb, y, epochs, lr):
    np.random.seed(42)
    w = np.random.uniform(-0.5, 0.5, size=(Xb.shape[1],))
    historial_pesos = [w.copy()]
    historial_errores = []

    for epoch in range(epochs):
        errores = 0
        for xi, yi in zip(Xb, y):
            z = np.dot(xi, w)
            yout = step(z)
            delta = yi - yout
            if delta != 0:
                w += lr * delta * xi
                errores += 1
        historial_pesos.append(w.copy())
        historial_errores.append(errores)

    return w, historial_pesos, historial_errores

# 4. Entrenar cada problema
resultados = {}
for nombre, y in problemas.items():
    w, hist_w, hist_err = entrenar(Xb, y, epochs, lr)
    resultados[nombre] = {
        "pesos": w,
        "historial_pesos": hist_w,
        "historial_errores": hist_err,
        "etiquetas": y,
    }

# ============================================================
#  GRÁFICO ESTÁTICO: Resultado final de los tres problemas
# ============================================================

fig_estatico, axes = plt.subplots(1, 3, figsize=(15, 5))
fig_estatico.suptitle("Perceptrón Simple: Comparación AND vs OR vs XOR", fontsize=14, fontweight="bold")

for ax, (nombre, res) in zip(axes, resultados.items()):
    y = res["etiquetas"]
    w = res["pesos"]
    converge = res["historial_errores"][-1] == 0

    # Puntos
    ax.scatter(X[y == 0, 0], X[y == 0, 1], color="red", s=150, zorder=5, label="Clase 0", edgecolors="black")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", s=150, zorder=5, label="Clase 1", edgecolors="black")

    # Línea de decisión: w1*x1 + w2*x2 + b = 0
    x_line = np.linspace(-0.5, 1.5, 200)
    if abs(w[1]) > 1e-6:
        y_line = -(w[0] * x_line + w[2]) / w[1]
        ax.plot(x_line, y_line, "g--", linewidth=2, label="Frontera de decisión")

    estado = "✅ Converge" if converge else "❌ No converge"
    ax.set_title(f"{nombre}\n{estado}", fontsize=13)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

plt.tight_layout()

# ============================================================
#  GRÁFICO DE ERRORES POR ÉPOCA
# ============================================================

fig_errores, ax_err = plt.subplots(figsize=(8, 4))
fig_errores.suptitle("Convergencia del Perceptrón por Época", fontsize=14, fontweight="bold")

colores = {"AND": "green", "OR": "blue", "XOR": "red"}
for nombre, res in resultados.items():
    ax_err.plot(range(1, epochs + 1), res["historial_errores"],
                marker="o", label=nombre, color=colores[nombre], linewidth=2)

ax_err.set_xlabel("Época")
ax_err.set_ylabel("Número de errores")
ax_err.legend()
ax_err.grid(True, alpha=0.3)
ax_err.set_xticks(range(1, epochs + 1))
plt.tight_layout()

# ============================================================
#  ANIMACIÓN: Evolución de la frontera de decisión por época
# ============================================================

fig_anim, axes_anim = plt.subplots(1, 3, figsize=(15, 5))
fig_anim.suptitle("Animación del Entrenamiento del Perceptrón", fontsize=14, fontweight="bold")

lineas = []
titulos = []
x_line = np.linspace(-0.5, 1.5, 200)

for ax, (nombre, res) in zip(axes_anim, resultados.items()):
    y = res["etiquetas"]
    ax.scatter(X[y == 0, 0], X[y == 0, 1], color="red", s=150, zorder=5, label="Clase 0", edgecolors="black")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", s=150, zorder=5, label="Clase 1", edgecolors="black")
    linea, = ax.plot([], [], "g--", linewidth=2, label="Frontera")
    lineas.append(linea)
    titulo = ax.set_title(f"{nombre} - Época 0", fontsize=13)
    titulos.append(titulo)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

nombres_lista = list(resultados.keys())

def actualizar(frame):
    for i, nombre in enumerate(nombres_lista):
        w = resultados[nombre]["historial_pesos"][frame]
        errores = resultados[nombre]["historial_errores"][min(frame, len(resultados[nombre]["historial_errores"]) - 1)]
        if abs(w[1]) > 1e-6:
            y_line = -(w[0] * x_line + w[2]) / w[1]
            lineas[i].set_data(x_line, y_line)
        else:
            lineas[i].set_data([], [])
        titulos[i].set_text(f"{nombre} - Época {frame} | Errores: {errores}")
    return lineas + titulos

anim = FuncAnimation(fig_anim, actualizar, frames=epochs + 1, interval=600, repeat=True, blit=False)

plt.tight_layout()

# ============================================================
#  TABLA DE RESULTADOS EN CONSOLA
# ============================================================

print("=" * 55)
print("  RESULTADOS DEL PERCEPTRÓN SIMPLE")
print("=" * 55)
for nombre, res in resultados.items():
    converge = res["historial_errores"][-1] == 0
    estado = "CONVERGE" if converge else "NO CONVERGE"
    print(f"\n  {nombre}:")
    print(f"    Estado:  {estado}")
    print(f"    Pesos:   w1={res['pesos'][0]:.3f}, w2={res['pesos'][1]:.3f}, b={res['pesos'][2]:.3f}")
    print(f"    Errores última época: {res['historial_errores'][-1]}")
    print(f"    Predicciones:")
    for xi, yi in zip(X, res["etiquetas"]):
        xb = np.append(xi, 1)
        pred = step(np.dot(xb, res["pesos"]))
        marca = "✓" if pred == yi else "✗"
        print(f"      {xi} → esperado: {yi}, predicho: {pred}  {marca}")
print("=" * 55)

plt.show()
