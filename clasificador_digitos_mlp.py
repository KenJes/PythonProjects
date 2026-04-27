import numpy as np
import os
import sys
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ==================== CONFIGURACIÓN ====================
MODELO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_digitos.npz")
HIDDEN_NEURONS = 64
LEARNING_RATE = 0.1
EPOCHS = 500
N_EPOCAS_PRINT = 50  # Mostrar progreso cada n épocas

# ==================== FUNCIONES DE LA RED NEURONAL ====================

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def one_hot(y, num_classes=10):
    """Convierte etiquetas enteras a vectores one-hot."""
    oh = np.zeros((len(y), num_classes))
    oh[np.arange(len(y)), y] = 1
    return oh

def entrenar_modelo():
    """Entrena la red neuronal MLP con el dataset de dígitos de sklearn."""
    print("=" * 50)
    print("ENTRENANDO MODELO DE RECONOCIMIENTO DE DÍGITOS")
    print("=" * 50)

    # Cargar el conjunto de datos de dígitos
    digits = load_digits()
    X, y = digits.data, digits.target

    # Preprocesamiento
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # One-hot encoding para las etiquetas
    y_train_oh = one_hot(y_train)

    # Inicialización de pesos
    input_neurons = X_train.shape[1]  # 64
    np.random.seed(42)
    weights_input_hidden = np.random.uniform(-0.5, 0.5, size=(input_neurons, HIDDEN_NEURONS))
    weights_hidden_output = np.random.uniform(-0.5, 0.5, size=(HIDDEN_NEURONS, 10))

    # Listas para almacenar métricas
    accuracy_list = []
    loss_list = []

    # Entrenamiento con backpropagation
    for epoch in range(EPOCHS):
        # Forward propagation
        hidden_input = np.dot(X_train, weights_input_hidden)
        hidden_output = sigmoid(hidden_input)
        output_input = np.dot(hidden_output, weights_hidden_output)
        output = sigmoid(output_input)

        # Métricas
        accuracy = accuracy_score(y_train, np.argmax(output, axis=1))
        loss = np.mean(np.square(y_train_oh - output))
        accuracy_list.append(accuracy)
        loss_list.append(loss)

        # Backpropagation
        output_error = y_train_oh - output
        output_delta = output_error * sigmoid_derivative(output)

        hidden_error = output_delta.dot(weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

        # Actualización de pesos
        weights_hidden_output += hidden_output.T.dot(output_delta) * LEARNING_RATE
        weights_input_hidden += X_train.T.dot(hidden_delta) * LEARNING_RATE

        if epoch % N_EPOCAS_PRINT == 0:
            print(f"Época {epoch:4d}/{EPOCHS} - Precisión: {accuracy:.4f} - Pérdida: {loss:.4f}")

    # Precisión final en entrenamiento
    print(f"Época {EPOCHS:4d}/{EPOCHS} - Precisión: {accuracy_list[-1]:.4f} - Pérdida: {loss_list[-1]:.4f}")

    # Predicción en el conjunto de prueba
    hidden_output = sigmoid(np.dot(X_test, weights_input_hidden))
    predicted_output = sigmoid(np.dot(hidden_output, weights_hidden_output))
    predicted_labels = np.argmax(predicted_output, axis=1)
    accuracy_test = accuracy_score(y_test, predicted_labels)
    print(f"\nPrecisión en el conjunto de prueba: {accuracy_test:.4f}")

    # Guardar modelo
    np.savez(
        MODELO_PATH,
        weights_input_hidden=weights_input_hidden,
        weights_hidden_output=weights_hidden_output,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
    )
    print(f"Modelo guardado en: {MODELO_PATH}")

    # Gráficas de entrenamiento
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(EPOCHS), accuracy_list, color='b')
    plt.title('Precisión durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(EPOCHS), loss_list, color='r')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return weights_input_hidden, weights_hidden_output, scaler.mean_, scaler.scale_


def cargar_modelo():
    """Carga los pesos y parámetros del scaler desde archivo."""
    data = np.load(MODELO_PATH)
    print(f"Modelo cargado desde: {MODELO_PATH}")
    return (
        data["weights_input_hidden"],
        data["weights_hidden_output"],
        data["scaler_mean"],
        data["scaler_scale"],
    )


def predecir(imagen_flat, w_ih, w_ho, scaler_mean, scaler_scale):
    """Realiza la predicción de un dígito dado un vector de 64 características."""
    # Aplicar el mismo escalado que en entrenamiento
    imagen_scaled = (imagen_flat - scaler_mean) / scaler_scale

    # Forward propagation
    hidden = sigmoid(np.dot(imagen_scaled, w_ih))
    output = sigmoid(np.dot(hidden, w_ho))

    return output  # Probabilidades para cada dígito 0-9


def preprocesar_imagen(ruta_imagen):
    """Carga una imagen y la convierte al formato 8x8 compatible con sklearn digits."""
    from PIL import Image

    # Cargar y convertir a escala de grises
    img = Image.open(ruta_imagen).convert('L')

    # Redimensionar a 8x8 con antialiasing
    img_resized = img.resize((8, 8), Image.LANCZOS)

    # Convertir a array numpy
    img_array = np.array(img_resized, dtype=np.float64)

    # sklearn digits: fondo oscuro (0), trazo claro (16)
    # Fotos normales: fondo claro (255), trazo oscuro (0)
    # Detectar automáticamente: si el promedio es alto, el fondo es claro → invertir
    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    # Normalizar al rango 0-16 (como sklearn digits)
    if img_array.max() > 0:
        img_array = (img_array / img_array.max()) * 16

    return img_array


def seleccionar_imagen():
    """Abre un diálogo para seleccionar una imagen."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()  # Ocultar ventana principal
    root.attributes('-topmost', True)

    ruta = filedialog.askopenfilename(
        title="Selecciona una imagen de un dígito",
        filetypes=[
            ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"),
            ("PNG", "*.png"),
            ("JPEG", "*.jpg *.jpeg"),
            ("BMP", "*.bmp"),
            ("Todos los archivos", "*.*"),
        ],
    )

    root.destroy()
    return ruta


def mostrar_prediccion(ruta_imagen, img_8x8, probabilidades, digito_predicho):
    """Muestra la imagen original, la versión 8x8 y las probabilidades."""
    from PIL import Image

    img_original = Image.open(ruta_imagen)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Imagen original
    axes[0].imshow(img_original, cmap='gray')
    axes[0].set_title("Imagen original")
    axes[0].axis('off')

    # Imagen preprocesada 8x8
    axes[1].imshow(img_8x8, cmap=plt.cm.gray_r, interpolation='nearest')
    axes[1].set_title(f"Preprocesada 8x8\nPredicción: {digito_predicho}")
    axes[1].axis('off')

    # Barras de confianza
    colores = ['gray'] * 10
    colores[digito_predicho] = 'green'
    axes[2].barh(range(10), probabilidades, color=colores)
    axes[2].set_yticks(range(10))
    axes[2].set_yticklabels([str(i) for i in range(10)])
    axes[2].set_xlabel('Confianza')
    axes[2].set_title('Probabilidad por dígito')
    axes[2].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()


# ==================== FLUJO PRINCIPAL ====================

def main():
    # Paso 1: Obtener o entrenar modelo
    if os.path.exists(MODELO_PATH):
        print("Modelo existente encontrado.")
        respuesta = input("¿Deseas re-entrenar el modelo? (s/n): ").strip().lower()
        if respuesta == 's':
            w_ih, w_ho, s_mean, s_scale = entrenar_modelo()
        else:
            w_ih, w_ho, s_mean, s_scale = cargar_modelo()
    else:
        print("No se encontró un modelo entrenado. Iniciando entrenamiento...")
        w_ih, w_ho, s_mean, s_scale = entrenar_modelo()

    # Paso 2: Ciclo de predicción
    while True:
        print("\n" + "=" * 50)
        print("RECONOCIMIENTO DE DÍGITOS")
        print("=" * 50)
        print("1. Seleccionar imagen para predecir")
        print("2. Re-entrenar modelo")
        print("3. Salir")
        opcion = input("Selecciona una opción: ").strip()

        if opcion == '1':
            ruta = seleccionar_imagen()
            if not ruta:
                print("No se seleccionó ninguna imagen.")
                continue

            print(f"Imagen seleccionada: {ruta}")

            # Preprocesar
            img_8x8 = preprocesar_imagen(ruta)
            img_flat = img_8x8.flatten().reshape(1, -1)

            # Predecir
            probabilidades = predecir(img_flat, w_ih, w_ho, s_mean, s_scale)[0]
            digito = np.argmax(probabilidades)

            print(f"\n>>> Dígito predicho: {digito}")
            print(f">>> Confianza: {probabilidades[digito]:.2%}")

            # Mostrar resultado visual
            mostrar_prediccion(ruta, img_8x8, probabilidades, digito)

        elif opcion == '2':
            w_ih, w_ho, s_mean, s_scale = entrenar_modelo()

        elif opcion == '3':
            print("¡Hasta luego!")
            break
        else:
            print("Opción no válida.")


if __name__ == "__main__":
    main()
