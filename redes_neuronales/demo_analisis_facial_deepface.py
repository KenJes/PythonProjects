"""
Ejecutar con el venv de Python 3.12:
    .venv312\\Scripts\\python.exe demo_analisis_facial_deepface.py

Métodos demostrados:
  1. analyze()       - Detecta edad, género, emoción y raza
  2. verify()        - Verifica si dos rostros son de la misma persona
  3. find()          - Busca un rostro en una base de datos de imágenes
  4. represent()     - Genera el embedding (vector) de un rostro
  5. extract_faces() - Detecta y recorta rostros de una imagen
  6. Tiempo real     - Análisis en vivo con la cámara web
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace

# ==================== CONFIGURACIÓN ====================
_BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_BASE, "rostros_db")
CAPTURAS_PATH = os.path.join(_BASE, "capturas")
os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(CAPTURAS_PATH, exist_ok=True)


# ==================== UTILIDADES ====================
def seleccionar_imagen_archivo(titulo="Selecciona una imagen"):
    """Abre un diálogo para seleccionar una imagen."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    ruta = filedialog.askopenfilename(
        title=titulo,
        filetypes=[
            ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.webp"),
            ("Todos", "*.*"),
        ],
    )
    root.destroy()
    return ruta


def seleccionar_carpeta(titulo="Selecciona una carpeta con rostros"):
    """Abre un diálogo para seleccionar una carpeta."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    ruta = filedialog.askdirectory(title=titulo)
    root.destroy()
    return ruta


def capturar_con_camara(titulo="Captura"):
    """Abre la webcam. ESPACIO para capturar, ESC para cancelar."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la cámara.")
        return None

    print("Cámara abierta — ESPACIO = capturar | ESC = cancelar")
    ruta = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(titulo, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == 32:  # ESPACIO
            nombre = f"captura_{len(os.listdir(CAPTURAS_PATH)) + 1}.jpg"
            ruta = os.path.join(CAPTURAS_PATH, nombre)
            cv2.imwrite(ruta, frame)
            print(f"Imagen guardada: {ruta}")
            break

    cap.release()
    cv2.destroyAllWindows()
    return ruta


def obtener_imagen(titulo="Selecciona una imagen"):
    """Pregunta al usuario si quiere usar cámara o archivo."""
    print(f"\n  [{titulo}]")
    print("  C = Cámara  |  A = Archivo")
    opcion = input("  Opción: ").strip().upper()
    if opcion == "C":
        return capturar_con_camara(titulo)
    return seleccionar_imagen_archivo(titulo)


# ==================== 1. ANALYZE ====================
def demo_analyze():
    """
    DeepFace.analyze() - Analiza atributos faciales:
      - Edad estimada
      - Género (Hombre/Mujer)
      - Emoción dominante (feliz, triste, enojado, sorpresa, miedo, disgusto, neutral)
      - Raza/etnia dominante
    """
    print("\n" + "=" * 60)
    print("1. DeepFace.analyze() - Análisis de atributos faciales")
    print("=" * 60)

    ruta = obtener_imagen("Imagen para ANALIZAR")
    if not ruta:
        print("No se seleccionó imagen.")
        return

    print(f"Analizando: {os.path.basename(ruta)}...")

    resultados = DeepFace.analyze(
        img_path=ruta,
        actions=["age", "gender", "emotion", "race"],
        enforce_detection=False,
    )

    for i, rostro in enumerate(resultados):
        print(f"\n--- Rostro {i + 1} ---")
        print(f"  Edad estimada: {rostro['age']} años")
        print(f"  Género: {rostro['dominant_gender']} ({rostro['gender'][rostro['dominant_gender']]:.1f}%)")
        print(f"  Emoción: {rostro['dominant_emotion']} ({rostro['emotion'][rostro['dominant_emotion']]:.1f}%)")
        print(f"  Raza: {rostro['dominant_race']} ({rostro['race'][rostro['dominant_race']]:.1f}%)")

        # Visualización
        img = cv2.imread(ruta)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        region = rostro["region"]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Imagen con bounding box
        cv2.rectangle(
            img_rgb,
            (region["x"], region["y"]),
            (region["x"] + region["w"], region["y"] + region["h"]),
            (0, 255, 0), 3,
        )
        axes[0].imshow(img_rgb)
        axes[0].set_title(f"Rostro {i + 1}: {rostro['dominant_gender']}, {rostro['age']} años", fontsize=11)
        axes[0].axis("off")

        # Gráfico de emociones
        emociones = rostro["emotion"]
        colores_emo = ["#e74c3c" if k == rostro["dominant_emotion"] else "#95a5a6" for k in emociones]
        axes[1].barh(list(emociones.keys()), list(emociones.values()), color=colores_emo)
        axes[1].set_xlabel("Porcentaje")
        axes[1].set_title("Emociones detectadas")
        axes[1].set_xlim(0, 100)

        # Gráfico de raza
        razas = rostro["race"]
        colores_raza = ["#3498db" if k == rostro["dominant_race"] else "#95a5a6" for k in razas]
        axes[2].barh(list(razas.keys()), list(razas.values()), color=colores_raza)
        axes[2].set_xlabel("Porcentaje")
        axes[2].set_title("Raza/Etnia detectada")
        axes[2].set_xlim(0, 100)

        plt.suptitle(f"DeepFace.analyze() - Rostro {i + 1}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()


# ==================== 2. VERIFY ====================
def demo_verify():
    """
    DeepFace.verify() - Compara dos rostros y determina si son la misma persona.
    Retorna: distancia, umbral y si es verificado (True/False).
    """
    print("\n" + "=" * 60)
    print("2. DeepFace.verify() - Verificación de identidad")
    print("=" * 60)

    ruta1 = obtener_imagen("Imagen 1 - Primera persona")
    if not ruta1:
        print("No se seleccionó imagen.")
        return

    ruta2 = obtener_imagen("Imagen 2 - Segunda persona")
    if not ruta2:
        print("No se seleccionó imagen.")
        return

    print(f"Comparando: {os.path.basename(ruta1)} vs {os.path.basename(ruta2)}...")

    resultado = DeepFace.verify(
        img1_path=ruta1,
        img2_path=ruta2,
        enforce_detection=False,
    )

    verificado = resultado["verified"]
    distancia = resultado["distance"]
    umbral = resultado["threshold"]
    modelo = resultado["model"]

    print(f"\n  Modelo usado: {modelo}")
    print(f"  Distancia: {distancia:.4f}")
    print(f"  Umbral: {umbral:.4f}")
    print(f"  ¿Misma persona? {'✅ SÍ' if verificado else '❌ NO'}")

    # Visualización
    img1 = cv2.cvtColor(cv2.imread(ruta1), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(ruta2), cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img1)
    axes[0].set_title(os.path.basename(ruta1), fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(img2)
    axes[1].set_title(os.path.basename(ruta2), fontsize=10)
    axes[1].axis("off")

    color = "green" if verificado else "red"
    texto = "✅ MISMA PERSONA" if verificado else "❌ DIFERENTE PERSONA"
    plt.suptitle(
        f"{texto}\nDistancia: {distancia:.4f} (umbral: {umbral:.4f})",
        fontsize=14, fontweight="bold", color=color,
    )
    plt.tight_layout()
    plt.show()


# ==================== 3. FIND ====================
def demo_find():
    """
    DeepFace.find() - Busca un rostro en una carpeta de imágenes (base de datos).
    Retorna las coincidencias ordenadas por similitud.
    """
    print("\n" + "=" * 60)
    print("3. DeepFace.find() - Búsqueda en base de datos de rostros")
    print("=" * 60)

    ruta_query = obtener_imagen("Imagen del rostro a BUSCAR")
    if not ruta_query:
        print("No se seleccionó imagen.")
        return

    print("Selecciona la CARPETA con imágenes de rostros (base de datos)...")
    carpeta_db = seleccionar_carpeta("Carpeta con rostros")
    if not carpeta_db:
        print("No se seleccionó carpeta.")
        return

    print(f"Buscando {os.path.basename(ruta_query)} en {carpeta_db}...")

    resultados = DeepFace.find(
        img_path=ruta_query,
        db_path=carpeta_db,
        enforce_detection=False,
        silent=True,
    )

    if len(resultados) == 0 or resultados[0].empty:
        print("No se encontraron coincidencias.")
        return

    df = resultados[0]
    print(f"\nSe encontraron {len(df)} coincidencia(s):")
    print(df[["identity", "distance"]].to_string(index=False))

    # Mostrar top resultados
    n_mostrar = min(5, len(df))
    fig, axes = plt.subplots(1, n_mostrar + 1, figsize=(4 * (n_mostrar + 1), 4))
    if n_mostrar + 1 == 1:
        axes = [axes]

    # Imagen query
    img_q = cv2.cvtColor(cv2.imread(ruta_query), cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_q)
    axes[0].set_title("BUSCAR", fontsize=11, fontweight="bold", color="blue")
    axes[0].axis("off")

    # Resultados
    for idx in range(n_mostrar):
        ruta_match = df.iloc[idx]["identity"]
        dist = df.iloc[idx]["distance"]
        img_m = cv2.cvtColor(cv2.imread(ruta_match), cv2.COLOR_BGR2RGB)
        axes[idx + 1].imshow(img_m)
        axes[idx + 1].set_title(f"#{idx + 1} dist={dist:.3f}\n{os.path.basename(ruta_match)}", fontsize=9)
        axes[idx + 1].axis("off")

    plt.suptitle("DeepFace.find() - Resultados", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ==================== 4. REPRESENT ====================
def demo_represent():
    """
    DeepFace.represent() - Genera el vector de embedding de un rostro.
    Útil para almacenar y comparar rostros de forma eficiente.
    """
    print("\n" + "=" * 60)
    print("4. DeepFace.represent() - Embedding vectorial del rostro")
    print("=" * 60)

    ruta = obtener_imagen("Imagen para obtener su EMBEDDING")
    if not ruta:
        print("No se seleccionó imagen.")
        return

    print(f"Generando embedding de: {os.path.basename(ruta)}...")

    resultados = DeepFace.represent(
        img_path=ruta,
        enforce_detection=False,
    )

    for i, rostro in enumerate(resultados):
        embedding = rostro["embedding"]
        modelo = rostro["model"]

        print(f"\n--- Rostro {i + 1} ---")
        print(f"  Modelo: {modelo}")
        print(f"  Dimensiones del vector: {len(embedding)}")
        print(f"  Primeros 10 valores: {[round(v, 4) for v in embedding[:10]]}")
        print(f"  Norma del vector: {np.linalg.norm(embedding):.4f}")

        # Visualización
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        img = cv2.cvtColor(cv2.imread(ruta), cv2.COLOR_BGR2RGB)
        axes[0].imshow(img)
        axes[0].set_title(f"Rostro {i + 1}", fontsize=11)
        axes[0].axis("off")

        axes[1].plot(embedding, color="#3498db", linewidth=0.5)
        axes[1].set_title(f"Embedding ({len(embedding)} dimensiones)", fontsize=11)
        axes[1].set_xlabel("Dimensión")
        axes[1].set_ylabel("Valor")
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f"DeepFace.represent() - Modelo: {modelo}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()


# ==================== 5. EXTRACT FACES ====================
def demo_extract_faces():
    """
    DeepFace.extract_faces() - Detecta y recorta todos los rostros de una imagen.
    """
    print("\n" + "=" * 60)
    print("5. DeepFace.extract_faces() - Detección y recorte de rostros")
    print("=" * 60)

    ruta = obtener_imagen("Imagen para EXTRAER ROSTROS")
    if not ruta:
        print("No se seleccionó imagen.")
        return

    print(f"Extrayendo rostros de: {os.path.basename(ruta)}...")

    rostros = DeepFace.extract_faces(
        img_path=ruta,
        enforce_detection=False,
    )

    print(f"Se detectaron {len(rostros)} rostro(s).")

    # Visualización
    n = len(rostros)
    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))
    if n + 1 == 1:
        axes = [axes]

    # Imagen original con bounding boxes
    img = cv2.cvtColor(cv2.imread(ruta), cv2.COLOR_BGR2RGB)
    img_boxes = img.copy()
    for rostro in rostros:
        r = rostro["facial_area"]
        cv2.rectangle(img_boxes, (r["x"], r["y"]), (r["x"] + r["w"], r["y"] + r["h"]), (0, 255, 0), 3)
    axes[0].imshow(img_boxes)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    # Rostros recortados
    for i, rostro in enumerate(rostros):
        face_img = rostro["face"]
        confianza = rostro["confidence"]
        axes[i + 1].imshow(face_img)
        axes[i + 1].set_title(f"Rostro {i + 1}\nConf: {confianza:.2f}", fontsize=10)
        axes[i + 1].axis("off")
        print(f"  Rostro {i + 1}: confianza={confianza:.4f}, área={rostro['facial_area']}")

    plt.suptitle(f"DeepFace.extract_faces() - {n} rostro(s) detectado(s)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ==================== 6. TIEMPO REAL ====================
def demo_realtime():
    """
    Análisis facial en tiempo real con la webcam.
    Detecta edad, género y emoción y los muestra sobre el video.
    Analiza cada 30 frames para mantener fluidez.
    ESC para salir.
    """
    print("\n" + "=" * 60)
    print("6. Tiempo real - Análisis en vivo con la cámara")
    print("=" * 60)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la cámara.")
        return

    print("Cámara abierta — ESC para salir")
    frame_count = 0
    info_text = "Analizando..."
    region = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Analizar cada 30 frames
        if frame_count % 30 == 1:
            try:
                resultados = DeepFace.analyze(
                    img_path=frame,
                    actions=["age", "gender", "emotion"],
                    enforce_detection=False,
                    silent=True,
                )
                r = resultados[0]
                edad = r["age"]
                genero = r["dominant_gender"]
                emocion = r["dominant_emotion"]
                region = r["region"]
                info_text = f"{genero}, {edad} anios, {emocion}"
            except Exception:
                info_text = "Sin deteccion"
                region = None

        # Dibujar bounding box
        if region and region["w"] > 0:
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, info_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("DeepFace - Tiempo Real (ESC para salir)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Cámara cerrada.")


# ==================== MENÚ PRINCIPAL ====================
def main():
    while True:
        print("\n" + "=" * 60)
        print("        DeepFace - Demostración de la API")
        print("=" * 60)
        print("  1. analyze()      → Edad, género, emoción, raza")
        print("  2. verify()       → ¿Son la misma persona?")
        print("  3. find()         → Buscar rostro en carpeta")
        print("  4. represent()    → Generar embedding vectorial")
        print("  5. extract_faces()→ Detectar y recortar rostros")
        print("  6. Tiempo real    → Análisis en vivo con cámara")
        print("  7. Ejecutar TODOS los demos")
        print("  0. Salir")
        print("-" * 60)

        opcion = input("Selecciona una opción: ").strip()

        if opcion == "1":
            demo_analyze()
        elif opcion == "2":
            demo_verify()
        elif opcion == "3":
            demo_find()
        elif opcion == "4":
            demo_represent()
        elif opcion == "5":
            demo_extract_faces()
        elif opcion == "6":
            demo_realtime()
        elif opcion == "7":
            demo_analyze()
            demo_verify()
            demo_find()
            demo_represent()
            demo_extract_faces()
            demo_realtime()
        elif opcion == "0":
            print("¡Hasta luego!")
            break
        else:
            print("Opción no válida.")


if __name__ == "__main__":
    main()
