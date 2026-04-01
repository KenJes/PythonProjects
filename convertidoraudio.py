import sys
import subprocess
import os

def extraer_audio_de_webm(input_file):
    base, _ = os.path.splitext(input_file)
    output_file = base + ".ogg"

    command = [
        "ffmpeg",
        "-i", input_file,
        "-map", "0:a:0",       # fuerza a usar la primera pista de audio
        "-c:a", "libvorbis",   # convierte a Vorbis
        "-ar", "44100",
        "-b:a", "128k",
        output_file
    ]

    try:
        subprocess.run(command, check=True)
        print(f"✅ Audio extraído y convertido: {output_file}")
    except subprocess.CalledProcessError:
        print("❌ Error: No se pudo extraer la pista de audio. Verifica que el archivo tenga audio.")
    except FileNotFoundError:
        print("❌ Error: FFmpeg no está instalado o no está en el PATH.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: py convertidor_webm_audio.py archivo.webm")
    else:
        extraer_audio_de_webm(sys.argv[1])