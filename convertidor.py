import sys
import subprocess
import os

def convertir_a_webm(input_file):
    base, _ = os.path.splitext(input_file)
    output_file = base + ".webm"

    command = [
        "ffmpeg",
        "-i", input_file,
        "-c:v", "libvpx",      # VP8
        "-c:a", "libvorbis",   # Vorbis
        "-ar", "44100",        # Frecuencia estándar
        "-b:a", "128k",        # Bitrate estándar
        output_file
    ]

    try:
        subprocess.run(command, check=True)
        print(f"✅ Conversión completa: {output_file}")
    except FileNotFoundError:
        print("❌ Error: FFmpeg no está instalado o no está en el PATH.")
    except subprocess.CalledProcessError:
        print("❌ Error: Falló la conversión del archivo.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: py convertidor.py archivo.mp4")
    else:
        convertir_a_webm(sys.argv[1])