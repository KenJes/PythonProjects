# ============================================================
# DICCIONARIO DE DELITOS INFORMÁTICOS
# Cada clave es el nombre del delito.
# Cada valor es otro diccionario con: descripción, ejemplo y consecuencia.
# ============================================================

delitos_informaticos = {
    "acceso_no_autorizado": {
        "descripcion": "Entrar a un sistema, red o computadora sin permiso del dueño.",
        "ejemplo": "Adivinar la contraseña del correo de alguien para leer sus mensajes.",
        "consecuencia": "Hasta 2 años de prisión y multa (Art. 211 bis, CPF México)."
    },
    "phishing": {
        "descripcion": "Engañar a una persona haciéndose pasar por una entidad confiable para robar datos.",
        "ejemplo": "Un correo falso del banco que pide ingresar tu usuario y contraseña.",
        "consecuencia": "Fraude electrónico: hasta 9 años de prisión según el monto defraudado."
    },
    "fraude_electronico": {
        "descripcion": "Obtener dinero o beneficios de forma ilegal usando medios digitales.",
        "ejemplo": "Crear una tienda en línea falsa para cobrar productos que nunca se envían.",
        "consecuencia": "De 3 meses a 12 años de prisión según el valor del fraude (Art. 386, CPF)."
    },
    "malware": {
        "descripcion": "Software malicioso diseñado para dañar, espiar o tomar control de un sistema.",
        "ejemplo": "Un virus que cifra todos los archivos de una empresa y pide dinero para liberarlos (ransomware).",
        "consecuencia": "Daño informático: hasta 8 años de prisión (Art. 211 bis 1, CPF México)."
    },
    "robo_de_identidad": {
        "descripcion": "Usar los datos personales de otra persona sin su consentimiento.",
        "ejemplo": "Solicitar un crédito bancario usando el nombre, CURP y RFC de otra persona.",
        "consecuencia": "Hasta 5 años de prisión (Ley Federal de Protección de Datos Personales)."
    },
    "phishing_dedicado":{
        "descripcion": "Ataque de phishing más sofisticado.",
        "ejemplo": "Utilizar redes sociales para vigilar a una persona para saber sus gustos, datos basicos y algo que pueda vulnerar un acceso a un sistema",
        "consecuencia": "Hasta 10 años de prisión (Codigo Penal Federal)."
    }
}

# ============================================================
# Ahora recorremos el diccionario para mostrar la información
# de forma ordenada. Usamos un ciclo 'for' sobre los elementos.
# ============================================================

print("=" * 60)
print("       CATÁLOGO DE DELITOS INFORMÁTICOS")
print("=" * 60)

# enumerate() nos da un número (i) y el par (clave, valor) al mismo tiempo
for i, (clave, info) in enumerate(delitos_informaticos.items(), start=1):
    # Reemplazamos los guiones bajos por espacios y ponemos en mayúsculas
    nombre_legible = clave.replace("_", " ").upper()
    print(f"\n{i}. {nombre_legible}")
    print(f"   📌 Descripción : {info['descripcion']}")
    print(f"   🔍 Ejemplo     : {info['ejemplo']}")
    print(f"   ⚖️  Consecuencia: {info['consecuencia']}")

print("\n" + "=" * 60)