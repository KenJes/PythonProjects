# PythonProjects

Repositorio personal de practicas en Python y Jupyter Notebook para aprender, experimentar y documentar implementaciones de algoritmos geneticos, redes neuronales, machine learning, seguridad informatica y utilidades pequenas.

El enfoque del repositorio es convertir teoria en codigo ejecutable, practicar desde cero cuando tiene sentido, y dejar evidencia clara del proceso de aprendizaje con scripts, notebooks y mini experimentos reutilizables.

## Proposito del repositorio

- Practicar fundamentos de IA, optimizacion y analisis de datos con implementaciones propias.
- Construir una base tecnica solida en Python, notebooks y visualizacion de resultados.
- Documentar mi avance de aprendizaje con ejercicios pequenos, medianos y proyectos exploratorios.
- Convertir practicas academicas y personales en portafolio tecnico publico.
- Reutilizar codigo y conceptos en proyectos mas grandes de IA aplicada, educacion y automatizacion.

## Que hay en este repositorio

### [algoritmos_geneticos](algoritmos_geneticos/)

Practicas y notebooks sobre algoritmos geneticos orientados a entender seleccion, cruza, mutacion, representacion binaria y optimizacion.

Incluye ejemplos como:
- [ag_general.ipynb](algoritmos_geneticos/ag_general.ipynb): algoritmo genetico general con hiperparametros ajustables.
- [ag_funcion_n_variables.py](algoritmos_geneticos/ag_funcion_n_variables.py): optimizacion genetica de una funcion con multiples variables.
- [ag_maximizar_funcion_xy.py](algoritmos_geneticos/ag_maximizar_funcion_xy.py): maximizacion de una funcion de dos variables con codificacion binaria.
- [ag_optimizacion_funcion_seno.py](algoritmos_geneticos/ag_optimizacion_funcion_seno.py): optimizacion de una funcion no lineal en una dimension.
- [ag_encontrar_numero_mayor_simple.py](algoritmos_geneticos/ag_encontrar_numero_mayor_simple.py) y [ag_encontrar_googol_elitista.py](algoritmos_geneticos/ag_encontrar_googol_elitista.py): ejercicios introductorios para entender operadores y evolucion generacional.

### [redes_neuronales](redes_neuronales/)

Implementaciones enfocadas en perceptrones, MLP, activaciones, compuertas logicas, clasificacion de digitos y comparativas de entrenamiento.

Incluye ejemplos como:
- [perceptron_and_basico.py](redes_neuronales/perceptron_and_basico.py)
- [perceptron_or_activacion_configurable.py](redes_neuronales/perceptron_or_activacion_configurable.py)
- [mlp_xor_retropropagacion.py](redes_neuronales/mlp_xor_retropropagacion.py)
- [clasificador_digitos_mlp.py](redes_neuronales/clasificador_digitos_mlp.py)
- [exploracion_digitos_y_mlp.ipynb](redes_neuronales/exploracion_digitos_y_mlp.ipynb)

### [machine_learning](machine_learning/)

Practicas de analisis de datos, carga de informacion, exploracion y pequenos tableros.

Incluye ejemplos como:
- [analisis_propinas_machine_learning.ipynb](machine_learning/analisis_propinas_machine_learning.ipynb)
- [ventas_videojuegos.ipynb](machine_learning/ventas_videojuegos.ipynb)
- [dashboard_ventas.py](machine_learning/dashboard_ventas.py): dashboard en Streamlit conectado a SQLite.

### [seguridad](seguridad/)

Practicas de delitos informaticos, ciberseguridad y criptografia introductoria.

Incluye ejemplos como:
- [catalogo_delitos_informaticos.py](seguridad/catalogo_delitos_informaticos.py)
- [diccionario_delitos.py](seguridad/diccionario_delitos.py)
- [practica_delitos_informaticos.ipynb](seguridad/practica_delitos_informaticos.ipynb)
- [introduccion_criptografia.ipynb](seguridad/introduccion_criptografia.ipynb)

### [convertidores](convertidores/)

Scripts utilitarios para conversion de formatos multimedia.

- [convertir_video_a_webm.py](convertidores/convertir_video_a_webm.py)
- [extraer_audio_webm_a_ogg.py](convertidores/extraer_audio_webm_a_ogg.py)

## Como recorrer el repositorio

- Si quieres ver material explicativo y proceso paso a paso, empieza por los notebooks.
- Si quieres ver implementaciones mas directas, revisa los scripts `.py`.
- Si quieres explorar proyectos con salida visual, revisa `machine_learning/` y `redes_neuronales/`.
- Si quieres seguir mi linea de aprendizaje en optimizacion evolutiva, empieza por `algoritmos_geneticos/`.

## Enfoque de trabajo

Este repositorio no busca ser una libreria empaquetada ni un framework. Su objetivo es servir como laboratorio tecnico personal para:

- probar ideas rapidamente,
- comparar tecnicas,
- documentar resultados,
- dejar practicas reutilizables,
- y construir una trayectoria visible de aprendizaje y mejora continua.

## Plan de implementaciones para aumentar contribuciones

La forma mas realista de aumentar contribuciones diarias no es hacer cambios enormes, sino dividir el trabajo en entregables pequenos y constantes. Este plan esta pensado para producir commits frecuentes con valor tecnico real.

### Regla base por contribucion

Cada contribucion diaria debe dejar al menos uno de estos entregables:

- un script nuevo,
- un notebook nuevo,
- una mejora visual o comparativa,
- una refactorizacion con mejor nombre o estructura,
- una mejora de README o documentacion,
- una grafica, tabla o resultado reproducible.

### Cadencia semanal sugerida

- Lunes: practica corta en `.py` con un concepto puntual.
- Martes: notebook explicativo con teoria, codigo y resultados.
- Miercoles: comparativa entre dos tecnicas o hiperparametros.
- Jueves: refactor, limpieza, renombrado o mejora de salidas.
- Viernes: mini proyecto aplicado o visualizacion.
- Sabado: documentacion, conclusiones y organizacion del repo.
- Domingo: commit ligero de seguimiento, backlog o ajuste pequeno.

### Backlog de practicas por bloques

#### Bloque 1: Algoritmos geneticos

- [ ] Implementar seleccion por torneo y compararla con ruleta.
- [ ] Agregar elitismo configurable al AG general.
- [ ] Crear notebook de comparacion entre cruce de un punto, dos puntos y uniforme.
- [ ] Graficar la evolucion del fitness maximo y promedio por generacion.
- [ ] Resolver una funcion con restricciones y operador de reparacion.
- [ ] Crear una practica de problema de la mochila con AG.
- [ ] Crear una practica de recorrido tipo TSP con representacion por permutaciones.

#### Bloque 2: Redes neuronales

- [ ] Implementar perceptron para NAND, NOR y XOR con analisis de por que XOR no converge en una sola capa.
- [ ] Crear notebook de descenso por gradiente para regresion logistica desde cero.
- [ ] Agregar visualizacion de frontera de decision para perceptrones.
- [ ] Comparar funciones de activacion en una tarea de clasificacion simple.
- [ ] Agregar matriz de confusion y metricas basicas a los clasificadores de digitos.
- [ ] Implementar early stopping en un MLP sencillo.
- [ ] Crear notebook de normalizacion y su efecto en el entrenamiento.

#### Bloque 3: Machine learning y analisis de datos

- [ ] Practica de regresion lineal desde cero con visualizacion.
- [ ] Practica de KNN para clasificacion con dataset pequeno.
- [ ] Practica de k-means con graficas de clusters.
- [ ] Notebook de PCA para reduccion de dimensionalidad.
- [ ] Mejorar el dashboard de ventas con filtros por genero, anio y publisher.
- [ ] Crear un notebook de limpieza de datos paso a paso antes del dashboard.
- [ ] Agregar exportacion de reportes o tablas resumen.

#### Bloque 4: Seguridad y criptografia

- [ ] Crear practica de cifrado Vigenere y compararlo con Cesar.
- [ ] Crear notebook de hashing, salting y almacenamiento seguro de contrasenas.
- [ ] Implementar una introduccion a RSA con numeros pequenos.
- [ ] Hacer una practica de analisis de frecuencia para romper cifrados clasicos.
- [ ] Crear un mini detector de phishing con reglas simples.
- [ ] Crear practica de validacion de contrasenas fuertes con puntuacion.
- [ ] Agregar una practica de inspeccion de certificados TLS y explicacion de hallazgos.

#### Bloque 5: Utilidades y portafolio tecnico

- [ ] Crear un notebook indice que enlace las mejores practicas del repositorio.
- [ ] Agregar capturas o resultados de salida a los proyectos mas importantes.
- [ ] Crear un script para ejecutar practicas por categoria.
- [ ] Agregar un pequeño menu CLI para lanzar demos del repositorio.
- [ ] Crear una pagina resumen futura con los proyectos mas representativos.

## Meta sugerida de contribucion

Si conviertes cada punto del backlog en entre 1 y 3 commits pequenos, este repositorio puede darte varias semanas de contribuciones consistentes sin forzar cambios artificiales.

Una meta razonable es:

- 1 contribucion diaria minima de lunes a viernes.
- 1 contribucion ligera de mantenimiento el fin de semana.
- 8 a 12 contribuciones tecnicas por quincena.
- 1 notebook o practica fuerte por semana.

## Siguiente criterio para cada practica nueva

Antes de subir una nueva practica, intenta que cumpla esto:

- explique el objetivo,
- tenga una salida clara,
- use nombres consistentes,
- deje al menos una conclusion o aprendizaje,
- y, cuando aplique, compare resultados entre variantes.

## Estado actual

El repositorio sigue creciendo como laboratorio de aprendizaje. La idea no es solo acumular archivos, sino construir una coleccion cada vez mejor organizada de practicas tecnicas que muestren progreso real.
