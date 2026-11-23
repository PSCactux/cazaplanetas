# Hito 2 · Astronet reentrenado y Astronet de bolsillo

Reproducimos Astronet (Shallue y Vanderburg, 2018) con las vistas de Kepler DR24 que
publicaron sus autores y entrenamos, con los mismos datos y la misma partición, el
modelo pequeño que propone el [plan](plan.html#modelo). Todo en un Mac con M3 Max,
PyTorch y la GPU de Apple (MPS).

**Resultado:** el Astronet reentrenado da AUC 0,9883 ± 0,0007 en la partición de prueba
(artículo: 0,988) y el Astronet de bolsillo da 0,9858 ± 0,0009 con **275 veces menos
parámetros** (36.161 frente a 9.940.193) y entrenando 3,5 veces más rápido. Los dos
criterios del hito (≥ 0,983 y ≥ 0,975 con ≤ 1/50 de los parámetros) se cumplen. La
distancia entre los dos, 0,0025 de AUC, cae dentro del margen de 0,005 que el plan
da por éxito frente a Astronet.

## Cómo se reproduce

```bash
uv sync
uv run python scripts/hito2_astronet.py          # unos 22 min en un M3 Max
uv run python scripts/hito2_astronet.py --quick  # prueba de humo: 1 semilla, 1 época, no toca docs/
```

El guion baja los 10 ficheros TFRecord de la carpeta de Google Drive enlazada en el
README de [exoplanet-ml](https://github.com/google-research/exoplanet-ml) (unos 145 MB;
el plan decía ~106 MB), comprueba su SHA-256, los lee sin TensorFlow y los guarda como un
solo `.npz` en `data/astronet/` (fuera de git). Después entrena cada modelo con las
semillas 0 a 4, mide tiempos y regenera las tablas de este documento (entre las marcas
`tabla:inicio` y `tabla:fin`) y `docs/hito2_results.json`, que guarda todas las cifras por
semilla, la configuración y las sumas de comprobación de los datos. La semilla fija la
inicialización, el orden de los lotes y los volteos: dos ejecuciones con la semilla 0
dieron el mismo AUC (0,9871).

## Datos

| Partición | Señales (TCE) | Planetas candidatos (PC) | Falsos positivos (AFP + NTP) |
|---|---|---|---|
| Entrenamiento | 12.589 | 2.885 | 9.704 |
| Validación | 1.574 | 355 | 1.219 |
| Prueba | 1.574 | 360 | 1.214 |

En total, las 15.737 señales con etiqueta de DR24. Cada señal trae la vista global (la
órbita entera plegada, 2.001 tramos) y la local (el tránsito ampliado, 201 tramos), ya
normalizadas por los autores. Etiqueta 1 = PC; 0 = AFP o NTP, como en Astronet.

## Resultados en la partición de prueba

<!-- tabla:inicio -->
Generado por `uv run python scripts/hito2_astronet.py` el 2026-11-19 (mps, PyTorch 2.15.0). Media ± desviación típica de 5 semillas; «media de 5» es el promedio de las probabilidades de los 5 modelos. Prueba: 1.574 señales, 360 planetas candidatos. Precisión y recall con umbral 0,5.

| Modelo | AUC | Precisión | Recall | Exactitud | AUC-PR | Parámetros | Entrenamiento | Inferencia por curva (MPS, lotes de 1024) | Inferencia por curva (CPU, 1 hilo, de una en una) |
|---|---|---|---|---|---|---|---|---|---|
| Astronet reentrenado | 0,9883 ± 0,0007 | 0,892 ± 0,018 | 0,947 ± 0,034 | 0,962 ± 0,004 | 0,946 ± 0,003 | 9.940.193 | 197 ± 5 s | 69,5 µs | 2,04 ms |
| Astronet de bolsillo | 0,9858 ± 0,0009 | 0,895 ± 0,012 | 0,902 ± 0,021 | 0,953 ± 0,003 | 0,937 ± 0,010 | 36.161 | 57 ± 3 s | 42,1 µs | 1,58 ms |
| Astronet reentrenado, media de 5 | 0,9886 | 0,898 | 0,956 | 0,965 | 0,947 | 49.700.965 | — | — | — |
| Astronet de bolsillo, media de 5 | 0,9874 | 0,914 | 0,914 | 0,961 | 0,945 | 180.805 | — | — | — |
| Astronet publicado: artículo, media de 10 | 0,988 | — | — | 0,960 | — | — | 90 min en una CPU (por modelo) | — | — |
| Astronet publicado: README, un modelo | 0,9889 | 0,877 | 0,972 | 0,963 | — | — | — | — | — |

Coste relativo: el modelo de bolsillo tiene **275 veces menos parámetros** (141 kB de pesos frente a 37,9 MB), 14 veces menos operaciones por curva (9,0 frente a 126,3 millones de multiplicaciones-sumas) y entrena 3,5 veces más rápido.

Solo las 690 señales de prueba cuya estrella no tiene ninguna señal en entrenamiento:

| Modelo | AUC | Precisión | Recall | AUC-PR |
|---|---|---|---|---|
| Astronet reentrenado | 0,9685 ± 0,0022 | 0,851 ± 0,021 | 0,929 ± 0,043 | 0,909 ± 0,005 |
| Astronet de bolsillo | 0,9653 ± 0,0014 | 0,862 ± 0,014 | 0,870 ± 0,030 | 0,903 ± 0,010 |
<!-- tabla:fin -->

Notas sobre las columnas:

* **Entrenamiento:** segundos de reloj por modelo en MPS, incluida la evaluación en
  validación al final de cada época. Bajar y convertir los datos lleva unos 30 s más, una
  sola vez.
* **Inferencia por curva:** mejor de 3 pasadas sobre las 1.574 curvas de prueba tras
  calentar, en lotes de 1024 en MPS; y de una en una en la CPU con un solo hilo
  (200 curvas). Solo el modelo: las vistas ya están hechas.
* **Exhaustividad con precisión 0,99** (en `hito2_results.json`): ~5 % para los dos. En
  este conjunto ninguno de los dos llega a una lista casi pura de planetas; es la métrica
  de ExoMiner y la retomaremos en DR25 (hito 4).

## Comparación con Astronet publicado

Lo reproducimos: la media de 5 modelos da AUC 0,9886 y exactitud 0,965, frente a 0,988 y
0,960 en el artículo (media de 10 modelos) y 0,9889 y 0,963 en el ejemplo del README de
exoplanet-ml (un modelo). Las diferencias que hay, y por qué:

1. **Tamaño de la prueba.** El artículo habla de 1.523 señales de prueba; los ficheros
   publicados traen 1.574, las mismas que la salida de ejemplo del README (350 + 10 planetas
   = 360). Usamos los ficheros publicados tal cual; no sabemos qué 51 señales cambian.
2. **Precisión y recall dependen del umbral y varían mucho entre semillas.** Con umbral 0,5,
   nuestras 5 semillas de Astronet van de 0,865 a 0,914 en precisión y de 0,889 a 0,975 en
   recall; el modelo del README (0,877 y 0,972) cae dentro de ese abanico. El AUC, que no
   depende del umbral, apenas varía (± 0,0007).
3. **Tiempo de entrenamiento.** El artículo da 90 min por modelo en una CPU de 2017 con
   TensorFlow; aquí son 197 s en la GPU de un M3 Max. No son comparables: por eso el plan
   reentrena Astronet aquí y mide los dos modelos en la misma máquina.
4. **Implementación.** Misma arquitectura y receta que el artículo (sección 5.2) y los
   ficheros de configuración de exoplanet-ml: columna global de 5 bloques (16 a 256 filtros,
   núcleo 5, max-pool 5 con paso 2), columna local de 2 bloques (16 y 32 filtros, max-pool 7),
   4 capas densas de 512 y salida sigmoide; Adam con tasa 10⁻⁵, lotes de 64, 50 épocas, sin
   dropout, volteo temporal aleatorio con probabilidad 0,5 y los pesos de la última época.
   Está escrita en PyTorch desde esa descripción (no se ha copiado código). El recuento exacto
   es de 9.940.193 parámetros, el 85 % en la primera capa densa.

## El Astronet de bolsillo

`PocketAstronet` en `src/cazaplanetas/models.py`:

* Una rama por vista: convolución inicial de 8 canales y bloques de dos convoluciones
  separables (por canal y luego mezcla 1×1, con BatchNorm y ReLU) seguidos de max-pool con
  paso 2. Global: 16 → 32 → 64 → 64 canales; local: 16 → 32 → 64.
* Al final de cada rama, media y máximo sobre toda la longitud (en vez de aplanar y meter
  capas densas enormes, que es donde Astronet tiene casi todos sus parámetros).
* Cabeza: dropout 0,1, una capa de 32 neuronas y la salida.
* AdamW (tasa 10⁻³, decaimiento 10⁻⁴) con tasa en coseno, lotes de 64, 20 épocas, el mismo
  volteo temporal que Astronet, y se queda con la época de mejor AUC en validación (salió
  entre la 9 y la 15).
* La configuración se eligió mirando solo la validación (3 variantes, todas en torno a
  0,981 de AUC). La única vez que se calculó la prueba antes de la ejecución final fue en una
  prueba de humo de 2 épocas, que no sirvió para decidir nada.

**Dónde queda.** A 0,0025 de AUC de Astronet con 1/275 de los parámetros. Con el mismo
umbral 0,5 tiene la misma precisión (0,895) pero menos recall (0,902 frente a 0,947): se
le escapan unos 16 planetas más de 360. La media de 5 modelos pequeños (180.805
parámetros, 55 veces menos que un solo Astronet) sube a 0,9874 de AUC y a 0,961 de
exactitud, ya por encima de la exactitud del artículo.

**Lo que no gana todavía: velocidad al predecir.** Aunque hace 14 veces menos
operaciones, en PyTorch tal cual solo es 1,7 veces más rápido en MPS (42 frente a 70 µs
por curva) y casi igual en CPU con un hilo (1,6 frente a 2,0 ms). Las convoluciones por
canal y las muchas operaciones pequeñas pesan más que las cuentas; en la CPU con varios
hilos va incluso peor (~40 ms por curva), por eso se mide con uno. Las dos cifras quedan
muy por debajo del milisegundo en GPU, así que no limita nada en este proyecto, pero no
hay que vender «14 veces menos operaciones» como «14 veces más rápido».

## Ojo: la partición de Astronet es por señal, no por estrella

884 de las 1.574 señales de prueba (el 56 %) están en estrellas que también tienen señales
en entrenamiento: sistemas con varios planetas, o binarias cuyo eclipse primario y
secundario salen como señales distintas. Si nos quedamos con las 690 señales de estrellas
nuevas, el AUC baja a ~0,968 para los dos modelos (última tabla). Parte del 0,988 publicado
viene, por tanto, de reconocer estrellas ya vistas. Para comparar con Astronet hay que usar
su partición, y así se ha hecho; pero en nuestro propio conjunto DR25 (hitos 3 y 4) se
partirá por estrella, como ya dice el plan, y esas cifras saldrán más bajas sin que el
modelo sea peor.

## Siguiente paso propuesto

* Cerrar la distancia de 0,0025: más aumentos de datos (desplazar la fase, ruido) y probar
  el umbral elegido en validación en vez de 0,5 para recuperar recall.
* Velocidad real: juntar BatchNorm con las convoluciones, probar `torch.compile` u ONNX, y
  una variante con convoluciones normales pequeñas (más parámetros, quizá más rápida en MPS).
* Las vistas pares/impares, secundario, centroide y los escalares llegan con el conjunto
  propio de DR25 (hito 3); son las que deberían separar mejor las binarias.

## Fuentes

* Shallue, C. J. y Vanderburg, A. (2018). *Identifying Exoplanets with Deep Learning*. AJ
  155, 94. [arXiv:1712.05044](https://arxiv.org/abs/1712.05044). Tablas 1 y 2 y sección 5.2.
* [google-research/exoplanet-ml](https://github.com/google-research/exoplanet-ml)
  (Apache-2.0): enlace a los datos preparados, configuración `local_global` y salida de
  ejemplo de `evaluate` en el README de `astronet`.
* Datos de Kepler: MAST (STScI); etiquetas de DR24: NASA Exoplanet Archive.
