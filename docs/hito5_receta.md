# Hito 5, paso 2 · Mejoras de receta sin datos nuevos

<!-- resumen:inicio -->
**Resultado: con las mismas señales, etiquetas, particiones y medida, el conjunto de 10 modelos de
bolsillo pasa a ExoMiner en las tres cifras, y un solo modelo de 25.745 parámetros iguala a su
conjunto en R@P0,99 pero no en AUC-PR.** Todo se ha elegido con la validación de cada partición, sin
curvas de luz nuevas. Media de las 10 particiones (R@P0,99 exacto · como lo mide ExoMiner · AUC-PR):

| | Parámetros | R@P0,99 exacto | Como ExoMiner | AUC-PR |
|---|---|---|---|---|
| *ExoMiner, conjunto de 10 (nuestras etiquetas y particiones)* | ≈ 2,5 millones | 0,932 | 0,950 | 0,9954 |
| Antes (hito 4): un modelo | 38.441 | 0,822 | 0,884 | 0,9894 |
| Antes: conjunto de 10 | 384.410 | 0,870 | 0,918 | 0,9922 |
| **Después: un modelo** (la receta elegida con la validación) | **25.745** | **0,920** | **0,957** | **0,9935** |
| Después: un modelo destilado del conjunto | 38.873 | 0,935 | 0,954 | 0,9939 |
| **Después: conjunto de 10** | **257.450** | **0,956** | **0,969** | **0,9961** |

* **Lo que lo hace es la señal hermana** (hipótesis 1): darle a la red la otra señal de la misma
  estrella con el mismo periodo (5 escalares y una vista reconstruida de la vista global guardada).
  Sola sube un modelo de 0,807 a 0,913 de R@P0,99 exacto (+0,106) y de 0,9892 a 0,9927 de AUC-PR
  (mismas semillas 0–4). Ningún secundario de binaria queda ya por encima del corte de precisión 0,99.
* La coincidencia de efemérides suma poco (+0,016 · +0,014 · +0,0006); la magnitud Kepler y la vista a
  P/2, nada medible. La búsqueda (17 cambios, solo validación) deja 8 cambios pequeños que juntos
  bajan el modelo a 25.745 parámetros (ramas por prueba, como ExoMiner) y suben sobre todo el conjunto.
  Pesos por clase, pérdida focal, desplazamiento de fase, media móvil de pesos y la normalización de
  ExoMiner empeoran o no cambian nada.
* **Distancia a ExoMiner**, partición a partición: conjunto contra conjunto, **+0,024 ± 0,020** exacto,
  +0,019 ± 0,019 con su cuenta y **+0,0007 ± 0,0019** de AUC-PR (por delante en 8, 8 y 5 de las 10
  particiones). Un modelo nuestro contra su conjunto de 10: −0,012 ± 0,049, +0,007 ± 0,025 y
  −0,0019 ± 0,0028. La ventaja en AUC-PR del conjunto está dentro del ruido; lo que sigue faltando en
  un modelo son los planetas de estrellas saturadas y los de periodo detectado al doble.
* El criterio de éxito del paso se cumple: el conjunto de 10 pasa el escalón de 0,909 con la cuenta
  exacta (0,956) y un solo modelo de ≤ 50.000 parámetros pasa al conjunto de 10 del hito 4 (0,920
  frente a 0,870). Configuraciones probadas: 22 recetas, 452 modelos. El veredicto del hito (paso 4)
  debe repetirse con otro reparto de particiones que no se ha mirado (sección 7).
<!-- resumen:fin -->

No se ha descargado ninguna curva de luz. Las únicas tablas nuevas son dos columnas de la tabla
estelar de DR25 (`kepid` y `kepmag`, 3 MB).

## Cómo se reproduce

```bash
uv sync
S=scripts/hito5_receta.py
uv run python $S run --configs sib sib_mag sib_mag_em          # la escalera: 150 modelos (reanudable)
uv run python $S run --search --configs s_zscore s_centroid s_posw s_focal s_smooth s_shift s_noise \
    s_ema s_lr2 s_lr05 s_wd s_drop s_ep30 s_bs128 s_wide s_branches s_trim   # la búsqueda: 102 modelos
uv run python $S run --configs combo                           # la combinación: 50 modelos
uv run python $S run --configs sib_mag_em combo --seed-start 5 # semillas 5–9: 100 modelos
uv run python $S run --configs distilled                       # destilación: 50 modelos (pide las semillas 0–9 del maestro)
uv run python $S report                                        # las tablas de este documento y hito5_receta.json
uv run python $S run --configs sib --quick                     # prueba de humo: 1 partición, 1 semilla, 1 época
```

`run` acepta `--shard i/n` y `--reverse` para repartir el trabajo entre varios procesos. Todo se
guarda en `data/hito5/receta/` (predicciones de validación, de prueba y de todas las señales, y los
pesos). Con otros trabajos en marcha en la misma máquina, los 452 modelos llevaron unas 20 horas (entre 100 y 400 s por modelo).

## 1. Qué se ha cambiado

Todo sale de lo que ya había: el conjunto de vistas del hito 3, la tabla de señales de DR25 y, como
única tabla nueva, la magnitud Kepler de la tabla estelar de DR25. El código nuevo está en
`cazaplanetas.derived` (entradas), `models.PocketVet` y `models.PocketBranches` (modelos),
`training.TrainConfig` (pérdida, aumentos, media de pesos, destilación; con los valores por defecto el
entrenamiento es idéntico al del hito 4, y se ha comprobado: la partición 0 con la semilla 0 da las
mismas puntuaciones, al bit) y `scripts/hito5_receta.py`.

| Cambio | Qué entra en el modelo | De dónde sale |
|---|---|---|
| **Señal hermana** (hipótesis 1) | 5 escalares: si hay otra señal en la misma estrella con el mismo periodo (±1 %), cuánto más fuerte es (log del cociente de MES), cuánto más honda (log del cociente de profundidades), su fase y su duración relativa. Y una vista local más (201 tramos) centrada en la fase de la hermana, en la escala de profundidad del tránsito, como la del secundario | La tabla de señales (periodo, época, duración, profundidad, MES) y la vista global guardada: la vista de la hermana se reconstruye promediando (o interpolando, si la ventana es más fina que un tramo) la vista global en la fase de la otra señal. Es la «corrección de datos» del secundario de ExoMiner (§ 3.5), pero como entrada y no como regla |
| **Magnitud Kepler** (hipótesis 4, primera parte) | 1 escalar, `kepmag` | Tabla estelar de DR25 (`q1_q17_dr25_ks`): solo `kepid` y `kepmag`. Esa tabla trae también cuántos KOI y planetas tiene cada estrella, que son etiqueta; no se bajan |
| **Vista global a P/2** | Un canal más en la rama global: la vista global plegada a la mitad del periodo (las dos mitades de la órbita apiladas) | La vista global guardada |
| **Coincidencia de efemérides** (hipótesis 3) | 5 escalares: si una señal de **otra** estrella a menos de 1° tiene el mismo periodo (o el doble, o la mitad; la deriva acumulada en los 4 años, menor que una duración) y transita a la vez (épocas a menos de media duración), la distancia a esa estrella, el cociente de profundidades y de MES, y la diferencia de magnitudes | Tabla de señales y coordenadas. Las coordenadas solo miden la distancia entre estrellas; no entran en el modelo. El radio de 1° se fijó antes de mirar nada |

Comprobaciones de fugas: las 11 columnas nuevas y las 2 vistas nuevas están en la lista de permitidas
(`features.DERIVED_COLUMNS`, `features.DERIVED_VIEWS`) y `tests/test_derived.py` comprueba que solo se
calculan con columnas permitidas (más `kepid`, `ra` y `dec` para emparejar) y que no dependen del orden
de las filas. La magnitud Kepler sola no separa planetas de falsos positivos certificados (AUC 0,49);
sí separa algo los no transitorios, que están en estrellas más brillantes (mediana 13,4 frente a 14,6;
AUC 0,66 planeta frente a resto), que es lo esperable de la física (las estrellas brillantes saturan y
dan más señales espurias), no de la etiqueta. Con todas las entradas nuevas, el modelo del hito 4 pasa de 38.441 a 38.873
parámetros; la receta final (ramas por prueba, sin vistas de impares/pares ni local) tiene **25.745**.

Cómo se decide cada cambio, fijado antes de entrenar: la **puntuación de validación** de cada modelo es
su AUC-PR en la parte de validación de su partición más 0,1 × la media de sus dos R@P0,99 (exacto y
como lo cuenta ExoMiner) en esa misma parte; el peso hace equivaler 0,03 de recall a 0,003 de AUC-PR.
Un peldaño se queda si su puntuación media, emparejada por partición y semilla con el peldaño anterior,
sube. En la búsqueda, un cambio entra en la combinación si sube la puntuación y el AUC-PR de validación;
la combinación se mide después con 10 particiones × 5 semillas y se queda con la misma regla.

## 2. La escalera del diagnóstico

<!-- escalera:inicio -->
Cada peldaño se entrena con 10 particiones × 5 semillas (las mismas semillas 0–4 en todos) y se mide en las partes de prueba. Media sobre las 10 particiones ± intervalo del 95 %. «Un modelo» es la media de las medidas de las semillas; «elegido» es, en cada partición, la semilla con mejor AUC-PR en **validación**; «media de 5» promedia las puntuaciones de las semillas. «Validación» es la puntuación con la que se decide (AUC-PR + 0,1 × la media de las dos cuentas de R@P0,99, en la parte de validación de cada partición) y su diferencia con el peldaño anterior, emparejada por partición y semilla.

| Peldaño | Parámetros | Un modelo: R@P0,99 exacto | Como ExoMiner | AUC-PR | Elegido con la validación: exacto | Como ExoMiner | AUC-PR | Media de 5: exacto | Como ExoMiner | AUC-PR | Validación | Δ validación | ¿Se queda? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Bolsillo del hito 4 (referencia) | 38.441 | 0,807 ± 0,070 | 0,884 ± 0,030 | 0,9892 ± 0,0029 | 0,860 ± 0,047 | 0,898 ± 0,032 | 0,9907 ± 0,0028 | 0,844 ± 0,090 | 0,913 ± 0,035 | 0,9912 ± 0,0028 | 1,0768 | — | — |
| + señal hermana (5 escalares y su vista) | 38.641 | 0,913 ± 0,022 | 0,937 ± 0,020 | 0,9927 ± 0,0021 | 0,930 ± 0,017 | 0,948 ± 0,010 | 0,9934 ± 0,0024 | 0,927 ± 0,028 | 0,948 ± 0,018 | 0,9938 ± 0,0023 | 1,0872 | +0,0104 ± 0,0013 | sí |
| + magnitud Kepler y vista global a P/2 | 38.713 | 0,910 ± 0,030 | 0,938 ± 0,021 | 0,9927 ± 0,0022 | 0,905 ± 0,057 | 0,937 ± 0,038 | 0,9936 ± 0,0029 | 0,933 ± 0,025 | 0,956 ± 0,017 | 0,9945 ± 0,0022 | 1,0873 | +0,0001 ± 0,0008 | sí |
| + coincidencia de efemérides con otras estrellas (5 escalares) | 38.873 | 0,926 ± 0,028 | 0,953 ± 0,018 | 0,9933 ± 0,0025 | 0,928 ± 0,052 | 0,963 ± 0,014 | 0,9941 ± 0,0025 | 0,945 ± 0,031 | 0,967 ± 0,015 | 0,9945 ± 0,0022 | 1,0881 | +0,0008 ± 0,0007 | sí |
| + los 8 cambios de la búsqueda que ganan en validación, juntos | 25.745 | 0,931 ± 0,034 | 0,956 ± 0,015 | 0,9936 ± 0,0027 | 0,932 ± 0,046 | 0,949 ± 0,038 | 0,9943 ± 0,0025 | 0,956 ± 0,019 | 0,967 ± 0,013 | 0,9957 ± 0,0019 | 1,0903 | +0,0022 ± 0,0006 | sí |

Lo que aporta cada peldaño en las partes de prueba (un modelo, partición a partición, frente al peldaño anterior):

| Peldaño | Δ R@P0,99 exacto | Δ como ExoMiner | Δ AUC-PR | Δ validación: AUC-PR | Δ validación: exacto | Δ validación: como ExoMiner |
|---|---|---|---|---|---|---|
| + señal hermana (5 escalares y su vista) | +0,106 ± 0,065 | +0,053 ± 0,030 | +0,0035 ± 0,0015 | +0,0029 | +0,099 | +0,051 |
| + magnitud Kepler y vista global a P/2 | −0,003 ± 0,020 | +0,002 ± 0,014 | +0,0001 ± 0,0009 | +0,0005 | −0,005 | −0,004 |
| + coincidencia de efemérides con otras estrellas (5 escalares) | +0,016 ± 0,021 | +0,014 ± 0,013 | +0,0006 ± 0,0010 | +0,0001 | +0,005 | +0,009 |
| + los 8 cambios de la búsqueda que ganan en validación, juntos | +0,006 ± 0,017 | +0,003 ± 0,013 | +0,0003 ± 0,0011 | +0,0020 | +0,003 | +0,001 |
<!-- escalera:fin -->

**Lectura.**

* **La señal hermana es casi todo.** Un solo modelo pasa de 0,807 a 0,913 de R@P0,99 exacto (+0,106 ±
  0,065), de 0,884 a 0,937 con la cuenta de ExoMiner y de 0,9892 a 0,9927 de AUC-PR, con 200
  parámetros más. Es más que la regla sin entrenar del diagnóstico (un modelo + regla, 0,869 · 0,914 ·
  0,9909) y además **estabiliza** el modelo: el intervalo de R@P0,99 exacto entre particiones baja de
  ±0,070 a ±0,022. Los secundarios de binaria dejan de ser un problema: con la media de 10 del hito 4,
  24 «señales más débiles de una pareja de la misma estrella y periodo» que no son planetas tenían
  más de 0,5 y 4 estaban por encima del corte de precisión 0,99; con la del último peldaño, ninguna.
* **La magnitud Kepler y la vista a P/2 no aportan nada medible** (validación +0,0001 ± 0,0008; en
  prueba, −0,003 · +0,002 · +0,0001). Por la regla fijada se quedan (la puntuación sube, aunque sea
  dentro del ruido). Lo que debían arreglar sigue ahí: los planetas de estrellas saturadas pierden
  0,0013 de AUC-PR en la media de 10 (ExoMiner, 0,0005; el hito 4, 0,0018) y Kepler-42 c y
  Kepler-1604 b (periodo detectado al doble) siguen con puntuación 0,000.
* **La coincidencia de efemérides suma poco y en el buen sentido** (+0,016 · +0,014 · +0,0006 en prueba;
  +0,0008 ± 0,0007 en validación).
* **La combinación de la búsqueda** sube la validación con claridad (+0,0022 ± 0,0006, sobre todo
  AUC-PR) con un tercio menos de parámetros (25.745), pero en prueba su ganancia en un modelo es
  pequeña (+0,006 · +0,003 · +0,0003). Donde más se nota es en la media de 5: 0,956 · 0,967 · 0,9957.
* La elección de semilla con la validación, que en el hito 4 subía mucho un modelo, ya no aporta: con
  modelos más estables, la semilla con mejor AUC-PR de validación no es la mejor en prueba
  (último peldaño: 0,928 ± 0,052 elegida, 0,926 ± 0,028 cualquiera).

## 3. Búsqueda de receta, un cambio cada vez

<!-- busqueda:inicio -->
Sobre el peldaño «+ coincidencia de efemérides con otras estrellas (5 escalares)», un cambio cada vez, con 3 particiones (1, 4, 7) × 2 semillas y **solo la parte de validación** (la prueba de esas particiones no se lee). Δ: diferencia con la referencia, emparejada por partición y semilla, ± su error típico.

| Cambio | Parámetros | Δ validación | Δ AUC-PR | Δ R@P0,99 exacto | Δ como ExoMiner | ¿Entra en la combinación? |
|---|---|---|---|---|---|---|
| Una rama por prueba con cuello de 8 neuronas (como ExoMiner) | 25.369 | +0,0011 ± 0,0011 | +0,0006 | +0,004 | +0,004 | sí |
| Sin vistas de impares/pares ni vista local (lo que la ablación dice que sobra) | 38.753 | +0,0009 ± 0,0009 | +0,0009 | +0,002 | −0,003 | sí |
| Suavizado de etiquetas (0,05) | 38.873 | +0,0009 ± 0,0014 | +0,0007 | +0,003 | +0,000 | sí |
| 30 épocas | 38.873 | +0,0009 ± 0,0007 | +0,0005 | +0,006 | +0,002 | sí |
| Aumento: ruido gaussiano (0,05 de la profundidad) en las vistas de flujo | 38.873 | +0,0005 ± 0,0009 | +0,0000 | +0,006 | +0,003 | sí |
| Normalización: centroide dividido por una escala del entrenamiento | 38.873 | +0,0005 ± 0,0009 | +0,0006 | +0,002 | −0,003 | sí |
| Paso de aprendizaje 2·10⁻³ | 38.873 | +0,0004 ± 0,0011 | +0,0005 | −0,004 | +0,002 | sí |
| Cabeza más ancha (escalares 64, capa final 48) | 46.457 | +0,0004 ± 0,0006 | +0,0004 | +0,003 | −0,003 | sí |
| Media móvil de los pesos (0,999 por paso) | 38.873 | −0,0005 ± 0,0009 | −0,0001 | −0,006 | −0,004 | no |
| Aumento: desplazamiento de fase (±3 tramos locales, ±10 globales) | 38.873 | −0,0006 ± 0,0008 | +0,0002 | −0,008 | −0,007 | no |
| Decaimiento de pesos 10⁻³ | 38.873 | −0,0011 ± 0,0011 | −0,0003 | −0,008 | −0,008 | no |
| Abandono 0,3 | 38.873 | −0,0015 ± 0,0012 | −0,0003 | −0,012 | −0,011 | no |
| Pérdida focal (γ = 2) | 38.873 | −0,0017 ± 0,0012 | −0,0002 | −0,021 | −0,008 | no |
| Desequilibrio: peso 3 para los planetas | 38.873 | −0,0017 ± 0,0015 | −0,0004 | −0,010 | −0,016 | no |
| Normalización: media y desviación típica, recorte a 20 (como ExoMiner) | 38.873 | −0,0030 ± 0,0010 | −0,0015 | −0,021 | −0,009 | no |
| Paso de aprendizaje 5·10⁻⁴ | 38.873 | −0,0038 ± 0,0011 | −0,0023 | −0,013 | −0,016 | no |
| Lote de 128 | 38.873 | −0,0062 ± 0,0013 | −0,0030 | −0,030 | −0,033 | no |

Referencia (validación, media de los 6 modelos): puntuación 1,0903, AUC-PR 0,9950, R@P0,99 exacto 0,948, como ExoMiner 0,958.
<!-- busqueda:fin -->

**Lectura.** Con 6 modelos por cambio, el error típico de una diferencia es de ±0,0006 a ±0,0015 en la
puntuación: casi todo lo que sale está dentro del ruido, y por eso se mide la combinación entera con
10 × 5 antes de quedársela. Lo que sí se distingue:

* **Empeoran con claridad**: lote de 128 (−0,0062), paso de aprendizaje 5·10⁻⁴ (−0,0038) y la
  normalización de los escalares de ExoMiner, media y desviación típica con recorte a 20 (−0,0030). La
  receta del hito 3 (mediana y rango intercuartílico, recorte a 8) se queda.
* **Pérdida y desequilibrio**: ni pesos por clase (peso 3 a los planetas: −0,0017) ni pérdida focal
  (−0,0017) ayudan, como ya indicaba que ExoMiner no los use. El suavizado de etiquetas (0,05) sale
  ligeramente a favor.
* **Aumentos**: el ruido gaussiano sale a favor por poco; el desplazamiento de fase, en contra.
* **Estabilizar promediando pesos** (media móvil de 0,999 por paso) no ayuda (−0,0005): el modelo ya se
  queda con la mejor época de validación.
* **Ramas por prueba** con cuello de 8 neuronas (la arquitectura de ExoMiner, con nuestras convoluciones
  separables) es lo que más sube (+0,0011) con un tercio menos de parámetros (25.369), y **quitar las
  vistas de impares/pares y local** (lo que la ablación del diagnóstico decía que sobraba) no cuesta
  nada (+0,0009).

## 4. Qué modelo único se queda

<!-- eleccion:inicio -->
Puntuación de validación (10 particiones × 5 semillas, semillas 0–4), la misma regla de los peldaños:

| Receta | Puntuación | AUC-PR | R@P0,99 exacto | Como ExoMiner |
|---|---|---|---|---|
| + los 8 cambios de la búsqueda que ganan en validación, juntos **(se queda)** | 1,0903 | 0,9960 | 0,931 | 0,955 |
| Destilado del conjunto de 10 (objetivo: ½ etiqueta + ½ media del conjunto) | 1,0892 | 0,9942 | 0,941 | 0,958 |
| + coincidencia de efemérides con otras estrellas (5 escalares) | 1,0881 | 0,9940 | 0,928 | 0,954 |
<!-- eleccion:fin -->

El modelo único del paso es, por tanto, **la combinación: 25.745 parámetros**, la mitad del tope de
50.000. El destilado queda segundo en validación, aunque en prueba da algo más (sección 5): se dice
tal cual y no se cambia la elección, porque se hizo con la validación.

**Destilación.** El maestro es la media de las 10 semillas del último peldaño (las semillas 5–9 se
entrenaron para eso, y la receta del maestro se fijó al acabar la búsqueda, antes de medir la
combinación); el alumno es un modelo de la misma receta entrenado con objetivo ½ etiqueta + ½ puntuación
del maestro, y solo con las señales de la parte de entrenamiento de su partición (el maestro de cada
partición tampoco vio ni su validación ni su prueba). El alumno mejora al modelo suelto de su misma
receta en las tres cifras (0,935 · 0,954 · 0,9939 frente a 0,926 · 0,953 · 0,9933, con las mismas semillas 0–4) y se queda a mitad
de camino de su conjunto (0,943 · 0,965 · 0,9947) con 10 veces menos parámetros.

## 5. Antes y después, y la distancia a ExoMiner

<!-- final:inicio -->
Media sobre las 10 particiones ± intervalo del 95 % (entre paréntesis, ± intervalo del 99 %, el que usa ExoMiner), todas con nuestras etiquetas y particiones y el mismo código de medida.

| Qué se mide | Parámetros al predecir | R@P0,99 exacto | R@P0,99 como lo mide ExoMiner | AUC-PR |
|---|---|---|---|---|
| *ExoMiner, media de 10 modelos (nuestras etiquetas y particiones)* | ≈ 2,5 millones | 0,932 ± 0,022 (0,032) | 0,950 ± 0,023 (0,033) | 0,9954 ± 0,0019 (0,0028) |
| Antes: bolsillo del hito 4, un modelo (media de 10 semillas) | 38.441 | 0,822 ± 0,059 (0,085) | 0,884 ± 0,028 (0,040) | 0,9894 ± 0,0029 (0,0042) |
| Antes: un modelo elegido con la validación entre 10 | 38.441 | 0,871 ± 0,047 (0,068) | 0,899 ± 0,037 (0,053) | 0,9907 ± 0,0026 (0,0037) |
| Antes: un modelo + la regla del diagnóstico (sin entrenar) | 38.441 | 0,869 ± 0,045 (0,065) | 0,914 ± 0,019 (0,027) | 0,9909 ± 0,0027 (0,0039) |
| Antes: media de 10 | 384.410 | 0,870 ± 0,058 (0,083) | 0,918 ± 0,022 (0,031) | 0,9922 ± 0,0026 (0,0037) |
| Antes: media de 10 + la regla | 384.410 | 0,902 ± 0,053 (0,076) | 0,947 ± 0,020 (0,028) | 0,9931 ± 0,0025 (0,0036) |
| Después: último peldaño, un modelo (media de 10 semillas) | 38.873 | 0,923 ± 0,030 (0,043) | 0,950 ± 0,017 (0,024) | 0,9931 ± 0,0027 (0,0039) |
| Después: último peldaño, elegido con la validación entre 10 | 38.873 | 0,915 ± 0,052 (0,074) | 0,955 ± 0,017 (0,024) | 0,9938 ± 0,0024 (0,0034) |
| Después: último peldaño, media de 10 | 388.730 | 0,943 ± 0,026 (0,038) | 0,965 ± 0,014 (0,020) | 0,9947 ± 0,0023 (0,0033) |
| Después: destilado del conjunto de 10 (media de 5 semillas) | 38.873 | 0,935 ± 0,028 (0,040) | 0,954 ± 0,019 (0,027) | 0,9939 ± 0,0024 (0,0034) |
| Después: destilado, elegido con la validación entre 5 | 38.873 | 0,939 ± 0,027 (0,038) | 0,956 ± 0,021 (0,030) | 0,9945 ± 0,0023 (0,0033) |
| Después: la combinación, un modelo (media de 10 semillas) | 25.745 | 0,920 ± 0,048 (0,069) | 0,957 ± 0,016 (0,022) | 0,9935 ± 0,0025 (0,0036) |
| Después: la combinación, elegido con la validación entre 10 | 25.745 | 0,930 ± 0,041 (0,059) | 0,950 ± 0,038 (0,055) | 0,9937 ± 0,0025 (0,0035) |
| Después: la combinación, media de 10 | 257.450 | 0,956 ± 0,021 (0,031) | 0,969 ± 0,013 (0,018) | 0,9961 ± 0,0015 (0,0022) |

**Distancia a ExoMiner** (nosotros menos ExoMiner, partición a partición: media ± intervalo del 95 %, y entre paréntesis en cuántas de las 10 particiones quedamos igual o por encima; positivo es mejor):

| Comparación | R@P0,99 exacto | Como ExoMiner | AUC-PR |
|---|---|---|---|
| Antes, un modelo frente a su conjunto de 10 | −0,110 ± 0,063 (0/10) | −0,066 ± 0,041 (1/10) | −0,0060 ± 0,0030 (1/10) |
| Antes, conjunto de 10 frente a su conjunto de 10 | −0,063 ± 0,061 (4/10) | −0,033 ± 0,034 (4/10) | −0,0032 ± 0,0028 (2/10) |
| Después, la combinación (un modelo) frente a su conjunto de 10 | −0,012 ± 0,049 (7/10) | +0,007 ± 0,025 (7/10) | −0,0019 ± 0,0028 (3/10) |
| Después, la combinación elegida con la validación frente a su conjunto de 10 | −0,003 ± 0,037 (6/10) | −0,001 ± 0,045 (7/10) | −0,0017 ± 0,0026 (3/10) |
| Después, conjunto de la combinación frente a su conjunto de 10 | +0,024 ± 0,020 (8/10) | +0,019 ± 0,019 (8/10) | +0,0007 ± 0,0019 (5/10) |
| Después, último peldaño, un modelo frente a su conjunto de 10 | −0,010 ± 0,026 (4/10) | +0,000 ± 0,020 (4/10) | −0,0023 ± 0,0026 (3/10) |
| Después, último peldaño, elegido con la validación frente a su conjunto de 10 | −0,017 ± 0,049 (5/10) | +0,005 ± 0,024 (4/10) | −0,0016 ± 0,0023 (3/10) |
| Después, destilado frente a su conjunto de 10 | +0,003 ± 0,020 (5/10) | +0,003 ± 0,023 (6/10) | −0,0015 ± 0,0024 (3/10) |
| Después, destilado elegido con la validación frente a su conjunto de 10 | +0,007 ± 0,020 (8/10) | +0,006 ± 0,029 (7/10) | −0,0009 ± 0,0023 (4/10) |
| Después, conjunto del último peldaño frente a su conjunto de 10 | +0,011 ± 0,018 (6/10) | +0,015 ± 0,019 (8/10) | −0,0007 ± 0,0024 (4/10) |
<!-- final:fin -->

**Lectura.**

* **Conjunto contra conjunto, por delante en las tres cifras**: 0,956 · 0,969 · 0,9961 frente a 0,932 ·
  0,950 · 0,9954, con unas 10 veces menos parámetros (257.450 frente a ≈ 2,5 millones). En R@P0,99
  la ventaja es clara (8 de 10 particiones, con las dos cuentas; el intervalo del 95 % de la exacta
  no toca el cero). En AUC-PR es +0,0007 ± 0,0019: empate dentro del ruido, por delante en 5 de 10.
  El conjunto del último peldaño de la escalera (sin los cambios de la búsqueda) se queda en 0,9947.
* **Un modelo contra su conjunto de 10**: la combinación (25.745 parámetros, 97 veces menos que su
  conjunto) iguala en R@P0,99 (−0,012 ± 0,049 exacto, +0,007 ± 0,025 con su cuenta) y pierde 0,0019 de
  AUC-PR. El destilado (38.873) es algo mejor en prueba (0,935 · 0,954 · 0,9939) pero peor en
  validación, así que no se elige. Un modelo único de ExoMiner no se puede medir: solo publica su
  conjunto.
* **El AUC-PR que falta en un modelo** está donde decía el diagnóstico. En la media de 10 del último
  peldaño, los 66 planetas de estrellas saturadas pierden 0,0013 de área (ExoMiner, 0,0005) y los 6 de
  periodo al doble, 0,0006 (ExoMiner, 0,0000); Kepler-444 b y d, Kepler-42 c y Kepler-1604 b siguen
  en 0,000. Ni la magnitud Kepler ni la vista a P/2 los han movido: si se quiere ganar también un
  modelo contra su conjunto, eso es lo que pide el paso 3 (masa y densidad de la estrella, tránsitos
  impares y pares con datos) y, si no basta, releer las curvas (centroide por componentes y plegados a
  P/2 y 2P exactos).

## 6. Cuántas configuraciones se han probado

<!-- cuenta:inicio -->
| Qué | Configuraciones | Modelos entrenados |
|---|---|---|
| Peldaños de la escalera (10 particiones × 5 semillas) | 4 | 200 |
| Búsqueda, un cambio cada vez (3 particiones × 2 semillas, solo validación) | 17 | 102 |
| Semillas 5–9 de la receta final (conjunto de 10 y maestro de la destilación) | 0 | 50 |
| Semillas 5–9 de la combinación (su conjunto de 10) | 0 | 50 |
| Destilación | 1 | 50 |
| **Total** | **22** | **452** |
<!-- cuenta:fin -->

## 7. Qué se ha elegido mirando qué

* **Las partes de prueba no se han usado para elegir nada.** Cada peldaño, cada cambio de la búsqueda,
  la combinación, la semilla «elegida» y el modelo único se han decidido con la puntuación de
  validación, con reglas escritas antes de entrenar (sección 1). Las tablas de prueba se han mirado al
  terminar cada peldaño, pero ninguna decisión depende de ellas; donde prueba y validación no
  coinciden (el destilado sale mejor en prueba y peor en validación que la combinación) se ha
  respetado la validación.
* **La validación de una partición es la prueba de otra.** Con el protocolo del hito 4, la partición
  *f* valida con la parte *f*+1, que es la de prueba de la partición *f*+1. Elegir una receta con la
  validación media de las 10 particiones supone haber visto, a través de **otros** modelos, las
  etiquetas de todas las partes. Es el protocolo habitual de validación cruzada con parte de
  validación fija y el mismo que siguió el diagnóstico, pero no es una validación anidada. La
  búsqueda usa solo las particiones 1, 4 y 7 (validación en las partes 2, 5 y 8). Por eso el paso 4
  del hito repite el veredicto con **otro reparto de particiones**, que no se ha mirado.
* **Lo que sí sale de mirar la prueba** son las hipótesis, en el diagnóstico (la señal hermana, la
  magnitud Kepler, la vista a P/2 y las efemérides salieron del análisis de errores en las partes de
  prueba). La lista de cambios de este paso y su orden estaban fijados antes de entrenar.
* **Constantes fijadas sin mirar datos:** la tolerancia de periodo de la hermana (1 %, la del
  diagnóstico), el radio de 1° y los criterios de deriva y época de las efemérides, el peso 0,1 de la
  puntuación de validación, las particiones y semillas de la búsqueda, y los valores de cada cambio de
  la búsqueda (uno por cambio, sin barrido).
* Configuraciones: 22 recetas entrenadas (sección 6), sin contar la comprobación de que la nueva
  canalización reproduce el modelo del hito 4 (1 modelo) ni las pruebas de humo de una época.
