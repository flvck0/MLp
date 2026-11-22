# PlantVillage: MLP y CNN

Clasificación de enfermedades en hojas de tomate con Keras/TensorFlow. Este
repositorio reúne nuestras tres evaluaciones parciales de TLY1102 — Técnicas
Avanzadas de Machine Learning I:

- **EP1:** un Perceptrón Multicapa (MLP) como modelo base.
- **EP2:** una Red Neuronal Convolucional (CNN) para el mismo problema, con los mismos datos, comparada con el MLP.
- **EP3:** un **pipeline integrado** que regulariza, optimiza y audita ambos modelos (ablación de regularización, 63 experimentos de hiperparámetros, Grad-CAM, LIME y análisis de sesgos).

## Resultados

| Modelo | Accuracy (prueba, 300 img.) | F1 macro (prueba) | F1 macro (prueba ampliada, 5.350 img.) | Parámetros | Informe |
|---|---|---|---|---|---|
| MLP (EP1) | 82,33 % | 81,92 % | 82,30 % | 3.181.061 | [`informe_tecnico.md`](informe_tecnico.md) |
| CNN (EP2) | 91,00 % | 91,05 % | 91,91 % | 1.143.493 | [`informe_ep2.md`](informe_ep2.md) |
| MLP optimizado (EP3) | 82,33 % | 82,15 % | 83,07 % | 3.181.061 | [`informe_ep3.md`](informe_ep3.md) |
| **CNN optimizada (EP3)** | **90,67 %** | **90,67 %** | **91,44 %** | **1.143.493** | [`informe_ep3.md`](informe_ep3.md) |

Las cinco clases de la muestra están balanceadas, así que la línea base al azar es
20 %. La CNN supera al MLP con diferencias estadísticamente significativas
(McNemar: p = 0,0001 en la prueba oficial y p ≈ 10⁻⁵⁶ en la ampliada).

**Lo principal de la EP3:**

- La optimización (63 experimentos y confirmación con 3 semillas) **confirmó como mejores las configuraciones de la EP1 y la EP2**. La única mejora aparente (L2 en la CNN) era efecto de la semilla.
- Encontramos un **sesgo de captura** en PlantVillage: cada clase viene de sus propias sesiones de foto, y los modelos aprenden en parte el fondo y el tinte de la sesión. Al neutralizar el fondo, la CNN pierde 55 puntos de recall en Leaf_Mold.
- Las Late_blight de una de las sesiones solo se detectan en el 68 % de los casos.
- Recomendamos la CNN como **herramienta de apoyo en condiciones de laboratorio**, todavía no para fotos de campo.

## Estructura del proyecto

```text
MLp/
├── README.md                        este archivo
├── informe_tecnico.md               informe técnico de la EP1 (MLP)
├── informe_ep2.md                   informe técnico de la EP2 (CNN vs MLP)
├── informe_ep3.md                   informe técnico de la EP3 (pipeline, optimización y auditoría)
├── notebooks/
│   ├── plantvillage_mlp.ipynb       notebook de la EP1
│   ├── plantvillage_cnn.ipynb       notebook de la EP2
│   └── plantvillage_ep3.ipynb       notebook de la EP3 (guardado con todas sus salidas)
├── data/
│   └── PlantVillage/                dataset (no se sube al repo, ver más abajo)
├── images/                          figuras: EP1 (0*), EP2 (cnn_*) y EP3 (ep3_*)
└── models/                          modelos .keras, particiones y métricas
    ├── *_ep3.csv / *_ep3.keras      resultados y modelos finales de la EP3
    └── ep3_experimentos/            un .json por experimento de la EP3 (configuración, historial y métricas)
```

## Entregables

| Entregable | EP1 | EP2 | EP3 |
|---|---|---|---|
| Informe técnico (.md) | [`informe_tecnico.md`](informe_tecnico.md) | [`informe_ep2.md`](informe_ep2.md) | [`informe_ep3.md`](informe_ep3.md) |
| Notebook (.ipynb) | [`plantvillage_mlp.ipynb`](notebooks/plantvillage_mlp.ipynb) | [`plantvillage_cnn.ipynb`](notebooks/plantvillage_cnn.ipynb) | [`plantvillage_ep3.ipynb`](notebooks/plantvillage_ep3.ipynb) |
| Modelos | `models/mlp_plantvillage.keras` | `models/cnn_plantvillage.keras` | `models/mlp_ep3.keras`, `models/cnn_ep3.keras` |
| Métricas y tablas | `models/*.csv` | `models/*_cnn.csv` | `models/*_ep3.csv`, `models/ep3_experimentos/` |
| Figuras | `images/0*.png` | `images/cnn_*.png` | `images/ep3_*.png` |

## Variante del grupo

Es la misma en las tres evaluaciones:

- 5 clases de tomate: `Tomato_healthy`, `Tomato_Early_blight`, `Tomato_Late_blight`, `Tomato_Leaf_Mold` y `Tomato_Bacterial_spot`.
- 400 imágenes por clase (2.000 en total), muestreadas al azar.
- Semilla 1102 en todo el flujo.
- Imágenes de 64 × 64 px.
- Partición estratificada 70 / 15 / 15.

La EP2 y la EP3 no vuelven a muestrear: leen la partición exacta que guardó la EP1
(`models/particion_*.csv`). La EP3 agrega una **prueba ampliada** con las 5.350
imágenes de esas cinco clases que no entraron al muestreo, sin duplicados ni fotos
de hojas que ya estuvieran en la muestra (`models/particion_ampliada_ep3.csv`).

## Dataset

El dataset no está en el repositorio por su tamaño (`data/` está en el
`.gitignore`). Es **PlantVillage**, de Kaggle:
[`emmarex/plantdisease`](https://www.kaggle.com/datasets/emmarex/plantdisease).
Los notebooks solo usan las cinco carpetas de tomate de la variante (7.579 imágenes).

## Cómo ejecutar el notebook de la EP3

### En local

1. Clona el repositorio y deja el dataset de una de estas dos formas:
   - el zip en `data/raw/PlantVillage.zip` (el notebook extrae solo las 5 clases), o
   - las carpetas ya descomprimidas en `data/PlantVillage/<clase>/*.jpg`.
2. Instala las dependencias:
   ```bash
   pip install tensorflow scikit-learn scipy seaborn pandas matplotlib pillow scikit-image lime jupyter
   ```
3. Abre `notebooks/plantvillage_ep3.ipynb` y ejecuta todas las celdas.

### En Google Colab

1. Opcional: activa la GPU en `Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU`.
2. Sube `PlantVillage.zip` al panel de archivos de la sesión.
3. Ejecuta todas las celdas. El notebook clona este repositorio para obtener la partición, los modelos de la EP1 y la EP2 y la caché de experimentos de la EP3. Si no hay conexión, puedes subir también el zip del repositorio (`MLp-main.zip`).

### Tiempo de ejecución

- Con `REUTILIZAR_RESULTADOS = True` (valor por defecto), los 63 experimentos se leen de `models/ep3_experimentos/`. Los pesos no se suben al repositorio (pesan 1,7 GB), así que el notebook reentrena solo los ~10 modelos que necesita cargar para la evaluación y la auditoría: unos 25 minutos en CPU y pocos minutos con GPU.
- Con `REUTILIZAR_RESULTADOS = False` se reentrena todo: 2 h 42 min en una CPU de 2 núcleos.
- En CPU las operaciones son deterministas y los números coinciden con los del informe. En GPU pueden variar en torno a ±1 punto porcentual.

Para cargar `models/cnn_ep3.keras` o `models/mlp_ep3.keras` fuera del notebook,
hay que definir antes la capa propia `BalanceBlancoAleatorio` (sección 3.4 del
notebook).

## Contenido del notebook de la EP3

| # | Sección | Contenido |
|---|---|---|
| 1 | Problema | Continuidad EP1 → EP2 → EP3, negocio, objetivos, KPIs, CRISP-DM, ética |
| 2 | Arquitectura del pipeline | Diagrama, etapas, funciones y principios de diseño |
| 3 | Implementación | Las funciones del pipeline (`cargar_datos` … `optimizar_modelo`, `main`) documentadas |
| 4 | Etapas 1–3 | Carga, EDA (calidad, duplicados, hojas repetidas, sesiones de captura, fondo) y preprocesamiento |
| 5 | Líneas base | Verificación de los modelos de la EP1 y la EP2 |
| 6 | Regularización | Ablación de Dropout, BatchNorm, aumento, EarlyStopping y L2 en ambos modelos |
| 7 | Optimización | Plan con hipótesis, OFAT, Random Search, confirmación con 3 semillas y selección |
| 8 | Evaluación final | `main()` para ambos modelos, prueba oficial y ampliada, ROC, McNemar |
| 9 | Auditoría | Grad-CAM, LIME, sesgos por sesión, intervención del fondo, robustez, mitigación y errores |
| 10 | Comparación | MLP vs CNN, KPIs y modelo recomendado |
| 11 | Evaluación crítica | Fortalezas, limitaciones, riesgos y mejoras del pipeline |
| 12–13 | Artefactos y conclusiones | Guardado y respuestas a la discusión de las actividades 3.4 y 3.5 |

### Notebooks de la EP1 y la EP2

Siguen siendo válidos. El de la EP2 se ejecuta igual que el de la EP3. El de la
EP1 lee el dataset desde `Mi unidad/PlantVillage_MLP/PlantVillage.zip` en Google
Drive (en Colab) o desde `data/` (en local).
