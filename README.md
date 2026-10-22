# PlantVillage: MLP y CNN

Clasificación de enfermedades en hojas de tomate con Keras/TensorFlow. Este
repositorio reúne nuestras dos evaluaciones parciales de TLY1102 — Técnicas
Avanzadas de Machine Learning I:

- **EP1:** un Perceptrón Multicapa (MLP) como modelo base.
- **EP2:** una Red Neuronal Convolucional (CNN) para el mismo problema, con los mismos datos, comparada con el MLP.

## Resultados

| Modelo | Accuracy (prueba) | F1 macro | Parámetros | Informe |
|---|---|---|---|---|
| MLP (EP1) | 82,33 % | 81,92 % | 3.181.061 | [`informe_tecnico.md`](informe_tecnico.md) |
| CNN (EP2) | 91,00 % | 91,05 % | 1.143.493 | [`informe_ep2.md`](informe_ep2.md) |

Las cinco clases están balanceadas, así que la línea base al azar es 20 %. La
diferencia entre los dos modelos es estadísticamente significativa (prueba de
McNemar, p = 0,0002).

## Estructura del proyecto

```text
MLp/
├── README.md                        este archivo
├── informe_tecnico.md               informe técnico de la EP1 (MLP)
├── informe_ep2.md                   informe técnico de la EP2 (CNN vs MLP)
├── notebooks/
│   ├── plantvillage_mlp.ipynb       notebook de la EP1
│   └── plantvillage_cnn.ipynb       notebook de la EP2 (guardado con salidas)
├── data/
│   └── PlantVillage/                dataset (no se sube al repo, ver más abajo)
├── images/                          figuras: las de la EP1 y las de la EP2 (prefijo cnn_)
└── models/                          modelos .keras, particiones y métricas (EP2 con sufijo _cnn)
```

## Entregables

| Entregable | EP1 | EP2 |
|---|---|---|
| Informe técnico (.md) | [`informe_tecnico.md`](informe_tecnico.md) | [`informe_ep2.md`](informe_ep2.md) |
| Notebook (.ipynb) | [`notebooks/plantvillage_mlp.ipynb`](notebooks/plantvillage_mlp.ipynb) | [`notebooks/plantvillage_cnn.ipynb`](notebooks/plantvillage_cnn.ipynb) |
| Modelo y métricas | `models/mlp_plantvillage.keras`, `models/*.csv` | `models/cnn_plantvillage.keras`, `models/*_cnn.csv` |
| Figuras | `images/0*.png` | `images/cnn_*.png` |

## Variante del grupo

Es la misma en las dos evaluaciones:

- 5 clases de tomate: `Tomato_healthy`, `Tomato_Early_blight`, `Tomato_Late_blight`, `Tomato_Leaf_Mold` y `Tomato_Bacterial_spot`.
- 400 imágenes por clase (2.000 en total), muestreadas al azar.
- Semilla 1102 en todo el flujo.
- Imágenes de 64 × 64 px.
- Partición estratificada 70 / 15 / 15.

La EP2 no vuelve a muestrear: lee la partición exacta que guardó la EP1
(`models/particion_*.csv`), así que los dos modelos se evalúan sobre las mismas
300 imágenes de prueba.

## Dataset

El dataset no está en el repositorio por su tamaño (`data/` está en el
`.gitignore`). Es **PlantVillage**, de Kaggle:
[`emmarex/plantdisease`](https://www.kaggle.com/datasets/emmarex/plantdisease).
Los notebooks solo usan las cinco carpetas de tomate de la variante.

## Cómo ejecutar el notebook de la EP2

### En local

1. Clona el repositorio y deja el dataset de una de estas dos formas:
   - el zip en `data/raw/PlantVillage.zip` (el notebook extrae solo las 5 clases), o
   - las carpetas ya descomprimidas en `data/PlantVillage/<clase>/*.jpg`.
2. Instala las dependencias:
   ```bash
   pip install tensorflow scikit-learn scipy seaborn pandas matplotlib pillow jupyter
   ```
3. Abre `notebooks/plantvillage_cnn.ipynb` y ejecuta todas las celdas. Las rutas se detectan solas y los resultados de la EP1 se leen desde `models/`.

En un equipo sin GPU la ejecución completa toma unos 4 minutos.

### En Google Colab

1. Opcional: activa la GPU en `Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU`.
2. Sube `PlantVillage.zip` al panel de archivos de la sesión.
3. Ejecuta todas las celdas (`Entorno de ejecución → Ejecutar todas`). El notebook clona este repositorio para obtener los resultados de la EP1. Si no hay conexión, puedes subir también el zip del repositorio (`MLp-main.zip`).

Las figuras y los modelos quedan en `images/` y `models/` dentro de la sesión.
Descárgalos antes de cerrarla.

### Notebook de la EP1

Las instrucciones originales siguen siendo válidas: en Colab se lee
`Mi unidad/PlantVillage_MLP/PlantVillage.zip` desde Google Drive, y en local el
dataset va en `data/`.

## Contenido del notebook de la EP2

| # | Sección | Contenido |
|---|---|---|
| 1 | Problema | Continuidad con la EP1, problema de negocio, objetivos, KPIs, CRISP-DM, ética |
| 2 | Datos y EDA | Variante, partición de la EP1, balance, calidad, ejemplos, color por clase |
| 3 | Preparación | Tensores 64×64×3 normalizados, aumento de datos |
| 4 | Diseño de la CNN | Arquitectura justificada, parámetros, campo receptivo |
| 5 | Entrenamiento | Curvas de aprendizaje y prueba de cinco configuraciones |
| 6 | Evaluación | Accuracy, Precision, Recall, F1, matriz de confusión, KPIs |
| 7 | MLP vs CNN | Tabla y gráficos comparativos, tiempo en el mismo equipo, prueba de McNemar |
| 8 | Feature maps | Filtros, feature maps por capa y Grad-CAM por clase |
| 9 | Errores | Confianza, aciertos, errores con Grad-CAM, sesgo del fondo, hipótesis |
| 10 | Conclusiones | Diferencias, ventajas, limitaciones y mejoras por regularización |

## Licencia de los datos

PlantVillage es un dataset público para investigación. El detalle de la licencia
está en la página de Kaggle.
