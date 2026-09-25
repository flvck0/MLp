# PlantVillage MLP

Clasificación de enfermedades en hojas de tomate mediante un Perceptrón
Multicapa (MLP), implementado en Keras/TensorFlow. Proyecto para la Evaluación
Parcial N°1 de TLY1102 — Técnicas Avanzadas de Machine Learning I.

**Resultado:** 82,33 % de accuracy en prueba (5 clases balanceadas, línea base
al azar 20 %). Detalle completo de métricas, gráficos e interpretación en
[`informe_tecnico.md`](informe_tecnico.md) y en el notebook.

## Estructura del proyecto

```text
MLp/
├── README.md                          este archivo
├── informe_tecnico.md                 informe técnico (problema, KPIs, EDA, CRISP-DM)
├── preguntas_defensa_borrador.md      borrador de apoyo para la defensa (documento aparte, en progreso)
├── notebooks/
│   └── plantvillage_mlp.ipynb         notebook principal, ejecutable de punta a punta
├── data/
│   └── PlantVillage/                  dataset (NO se sube al repo, ver más abajo)
├── images/                            figuras generadas por el notebook (EDA, curvas, matriz de confusión)
└── models/                            modelo entrenado (.keras) y CSV de métricas/particiones
```

## Entregables de la evaluación

| Entregable | Ubicación |
|---|---|
| Informe técnico (.md) | [`informe_tecnico.md`](informe_tecnico.md) |
| Notebook (.ipynb) | [`notebooks/plantvillage_mlp.ipynb`](notebooks/plantvillage_mlp.ipynb) |
| Dataset y archivos complementarios | `data/` (instrucciones de descarga abajo) |
| Carpeta organizada | esta misma estructura |

## Variante del grupo

- **5 clases** de tomate: `Tomato_healthy`, `Tomato_Early_blight`,
  `Tomato_Late_blight`, `Tomato_Leaf_Mold`, `Tomato_Bacterial_spot`.
- **400 imágenes por clase** como máximo (2.000 en total), muestreadas al azar.
- **Semilla 1102** en todo el flujo: muestreo, particiones e inicialización.
- **Resolución 64 × 64 px** de entrada.
- **Partición estratificada 70 / 15 / 15**.

## Dataset

El dataset no se incluye en el repositorio por su tamaño (`data/` está en
`.gitignore`). Es el dataset **PlantVillage** de Kaggle:
[`emmarex/plantdisease`](https://www.kaggle.com/datasets/emmarex/plantdisease).

El notebook solo necesita las cinco carpetas de tomate de la variante; no hace
falta el dataset completo (38 clases).

### Ejecutar en Google Colab (recomendado)

1. Descarga `PlantVillage.zip` desde Kaggle.
2. Súbelo a tu Google Drive en esta ruta exacta:
   ```text
   Mi unidad/PlantVillage_MLP/PlantVillage.zip
   ```
3. Abre `notebooks/plantvillage_mlp.ipynb` en Colab y ejecuta todas las celdas
   (`Entorno de ejecución → Ejecutar todas`). La primera celda monta Drive
   (pide autorización una vez) y el notebook descomprime automáticamente solo
   las 5 clases de la variante.
4. Las figuras y el modelo se guardan en
   `Mi unidad/PlantVillage_MLP/images/` y `.../models/` para no perderse al
   cerrar la sesión.

### Ejecutar en local

1. Descarga `PlantVillage.zip` y colócalo en `data/raw/PlantVillage.zip`, o
   descomprímelo directamente en `data/PlantVillage/<clase>/*.jpg`.
2. Instala las dependencias:
   ```bash
   pip install tensorflow scikit-learn seaborn pandas matplotlib pillow jupyter
   ```
3. Ejecuta el notebook con Jupyter o `jupyter nbconvert --execute`.

El notebook detecta automáticamente si corre en Colab o en local y ajusta las
rutas — no requiere editar ninguna ruta a mano.

## Flujo del notebook

El notebook sigue el orden de trabajo pedido para la evaluación:

| # | Sección | Contenido |
|---|---------|-----------|
| 1 | Problema | Problema de negocio, objetivos, KPIs, CRISP-DM, consideraciones éticas |
| 2 | Datos | Inventario, variante, distribución, calidad, EDA visual, color por clase |
| 3 | Preprocesamiento | Partición estratificada, normalización, verificación |
| 4 | MLP | Arquitectura justificada capa por capa, conteo de parámetros |
| 5 | Entrenamiento | Curvas de aprendizaje, diagnóstico, barrido de arquitecturas/optimizadores/learning rate |
| 6 | Evaluación | Accuracy, Precision, Recall, F1, matriz de confusión, cumplimiento de KPIs |
| 7 | Errores | Confianza, aciertos/errores, clases confundidas, limitaciones del MLP en imágenes |
| 8 | Conclusiones | Resumen, decisiones de diseño, fortalezas, limitaciones, mejoras futuras |

## Licencia de los datos

PlantVillage es un dataset público para investigación. Ver la página de Kaggle
para el detalle de licencia.
