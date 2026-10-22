# Informe técnico EP2 — Comparación CNN vs MLP en PlantVillage

**Asignatura:** TLY1102 — Técnicas Avanzadas de Machine Learning I  
**Evaluación:** Parcial N°2 — Presentación y defensa técnica  
**Notebook:** [`notebooks/plantvillage_cnn.ipynb`](notebooks/plantvillage_cnn.ipynb)

## 1. Continuidad del problema y variante propia

La Evaluación Parcial 2 mantiene el problema trabajado en la EP1: clasificar imágenes de hojas de tomate según su condición. Se conserva la misma variante para que la comparación entre arquitecturas sea objetiva.

Se utilizan cinco clases:

- `Tomato_healthy`
- `Tomato_Early_blight`
- `Tomato_Late_blight`
- `Tomato_Leaf_Mold`
- `Tomato_Bacterial_spot`

La variante utiliza como máximo 400 imágenes por clase, para un total de 2.000 imágenes. Se mantiene la semilla 1102, la resolución 64 × 64 RGB y la partición estratificada 70% entrenamiento, 15% validación y 15% prueba.

La continuidad permite atribuir las diferencias de desempeño principalmente al cambio de arquitectura: el MLP aplana la imagen, mientras que la CNN conserva la estructura espacial.

## 2. Problema de negocio, objetivo y KPIs

Una herramienta de clasificación de hojas puede apoyar el monitoreo agrícola y priorizar inspecciones en terreno. El modelo es experimental y no reemplaza el diagnóstico de un especialista.

El objetivo de esta evaluación es implementar una CNN, evaluar su desempeño y compararlo objetivamente con el MLP desarrollado en la EP1.

| KPI | Meta | Justificación |
|---|---:|---|
| Accuracy de prueba | ≥ 60% | Superar la línea base aleatoria de 20% en cinco clases balanceadas |
| F1 macro | ≥ 60% | Dar igual importancia a todas las clases |
| Recall de clases enfermas | ≥ 60% | Reducir falsos negativos, que son más costosos para el cultivo |
| Brecha entrenamiento-prueba | ≤ 0,15 | Evaluar la capacidad de generalización |

## 3. Preparación de datos

Las imágenes se convierten a tensores RGB de forma `(64, 64, 3)` y se normalizan dividiendo cada píxel por 255. A diferencia del MLP, no se aplica `Flatten` durante el preprocesamiento; la CNN recibe la matriz espacial completa.

La partición es estratificada y no presenta solapamiento entre entrenamiento, validación y prueba. El notebook verifica el rango `[0, 1]`, la ausencia de valores NaN y la forma de los tensores.

## 4. Diseño de la CNN

La arquitectura implementada es:

```text
Entrada (64, 64, 3)
├── Conv2D(32, 3x3) + BatchNormalization + ReLU + MaxPooling
├── Conv2D(64, 3x3) + BatchNormalization + ReLU + MaxPooling
├── Conv2D(128, 3x3) + BatchNormalization + ReLU + MaxPooling
├── GlobalAveragePooling2D
├── Dense(128, ReLU)
├── Dropout(0,40)
└── Dense(5, Softmax)
```

Las capas convolucionales detectan patrones locales. MaxPooling reduce la resolución espacial y conserva activaciones relevantes. GlobalAveragePooling reduce el número de parámetros antes de la clasificación. Batch Normalization estabiliza el entrenamiento y Dropout reduce el sobreajuste.

Se utiliza Adam con learning rate inicial 0,001 y `sparse_categorical_crossentropy`.

## 5. Entrenamiento

Se registran las curvas de pérdida y accuracy de entrenamiento y validación. Early Stopping conserva los mejores pesos según `val_loss`, mientras que ReduceLROnPlateau ajusta la tasa de aprendizaje cuando la validación deja de mejorar.

## 6. Evaluación de la CNN

Los valores deben actualizarse después de ejecutar el notebook:

| Métrica | Resultado CNN |
|---|---:|
| Accuracy | COMPLETAR |
| Precision macro | COMPLETAR |
| Recall macro | COMPLETAR |
| F1 macro | COMPLETAR |

La matriz de confusión se encuentra en `images/08_cnn_matriz_confusion.png` y el reporte por clase se genera en la sección de evaluación del notebook.

## 7. Comparación MLP vs CNN

Como referencia, la EP1 reportó para el MLP un Accuracy de 82,33% y un F1 macro de 81,92%. La comparación reproducible de la EP2 se genera en `models/comparacion_mlp_cnn.csv`.

| Modelo | Accuracy | F1 macro | Parámetros | Tiempo entrenamiento | Brecha generalización |
|---|---:|---:|---:|---:|---:|
| MLP EP1 | 0,8233 | 0,8192 | 3.181.061 | COMPLETAR | 0,1481 |
| MLP reentrenado | COMPLETAR | COMPLETAR | COMPLETAR | COMPLETAR | COMPLETAR |
| CNN | COMPLETAR | COMPLETAR | COMPLETAR | COMPLETAR | COMPLETAR |

La comparación debe interpretar no solo qué modelo obtiene mayor Accuracy, sino también el equilibrio entre F1, tiempo, cantidad de parámetros y brecha de generalización.

## 8. Feature Maps y mapas de activación

El notebook visualiza Feature Maps de las capas `conv_1`, `conv_2` y `conv_3`. En las primeras capas se esperan bordes y texturas simples; en capas más profundas se observan combinaciones de patrones relacionadas con lesiones y regiones de la hoja.

También se generan mapas Grad-CAM para una imagen representativa de cada clase. Estos mapas permiten observar qué regiones influyeron en la decisión de la CNN.

Archivos generados:

- `images/10_feature_maps.png`
- `images/11_mapas_activacion_por_clase.png`

## 9. Análisis de errores

Se deben comentar:

- Las clases con mayor cantidad de errores.
- Las confusiones entre `Early_blight` y `Late_blight`.
- Los falsos negativos de hojas enfermas clasificadas como saludables.
- Ejemplos correctos e incorrectos.
- Diferencias entre las confianzas de aciertos y errores.

La CNN conserva información espacial, por lo que debería representar mejor la forma y ubicación de las lesiones que el MLP. Esta afirmación debe comprobarse con las métricas y los mapas generados, no asumirse previamente.

## 10. Conclusiones

Completar esta sección después de ejecutar el notebook:

1. La CNN obtuvo un F1 macro de **COMPLETAR** frente a **81,92%** del MLP de la EP1.
2. El modelo con mejor generalización fue **COMPLETAR**, considerando la brecha entrenamiento-prueba.
3. La principal ventaja de la CNN fue **COMPLETAR**.
4. La principal limitación observada fue **COMPLETAR**.
5. Como mejora futura se propone utilizar transfer learning, más imágenes de campo y técnicas de regularización adicionales.

## 11. Reproducibilidad

El notebook mantiene la semilla 1102, reutiliza las cinco clases de la EP1, guarda las particiones y exporta el modelo CNN, métricas, historial, matriz de confusión, predicciones, Feature Maps y mapas de activación.
