# PlantVillage MLP

Proyecto de clasificación de imágenes de hojas de tomate mediante un Perceptrón Multicapa (MLP), preparado para ejecutarse en Jupyter Notebook o Google Colab.

## Contenido

- `plantvillage_mlp_proyecto.ipynb`: notebook reproducible con EDA, preprocesamiento, entrenamiento, métricas, matriz de confusión y análisis de errores.

## Dataset

El dataset no se incluye en el repositorio porque contiene miles de imágenes y ocupa demasiado espacio. Se deben seleccionar cinco clases:

- `Tomato_healthy`
- `Tomato_Early_blight`
- `Tomato_Late_blight`
- `Tomato_Leaf_Mold`
- `Tomato_Bacterial_spot`

En ejecución local, el notebook busca inicialmente:

```text
/Users/joseconcha/Downloads/PlantVillage/PlantVillage
```

En Google Colab, se puede subir `PlantVillage.zip` a:

```text
Mi unidad/PlantVillage_MLP/PlantVillage.zip
```

El notebook descomprime el archivo automáticamente.

## Ejecución

1. Abrir `plantvillage_mlp_proyecto.ipynb` en Jupyter o Google Colab.
2. Ejecutar las celdas en orden.
3. Si la ruta del dataset es diferente, modificar `DATA_ROOT_OVERRIDE`.
4. Revisar las métricas y gráficos generados en la carpeta `plantvillage_outputs` o `PlantVillage_MLP/outputs`.

## Variante experimental

Se utilizan como máximo 400 imágenes por clase, con semilla reproducible `1102`, redimensionamiento a `64 x 64` y partición estratificada de entrenamiento, validación y prueba.
