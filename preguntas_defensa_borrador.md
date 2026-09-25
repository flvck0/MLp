# Preguntas frecuentes para la defensa — borrador

> **Estado:** borrador de apoyo, en progreso. Las respuestas están alineadas con
> los resultados de la última ejecución del notebook. Si se vuelve a entrenar
> con otra semilla o configuración, revisar las cifras antes de usarlas.

Respuestas breves a lo que probablemente se pregunte. La defensa es individual y
con **preguntas cruzadas**: hay que poder responder sobre cualquier sección, no
solo la que cada integrante expuso.

## Preguntas sobre el modelo y las decisiones

**¿Por qué un MLP y no una CNN?** Porque la unidad evalúa los fundamentos de
redes densas. El MLP es el *modelo base* que establece el punto de comparación y
permite demostrar con evidencia por qué se necesita una CNN.

**¿Por qué 64 × 64?** Porque a 256 × 256 la primera capa densa tendría más de 50
millones de parámetros frente a 1.400 imágenes de entrenamiento. Es un compromiso
explícito entre detalle y viabilidad.

**¿Qué hace `Flatten()` y qué se pierde?** Convierte la matriz 64 × 64 × 3 en un
vector de 12.288 valores, porque `Dense` solo acepta vectores. Se pierde la
vecindad entre píxeles: la red ya no puede saber qué píxeles eran adyacentes.

**¿Por qué softmax y no sigmoide en la salida?** Porque las clases son
mutuamente excluyentes y softmax garantiza que las probabilidades sumen 1.
Sigmoide trataría cada clase de forma independiente, que es el caso multietiqueta.

**¿Por qué `sparse_categorical_crossentropy`?** Crossentropy mide la distancia
entre la distribución real y la predicha, que es lo que corresponde en
clasificación. La variante *sparse* acepta etiquetas enteras (0-4) en lugar de
exigir vectores one-hot.

**¿Cuál es la diferencia entre función de activación y función de salida?** La
activación va entre capas ocultas y aporta la no linealidad (ReLU). La función de
salida va solo al final y adapta el resultado al tipo de problema (softmax para
clasificación multiclase, lineal para regresión). Son conceptos distintos: una
sigmoide no sirve como activación oculta pero sí como función de salida.

**¿Qué hace Adam y por qué no SGD?** Adam combina momentum con tasas de
aprendizaje adaptativas por parámetro, lo que lo hace menos sensible al valor
inicial de λ. Importante: en nuestro barrido (sección 5.3) Adam y SGD+momentum
quedaron prácticamente empatados — la diferencia fue de 4 imágenes de 300, que
está dentro del ruido. Además se probaron a tasas distintas, así que esa
comparación no aísla el efecto del optimizador. **No afirmar que Adam ganó.**
Se eligió Adam por robustez general, no porque el experimento lo demuestre.

**¿Qué es el batch size y qué relación tiene con la época?** El batch es el grupo
de ejemplos que pasa por la red antes de actualizar los pesos. Una época se
completa cuando pasaron todos los batches. Con 1.400 imágenes y batch de 32 hay
44 actualizaciones de pesos por época.

**¿Hay overfitting?** Hay señales, contenidas. La brecha train − val en la época
conservada es baja (~0,05), pero creció hacia el final del entrenamiento
(~0,09): la red empezó a memorizar y `restore_best_weights` evitó quedarse con
ese modelo. La brecha train − test del modelo final es 0,148, bajo la meta de
0,15. **Ojo con la pregunta capciosa:** en nuestra corrida EarlyStopping *no*
llegó a activarse — se agotó el tope de 60 épocas. Lo que actuó fue
`restore_best_weights`, que recuperó la época 53. Son dos mecanismos distintos
del mismo callback.

**¿Por qué la brecha de la sección 5.2 (0,05) no coincide con la del KPI
(0,148)?** Porque miden cosas distintas. La de 5.2 es train vs validación
*durante* el entrenamiento, con el aumento de datos activo (que dificulta el
entrenamiento y baja su accuracy). La del KPI es train vs test del modelo ya
entrenado, evaluado sobre imágenes sin alterar. No es una contradicción.

**¿Por qué no basta el accuracy?** Porque no dice *qué* clase falla. Un modelo
puede tener buen accuracy global e ignorar completamente una enfermedad. Por eso
se reportan Precision, Recall y F1 por clase.

**¿Por qué el recall importa más que la precision aquí?** Porque los errores no
son simétricos: un falso positivo cuesta una inspección innecesaria, un falso
negativo deja una planta enferma sin tratar y contagiando.

**¿Qué significa que las métricas macro y weighted coincidan?** Que las clases
están balanceadas. Si difirieran mucho, indicaría que el modelo funciona bien
solo en las clases más frecuentes.

**¿Por qué se estratifica la partición?** Para que las cinco clases mantengan su
proporción en entrenamiento, validación y prueba. Sin estratificar, el azar podría
dejar una clase sub-representada en prueba y sus métricas serían ruido.

**¿Para qué sirve el conjunto de validación si ya hay uno de prueba?** Validación
se usa durante el entrenamiento para decidir cuándo detener y comparar
configuraciones. Prueba se toca una sola vez al final; si se usara para tomar
decisiones, dejaría de ser una estimación honesta de generalización.

**¿Qué cambiarían con más tiempo?** Pasar a una CNN, que ataca directamente las
tres limitaciones identificadas en la sección 7.6. Dentro del MLP, aumento de
datos y el dataset completo.

**¿Por qué la semilla 1102?** Es la variante del grupo. Fija muestreo, particiones
e inicialización de pesos, de modo que cualquiera pueda reproducir exactamente los
mismos resultados.


for n, celda in enumerate(cells):
    celda["id"] = f"cell{n:03d}"
