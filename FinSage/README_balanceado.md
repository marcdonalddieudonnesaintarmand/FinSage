# Modelo Balanceado

Descripción
-----------
El "Modelo Balanceado" busca un equilibrio entre cantidad de datos y calidad. El objetivo es aumentar la muestra manteniendo cierto control de calidad para mejorar generalización sin sacrificar completamente la precisión.

Criterios de selección
----------------------
- Se incluye un filtrado menos estricto que el "ultra-filtrado": los registros que cumplan al menos 1 de los criterios de calidad (p. ej. `quality_score >= 1`) se consideran para entrenamiento.

Dónde está el código
---------------------
- Archivo principal: `train_balanced_model.py`.
- En ese script se aplica el filtrado moderado, se balancean las clases y luego se entrena un pipeline parecido al del modelo ultra-filtrado.

Pipeline de entrenamiento
-------------------------
1. Limpieza y extracción de features (mismas funciones auxiliares: `clean_text`, `extract_features`).
2. Preprocesado similar: TF-IDF + escalado + one-hot.
3. Clasificador/ensemble: similar a los otros scripts (Stacking/ensemble según configuración local del archivo).
4. Métricas: `accuracy_score`, `f1_score`, `classification_report`, `confusion_matrix`.

Ventajas y limitaciones
-----------------------
- Ventaja: más datos útiles para entrenar → mejor robustez / generalización.
- Limitación: precisión inferior al modelo ultra-filtrado pero mejor cobertura que éste.

Archivos de salida
------------------
- Modelo guardado: `financial_sentiment_balanced_model.joblib`
- Datos usados/guardados: `financial_sentiment_balanced_data.csv`

Uso rápido
----------
Ejecutar el script de entrenamiento para reproducir los pasos:

```powershell
& "./venv/Scripts/python.exe" train_balanced_model.py
```

Recomendación: revisar el valor final de `accuracy` impreso por el script para evaluar si se requieren ajustes de hiperparámetros.

Detalles del código
-------------------
- Archivo principal: `train_balanced_model.py`.
- Selección de datos: el script aplica un filtro moderado (en el repo actual se mantuvo la lógica para aceptar registros con `quality_score >= 1` o equivalente). Revisa la variable `quality_score` en el script para ver el umbral concreto.
- Balanceo: tras filtrar, el script realiza un muestreo para igualar clases (`sample(n=min_class_size, random_state=42)`) y así evitar sesgo de clase.

Funciones y pasos principales
----------------------------
1. `clean_text` y `extract_features`: mismas funciones auxiliares usadas en el resto de scripts — se encargan de normalizar los titulares y generar features de conteo/score.
2. Preparación de `X` e `y`: selección de columna de texto `Headline_clean`, columnas numéricas y categóricas.
3. `ColumnTransformer` con `TfidfVectorizer`, `SimpleImputer`+`StandardScaler` y `OneHotEncoder`.
4. Ensemble / Classifier: configuración similar a `train_best_model.py` (Stacking / RandomForest / GradientBoosting / SVC), revisa el bloque donde se crea `model = Pipeline(steps=[...])`.

Hiperparámetros y reproducibilidad
--------------------------------
- El script usa `random_state=42` para muestreos y clasificadores cuando aplica sampling o se inicializan estimadores.
- Ajustes típicos que puedes probar: `n_estimators` en RandomForest, `max_depth`, `learning_rate` en GradientBoosting y `C` en LogisticRegression/SVC.

Salida
-----
- Modelo serializado en `financial_sentiment_balanced_model.joblib`.
- Datos usados guardados en `financial_sentiment_balanced_data.csv`.

Ejecución
--------
```powershell
& "./venv/Scripts/python.exe" train_balanced_model.py
```

Notas
-----
- Este README amplía el anterior con detalles de implementación; el archivo `train_balanced_model.py` contiene comentarios y print statements que muestran los pasos y tamaños intermedios del dataset.
