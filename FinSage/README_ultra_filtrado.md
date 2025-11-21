# Modelo Ultra-filtrado

Resumen
-------
El "Modelo Ultra-filtrado" (implementado en `train_best_model.py`) entrena usando únicamente los ejemplos de más alta calidad. Se aplica un filtro agresivo para mantener ejemplos que cumplan al menos 2 de 3 criterios de calidad, de modo de maximizar la precisión del conjunto final.

Objetivo
--------
- Priorizar calidad por sobre cantidad para obtener un modelo con alta precisión en test.

Filtro de calidad (implementación)
----------------------------------
La lógica está en `train_best_model.py` (sección "FILTRO DE CALIDAD AGRESIVO"). Los pasos clave del código son:

- `df['headline_match']`:
   - (score >= 2) & Sentiment == 'positive'
   - (score <= -2) & Sentiment == 'negative'

- `df['index_match']`:
   - (Index_Change_Percent_num > 0.5) & Sentiment == 'positive'
   - (Index_Change_Percent_num < -0.5) & Sentiment == 'negative'
   - `NaN` en Index_Change_Percent_num se considera aceptable

- `df['has_strong_keyword']`:
   - `has_ultra_pos == 1` o `has_ultra_neg == 1`

- `df['quality_score']` se calcula como suma de los tres indicadores (cada uno convertido a `int`).
- Filtrado final: `df = df[df['quality_score'] >= 2].copy()` → queda el conjunto ultra-filtrado.

Feature engineering y pipeline
------------------------------
Código relevante (en el mismo archivo):

- `clean_text(text)` — limpieza básica (lowercase, eliminar URLs, caracteres no alfanuméricos)  
- `extract_features(headline)` — calcula conteo de palabras positivas/negativas y un `score` ponderado (ultra-palabras cuentan 3x).  

Pipeline de entrenamiento:

1. `TfidfVectorizer` para `Headline_clean` (ngramas hasta 4, sublinear_tf, max_features limitado).  
2. Transformaciones numéricas: `SimpleImputer(median)` + `StandardScaler`.  
3. Transformaciones categóricas: `SimpleImputer(constant='unknown')` + `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.  
4. Clasificador: `StackingClassifier` con estimadores base (`LogisticRegression`, `RandomForest`, `GradientBoosting`, `SVC`) y `LogisticRegression` final.

Hiperparámetros principales (desde el script):

- `RandomForestClassifier(n_estimators=700, max_depth=35, max_features='sqrt')`
- `GradientBoostingClassifier(n_estimators=700, learning_rate=0.02, max_depth=12)`
- `LogisticRegression(C=5.0, solver='saga', max_iter=10000)` (como base)  
- `StackingClassifier(cv=15)`

Evaluación y salidas
--------------------
- Métricas impresas: `accuracy` (train/test), `f1_score` (macro), `classification_report`, `confusion_matrix`.  
- Archivos generados:
   - `financial_sentiment_best_model.joblib` (modelo entrenado)  
   - `financial_sentiment_best_data.csv` (datos filtrados usados para entrenamiento)

Cómo ejecutar
-------------
1. Asegúrate de tener el entorno virtual activado y dependencias instaladas (`joblib`, `pandas`, `scikit-learn`, `numpy`).
2. Ejecuta:

```powershell
& "./venv/Scripts/python.exe" train_best_model.py
```

Notas y recomendaciones
----------------------
- Este script sacrifica cobertura por precisión. Úsalo cuando la fiabilidad de la predicción sea prioritaria.
- Para reproducibilidad, el script fija `random_state=42` en muestreos y clasificadores.
- Si quieres aumentar la cobertura, considera usar `train_balanced_model.py` o `train_max_data_model.py`.

Referencias en el código
------------------------
- Filtrado: busca `quality_score` en `train_best_model.py`.  
- Funciones auxiliares: `clean_text`, `extract_features`, `prepare_input` (estas últimas son definidas también en `demo_model.py` para inferencia).

# Modelo Ultra-filtrado

Descripción
-----------
El "Modelo Ultra-filtrado" entrena usando únicamente los ejemplos de más alta calidad extraídos del dataset. Estos ejemplos se seleccionan aplicando tres criterios estrictos y manteniendo solo los registros que cumplen al menos 2 de 3 criterios (quality_score >= 2).

Criterios de calidad
--------------------
1. `headline_match`: El `score` calculado del headline debe coincidir con la etiqueta `Sentiment`:
   - `score >= 2` → `Sentiment == 'positive'`
   - `score <= -2` → `Sentiment == 'negative'`
2. `index_match`: El `Index_Change_Percent_num` (si existe) debe ser coherente:
   - `> +0.5` → `positive` ; `< -0.5` → `negative` ; `NaN` se considera válido.
3. `has_strong_keyword`: El headline debe contener una palabra ultra-específica (p. ej. `surge`, `crash`, etc.).

Dónde está el código
---------------------
- Archivo principal: `train_best_model.py`
- Lógica de filtrado: sección "FILTRO DE CALIDAD AGRESIVO" (líneas alrededor de la creación de `quality_score`).
  - `df['headline_match']`, `df['index_match']`, `df['has_strong_keyword']`
  - `df['quality_score'] = df['headline_match'].astype(int) + df['index_match'].astype(int) + df['has_strong_keyword'].astype(int)`
  - Filtrado final: `df = df[df['quality_score'] >= 2].copy()`

Pipeline de entrenamiento
-------------------------
1. Feature engineering: extracción de `Headline_clean`, `ultra_pos`, `ultra_neg`, `score`, `ratio`, y varias características numéricas.
2. Preprocesado: `TfidfVectorizer` para texto, escalado e imputación para numéricos, `OneHotEncoder` para categóricas.
3. Clasificador: `StackingClassifier` con `LogisticRegression`, `RandomForest`, `GradientBoosting`, `SVC` y `LogisticRegression` final.
4. Métricas: `accuracy_score`, `f1_score`, `classification_report`, `confusion_matrix`.

Ventajas y limitaciones
-----------------------
- Ventaja: alta precisión (ej. ~90% en test en este repo) gracias a datos muy confiables.
- Limitación: pequeño conjunto de entrenamiento (poca cantidad → menor cobertura/robustez frente a variabilidad).

Archivos de salida
------------------
- Modelo guardado: `financial_sentiment_best_model.joblib`
- Datos filtrados: `financial_sentiment_best_data.csv`

Uso rápido (ejecutar demo)
--------------------------
```
python demo_model.py
```
El demo carga `financial_sentiment_best_model.joblib` y permite probar titulares en modo interactivo.
