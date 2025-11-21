# Modelo Máximos Datos

Descripción
-----------
El "Modelo Máximos Datos" utiliza la mayor cantidad de registros posible tras una limpieza básica: se eliminan nulos y registros neutrales, pero no se aplica el filtro de calidad agresivo.

Dónde está el código
---------------------
- Archivo principal: `train_max_data_model.py`.
- En este script se carga `financial_news_events_clean.csv`, se extraen features y se entrena el modelo sin aplicar `quality_score` ni filtros estrictos.

Pipeline de entrenamiento
-------------------------
1. Limpieza mínima (drop nulos, filtrar `positive` y `negative`).
2. Extracción de features (mismas funciones auxiliares).
3. Preprocesado: TF-IDF + escalado + one-hot.
4. Clasificador: ensemble similar a los otros scripts.

Ventajas y limitaciones
-----------------------
- Ventaja: máxima cobertura de datos (mejor para situaciones donde la variedad importa).
- Limitación: menor precisión en test (en este repo ~60-70%), puede necesitar más regularización o más datos etiquetados limpios.

Archivos de salida
------------------
- Modelo guardado: `financial_sentiment_max_data_model.joblib`
- Datos guardados: `financial_sentiment_ultimate_filtered_data.csv` (nombre usado por el script para la salida).

Uso rápido
----------
```powershell
& "./venv/Scripts/python.exe" train_max_data_model.py
```

Notas
-----
- Este script incluye recomendaciones locales impresas, pero no referencia ni compara explícitamente a otros modelos en el resultado (modificado para independencia). Si quieres que elimine cualquier texto remanente, puedo hacerlo.
Detalles del código
-------------------
- Archivo principal: `train_max_data_model.py`.
- Selección de datos: se parte de `financial_news_events_clean.csv` y se filtran los neutrales y nulos, sin aplicar `quality_score` ni filtros estrictos.

Funciones y pasos principales
----------------------------
1. `clean_text` y `extract_features` para normalizar y extraer features del titular.  
2. Preparación de características: columnas numéricas (Index change, Trading volume), `headline_length`, `word_count`, `has_percent`, `has_number`, etc.  
3. Preprocesado y pipeline: `TfidfVectorizer` + transformaciones numéricas + `OneHotEncoder`.  
4. Clasificador/ensemble: igual estructura que en los otros scripts (ver creación de `model = Pipeline(steps=[...])`).

Hiperparámetros y notas de ajuste
--------------------------------
- Usar `random_state=42` para reproducibilidad.  
- Este enfoque puede requerir regularización más fuerte o ajuste de `max_depth`/`n_estimators` para evitar overfitting debido a mayor ruido en los datos.

Salida
-----
- Modelo guardado: `financial_sentiment_max_data_model.joblib`.
- Datos guardados: `financial_sentiment_ultimate_filtered_data.csv` (nombre que usa el script para persistir su versión del dataset).

Ejecución
--------
```powershell
& "./venv/Scripts/python.exe" train_max_data_model.py
```

