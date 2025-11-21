# Modelo Ultra-Filtrado (Best Model)

## 📊 Descripción General

Este es el modelo de sentimiento financiero **más preciso** con **90.24% accuracy**. Utiliza un filtro de calidad agresivo que selecciona solo los datos más confiables para el entrenamiento.

## 🎯 Objetivo

Alcanzar la máxima precisión en la clasificación de sentimiento (POSITIVE/NEGATIVE) aplicando criterios estrictos de calidad de datos.

## 📁 Archivos Principales

- **`train_best_model.py`** - Script de entrenamiento del modelo
- **`financial_sentiment_best_model.joblib`** - Modelo entrenado guardado
- **`financial_sentiment_best_data.csv`** - Datos filtrados y usados para entrenar

## 🔧 Arquitectura del Código

### 1. **Carga de Datos** (Líneas 95-110)
```python
df = pd.read_csv("financial_news_events_clean.csv")
# Entrada: 1,852 registros con sentimientos positive/negative
```

### 2. **Feature Engineering** (Líneas 112-125)
- Limpieza de texto: `clean_text()`
- Extracción de características: `extract_features()`
  - `ultra_pos`: Palabras ultra-positivas (surge, soar, boom, etc.)
  - `ultra_neg`: Palabras ultra-negativas (crash, plunge, collapse, etc.)
  - `pos`/`neg`: Palabras positivas/negativas comunes
  - `score`: Puntuación calculada (total_pos - total_neg)
  - `ratio`: Ratio positivo/negativo

### 3. **Filtro de Calidad Agresivo** (Líneas 127-152)
Aplica 3 criterios de validación:

**Criterio 1: Headline Match** (Línea 130-133)
```python
df["headline_match"] = (
    ((df["score"] >= 2) & (df["Sentiment"] == "positive")) |
    ((df["score"] <= -2) & (df["Sentiment"] == "negative"))
)
```
- Si el score es alto → debe ser positive
- Si el score es bajo → debe ser negative

**Criterio 2: Index Match** (Línea 136-141)
```python
df["index_match"] = (
    ((df["Index_Change_Percent_num"] > 0.5) & (df["Sentiment"] == "positive")) |
    ((df["Index_Change_Percent_num"] < -0.5) & (df["Sentiment"] == "negative")) |
    (df["Index_Change_Percent_num"].isna())
)
```
- Si cambio de índice > +0.5% → debe ser positive
- Si cambio de índice < -0.5% → debe ser negative

**Criterio 3: Strong Keywords** (Línea 143)
```python
df["has_strong_keyword"] = (df["has_ultra_pos"] == 1) | (df["has_ultra_neg"] == 1)
```
- Debe tener al menos una palabra clave ultra-importante

**Puntuación de Calidad** (Línea 146-149)
```python
df["quality_score"] = (
    df["headline_match"].astype(int) +
    df["index_match"].astype(int) +
    df["has_strong_keyword"].astype(int)
)
df = df[df["quality_score"] >= 2].copy()  # Mínimo 2 de 3 criterios
```

**Resultado:** 1,852 → **268 registros** (solo los mejores)

### 4. **Balanceo de Clases** (Línea 154-160)
```python
min_class_size = df["Sentiment"].value_counts().min()
df_balanced = pd.concat([
    df[df["Sentiment"] == "positive"].sample(n=min_class_size, random_state=42),
    df[df["Sentiment"] == "negative"].sample(n=min_class_size, random_state=42)
])
```
- Asegura distribución equitativa: 134 positive, 134 negative

### 5. **Preparación de Features Adicionales** (Línea 162-180)
```python
df["headline_length"] = df["Headline_clean"].str.len()
df["word_count"] = df["Headline_clean"].str.split().str.len()
df["is_pos_change"] = (df["Index_Change_Percent_num"] > 0).astype(int)
df["is_neg_change"] = (df["Index_Change_Percent_num"] < 0).astype(int)
df["abs_change"] = df["Index_Change_Percent_num"].abs()
df["log_volume"] = np.log1p(df["Trading_Volume_num"].fillna(0))
```

### 6. **Pipeline de Preprocesamiento** (Línea 195-225)
```python
preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(...), "Headline_clean"),        # Vectorización TF-IDF
        ("num", StandardScaler(), numeric_cols),                 # Normalización numérica
        ("cat", OneHotEncoder(), categorical_cols),              # Encoding categórico
    ]
)
```

### 7. **Ensemble Stacking** (Línea 228-270)
Combina 4 estimadores base con meta-aprendiz:

**Estimadores Base:**
- `LogisticRegression`: Modelo lineal base
- `RandomForestClassifier`: Ensemble de árboles (700 árboles)
- `GradientBoostingClassifier`: Boosting secuencial (700 estimadores)
- `SVC`: Support Vector Machine con kernel RBF

**Meta-Aprendiz:**
```python
StackingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('gb', gb), ('svm', svm)],
    final_estimator=LogisticRegression(C=3.0, max_iter=5000),
    cv=15,                    # Cross-validation con 15 folds
    n_jobs=1                  # Procesamiento secuencial
)
```

### 8. **Entrenamiento** (Línea 278-284)
```python
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", stacking_clf)
])
model.fit(X_train, y_train)  # 227 registros de entrenamiento
```

### 9. **Evaluación** (Línea 287-320)
```python
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test, average="macro")

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_test)
tn, fp, fn, tp = cm.ravel()
precision = tp/(tp+fp)
recall = tp/(tp+fn)
```

**Métricas Finales:**
- Train Accuracy: ~95%
- Test Accuracy: **90.24%**
- F1 Score: ~90%

### 10. **Guardado del Modelo** (Línea 324-329)
```python
joblib.dump(model, "financial_sentiment_best_model.joblib")
df.to_csv("financial_sentiment_best_data.csv", index=False)
```

## 📊 Flujo de Datos

```
financial_news_events_clean.csv (1,852)
    ↓
[Feature Engineering]
    ↓
[Filtro Calidad - Criterio 1,2,3] → 268 registros
    ↓
[Balanceo de Clases] → 268 (134 pos, 134 neg)
    ↓
[Train/Test Split 85/15] → 227 train, 41 test
    ↓
[Preprocesamiento TF-IDF + Scalers]
    ↓
[Ensemble Stacking 4 modelos]
    ↓
[Evaluación] → 90.24% Accuracy
    ↓
financial_sentiment_best_model.joblib
```

## 🎯 Características Clave

| Aspecto | Valor |
|--------|-------|
| **Registros Usados** | 268 |
| **Criterio de Filtro** | ≥2 de 3 criterios |
| **Train/Test Split** | 227/41 (85/15) |
| **Algoritmo** | Stacking Ensemble |
| **Estimadores Base** | 4 (LR, RF, GB, SVM) |
| **Accuracy Test** | **90.24%** |
| **F1 Score** | ~90% |

## 🚀 Cómo Usar

```python
import joblib
from demo_model import predict_sentiment

model = joblib.load("financial_sentiment_best_model.joblib")
sentiment, confidence = predict_sentiment(
    model,
    "Stock market surges to record highs",
    index_change=2.5,
    sector="Technology"
)
print(f"Sentimiento: {sentiment}, Confianza: {confidence:.2%}")
```

## 💡 Ventajas y Desventajas

**✅ Ventajas:**
- Máxima precisión (90.24%)
- Datos de altísima calidad
- Menos overfitting
- Modelos confiables

**⚠️ Desventajas:**
- Solo 268 registros (menos datos)
- Mucho tiempo de filtrado
- Puede ser muy restrictivo

## 📝 Notas

- El filtro agresivo elimina 86% de los datos (1,852 → 268)
- Solo los datos que cumplen ≥2 criterios se usan
- El modelo es muy específico pero altamente confiable
