"""
Modelo con Pseudo-Labeling - SIN FILTRO
Estrategia: Entrenar con datos limpios, predecir en datos ruidosos,
y re-entrenar con predicciones confiables
"""
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib


# Palabras clave
ULTRA_POSITIVE = {
    'surge', 'soar', 'rally', 'skyrocket', 'boom', 'breakthrough', 'record', 'milestone',
    'triumph', 'exceed', 'outperform', 'beat', 'strong', 'robust', 'impressive'
}

ULTRA_NEGATIVE = {
    'plunge', 'crash', 'collapse', 'tumble', 'plummet', 'slump', 'scandal', 'breach',
    'crisis', 'disaster', 'fail', 'miss', 'underperform', 'weak', 'disappointing'
}

POSITIVE = {
    'gain', 'rise', 'boost', 'climb', 'advance', 'profit', 'growth', 'increase',
    'positive', 'up', 'bullish', 'optimistic', 'expansion', 'solid', 'favorable',
    'approval', 'success', 'high', 'recovery', 'upward', 'strengthen', 'excite'
}

NEGATIVE = {
    'fall', 'drop', 'decline', 'sink', 'slide', 'loss', 'decrease', 'negative',
    'down', 'bearish', 'pessimistic', 'contraction', 'concern', 'worry', 'fear',
    'risk', 'threat', 'trouble', 'headwind', 'slowdown', 'weaken', 'dip', 'rattle'
}


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s\-']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_features(headline: str) -> dict:
    text = headline.lower()
    words = set(text.split())
    
    ultra_pos = len(words & ULTRA_POSITIVE)
    ultra_neg = len(words & ULTRA_NEGATIVE)
    pos = len(words & POSITIVE)
    neg = len(words & NEGATIVE)
    
    total_pos = ultra_pos * 3 + pos
    total_neg = ultra_neg * 3 + neg
    
    score = total_pos - total_neg
    ratio = total_pos / (total_neg + 1)
    
    return {
        'ultra_pos': ultra_pos,
        'ultra_neg': ultra_neg,
        'pos': pos,
        'neg': neg,
        'total_pos': total_pos,
        'total_neg': total_neg,
        'score': score,
        'ratio': ratio,
        'has_ultra_pos': 1 if ultra_pos > 0 else 0,
        'has_ultra_neg': 1 if ultra_neg > 0 else 0,
        'has_percent': 1 if '%' in headline or 'percent' in text else 0,
        'has_number': 1 if any(c.isdigit() for c in headline) else 0,
    }


def calculate_quality_score(df):
    """Calcula quality score sin filtrar"""
    headline_match = (
        ((df['score'] > 0) & (df['Sentiment'] == 'positive')) |
        ((df['score'] < 0) & (df['Sentiment'] == 'negative'))
    )
    
    index_match = (
        ((df['Index_Change_Percent_num'] > 0) & (df['Sentiment'] == 'positive')) |
        ((df['Index_Change_Percent_num'] < 0) & (df['Sentiment'] == 'negative')) |
        (df['Index_Change_Percent_num'].isna())
    )
    
    has_keyword = (df['pos'] >= 1) | (df['neg'] >= 1)
    
    quality_score = (
        headline_match.astype(int) +
        index_match.astype(int) +
        has_keyword.astype(int)
    )
    
    return quality_score


def main():
    print("="*70)
    print("MODELO CON PSEUDO-LABELING - SIN FILTRO")
    print("="*70)
    print("\nEstrategia:")
    print("1. Entrenar con datos de alta calidad")
    print("2. Predecir en datos de baja calidad")
    print("3. Agregar predicciones confiables al training set")
    print("4. Re-entrenar con dataset expandido")
    
    # Cargar datos
    print("\n[1/7] Cargando datos...")
    df = pd.read_csv("financial_news_events_clean.csv")
    df = df.dropna(subset=["Sentiment", "Headline"])
    df["Sentiment"] = df["Sentiment"].str.lower()
    df = df[df["Sentiment"].isin(["positive", "negative"])].copy()
    
    print(f"   Total datos: {len(df)}")
    
    # Procesar características
    print("\n[2/7] Procesando características...")
    df["Headline_clean"] = df["Headline"].apply(clean_text)
    
    features = df["Headline"].apply(extract_features)
    for key in ['ultra_pos', 'ultra_neg', 'pos', 'neg', 'total_pos', 'total_neg',
                'score', 'ratio', 'has_ultra_pos', 'has_ultra_neg', 'has_percent', 'has_number']:
        df[key] = features.apply(lambda x: x[key])
    
    # Index_Change
    if "Index_Change_Percent" in df.columns:
        df["Index_Change_Percent_num"] = pd.to_numeric(
            df["Index_Change_Percent"].astype(str).str.replace("%", "").str.replace(",", ""),
            errors='coerce'
        )
    else:
        df["Index_Change_Percent_num"] = np.nan
    
    # Trading Volume
    if "Trading_Volume" in df.columns:
        df["Trading_Volume_num"] = pd.to_numeric(
            df["Trading_Volume"].astype(str).str.replace(",", ""),
            errors='coerce'
        )
        df["log_volume"] = np.log1p(df["Trading_Volume_num"].fillna(0))
    else:
        df["Trading_Volume_num"] = 0
        df["log_volume"] = 0
    
    df["headline_length"] = df["Headline_clean"].str.len()
    df["word_count"] = df["Headline_clean"].str.split().str.len()
    df["is_pos_change"] = (df["Index_Change_Percent_num"] > 0).astype(int)
    df["is_neg_change"] = (df["Index_Change_Percent_num"] < 0).astype(int)
    df["abs_change"] = df["Index_Change_Percent_num"].abs()
    
    # Categóricas
    cat_cols = ["Sector", "Impact_Level"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")
        else:
            df[col] = "unknown"
    
    df = df[df["Headline_clean"].str.strip() != ""]
    
    # Calcular quality score
    df["quality_score"] = calculate_quality_score(df)
    
    print(f"\n[3/7] Separando datos por calidad...")
    # Datos de alta calidad (≥2 criterios)
    df_high_quality = df[df["quality_score"] >= 2].copy()
    # Datos de baja calidad (<2 criterios)
    df_low_quality = df[df["quality_score"] < 2].copy()
    
    print(f"   Alta calidad (≥2 criterios): {len(df_high_quality)}")
    print(f"   Baja calidad (<2 criterios): {len(df_low_quality)}")
    
    # Balancear datos de alta calidad
    min_class = df_high_quality["Sentiment"].value_counts().min()
    df_high_balanced = pd.concat([
        df_high_quality[df_high_quality["Sentiment"] == "positive"].sample(n=min_class, random_state=42),
        df_high_quality[df_high_quality["Sentiment"] == "negative"].sample(n=min_class, random_state=42)
    ])
    
    # Preparar features
    text_col = "Headline_clean"
    numeric_cols = [
        'ultra_pos', 'ultra_neg', 'pos', 'neg', 'total_pos', 'total_neg',
        'score', 'ratio', 'has_ultra_pos', 'has_ultra_neg', 'has_percent', 'has_number',
        "Index_Change_Percent_num", "Trading_Volume_num", "log_volume",
        "headline_length", "word_count", "is_pos_change", "is_neg_change", "abs_change"
    ]
    
    # FASE 1: Entrenar con datos de alta calidad
    print("\n[4/7] FASE 1: Entrenando con datos de alta calidad...")
    
    y_high = df_high_balanced["Sentiment"]
    X_high = df_high_balanced[[text_col] + numeric_cols + cat_cols]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_high, y_high, test_size=0.2, random_state=42, stratify=y_high
    )
    
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Pipeline
    text_transformer = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=1,
        max_df=0.85,
        max_features=50000,
        sublinear_tf=True
    )
    
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_transformer, text_col),
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )
    
    # Modelo inicial
    lr = LogisticRegression(max_iter=3000, C=2.0, solver='saga', class_weight='balanced', n_jobs=-1, random_state=42)
    rf = RandomForestClassifier(n_estimators=300, max_depth=25, class_weight='balanced', n_jobs=-1, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=8, random_state=42)
    
    stacking_clf = StackingClassifier(
        estimators=[('lr', lr), ('rf', rf), ('gb', gb)],
        final_estimator=LogisticRegression(C=1.5, max_iter=2000, random_state=42),
        cv=5,
        n_jobs=-1
    )
    
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", stacking_clf)
    ])
    
    print("   Entrenando modelo inicial...")
    model.fit(X_train, y_train)
    
    # Evaluar modelo inicial
    y_pred_test = model.predict(X_test)
    acc_initial = accuracy_score(y_test, y_pred_test)
    print(f"   Accuracy inicial (solo datos limpios): {acc_initial:.4f} ({acc_initial*100:.2f}%)")
    
    # FASE 2: Pseudo-labeling en datos de baja calidad
    print("\n[5/7] FASE 2: Aplicando pseudo-labeling...")
    
    if len(df_low_quality) > 0:
        X_low = df_low_quality[[text_col] + numeric_cols + cat_cols]
        
        # Predecir con probabilidades
        print("   Prediciendo en datos de baja calidad...")
        y_low_pred_proba = model.predict_proba(X_low)
        y_low_pred = model.predict(X_low)
        
        # Seleccionar solo predicciones confiables (probabilidad > 0.8)
        confidence_threshold = 0.8
        max_proba = np.max(y_low_pred_proba, axis=1)
        confident_mask = max_proba >= confidence_threshold
        
        print(f"   Predicciones confiables (prob ≥ {confidence_threshold}): {confident_mask.sum()}")
        
        if confident_mask.sum() > 0:
            # Agregar predicciones confiables al training set
            df_pseudo = df_low_quality[confident_mask].copy()
            df_pseudo["Sentiment"] = y_low_pred[confident_mask]
            
            # Combinar datos originales + pseudo-labels
            df_combined = pd.concat([df_high_balanced, df_pseudo])
            
            print(f"   Dataset expandido: {len(df_high_balanced)} → {len(df_combined)}")
            
            # FASE 3: Re-entrenar con dataset expandido
            print("\n[6/7] FASE 3: Re-entrenando con pseudo-labels...")
            
            y_combined = df_combined["Sentiment"]
            X_combined = df_combined[[text_col] + numeric_cols + cat_cols]
            
            X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
                X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
            )
            
            print("   Re-entrenando modelo...")
            model.fit(X_train_new, y_train_new)
            
            # Evaluar modelo final
            y_pred_final = model.predict(X_test_new)
            acc_final = accuracy_score(y_test_new, y_pred_final)
            f1_final = f1_score(y_test_new, y_pred_final, average='macro')
            
            print("\n[7/7] Evaluando modelo final...")
            
            print("\n" + "="*70)
            print("RESULTADOS: PSEUDO-LABELING (SIN FILTRO)")
            print("="*70)
            print(f"Datos iniciales (alta calidad): {len(df_high_balanced)}")
            print(f"Pseudo-labels agregados: {confident_mask.sum()}")
            print(f"Dataset final: {len(df_combined)}")
            print(f"\nAccuracy inicial: {acc_initial:.4f} ({acc_initial*100:.2f}%)")
            print(f"Accuracy final:   {acc_final:.4f} ({acc_final*100:.2f}%)")
            print(f"Mejora: {(acc_final - acc_initial)*100:+.2f}%")
            print(f"\nF1 Macro: {f1_final:.4f}")
            
            print("\nClassification Report:")
            print(classification_report(y_test_new, y_pred_final))
            
            print("\nMatriz de Confusión:")
            cm = confusion_matrix(y_test_new, y_pred_final)
            print(cm)
            
            # Guardar
            model_path = "financial_sentiment_pseudo_labeling.joblib"
            joblib.dump(model, model_path)
            print(f"\n✓ Modelo guardado: {model_path}")
            
            print("\n" + "="*70)
            print("RESUMEN")
            print("="*70)
            print(f"{'Modelo':<35} {'Datos':<10} {'Accuracy':<10}")
            print("-"*70)
            print(f"{'Solo datos limpios':<35} {len(df_high_balanced):<10} {f'{acc_initial*100:.2f}%':<10}")
            print(f"{'Con pseudo-labeling':<35} {len(df_combined):<10} {f'{acc_final*100:.2f}%':<10}")
            print("="*70)
            
        else:
            print("   ⚠️  No hay predicciones confiables suficientes")
    else:
        print("   ⚠️  No hay datos de baja calidad")


if __name__ == "__main__":
    main()
