"""
Modelo con MÁXIMOS datos: Sin filtro de calidad
Usa todos los datos disponibles para maximizar el aprendizaje
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


# Palabras clave expandidas
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
    'approval', 'success', 'high', 'recovery', 'upward', 'strengthen', 'excite',
    'benefit', 'improve', 'win', 'achieve', 'prosper', 'thrive'
}

NEGATIVE = {
    'fall', 'drop', 'decline', 'sink', 'slide', 'loss', 'decrease', 'negative',
    'down', 'bearish', 'pessimistic', 'contraction', 'concern', 'worry', 'fear',
    'risk', 'threat', 'trouble', 'headwind', 'slowdown', 'weaken', 'dip', 'rattle',
    'hurt', 'damage', 'lose', 'struggle', 'suffer', 'deteriorate'
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


def main():
    print("="*70)
    print("MODELO CON MÁXIMOS DATOS: Sin Filtro de Calidad")
    print("="*70)
    
    # Cargar datos
    print("\n[1/6] Cargando datos...")
    df = pd.read_csv("financial_news_events_clean.csv")
    print(f"   Registros originales: {len(df)}")
    
    # Solo filtrar nulls y quedarse con positive/negative
    df = df.dropna(subset=["Sentiment", "Headline"])
    df["Sentiment"] = df["Sentiment"].str.lower()
    df = df[df["Sentiment"].isin(["positive", "negative"])].copy()
    print(f"   Registros positive/negative: {len(df)}")
    print(f"   - Positive: {len(df[df['Sentiment'] == 'positive'])}")
    print(f"   - Negative: {len(df[df['Sentiment'] == 'negative'])}")
    
    # Procesar características
    print("\n[2/6] Procesando características...")
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
    
    # Características adicionales
    df["headline_length"] = df["Headline_clean"].str.len()
    df["word_count"] = df["Headline_clean"].str.split().str.len()
    df["avg_word_length"] = df["Headline_clean"].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
    )
    df["is_pos_change"] = (df["Index_Change_Percent_num"] > 0).astype(int)
    df["is_neg_change"] = (df["Index_Change_Percent_num"] < 0).astype(int)
    df["abs_change"] = df["Index_Change_Percent_num"].abs()
    
    # Categóricas
    cat_cols = ["Sector", "Impact_Level", "Market_Event"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")
        else:
            df[col] = "unknown"
    
    # Solo eliminar headlines vacíos
    df = df[df["Headline_clean"].str.strip() != ""]
    
    print(f"\n[3/6] Sin filtro de calidad - usando TODOS los datos")
    print(f"   Total datos: {len(df)}")
    print(f"   - Positive: {len(df[df['Sentiment'] == 'positive'])}")
    print(f"   - Negative: {len(df[df['Sentiment'] == 'negative'])}")
    
    # Balancear clases (para evitar bias)
    min_class_size = df["Sentiment"].value_counts().min()
    df_balanced = pd.concat([
        df[df["Sentiment"] == "positive"].sample(n=min_class_size, random_state=42),
        df[df["Sentiment"] == "negative"].sample(n=min_class_size, random_state=42)
    ])
    df = df_balanced.copy()
    print(f"   Después de balancear: {len(df)}")
    
    # Preparar X e y
    print("\n[4/6] Preparando datos...")
    text_col = "Headline_clean"
    numeric_cols = [
        'ultra_pos', 'ultra_neg', 'pos', 'neg', 'total_pos', 'total_neg',
        'score', 'ratio', 'has_ultra_pos', 'has_ultra_neg', 'has_percent', 'has_number',
        "Index_Change_Percent_num", "Trading_Volume_num", "log_volume",
        "headline_length", "word_count", "avg_word_length",
        "is_pos_change", "is_neg_change", "abs_change"
    ]
    
    y = df["Sentiment"]
    X = df[[text_col] + numeric_cols + cat_cols]
    
    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Pipeline
    print("\n[5/6] Construyendo y entrenando modelo...")
    
    text_transformer = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.90,
        max_features=100000,
        sublinear_tf=True,
        use_idf=True,
        norm='l2'
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
    
    # Ensemble robusto
    lr = LogisticRegression(
        max_iter=3000, C=2.0, solver='saga',
        class_weight='balanced', n_jobs=-1, random_state=42
    )
    
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=25, min_samples_split=4,
        min_samples_leaf=2, max_features='sqrt', class_weight='balanced',
        n_jobs=-1, random_state=42
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=8,
        min_samples_split=4, min_samples_leaf=2, subsample=0.85,
        max_features='sqrt', random_state=42
    )
    
    svm = SVC(
        C=1.5, kernel='rbf', gamma='scale',
        class_weight='balanced', probability=True, random_state=42
    )
    
    stacking_clf = StackingClassifier(
        estimators=[
            ('lr', lr),
            ('rf', rf),
            ('gb', gb),
            ('svm', svm)
        ],
        final_estimator=LogisticRegression(C=1.5, max_iter=2000, random_state=42),
        cv=8,
        n_jobs=-1
    )
    
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", stacking_clf)
    ])
    
    print("   Entrenando (esto tomará varios minutos)...")
    model.fit(X_train, y_train)
    
    # Evaluación
    print("\n[6/6] Evaluando...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test, average="macro")
    
    print("\n" + "="*70)
    print("RESULTADOS MODELO CON MÁXIMOS DATOS")
    print("="*70)
    print(f"Datos usados: {len(df)}")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"\nAccuracy Train: {acc_train:.4f} ({acc_train*100:.2f}%)")
    print(f"Accuracy Test:  {acc_test:.4f} ({acc_test*100:.2f}%)")
    print(f"F1 Macro Test:  {f1_test:.4f}")
    
    if acc_test >= 0.80:
        print("\n✓ Buen resultado (>80%) con MÁXIMOS datos")
    
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_pred_test))
    
    print("\nMatriz de Confusión (Test):")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTrue Negatives: {tn}, False Positives: {fp}")
    print(f"False Negatives: {fn}, True Positives: {tp}")
    if (tp + fp) > 0:
        print(f"Precision Positive: {tp/(tp+fp):.4f}")
    if (tp + fn) > 0:
        print(f"Recall Positive: {tp/(tp+fn):.4f}")
    print("="*70)
    
    # Guardar
    model_path = "financial_sentiment_max_data_model.joblib"
    joblib.dump(model, model_path)
    print(f"\n✓ Modelo guardado: {model_path}")
    
    data_path = "financial_sentiment_ultimate_filtered_data.csv"
    df.to_csv(data_path, index=False)
    print(f"✓ Datos guardados: {data_path}")
    
    print("\n💡 Recomendación:")
    if acc_test >= 0.85:
        print(f"   ✓ Excelente: {len(df)} datos y {acc_test*100:.2f}% accuracy")
    elif acc_test >= 0.80:
        print(f"   ✓ Bueno: {acc_test*100:.2f}% accuracy con {len(df)} datos")
    else:
        print(f"   ⚠️  Accuracy {acc_test*100:.2f}% podría mejorarse con más refinamiento")


if __name__ == "__main__":
    main()
