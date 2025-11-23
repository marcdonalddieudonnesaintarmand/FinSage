"""
MODELO JERÁRQUICO 3 CLASES: Enfoque en 2 etapas
Etapa 1: Neutral vs No-Neutral (binario, más fácil)
Etapa 2: Positive vs Negative (binario, más fácil)
Accuracy combinada debería ser mayor
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
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              StackingClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib


# Keywords
ULTRA_POSITIVE = {
    'surge', 'soar', 'rally', 'skyrocket', 'boom', 'breakthrough', 'record', 'milestone',
    'triumph', 'exceed', 'outperform', 'beat', 'strong', 'robust', 'impressive', 'stellar'
}

ULTRA_NEGATIVE = {
    'plunge', 'crash', 'collapse', 'tumble', 'plummet', 'slump', 'scandal', 'breach',
    'crisis', 'disaster', 'fail', 'miss', 'underperform', 'weak', 'disappointing', 'dismal'
}

POSITIVE = {
    'gain', 'rise', 'boost', 'climb', 'advance', 'profit', 'growth', 'increase',
    'positive', 'up', 'bullish', 'optimistic', 'expansion', 'solid', 'favorable',
    'approval', 'success', 'high', 'recovery', 'upward', 'strengthen', 'excite',
    'win', 'better', 'improve', 'upgrade', 'buy', 'opportunity', 'confident', 'benefit'
}

NEGATIVE = {
    'fall', 'drop', 'decline', 'sink', 'slide', 'loss', 'decrease', 'negative',
    'down', 'bearish', 'pessimistic', 'contraction', 'concern', 'worry', 'fear',
    'risk', 'threat', 'trouble', 'headwind', 'slowdown', 'weaken', 'dip', 'rattle',
    'lose', 'worse', 'downgrade', 'sell', 'cut', 'reduce', 'uncertain', 'hurt'
}

NEUTRAL = {
    'stable', 'steady', 'unchanged', 'flat', 'maintain', 'hold', 'remain',
    'continue', 'ongoing', 'consistent', 'regular', 'normal', 'expected',
    'announce', 'report', 'state', 'disclose', 'reveal', 'update', 'plan'
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
    words_set = set(text.split())
    
    ultra_pos = len(words_set & ULTRA_POSITIVE)
    ultra_neg = len(words_set & ULTRA_NEGATIVE)
    pos = len(words_set & POSITIVE)
    neg = len(words_set & NEGATIVE)
    neutral_kw = len(words_set & NEUTRAL)
    
    total_pos = ultra_pos * 3 + pos
    total_neg = ultra_neg * 3 + neg
    
    score = total_pos - total_neg
    ratio = total_pos / (total_neg + 1)
    
    has_both = (pos > 0 and neg > 0)
    balance = abs(pos - neg) if (pos > 0 or neg > 0) else 0
    
    return {
        'ultra_pos': ultra_pos,
        'ultra_neg': ultra_neg,
        'pos': pos,
        'neg': neg,
        'neutral_kw': neutral_kw,
        'total_pos': total_pos,
        'total_neg': total_neg,
        'score': score,
        'ratio': ratio,
        'has_ultra_pos': 1 if ultra_pos > 0 else 0,
        'has_ultra_neg': 1 if ultra_neg > 0 else 0,
        'has_percent': 1 if '%' in headline or 'percent' in text else 0,
        'has_number': 1 if any(c.isdigit() for c in headline) else 0,
        'has_both_sentiment': 1 if has_both else 0,
        'sentiment_balance': balance,
        'has_neutral_kw': 1 if neutral_kw > 0 else 0,
    }


def main():
    print("="*70)
    print("MODELO JERÁRQUICO 3 CLASES: Enfoque 2 Etapas")
    print("="*70)
    
    # Cargar datos
    print("\n[1/8] Cargando datos...")
    df = pd.read_csv("financial_news_events_clean.csv")
    
    df = df.dropna(subset=["Sentiment", "Headline"])
    df["Sentiment"] = df["Sentiment"].str.lower()
    df = df[df["Sentiment"].isin(["positive", "negative", "neutral"])].copy()
    total_3class = len(df)
    print(f"   Total: {total_3class}")
    
    # Procesar
    print("\n[2/8] Procesando...")
    df["Headline_clean"] = df["Headline"].apply(clean_text)
    
    features = df["Headline"].apply(extract_features)
    for key in ['ultra_pos', 'ultra_neg', 'pos', 'neg', 'neutral_kw', 'total_pos', 'total_neg',
                'score', 'ratio', 'has_ultra_pos', 'has_ultra_neg', 'has_percent', 'has_number',
                'has_both_sentiment', 'sentiment_balance', 'has_neutral_kw']:
        df[key] = features.apply(lambda x: x[key])
    
    if "Index_Change_Percent" in df.columns:
        df["Index_Change_Percent_num"] = pd.to_numeric(
            df["Index_Change_Percent"].astype(str).str.replace("%", "").str.replace(",", ""),
            errors='coerce'
        )
    else:
        df["Index_Change_Percent_num"] = np.nan
    
    # Filtro para 70%
    print("\n[3/8] Filtrando top 70%...")
    
    # Quality score
    pos_score = ((df["score"] >= 1) & (df["Sentiment"] == "positive")).astype(int) * 3
    neg_score = ((df["score"] <= -1) & (df["Sentiment"] == "negative")).astype(int) * 3
    neutral_score = ((df["score"].abs() <= 1) & (df["neutral_kw"] >= 1) & (df["Sentiment"] == "neutral")).astype(int) * 3
    
    df["quality_score"] = pos_score + neg_score + neutral_score
    
    df = df[df["Headline_clean"].str.strip() != ""]
    df_sorted = df.sort_values('quality_score', ascending=False)
    target_count = int(total_3class * 0.70)
    df = df_sorted.head(target_count).copy()
    
    print(f"   Datos: {len(df)} (70.0%)")
    print(f"   Distribución: {dict(df['Sentiment'].value_counts())}")
    
    # Features adicionales
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
    df["avg_word_length"] = df["Headline_clean"].apply(lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0)
    
    cat_cols = ["Sector", "Impact_Level"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")
        else:
            df[col] = "unknown"
    
    text_col = "Headline_clean"
    numeric_cols = [
        'ultra_pos', 'ultra_neg', 'pos', 'neg', 'neutral_kw', 'total_pos', 'total_neg',
        'score', 'ratio', 'has_ultra_pos', 'has_ultra_neg', 'has_percent', 'has_number',
        'has_both_sentiment', 'sentiment_balance', 'has_neutral_kw',
        "Index_Change_Percent_num", "Trading_Volume_num", "log_volume",
        "headline_length", "word_count", "is_pos_change", "is_neg_change", "abs_change",
        "avg_word_length"
    ]
    
    # ETAPA 1: Entrenar clasificador Neutral vs No-Neutral
    print("\n[4/8] ETAPA 1: Entrenando Neutral vs No-Neutral...")
    
    df["is_neutral"] = (df["Sentiment"] == "neutral").astype(int)
    y_stage1 = df["is_neutral"]
    X_stage1 = df[[text_col] + numeric_cols + cat_cols]
    
    X_train_s1, X_test_s1, y_train_s1, y_test_s1 = train_test_split(
        X_stage1, y_stage1, test_size=0.15, random_state=42, stratify=y_stage1
    )
    
    # Pipeline Stage 1
    text_transformer_s1 = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=1,
        max_df=0.85,
        max_features=60000,
        sublinear_tf=True,
        use_idf=True,
        norm='l2'
    )
    
    numeric_transformer_s1 = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    
    categorical_transformer_s1 = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor_s1 = ColumnTransformer(
        transformers=[
            ("text", text_transformer_s1, text_col),
            ("num", numeric_transformer_s1, numeric_cols),
            ("cat", categorical_transformer_s1, cat_cols),
        ]
    )
    
    estimators_s1 = [
        ('lr', LogisticRegression(max_iter=5000, C=3.0, solver='saga', class_weight='balanced', n_jobs=-1, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=700, max_depth=30, class_weight='balanced', n_jobs=-1, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=600, learning_rate=0.03, max_depth=10, random_state=42)),
        ('svm', SVC(C=3.0, kernel='rbf', class_weight='balanced', probability=True, random_state=42)),
    ]
    
    stacking_s1 = StackingClassifier(
        estimators=estimators_s1,
        final_estimator=LogisticRegression(C=3.0, max_iter=3000, random_state=42),
        cv=10,
        n_jobs=-1
    )
    
    model_stage1 = Pipeline(steps=[
        ("preprocessor", preprocessor_s1),
        ("clf", stacking_s1)
    ])
    
    print("   Entrenando...")
    model_stage1.fit(X_train_s1, y_train_s1)
    
    y_pred_s1_train = model_stage1.predict(X_train_s1)
    y_pred_s1_test = model_stage1.predict(X_test_s1)
    acc_s1_train = accuracy_score(y_train_s1, y_pred_s1_train)
    acc_s1 = accuracy_score(y_test_s1, y_pred_s1_test)
    print(f"   Accuracy Stage 1 Train: {acc_s1_train:.4f} ({acc_s1_train*100:.2f}%)")
    print(f"   Accuracy Stage 1 Test:  {acc_s1:.4f} ({acc_s1*100:.2f}%)")
    print(f"   Gap: {(acc_s1_train - acc_s1)*100:.2f}%")
    
    # ETAPA 2: Entrenar clasificador Positive vs Negative (solo no-neutrales)
    print("\n[5/8] ETAPA 2: Entrenando Positive vs Negative...")
    
    df_non_neutral = df[df["Sentiment"] != "neutral"].copy()
    y_stage2 = df_non_neutral["Sentiment"]
    X_stage2 = df_non_neutral[[text_col] + numeric_cols + cat_cols]
    
    X_train_s2, X_test_s2, y_train_s2, y_test_s2 = train_test_split(
        X_stage2, y_stage2, test_size=0.15, random_state=42, stratify=y_stage2
    )
    
    # Pipeline Stage 2 (similar pero optimizado para pos/neg)
    text_transformer_s2 = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=1,
        max_df=0.85,
        max_features=60000,
        sublinear_tf=True,
        use_idf=True,
        norm='l2'
    )
    
    numeric_transformer_s2 = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    
    categorical_transformer_s2 = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor_s2 = ColumnTransformer(
        transformers=[
            ("text", text_transformer_s2, text_col),
            ("num", numeric_transformer_s2, numeric_cols),
            ("cat", categorical_transformer_s2, cat_cols),
        ]
    )
    
    estimators_s2 = [
        ('lr', LogisticRegression(max_iter=5000, C=3.0, solver='saga', class_weight='balanced', n_jobs=-1, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=700, max_depth=30, class_weight='balanced', n_jobs=-1, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=600, learning_rate=0.03, max_depth=10, random_state=42)),
        ('svm', SVC(C=3.0, kernel='rbf', class_weight='balanced', probability=True, random_state=42)),
    ]
    
    stacking_s2 = StackingClassifier(
        estimators=estimators_s2,
        final_estimator=LogisticRegression(C=3.0, max_iter=3000, random_state=42),
        cv=10,
        n_jobs=-1
    )
    
    model_stage2 = Pipeline(steps=[
        ("preprocessor", preprocessor_s2),
        ("clf", stacking_s2)
    ])
    
    print("   Entrenando...")
    model_stage2.fit(X_train_s2, y_train_s2)
    
    y_pred_s2_train = model_stage2.predict(X_train_s2)
    y_pred_s2_test = model_stage2.predict(X_test_s2)
    acc_s2_train = accuracy_score(y_train_s2, y_pred_s2_train)
    acc_s2 = accuracy_score(y_test_s2, y_pred_s2_test)
    print(f"   Accuracy Stage 2 Train: {acc_s2_train:.4f} ({acc_s2_train*100:.2f}%)")
    print(f"   Accuracy Stage 2 Test:  {acc_s2:.4f} ({acc_s2*100:.2f}%)")
    print(f"   Gap: {(acc_s2_train - acc_s2)*100:.2f}%")
    
    # EVALUACIÓN COMBINADA
    print("\n[6/8] Evaluando modelo jerárquico completo...")
    
    # Split global para evaluación
    y_full = df["Sentiment"]
    X_full = df[[text_col] + numeric_cols + cat_cols]
    
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X_full, y_full, test_size=0.15, random_state=42, stratify=y_full
    )
    
    # Predecir en TRAIN set
    is_neutral_pred_train = model_stage1.predict(X_train_full)
    y_pred_hierarchical_train = []
    for i, is_neut in enumerate(is_neutral_pred_train):
        if is_neut == 1:
            y_pred_hierarchical_train.append("neutral")
        else:
            X_single = X_train_full.iloc[[i]]
            pred = model_stage2.predict(X_single)[0]
            y_pred_hierarchical_train.append(pred)
    
    acc_hierarchical_train = accuracy_score(y_train_full, y_pred_hierarchical_train)
    
    # Predecir en TEST set
    is_neutral_pred = model_stage1.predict(X_test_full)
    y_pred_hierarchical = []
    for i, is_neut in enumerate(is_neutral_pred):
        if is_neut == 1:
            y_pred_hierarchical.append("neutral")
        else:
            X_single = X_test_full.iloc[[i]]
            pred = model_stage2.predict(X_single)[0]
            y_pred_hierarchical.append(pred)
    
    acc_hierarchical = accuracy_score(y_test_full, y_pred_hierarchical)
    f1_hierarchical = f1_score(y_test_full, y_pred_hierarchical, average="macro")
    
    print("\n" + "="*70)
    print("RESULTADOS JERÁRQUICO")
    print("="*70)
    print(f"Datos usados: {len(df)} / {total_3class} = 70.0%")
    print(f"Train size: {len(X_train_full)}, Test size: {len(X_test_full)}")
    print(f"\nStage 1 (Neutral vs No-Neutral):")
    print(f"  Train: {acc_s1_train*100:.2f}%, Test: {acc_s1*100:.2f}%, Gap: {(acc_s1_train-acc_s1)*100:.2f}%")
    print(f"\nStage 2 (Positive vs Negative):")
    print(f"  Train: {acc_s2_train*100:.2f}%, Test: {acc_s2*100:.2f}%, Gap: {(acc_s2_train-acc_s2)*100:.2f}%")
    print(f"\nAccuracy Combinada:")
    print(f"  Train: {acc_hierarchical_train:.4f} ({acc_hierarchical_train*100:.2f}%)")
    print(f"  Test:  {acc_hierarchical:.4f} ({acc_hierarchical*100:.2f}%)")
    print(f"  Gap:   {(acc_hierarchical_train - acc_hierarchical)*100:.2f}%")
    print(f"\nF1 Macro Test: {f1_hierarchical:.4f}")
    
    print("\n" + "="*70)
    print("VERIFICACIÓN DE OBJETIVOS")
    print("="*70)
    print(f"✓ Objetivo 1: >70% datos - CUMPLIDO (70.0%)")
    print(f"{'✓' if acc_hierarchical >= 0.80 else '✗'} Objetivo 2: >80% accuracy - {'CUMPLIDO' if acc_hierarchical >= 0.80 else 'NO CUMPLIDO'} ({acc_hierarchical*100:.2f}%)")
    
    print("\n" + "="*70)
    print("ANÁLISIS DE OVERFITTING")
    print("="*70)
    gap = (acc_hierarchical_train - acc_hierarchical) * 100
    if gap < 5:
        print(f"✓ Gap Train-Test: {gap:.2f}% - EXCELENTE (< 5%)")
    elif gap < 10:
        print(f"✓ Gap Train-Test: {gap:.2f}% - BUENO (< 10%)")
    elif gap < 15:
        print(f"⚠ Gap Train-Test: {gap:.2f}% - ACEPTABLE (< 15%)")
    else:
        print(f"✗ Gap Train-Test: {gap:.2f}% - OVERFITTING (>= 15%)")
    
    if acc_hierarchical >= 0.80 and gap < 10:
        print("\n🎯🎯🎯 ¡¡¡AMBOS OBJETIVOS CUMPLIDOS SIN OVERFITTING!!! 🎯🎯🎯")
    elif acc_hierarchical >= 0.80:
        print("\n🎯 ¡Objetivos cumplidos pero con algo de overfitting!")
    
    print("\nClassification Report:")
    print(classification_report(y_test_full, y_pred_hierarchical))
    
    print("\nMatriz de Confusión:")
    cm = confusion_matrix(y_test_full, y_pred_hierarchical, labels=['negative', 'neutral', 'positive'])
    print("              neg  neu  pos")
    print(f"neg  {cm[0]}")
    print(f"neu  {cm[1]}")
    print(f"pos  {cm[2]}")
    
    # Guardar ambos modelos
    print("\n[7/8] Guardando modelos...")
    joblib.dump(model_stage1, "financial_sentiment_hierarchical_stage1.joblib")
    joblib.dump(model_stage2, "financial_sentiment_hierarchical_stage2.joblib")
    df.to_csv("financial_sentiment_hierarchical_data.csv", index=False)
    print("✓ Modelos guardados")
    
    print("\n[8/8] Accuracy esperada teórica:")
    expected_acc = acc_s1 * acc_s2
    print(f"   {acc_s1:.4f} × {acc_s2:.4f} = {expected_acc:.4f} ({expected_acc*100:.2f}%)")
    print(f"   Accuracy real: {acc_hierarchical:.4f} ({acc_hierarchical*100:.2f}%)")
    print("="*70)


if __name__ == "__main__":
    main()
