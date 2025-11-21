"""
Script de demostración del mejor modelo de sentimiento financiero
Muestra cómo usar el modelo entrenado para hacer predicciones
"""
import joblib
import pandas as pd
import numpy as np
import re


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s\-']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_features(headline: str) -> dict:
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


def prepare_input(headline, index_change=None, trading_volume=None, sector="unknown", impact_level="unknown"):
    """
    Prepara un headline para predicción
    
    Args:
        headline: Texto del titular financiero
        index_change: Cambio porcentual del índice (opcional)
        trading_volume: Volumen de trading (opcional)
        sector: Sector financiero (opcional)
        impact_level: Nivel de impacto (opcional)
    
    Returns:
        DataFrame listo para predicción
    """
    # Limpiar texto
    headline_clean = clean_text(headline)
    
    # Extraer características
    features = extract_features(headline)
    
    # Características numéricas adicionales
    headline_length = len(headline_clean)
    word_count = len(headline_clean.split())
    
    # Procesar index_change
    if index_change is not None:
        is_pos_change = 1 if index_change > 0 else 0
        is_neg_change = 1 if index_change < 0 else 0
        abs_change = abs(index_change)
    else:
        index_change = 0
        is_pos_change = 0
        is_neg_change = 0
        abs_change = 0
    
    # Procesar trading_volume
    if trading_volume is not None:
        log_volume = np.log1p(trading_volume)
    else:
        trading_volume = 0
        log_volume = 0
    
    # Crear DataFrame
    data = {
        'Headline_clean': [headline_clean],
        'ultra_pos': [features['ultra_pos']],
        'ultra_neg': [features['ultra_neg']],
        'pos': [features['pos']],
        'neg': [features['neg']],
        'total_pos': [features['total_pos']],
        'total_neg': [features['total_neg']],
        'score': [features['score']],
        'ratio': [features['ratio']],
        'has_ultra_pos': [features['has_ultra_pos']],
        'has_ultra_neg': [features['has_ultra_neg']],
        'has_percent': [features['has_percent']],
        'has_number': [features['has_number']],
        'Index_Change_Percent_num': [index_change],
        'Trading_Volume_num': [trading_volume],
        'log_volume': [log_volume],
        'headline_length': [headline_length],
        'word_count': [word_count],
        'is_pos_change': [is_pos_change],
        'is_neg_change': [is_neg_change],
        'abs_change': [abs_change],
        'Sector': [sector],
        'Impact_Level': [impact_level]
    }
    
    return pd.DataFrame(data)


def predict_sentiment(model, headline, index_change=None, trading_volume=None, sector="unknown", impact_level="unknown"):
    """
    Predice el sentimiento de un headline financiero
    
    Returns:
        tuple: (sentiment, confidence)
    """
    # Preparar input
    X = prepare_input(headline, index_change, trading_volume, sector, impact_level)
    
    # Predecir
    prediction = model.predict(X)[0]
    
    # Obtener probabilidades si está disponible
    try:
        probas = model.predict_proba(X)[0]
        confidence = max(probas)
    except:
        confidence = None
    
    return prediction, confidence


def main():
    print("="*70)
    print("DEMO: Modelo de Sentimiento Financiero (90.24% Accuracy)")
    print("="*70)
    
    # Cargar modelo
    print("\n[1/3] Cargando modelo...")
    try:
        model = joblib.load("financial_sentiment_best_model.joblib")
        print("   ✓ Modelo cargado exitosamente")
    except FileNotFoundError:
        print("   ❌ Error: No se encontró 'financial_sentiment_best_model.joblib'")
        print("   → Ejecuta primero: python train_best_model.py")
        return
    
    # Ejemplos de prueba
    print("\n[2/3] Probando con ejemplos...")
    
    test_cases = [
        {
            "headline": "Stock market surges to record highs on strong earnings",
            "index_change": 2.5,
            "sector": "Technology",
            "impact_level": "High"
        },
        {
            "headline": "Market crashes as investors flee amid economic fears",
            "index_change": -3.2,
            "sector": "Finance",
            "impact_level": "High"
        },
        {
            "headline": "Tech giant beats expectations with impressive quarterly results",
            "index_change": 1.8,
            "sector": "Technology",
            "impact_level": "Medium"
        },
        {
            "headline": "Banking sector faces headwinds from regulatory changes",
            "index_change": -1.5,
            "sector": "Finance",
            "impact_level": "Medium"
        },
        {
            "headline": "Company announces breakthrough in renewable energy technology",
            "index_change": 3.0,
            "sector": "Energy",
            "impact_level": "High"
        }
    ]
    
    print("\n" + "-"*70)
    for i, case in enumerate(test_cases, 1):
        headline = case["headline"]
        index_change = case.get("index_change")
        sector = case.get("sector", "unknown")
        impact_level = case.get("impact_level", "unknown")
        
        sentiment, confidence = predict_sentiment(
            model, headline, index_change, None, sector, impact_level
        )
        
        print(f"\nEjemplo {i}:")
        print(f"  Headline: {headline}")
        print(f"  Index Change: {index_change:+.1f}%" if index_change else "  Index Change: N/A")
        print(f"  Sector: {sector}")
        print(f"  → Predicción: {sentiment.upper()}")
        if confidence:
            print(f"  → Confianza: {confidence:.2%}")
        print("-"*70)
    
    # Modo interactivo
    print("\n[3/3] Modo interactivo")
    print("Ingresa tus propios headlines (o 'q' para salir)\n")
    
    while True:
        try:
            headline = input("Headline: ").strip()
            
            if headline.lower() in ['q', 'quit', 'exit', 'salir']:
                break
            
            if not headline:
                continue
            
            # Pedir datos opcionales
            try:
                index_str = input("Index Change % (Enter para omitir): ").strip()
                index_change = float(index_str) if index_str else None
            except:
                index_change = None
            
            sector = input("Sector (Enter para 'unknown'): ").strip() or "unknown"
            
            # Predecir
            sentiment, confidence = predict_sentiment(model, headline, index_change, None, sector)
            
            print(f"\n  → Predicción: {sentiment.upper()}")
            if confidence:
                print(f"  → Confianza: {confidence:.2%}")
            print()
            
        except KeyboardInterrupt:
            print("\n\nSaliendo...")
            break
        except Exception as e:
            print(f"  ❌ Error: {e}\n")
    
    print("\n" + "="*70)
    print("Demo completada")
    print("="*70)


if __name__ == "__main__":
    main()
