# Comparación de Modelos

Resumen
------
Este documento resume las diferencias entre los tres enfoques implementados en el repositorio:

- **Ultra-filtrado**: Entrena solo con los ejemplos de mayor calidad (filtro agresivo). Menos datos, mayor precisión.
- **Balanceado**: Filtrado moderado y balanceo de clases. Más datos y mejor generalización.
- **Máximos datos**: Usa todos los datos limpios disponibles sin filtros de calidad.

Tabla resumida (valores de ejemplo en este repo)
-----------------------------------------------
| Modelo           | Datos usados | Train | Test | Accuracy aproximada |
|------------------|--------------:|------:|-----:|---------------------:|
| Ultra-filtrado   | 268          | 227   | 41   | ~90.24%             |
| Balanceado       | 1,308        | 1,046 | 262  | ~84.73%             |
| Máximos datos    | 1,852        | 1,481 | 371  | ~67.65%             |

Interpretación
--------------
- El `Ultra-filtrado` prioriza calidad sobre cantidad; útil cuando la precisión es crítica.
- El `Balanceado` busca un compromiso útil para la mayoría de casos de producción.
- `Máximos datos` es apropiado para exploración y cuando se busca capturar la mayor variabilidad posible.

Recomendación de uso
---------------------
- Para pruebas rápidas o despliegue con alta fiabilidad: `Ultra-filtrado`.
- Para entrenamiento de modelos que deben generalizar a muchos casos: `Balanceado`.
- Para investigación y extracción de patrones raros: `Máximos datos`.

Cómo reproducir la comparación
1. Ejecuta cada script de entrenamiento por separado:
   - `train_best_model.py`
   - `train_balanced_model.py`
   - `train_max_data_model.py`
2. Anota los tamaños de `df`, `len(X_train)` y `len(X_test)` que imprime cada script y la `accuracy` final.
3. Completa la tabla con los valores resultantes.

Nota técnica
Nota técnica
------------
- Cada script es autónomo: no deben contener referencias cruzadas ni suposiciones sobre la existencia de los otros modelos. Se han modificado los scripts para cumplir esto.

Comparación detallada por código
--------------------------------
Cada modelo imprime durante su ejecución los tamaños de `df`, `len(X_train)` y `len(X_test)` y la `accuracy` final. Para reproducir la comparación exacta programáticamente sigue estos pasos:

1. Ejecuta cada script (en este orden si quieres reproducir los valores del README):
    - `train_best_model.py`  (ultra-filtrado)
    - `train_balanced_model.py` (balanceado)
    - `train_max_data_model.py` (máximos datos)

2. Captura los prints o modifica los scripts para que al final guarden un CSV resumen con columnas: `model_name, n_data, train_size, test_size, acc_test`.

Ejemplo rápido (pseudocódigo para crear el CSV comparativo):

```python
summary = []
# ejecutar cada script o invocar sus funciones internas
summary.append({'model_name': 'ultra', 'n_data': 268, 'train':227, 'test':41, 'acc_test':0.9024})
summary.append({'model_name': 'balanced', 'n_data': 1308, 'train':1046, 'test':262, 'acc_test':0.8473})
summary.append({'model_name': 'max', 'n_data': 1852, 'train':1481, 'test':371, 'acc_test':0.6765})
import pandas as pd
pd.DataFrame(summary).to_csv('models_comparison.csv', index=False)
```

Interpretación y recomendaciones (al nivel del código):
- El `ultra-filtrado` implementa su propio filtro (`quality_score >= 2`) antes de construir el pipeline — busca alta confianza en las etiquetas.  
- El `balanceado` aplica un filtrado moderado y luego balancea clases por muestreo; esto requiere `random_state` para reproducibilidad.  
- El `máximos datos` evita filtros de calidad y por tanto requerirá mayor regularización en los estimadores para evitar overfitting.

