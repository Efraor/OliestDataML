# Olist Data Science Challenge – Penguin Hatch

Proyecto end-to-end para la resolución del desafío de ciencia de datos de Olist. El objetivo es mejorar la experiencia de los clientes mediante analítica y modelos reproducibles. En esta implementación se desarrollaron tres de los casos propuestos (estimación de tiempos de entrega, recomendaciones de productos y análisis de sentimientos), cada uno documentado en un notebook dedicado dentro de `notebooks/`.

## 1. Contexto del desafío

- **Escenario**: rol de pasante en Olist con acceso al dataset público de e-commerce brasileño.
- **Misión**: construir soluciones accionables que permitan anticipar problemas logísticos, sugerir artículos relevantes y entender la voz del cliente.
- **Alcance mínimo**: notebook por caso, comparación de múltiples modelos clásicos de `scikit-learn`, separación explícita de entrenamiento/validación/test y visualizaciones interpretables.

## 2. Estructura del repositorio

```
.
├── data/                  # CSV originales (raw) y derivados (processed)
├── models/                # Artefactos entrenados listos para desplegar
├── notebooks/             # Series 00–03 con experimentos por caso
├── outputs/               # Tablas o reportes exportados
├── reports/               # Figuras e informes listos para stakeholders
├── src/                   # Código reutilizable (config, paths, utils)
├── requirements.txt       # Dependencias mínimas del entorno
└── README.md              # Este documento
```

Repos clave:
- `src/config.py`: semillas y tamaños de split.
- `src/paths.py`: rutas absolutas para datos, modelos y reportes.
- `src/utils/seeds.py`: helper para hacer reproducible cualquier notebook/script.

## 3. Datos

Los CSV originales viven en `data/raw/` (copiados del dataset “Brazilian E-Commerce Public Dataset by Olist” de Kaggle). Los principales archivos utilizados son:

- `olist_orders_dataset.csv`: tiempos de compra, aprobación, despacho y entrega.
- `olist_order_items_dataset.csv` + `olist_products_dataset.csv`: detalle de ítems, precios y atributos físicos.
- `olist_order_reviews_dataset.csv`: reseñas textuales con calificaciones 1–5.
- Tablas auxiliares: `olist_customers_dataset.csv`, `olist_sellers_dataset.csv`, `olist_geolocation_dataset.csv` y diccionario de categorías.

Las transformaciones intermedias (limpieza, features y atributos clusterizados) se documentan en cada notebook y pueden guardarse en `data/processed/` cuando se necesite persistirlas.

## 4. Configuración rápida

1. **Crear entorno** (Python ≥3.10 recomendado):
   ```powershell
   python -m venv .venv
   .\\.venv\\Scripts\\activate
   ```
2. **Instalar dependencias**:
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install scikit-learn==1.5.2 joblib==1.4.2
   ```
3. **Levantar Jupyter**:
   ```powershell
   jupyter lab  # o jupyter notebook
   ```
4. **Ejecutar** los notebooks en el orden `00_setup.ipynb` → casos particulares (`01`, `02`, `03`). Cada notebook detecta la raíz del proyecto automáticamente gracias a `src/paths.py`.

## 5. Casos implementados

### 5.1 Estimación de tiempos de entrega (ETA) — `notebooks/01_case_eta.ipynb`

- **Target**: `delivery_days`, días entre la compra y la entrega al cliente (rangos 1–46, media 11.6 sobre 95 563 pedidos limpios).
- **Features**:
  - Duraciones derivadas: `approval_delay_h`, `estimated_days`, `carrier_delay_h`.
  - Calendario: día de la semana/hora/mes de compra.
- **Metodología**: `train_test_split` (80/20, `SEED=42`), estandarización mínima donde aplica y evaluación con cross-validation (5 folds) sobre MAE.
- **Modelos probados**:

| Modelo               | MAE CV (↓) |
|----------------------|-----------:|
| Gradient Boosting    | **4.27**   |
| Random Forest        | 4.39       |
| Linear Regression    | 4.60       |
| K-Nearest Neighbors  | 4.66       |
| Decision Tree        | 5.95       |

- **Performance hold-out** (Gradient Boosting): MAE 4.27 días, RMSE 6.08, R² 0.40. El error aumenta en colas (pedidos extremadamente rápidos o lentos), por lo que se marcaron estos casos como candidatos a reglas de negocio adicionales.
- **Entregable**: tabla de pedidos en riesgo de retraso (predicción − estimación oficial > 3 días) para el equipo de operaciones.

### 5.2 Recomendador de productos — `notebooks/02_case_recomendations.ipynb`

- **Objetivo**: ofrecer sugerencias “antes de que el cliente note que no las necesita”, agrupando productos similares por atributos físicos/calidad.
- **Preparación**:
  - Merge entre `order_items`, `products`, traducción de categorías y puntajes medios de review.
  - Imputación con medianas para medidas físicas y score neutral (=3.0) si el producto no recibió reseñas.
  - Agrupación de las 10 categorías más vendidas (`cat_top`) + etiqueta `other`.
  - Escalado de variables numéricas (`StandardScaler`) + one-hot de categorías → matriz final `X` de 17 columnas.
- **Modelado**: `KMeans` con `k ∈ [2,11]`. Se escogió **k = 5** (elbow + silhouette), `silhouette = 0.2152`.
- **Insights de clusters**:
  - `Cluster 1` (45% del catálogo): perfumería/computación con altos puntajes (avg 4.6) y productos compactos.
  - `Cluster 0`: deportes/hogar con pesos medios y reviews estables (avg 4.3).
  - `Cluster 2`: categorías con insatisfacción recurrente (avg 1.94) → candidatos a auditorías de calidad.
  - `Cluster 3`: artículos voluminosos (>3 kg) que requieren revisar costos logísticos.
  - `Cluster 4`: mobiliario de oficina y decoración pesada (altura promedio 47 cm), nicho pero con tickets altos.
- **Motor de recomendación**: función `recommend_from_product(product_id, top_n)` que busca el cluster del producto base y retorna el top-N de ese grupo ordenado por `avg_review_score`, filtrando el mismo SKU. Permite integrarse rápidamente en APIs o dashboards.

### 5.3 Análisis de sentimientos — `notebooks/03_case_sentiment.ipynb`

- **Dataset**: 42 687 reseñas únicas. Se descartaron puntuaciones neutras (=3) y se definió `label=1` para reviews ≤2 (negativas) y `label=0` para ≥4 (positivas). Distribución: 72% positivas, 28% negativas.
- **Split**: 60% train (24 988), 15% validation (6 247), 25% test (7 809) manteniendo estratificación.
- **Representación**: `TfidfVectorizer` (min_df=5, ngramas 1–2, 3 000 features, sin stopwords para conservar modismos portugueses/españoles).
- **Modelos evaluados (macro F1 en validation):**

| Modelo           | F1 macro |
|------------------|---------:|
| Logistic Regression (balanced) | **0.923** |
| Linear SVC (balanced)          | 0.921     |
| Multinomial NB                 | 0.916     |
| Dummy (baseline)               | 0.418     |

- **Resultado final**: Logistic Regression con regularización L2 (`max_iter=2000`, `class_weight="balanced"`) obtiene **F1 macro 0.920** y accuracy 0.933 en test. Recupera 94.8% de reviews negativas con 83.7% de precisión.
- **Artefacto**: el pipeline (`TfidfVectorizer + LogReg`) se serializó en `notebooks/sentiment_pipeline.joblib` para facilitar despliegues batch o en streaming.

## 6. Ejecución y replicabilidad

1. Actualiza/descarga los CSV de Olist (si no están presentes) en `data/raw/`.
2. Corre `00_setup.ipynb` para validar rutas y crear carpetas vacías (`models`, `reports`, etc.).
3. Ejecuta cada notebook caso por caso. Todos inicializan `set_seeds(SEED)` para resultados deterministas.
4. Guarda los outputs relevantes en `models/`, `reports/` o `outputs/` según corresponda.

## 7. Próximos pasos sugeridos

1. Incorporar features geoespaciales (distancia cliente–seller) y métricas de courier en el modelo de ETA para superar MAE < 4 días.
2. Entrenar un recomendador híbrido que combine co-ocurrencia de productos (ALS o modelos de secuencia) con los clusters actuales.
3. Publicar un dashboard ligero (Streamlit) que conecte los tres casos y permita a Paulo decir “wow, estás contratado” con datos en vivo.

---

¿Tienes comentarios o un cuarto caso en mente (por ejemplo, precios dinámicos)? Abre un issue o extiende el repositorio siguiendo la misma convención de notebooks por caso.
