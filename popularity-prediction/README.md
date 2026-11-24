# ViralScope — Real-World Viral Topic Prediction

> "We extend multimodal popularity prediction by incorporating cascade dynamics,
> temporal attention, and ensemble learning, improving robustness and
> interpretability."

**Live:** `http://127.0.0.1:5000` · AUC 0.9983 · Accuracy 97.2% · R² 0.9964

---

## Architecture

```
Input: Topic + Platform + Followers + Early Signals + Image
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   QUAD-ENGINE HIERARCHY                         │
│                                                                 │
│  Engine 0 ── XGBoost + LightGBM Viral Model (AUC 0.9983)       │
│  Engine 1 ── Stacked Enhancement Layer (R² 0.9964)              │
│  Engine 2 ── Tabular Ensemble (HGB + RF + Ridge)                │
│  Engine 3 ── Multimodal GNN Fallback                            │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
    Viral Probability + 24h/7d/30d Predictions + Tier Badge
```

---

## Enhancement Layer (Add-on Modules — No Existing Files Modified)

### Step 1 · Advanced Temporal Features
**`src/advanced_temporal_features.py`**

| Feature | Formula |
|---------|---------|
| `growth_rate_1` | w2 − w1 |
| `growth_rate_2` | w3 − w2 |
| `acceleration` | growth_rate_2 − growth_rate_1 |
| `momentum` | (w1 + w2 + w3) / 3 |
| `exp_fit_coef` | log-linear slope fit to [w1, w2, w3] |
| `log_ratio_12` | log(w2+1) − log(w1+1) |
| `log_ratio_23` | log(w3+1) − log(w2+1) |

### Step 2 · Cascade Structure Features
**`src/cascade_features.py`**

| Feature | Description |
|---------|-------------|
| `effective_spread` | Sum of GNN embedding magnitudes (total reach proxy) |
| `concentration_index` | Gini coefficient of embedding distribution |
| `propagation_intensity` | L2/L1 norm ratio (packing efficiency) |
| `hub_dominance` | max_abs / mean_abs (star vs mesh structure) |

### Step 3 · Temporal Attention
**`src/temporal_attention.py`**

```python
attention_weights = softmax([w1, w2, w3])
weighted_sum      = w1*a1 + w2*a2 + w3*a3
recency_bias      = a3 - a1          # weight on recent vs early
```

### Step 4 · Feature Augmentation Layer
**`src/augmented_features.py`**

```
fused_features (29) + advanced_temporal (7) + cascade (4) + attention (5) + entropy (1)
= 46+ unified features
```

### Step 5 · Ensemble Model
**`src/ensemble_model.py`**

- Loads existing Ridge baseline from `saved/final_model.pkl`
- Adds `HistGradientBoostingRegressor` + `RandomForestRegressor`
- Returns weighted average prediction

### Step 6 · Uncertainty Estimation
```python
result = {
    "prediction":      float,   # mean across models
    "confidence":      float,   # std across models (log-space)
    "model_agreement": float,   # 1 / (1 + variance)
}
```

### Step 7 · Multi-Horizon Prediction
**`src/multi_horizon.py`**

Blends two extrapolation strategies anchored to the ensemble's H1 prediction:

| Strategy | Method |
|----------|--------|
| Log-linear | OLS fit on log(history), extrapolate forward |
| Holt's smoothing | Double exponential (trend-aware) |
| **Blend** | Weighted average (50/50 by default) |

```python
from src.multi_horizon import MultiHorizonProjector
mhp = MultiHorizonProjector()
result = mhp.project([w1, w2, w3], base_pred=2000.0)
# {horizon_1, horizon_2, horizon_3, growth_trend: 'accelerating|decelerating|stable'}
```

### Step 8 · Cascade Graph Features
**`src/cascade_graph_features.py`**

| Feature | Source |
|---------|--------|
| `cascade_depth` | Graph diameter (networkx), GNN-proxy fallback |
| `branching_factor` | Avg out-degree of non-leaf nodes |
| `spread_speed` | Edges per node (density proxy) |
| `num_unique_users` | Embedding entropy (diversity proxy) |
| `root_influence_score` | Fraction of nodes reachable from hub |

### Step 9 · Extended Inference Interface
**`src/extended_inference.py`**

Unified single-entry-point for the full enhanced pipeline:

```python
from src.extended_inference import ExtendedPredictor

ep = ExtendedPredictor()
ep.warmup()

result = ep.predict_from_raw({'w1': 120, 'w2': 340, 'w3': 890})
print(result.summary())
# pred=78.3  ±0.104  agree=0.989  H1=78 H2=154 H3=278  trend=accelerating

cmp = ep.compare_with_baseline(X_time, gnn_row)  # research comparison dict
```

---

## Image Upload Feature
**`src/image_features.py`**

Upload any post image for instant viral signal extraction:

| Signal | Method |
|--------|--------|
| **Brightness** | Mean HSV value channel |
| **Saturation** | Mean HSV saturation |
| **Edge Density** | PIL FIND_EDGES filter |
| **Format** | Aspect ratio → square/portrait/landscape/wide |
| **Person Detected** | Skin-tone pixel fraction heuristic |
| **Dominant Colors** | Octree quantization (top 5) |
| **Viral Boost** | Composite score 0–1 |
| **Auto-fill** | Urgency + Controversy pre-filled from image |

---

## Performance

| Model | Metric | Value |
|-------|--------|-------|
| Viral Model (Engine 0) | AUC-ROC | **0.9983** |
| Viral Model (Engine 0) | Within-2x Accuracy | **97.2%** |
| Enhancement Layer (Engine 1) | OOF R² | **0.9964** |
| Enhancement Layer (Engine 1) | Hold-out R² | **1.0000** |
| Enhancement Layer (Engine 1) | OOF MAE | **1.93** |
| Enhancement Layer (Engine 1) | Base Models | **5 (HGB+RF+Ridge+XGB+LGB)** |
| Enhancement Layer (Engine 1) | Features Selected | **42/62** |

---

## Saved Artifacts

| File | Contents |
|------|----------|
| `saved/viral_model.pkl` | Engine 0: XGBoost+LGB viral classifier |
| `saved/stacked_model.pkl` | Engine 1: stacked enhancement ensemble |
| `saved/popularity_model.pkl` | Engine 2: tabular HGB model |
| `saved/final_model.pkl` | Engine 3: Ridge baseline + scaler |
| `saved/ensemble_model.pkl` | **NEW**: HGB+RF+Ridge ensemble (extended layer) |
| `saved/fused_features.pkl` | Fused feature matrix (GNN+BERT+Entropy) |
| `saved/gnn_features.pkl` | Per-window GNN embeddings |
| `saved/bert_features.pkl` | BERT text embeddings |
| `saved/entropy_features.pkl` | Window entropy features |
| `saved/dataset.pkl` | Raw temporal windows + targets |

---

## Extended Enhancement Modules (Complete Inventory)

All files below are **purely additive** — zero existing files modified.

| File | Type | Purpose |
|------|------|---------|
| `src/advanced_temporal_features.py` | Feature | 7 growth/acceleration/exp-fit features |
| `src/temporal_attention.py` | Feature | Softmax attention over 3 windows |
| `src/cascade_features.py` | Feature | 4 GNN-proxy cascade structure features |
| `src/cascade_graph_features.py` | Feature | 5 topology-based cascade features (networkx) |
| `src/augmented_features.py` | Feature | 46-dim unified augmented feature matrix |
| `src/ensemble_model.py` | Model | HGB + RF + Ridge ensemble with uncertainty |
| `src/multi_horizon.py` | Model | Log-linear + Holt 3-window horizon projector |
| `src/extended_inference.py` | Interface | Unified inference + comparison API |
| `src/evaluate_extended.py` | Eval | Research comparison runner |
| `src/feature_drift.py` | Monitoring | Z-score + Mahalanobis drift detection |
| `src/app_extension.py` | API | Flask Blueprint: `/api/extended/*` endpoints |
| `evaluate_ensemble.py` | Script | CLI: `--smoke` / `--demo` / `--full` |
| `extended_app.py` | Script | ViralScope + extended API on port 5001 |
| `train_ensemble.py` | Script | Train extended ensemble (HGB+RF+Ridge, 5-fold CV) |
| `research_compare.py` | Script | Full ablation study + research report |

---

## Extended API Endpoints

Start with `python extended_app.py` (port 5001, no existing files changed):

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/extended/status`  | Module readiness check |
| `POST` | `/api/extended/predict` | Full extended pipeline inference |
| `POST` | `/api/extended/compare` | Baseline vs ensemble side-by-side |
| `POST` | `/api/extended/horizon` | Multi-horizon projection only |
| `POST` | `/api/extended/explain` | Feature breakdown with interpretation |

**Example:**
```bash
curl -X POST http://localhost:5001/api/extended/predict \
     -H "Content-Type: application/json" \
     -d '{"w1": 120, "w2": 340, "w3": 890}'
```

---

## Strict Rules Compliance

- ✅ `graph_builder.py` — NOT modified
- ✅ `gnn_model.py` — NOT modified
- ✅ `final_model.py` — NOT modified
- ✅ `bert_encoder.py` — NOT modified
- ✅ `app.py` — NOT modified
- ✅ All pipeline files — NOT modified
- ✅ All new modules are purely additive

---

## Run

```bash
# Original UI (unchanged)
python app.py                          # http://127.0.0.1:5000

# Extended UI + API (no files modified)
python extended_app.py                 # http://127.0.0.1:5001

# Research tools
python research_compare.py             # full ablation report (stdout)
python research_compare.py --concise   # key metrics only
python research_compare.py --save      # also write report.txt

# Evaluation suite
python evaluate_ensemble.py --smoke    # module smoke test (7 checks)
python evaluate_ensemble.py --demo     # multi-horizon demo
python evaluate_ensemble.py --full     # full vs-baseline comparison

# Training
python train_ensemble.py               # train + save ensemble (5-fold CV)
python train_ensemble.py --folds 10   # use 10-fold CV
python train_ensemble.py --n-aug 6000 # use 6000 augmented samples

# Individual module self-tests
python -m src.extended_inference       # extended predictor + drift detection
python -m src.feature_drift            # drift detector scenarios
python -m src.ensemble_model           # retrain ensemble
python -m src.multi_horizon            # horizon projector demo
```

Open original UI: **http://127.0.0.1:5000**
Open extended UI: **http://127.0.0.1:5001**
