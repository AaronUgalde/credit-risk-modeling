<!-- TODO before merging to the default branch:
     resolve every <<< FILL >>> marker below (screenshots, demo credentials,
     handles, and the metrics flagged inline). They are visible when rendered. -->

# Credit Risk Modeling — from notebook to serving API

> A credit-default classifier taken all the way to a production-shaped REST service:
> reproducible feature pipeline, model selection by cross-validation, SHAP explanations,
> and a FastAPI layer with single and batch endpoints.

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python_3.11-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="SHAP" src="https://img.shields.io/badge/SHAP-explainability-black">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

<!-- <<< FILL >>> A screenshot of the Swagger UI at /docs with a populated response,
     or a SHAP waterfall plot, goes here. Save to docs/ and uncomment. -->
<!-- ![Swagger UI](docs/screenshot-swagger.png) -->

**At a glance**

| | |
|---|---|
| **Problem** | Predict probability of loan default and bucket each application as LOW / MEDIUM / HIGH risk |
| **Data** | 32,581 loan applications, 11 raw features, 21.8% default rate (public credit-risk dataset) |
| **Model** | Random Forest, selected over Logistic Regression / XGBoost / LightGBM by 5-fold stratified CV |
| **Headline** | **AUC-ROC 0.934** on a 20% holdout — **+8.7 points** over the logistic baseline (0.847) |
| **Serving** | FastAPI with `/predict`, `/predict/batch` (≤ 500 records), `/health`, `/model/info` |
| **Scope** | Personal portfolio project. The business framing below is a stated assumption, not a real client. |

---

## The problem, stated honestly

This is a portfolio project built on a **public credit-risk dataset**. To make the engineering
decisions concrete I worked against an explicit set of assumptions rather than pretending there
were no stakeholders:

> *Assumed context:* a risk team reviewing on the order of a thousand applications a month,
> ~45 minutes of analyst time per file, with approval criteria that drift between reviewers.
> Under those assumptions the value of automation is consistency and throughput, and the cost of
> a false negative (an approved loan that defaults) far exceeds the cost of a false positive.

That assumption is what drives every choice below: **recall on the default class is the metric
that matters**, the model must be explainable to a human reviewer, and inference has to be fast
enough to sit inline in an existing workflow.

Three properties of the data shaped the modeling:

1. **Class imbalance** — only 21.8% of applications defaulted, so accuracy is a useless target.
2. **Informative missingness** — `person_emp_length` and `loan_int_rate` were missing in a pattern
   correlated with the label. The absence of the value *is* a signal.
3. **Rare categories** — `loan_intent` and `person_home_ownership` had levels under 1% frequency,
   which mostly contribute variance.

---

## Results

### Model selection (5-fold stratified CV)

| Model | AUC-ROC | F1 (default) | Inference |
|---|---|---|---|
| Logistic Regression *(baseline)* | 0.847 | 0.691 | < 1 ms |
| LightGBM | 0.918 | 0.788 | 2 ms |
| XGBoost | 0.921 | 0.793 | 3 ms |
| **Random Forest** *(selected)* | **0.934** | **0.841** | 8 ms |

Random Forest won on AUC, needed no feature scaling, was robust to the income outliers, and tuned
cleanly with `RandomizedSearchCV` (50 iterations, `class_weight='balanced'`).

### Holdout performance (6,517 applications)

```
                 precision   recall   f1-score   support
   No Default        0.949    0.957      0.953      5110
      Default        0.839    0.812      0.841      1407      <<< FILL >>> confirm these
```

<!-- <<< FILL >>> The old README's summary table claimed Recall 0.871 for the default
     class, but the confusion matrix (264 FN / 1,407 actual defaults) and the
     classification report both give 0.812. Re-run notebook 03/04, decide which is
     correct, and make the summary, the confusion matrix and this table agree.
     An interviewer who checks your arithmetic will check this one. -->

```
                    Predicted: No Default   Predicted: Default
Actual: No Default          4,891                   219
Actual: Default               264                 1,143
```

AUC-ROC **0.934** · optimal threshold by Youden's J: **0.41**.

### Serving latency

<!-- <<< FILL >>> Nothing in this repo measures latency today. Run this, then paste
     the real output. A number you can reproduce on demand is worth far more in an
     interview than a number you remember. -->

```bash
# with the API running on :8000
python - <<'PY'
import time, statistics, httpx
payload = {"person_age":28,"person_income":55000,"person_home_ownership":"RENT",
           "person_emp_length":4.0,"loan_intent":"PERSONAL","loan_grade":"C",
           "loan_amnt":15000,"loan_int_rate":13.5,"loan_percent_income":0.27,
           "cb_person_default_on_file":"N","cb_person_cred_hist_length":6}
c = httpx.Client(base_url="http://localhost:8000")
for _ in range(50): c.post("/predict", json=payload)          # warm up
lat = []
for _ in range(500):
    t = time.perf_counter(); c.post("/predict", json=payload)
    lat.append((time.perf_counter()-t)*1000)
lat.sort()
print(f"n=500  p50={statistics.median(lat):.1f}ms  p95={lat[474]:.1f}ms  p99={lat[494]:.1f}ms")
PY
```

> Measured on <<< FILL: your machine, e.g. "M2 MacBook Air, single uvicorn worker, localhost" >>>.
> Local numbers, not a load test — stated as such so nobody mistakes them for production SLOs.

---

## What I learned

The parts worth telling someone else:

1. **Feature engineering beat algorithm selection.** The derived-feature pipeline was worth about
   +0.04 AUC; Random Forest *with* derived features beat XGBoost *without* them. Time spent on
   features paid better than time spent on model search.
2. **Missingness flags are features.** Encoding "this field was empty" as two binary columns moved
   recall by ~3 points, because in this dataset not answering correlates with risk.
3. **`class_weight='balanced'` beat SMOTE here.** SMOTE's synthetic examples interpolated across
   categorical encodings and produced records that couldn't exist, costing ~0.8 AUC. Reweighting
   inside the trees was cleaner.
4. **Serializing the pipeline is as load-bearing as the model.** Custom transformers
   (`RareGrouper`, `FeatureCreator`) must be importable under the same module path at unpickle
   time or `joblib` fails — sometimes loudly, worse, sometimes silently. This is the bug that
   eats a production deploy, and it's why `api/transformers.py` exists as its own module.
5. **`loan_percent_income` dominates.** Above a 40% income-to-loan ratio, default probability runs
   ~3.2× the mean, largely independent of credit history.

---

## Architecture

```
  OFFLINE (notebooks)                    ARTIFACTS                 ONLINE (api/)
┌──────────────────────┐          ┌────────────────────┐      ┌───────────────────┐
│ 01 EDA               │          │ feature_engineering│      │ FastAPI           │
│ 02 Feature pipeline  │ ───────▶ │   _pipeline.pkl    │ ───▶ │  POST /predict    │
│ 03 Model selection   │          │ best_model_rf      │      │  POST /predict/   │
│ 04 SHAP interpretation│         │   _optimized.pkl   │      │        batch      │
└──────────────────────┘          └────────────────────┘      │  GET  /health     │
                                                              │  GET  /model/info │
                                                              └───────────────────┘
```

**Feature pipeline**

```
11 raw features
   ▼ FeatureCreator      8 derived: log(income), log(amount), income/loan ratio,
   │                     credit-history/age ratio, age buckets, amount × rate,
   │                     2 missingness flags
   ▼ RareGrouper         categories below 1% frequency → 'OTHER'
   ▼ ColumnTransformer   ordinal encoding + median imputation
   ▼ RandomForestClassifier
{default_probability, risk_level, prediction, confidence}
```

---

## Run it

```bash
git clone https://github.com/AaronUgalde/credit-risk-modeling.git
cd credit-risk-modeling
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Serve the pre-trained artifacts
cd api && uvicorn main:app --reload --port 8000
```

Interactive docs at <http://localhost:8000/docs>.

```bash
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "person_age": 28, "person_income": 55000, "person_home_ownership": "RENT",
  "person_emp_length": 4.0, "loan_intent": "PERSONAL", "loan_grade": "C",
  "loan_amnt": 15000, "loan_int_rate": 13.5, "loan_percent_income": 0.27,
  "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 6 }'
```

```json
{ "default_probability": 0.412, "risk_level": "MEDIUM", "prediction": 0, "confidence": 0.588 }
```

To retrain, run `notebooks/01` → `04` in order; they regenerate the `.pkl` artifacts.

**Risk buckets:** `LOW` < 30% · `MEDIUM` 30–60% (route to human review) · `HIGH` > 60%.

---

## Repository layout

```
credit-risk-modeling/
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb          <<< FILL >>> currently named *.ipynb.ipynb
│   ├── 02_feature_engineering_experiments.ipynb
│   ├── 03_model_prototyping_and_tuning.ipynb
│   ├── 04_model_interpretation_and_insights.ipynb
│   └── *.pkl                                       serialized pipeline + model
├── api/
│   ├── main.py            FastAPI app
│   ├── transformers.py    custom transformers (importable for unpickling)
│   └── test_api.py        endpoint tests
└── requirements.txt
```

---

## Known limitations

- **Single train/test split**, no nested CV — the reported AUC carries selection optimism.
- **No monitoring for drift.** A credit model degrades as the economy moves; production would need
  scheduled re-evaluation and a PSI/feature-drift check, neither of which is here.
- **Fairness is unexamined.** `person_age` is a direct input. In a real lending context that
  requires an explicit fairness review and probably a disparate-impact analysis before anyone
  deploys it. I'm naming it rather than leaving it implied.
- **Latency numbers are local**, single-worker, no concurrency.

---

## License

MIT — see [`LICENSE`](LICENSE).
