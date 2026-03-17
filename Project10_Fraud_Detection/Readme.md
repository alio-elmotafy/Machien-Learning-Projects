<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,30&height=200&section=header&text=Insurance%20Fraud%20Intelligence&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=38&desc=End-to-End%20Machine%20Learning%20%7C%20FLAML%20AutoML%20%7C%20Voting%20Ensemble&descAlignY=60&descSize=16" width="100%"/>

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FLAML](https://img.shields.io/badge/FLAML-AutoML-00C49A?style=for-the-badge&logo=leaflet&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-RUS-8E44AD?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production%20Ready-00eb93?style=for-the-badge)

<br/>

> **Detect insurance fraud before payouts are made — using a 4-phase ML optimization pipeline.**
> Full EDA → Feature Engineering → Sampling Strategy → Manual Tuning → AutoML → Ensemble

<br/>

</div>

---

## 🗂️ Table of Contents

- [📖 Project Overview](#-project-overview)
- [📊 Dataset](#-dataset)
- [🔍 Exploratory Data Analysis](#-exploratory-data-analysis)
- [🛠️ Data Preprocessing](#️-data-preprocessing)
- [🤖 Modeling Pipeline](#-modeling-pipeline)
- [🏆 Results & Performance](#-results--performance)
- [📦 Dependencies](#-dependencies)
- [✍️ Author](#️-author)

---

## 📖 Project Overview

<table>
<tr>
<td width="60%">

Insurance fraud costs the industry **billions of dollars** every year. This project builds a **complete end-to-end machine learning pipeline** that:

- 🔎 Performs rich EDA with count plots, heatmaps, and pie charts
- 🧹 Engineers new predictive features from raw claim data
- ⚖️ Tackles severe class imbalance (~11% fraud) with sampling strategies
- 🎯 Hand-tunes a **4-model Soft Voting Ensemble** via `HalvingGridSearchCV`
- 🤖 Benchmarks against **FLAML AutoML** (CatBoost winner)
- 📐 Finds the **optimal decision threshold** via Precision-Recall curve analysis

</td>
<td width="40%" align="center">

```
Dataset      → Car Insurance Claims
Features     → 24 columns
Target       → fraud_reported (Y / N)
Class Ratio  → ~11% fraud (severely imbalanced)
Best Model   → Manual Ensemble (F1: 0.33 / Acc: 83%)
AutoML       → FLAML → CatBoost (Recall: 0.61)
```

</td>
</tr>
</table>

---

## 📊 Dataset

**Source:** [Car Insurance Fraud Detection Dataset — Kaggle](https://www.kaggle.com/datasets/ahluwaliasaksham/car-insurance-fraud-detection-dataset)

The dataset covers car insurance claims with **24 features** spanning policyholder demographics, incident details, and financial information.

<details>
<summary><b>📋 Click to expand — Full Feature Reference</b></summary>

<br/>

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `policy_id` | ID | Unique policy identifier *(dropped)* |
| 2 | `policy_state` | Categorical | State where the policy was issued |
| 3 | `policy_deductible` | Numeric | Deductible amount in USD |
| 4 | `policy_annual_premium` | Numeric | Annual premium paid |
| 5 | `insured_age` | Numeric | Age of the insured person |
| 6 | `insured_sex` | Categorical | Gender of the insured |
| 7 | `insured_education_level` | Categorical | Education level |
| 8 | `insured_occupation` | Categorical | Occupation type |
| 9 | `insured_hobbies` | Categorical | Hobby type |
| 10 | `incident_date` | Date | Date of the incident *(engineered → dropped)* |
| 11 | `incident_type` | Categorical | Type of incident (collision, theft, etc.) |
| 12 | `collision_type` | Categorical | Collision direction (front/rear/side) |
| 13 | `incident_severity` | Ordinal | Level of damage → mapped to 1/2/3 |
| 14 | `authorities_contacted` | Categorical | Who was contacted (Police/Fire/etc.) |
| 15 | `incident_state` | Categorical | State where incident occurred |
| 16 | `incident_city` | Categorical | City name *(dropped — high cardinality)* |
| 17 | `incident_hour_of_the_day` | Numeric | Time of the incident |
| 18 | `number_of_vehicles_involved` | Numeric | Vehicles involved in the incident |
| 19 | `bodily_injuries` | Numeric | Number of injuries reported |
| 20 | `witnesses` | Numeric | Number of witnesses present |
| 21 | `police_report_available` | Categorical | Whether a police report exists |
| 22 | `claim_amount` | Numeric | Amount claimed in USD |
| 23 | `total_claim_amount` | Numeric | Estimated total damage in USD |
| 24 | `fraud_reported` | **TARGET** | Y = fraud, N = legitimate |

</details>

---

## 🔍 Exploratory Data Analysis

### 📉 Columns Dropped & Why

| Column | Reason for Removal |
|--------|--------------------|
| `policy_id` | Unique identifier — zero predictive signal |
| `incident_city` | Too many unique values — high cardinality noise |
| `incident_date` | Converted to `days_since_max_incident` then dropped |
| `policy_bind_date` | Date field with no direct predictive value |
| `insured_zip` | High-cardinality zip code — replaced by state-level risk score |

---

### 📊 Visualizations Produced in Notebook

| Visual | What It Shows | Key Insight |
|--------|---------------|-------------|
| 📊 **Count Plots** | Every categorical feature split by `fraud_reported` | Fraud clusters in specific incident types and occupations |
| 🔥 **Correlation Heatmap** | Pearson matrix for all numeric features | Strong correlation between `claim_amount` and `total_claim_amount` |
| 🥧 **Pie Charts** | Proportional share for every categorical feature | Reveals class and category imbalances at a glance |
| 🏆 **Feature Importance** | Top 10 fraud drivers from Gradient Boosting | Incident severity and claim amount dominate |
| 📉 **Precision-Recall Curve** | F1 score at every probability threshold | Used to find the mathematically optimal decision threshold |
| 📊 **Model Comparison Bar Chart** | Precision / Recall / F1 / Accuracy side-by-side | Manual Ensemble vs. AutoML trade-off made visual |

---

### 🔑 Key EDA Findings

```
✦  Only ~11% of claims are fraudulent → extreme class imbalance, accuracy is a "vanity metric"
✦  claim_amount and total_claim_amount are highly correlated → potential redundancy
✦  incident_severity is the #1 fraud signal — Total Loss claims are disproportionately fraudulent
✦  Witness presence is a key reality check — fraud cases often lack witnesses
✦  Authority contact (Police/Fire) correlates strongly with legitimate claims
✦  STD varies widely across numeric columns → StandardScaler is essential
✦  Synthetic oversampling (SMOTEENN) created hallucinated patterns → abandoned in favor of RUS
```

---

## 🛠️ Data Preprocessing

### Complete Step-by-Step Pipeline

```
STEP 1 → Feature Engineering (3 new columns created)
         └─ days_since_max_incident   : recency signal from incident_date
         └─ claim_to_premium_ratio    : financial stress indicator
         └─ location_risk_score       : per-state mean fraud rate encoding

STEP 2 → Drop low-value / high-cardinality columns
         └─ incident_date, incident_city, policy_bind_date, insured_zip

STEP 3 → Encode Target
         └─ "Y" → 1 (fraud)   |   "N" → 0 (legitimate)

STEP 4 → Handle Missing Values
         └─ Categorical → fill with mode
         └─ Numeric     → fill with median

STEP 5 → Ordinal Encoding for incident_severity
         └─ Minor Damage → 1  |  Major Damage → 2  |  Total Loss → 3

STEP 6 → pd.get_dummies() on all remaining categorical columns
         └─ drop_first=True to avoid multicollinearity

STEP 7 → Train / Test Split  (80% / 20%, stratified by target)
         └─ Stratified ensures both splits reflect the 11% fraud ratio

STEP 8 → StandardScaler  (fit ONLY on train, transform both)
         └─ Prevents data leakage from test statistics into training

STEP 9 → Random Under-Sampling  (applied ONLY to training data)
         └─ Reduces majority class to match minority (authentic data only)
         └─ Test set kept at original distribution for honest evaluation
```

> **Why Random Under-Sampling over SMOTE?** SMOTEENN was trialed first but generated synthetic fraud patterns that don't exist in reality, causing unacceptably high false positives (Precision: 0.18). Under-sampling uses only authentic data points — preserving the true signal of what fraud actually looks like.

> **Why threshold tuning?** With imbalanced classes, the default 0.5 threshold is suboptimal. The Precision-Recall curve was used to find the threshold that mathematically maximizes F1-Score, settling at **0.60**.

---

## 🤖 Modeling Pipeline

### Phase 1 — SMOTEENN Trial *(Abandoned)*

Initial attempt using hybrid oversampling + noise cleaning:

```
Result → Precision: 0.18  |  Recall: 0.53  |  F1: 0.27
Reason abandoned → Synthetic samples hallucinated non-existent fraud patterns
                   → Flooded model with false signals → poor precision
```

---

### Phase 2 — Random Under-Sampling Stabilization

Pivoted to authentic-data-only balancing:

```
Result → Precision: 0.19  |  Recall: 0.62  |  F1: 0.29
Breakthrough → Removed majority-class bias, model learned real fraud characteristics
```

---

### Phase 3 — Ensemble Pruning + Manual Tuning with HalvingGridSearchCV

Initial 7-model ensemble was degraded by weak learners (KNN: AUC 0.51, Naive Bayes: AUC 0.60). Pruned to the **top 4 performers** and tuned with `HalvingGridSearchCV` + **5-fold Stratified K-Fold**:

<table>
<tr>
<td>

**🌿 Gradient Boosting Grid**
```python
learning_rate: [0.1]
n_estimators:  [100, 200]
```

</td>
<td>

**🌲 Random Forest Grid**
```python
n_estimators: [100, 200]
class_weight: 'balanced'
```

</td>
<td>

**📐 Logistic Regression Grid**
```python
C: [0.1, 1, 10]
max_iter: 1000
```

</td>
<td>

**⚙️ SVM Grid**
```python
C: [0.1, 1]
probability: True
```

</td>
</tr>
</table>

---

### Phase 4 — Soft Voting Ensemble + Optimal Threshold

```python
VotingClassifier(
    estimators=[
        ('GradBoost',    best_gradboost),
        ('RandomForest', best_rf),
        ('LogReg',       best_logreg),
        ('SVM',          best_svm)
    ],
    voting='soft'   # averages probabilities — more nuanced than majority vote
)
```

**Threshold tuning via Precision-Recall curve:**

```python
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_threshold = thresholds[np.argmax(f1_scores)]  # → 0.60
```

---

### Phase 5 — AutoML with FLAML

```python
automl = AutoML()
automl.fit(X_train=X_res, y_train=y_res,
           time_budget=120,
           metric='f1',
           task='classification',
           seed=42)
# Winner → CatBoost
```

`FLAML` automatically explores the full model zoo within the time budget, tuning hyperparameters and selecting the best configuration. **CatBoost** emerged as its champion.

---

## 🏆 Results & Performance

### Final Benchmark — Manual Ensemble vs. AutoML (CatBoost)

| Metric | Manual Refined Ensemble | AutoML (CatBoost) |
|--------|------------------------|-------------------|
| **Precision** | **0.30** ✅ | 0.21 |
| **Recall** | 0.37 | **0.61** ✅ |
| **F1-Score** | **0.33** ✅ | 0.31 |
| **Accuracy** | **83%** ✅ | 68% |

### Strategic Interpretation

```
Manual Ensemble  →  Best for AUTOMATED FLAGGING
                    Higher precision = fewer false alarms
                    Investigators review only high-confidence fraud signals

AutoML CatBoost  →  Best for FRAUD SCREENING
                    Higher recall = catches more real fraud cases
                    Ensures minimal fraudulent payouts slip through
```

---

### 🏅 Top Fraud Drivers — Gradient Boosting Feature Importance

| Rank | Feature | Why It Matters |
|------|---------|----------------|
| 🥇 1 | `incident_severity` | Total Loss claims are the strongest fraud signal |
| 🥈 2 | `claim_amount` | Fraudulent claims tend to be disproportionately high |
| 🥉 3 | `witnesses` | Legitimate accidents typically have witnesses |
| 4 | `authorities_contacted` | Police/Fire involvement signals a real incident |
| 5 | `claim_to_premium_ratio` | Engineered feature — financial stress indicator |
| 6 | `days_since_max_incident` | Engineered feature — recency of the claim filing |
| 7 | `location_risk_score` | Engineered feature — historical fraud rate by state |
| 8 | `bodily_injuries` | Inflated injury counts are a common fraud tactic |
| 9 | `number_of_vehicles_involved` | Multi-vehicle staged accidents are a known fraud pattern |
| 10 | `policy_annual_premium` | High-premium policies are more frequently targeted |

---

## 📦 Dependencies

```txt
pandas>=2.0.0           → Data manipulation
numpy>=1.24.0           → Numerical computing
scikit-learn>=1.3.0     → Preprocessing, metrics, ensemble, HalvingGridSearchCV
flaml>=2.0.0            → AutoML framework (CatBoost winner)
imbalanced-learn>=0.11.0 → RandomUnderSampler
matplotlib>=3.7.0       → Plotting
seaborn>=0.12.0         → Statistical visualizations
rich                    → Styled terminal output
```

---

## ✍️ Author

<div align="center">

<br/>

**Ali Osama**

[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/alio-elmotafy)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/redhorse22)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ali-osama-240390274)

<br/>

*Built with 🔥 passion for data science and machine learning*

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,30&height=100&section=footer" width="100%"/>

</div>
