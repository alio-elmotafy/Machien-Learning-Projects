<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Customer%20Churn%20Intelligence&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=38&desc=End-to-End%20Machine%20Learning%20%7C%20LightGBM%20%7C%20Streamlit%20Cloud&descAlignY=60&descSize=16" width="100%"/>

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Best%20Model-00C49A?style=for-the-badge&logo=leaflet&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud%20Deploy-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production%20Ready-00eb93?style=for-the-badge)

<br/>

> **Predict which telecom customers are about to leave — before they do.**
> Full EDA → Preprocessing → AutoML → Manual Tuning → Voting Ensemble → Streamlit App

<br/>

</div>

---

## 🗂️ Table of Contents

- [📖 Project Overview](#-project-overview)
- [📁 Repository Structure](#-repository-structure)
- [📊 Dataset](#-dataset)
- [🔍 Exploratory Data Analysis](#-exploratory-data-analysis)
- [🛠️ Data Preprocessing](#️-data-preprocessing)
- [🤖 Modeling Pipeline](#-modeling-pipeline)
- [🏆 Results & Performance](#-results--performance)
- [🌐 Streamlit App](#-streamlit-app)
- [🚀 How to Run Locally](#-how-to-run-locally)
- [☁️ Streamlit Cloud Deployment](#️-streamlit-cloud-deployment)
- [📦 Dependencies](#-dependencies)
- [✍️ Author](#️-author)

---

## 📖 Project Overview

<table>
<tr>
<td width="60%">

Customer churn is one of the most costly problems in the telecom industry. This project builds a **complete end-to-end machine learning pipeline** that:

- 🔎 Deeply explores the Telco dataset with rich visualizations
- 🧹 Cleans, transforms and balances the data rigorously
- ⚡ Benchmarks **40+ classifiers** automatically via LazyPredict
- 🎯 Hand-tunes the **top 3 models** with `HalvingGridSearchCV`
- 🗳️ Builds a **Soft Voting Ensemble** for maximum accuracy
- 📡 Deploys a **beautiful real-time prediction dashboard** on Streamlit Cloud

</td>
<td width="40%" align="center">

```
Dataset      → 7,043 customers
Features     → 38 columns
Target       → Churn Label (Yes / No)
Class Ratio  → ~26% churned
Best Model   → LightGBM
Deployment   → Streamlit Cloud
```

</td>
</tr>
</table>

---

## 📁 Repository Structure

```
📦 customer-churn-intelligence/
│
├── 📄 APP.py                    ← Streamlit dashboard (main app)
├── 📓 customer_churn.ipynb      ← Full analysis & training notebook
├── 🤖 lgbm_model.pkl            ← Trained LightGBM model
├── ⚖️  scaler.pkl                ← Fitted StandardScaler
├── 📋 model_columns.pkl         ← Training column order (for alignment)
├── 📦 requirements.txt          ← Python dependencies
└── 📖 README.md                 ← You are here
```

> ⚠️ The `.pkl` files are committed directly to the repo so Streamlit Cloud can load them at runtime without any file upload step.

---

## 📊 Dataset

**Source:** [Telco Customer Churn 11.1.3 — Kaggle](https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3)

The dataset covers **7,043 telecom customers** across California with **38 features** spanning demographics, services, financials and satisfaction scores.

<details>
<summary><b>📋 Click to expand — Full Feature Reference</b></summary>

<br/>

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `Customer ID` | ID | Unique customer identifier *(dropped)* |
| 2 | `Gender` | Categorical | Male / Female |
| 3 | `Age` | Numeric | Customer's age in years |
| 4 | `Senior Citizen` | Categorical | Age ≥ 65: Yes / No |
| 5 | `Married` | Categorical | Marital status: Yes / No |
| 6 | `Dependents` | Categorical | Lives with dependents: Yes / No |
| 7 | `Number of Dependents` | Numeric | Count of dependents |
| 8 | `City` | Categorical | City of primary residence |
| 9 | `Zip Code` | Numeric | Zip code |
| 10 | `Latitude` | Numeric | Geo-coordinate |
| 11 | `Longitude` | Numeric | Geo-coordinate |
| 12 | `Population` | Numeric | Population of the zip area |
| 13 | `Referred a Friend` | Categorical | Has ever referred someone: Yes / No |
| 14 | `Number of Referrals` | Numeric | Total referral count |
| 15 | `Tenure in Months` | Numeric | Total months with the company |
| 16 | `Offer` | Categorical | Last marketing offer *(dropped — high nulls)* |
| 17 | `Phone Service` | Categorical | Subscribes to phone: Yes / No |
| 18 | `Multiple Lines` | Categorical | Multiple phone lines: Yes / No |
| 19 | `Internet Service` | Categorical | DSL / Fiber Optic / Cable / No |
| 20 | `Internet Type` | Categorical | *(dropped — high nulls)* |
| 21 | `Avg Monthly GB Download` | Numeric | Average monthly download in GB |
| 22 | `Online Security` | Categorical | Add-on service: Yes / No |
| 23 | `Online Backup` | Categorical | Add-on service: Yes / No |
| 24 | `Device Protection Plan` | Categorical | Add-on service: Yes / No |
| 25 | `Premium Tech Support` | Categorical | Add-on service: Yes / No |
| 26 | `Streaming TV` | Categorical | Streams TV via internet: Yes / No |
| 27 | `Streaming Movies` | Categorical | Streams movies via internet: Yes / No |
| 28 | `Streaming Music` | Categorical | Streams music via internet: Yes / No |
| 29 | `Unlimited Data` | Categorical | Unlimited data plan: Yes / No |
| 30 | `Contract` | Categorical | Month-to-Month / One Year / Two Year |
| 31 | `Paperless Billing` | Categorical | Yes / No |
| 32 | `Payment Method` | Categorical | Bank Withdrawal / Credit Card / Mailed Check |
| 33 | `Monthly Charge` | Numeric | Current total monthly charge |
| 34 | `Total Charges` | Numeric | Cumulative charges to date |
| 35 | `Total Refunds` | Numeric | Cumulative refunds |
| 36 | `Total Extra Data Charges` | Numeric | Extra data overage charges |
| 37 | `Total Long Distance Charges` | Numeric | Total long-distance charges |
| 38 | `Avg Monthly Long Distance Charges` | Numeric | Average long-distance per month |
| 39 | `Satisfaction Score` | Numeric | 1 (Very Unsatisfied) → 5 (Very Satisfied) |
| 40 | `Churn Score` | Numeric | IBM SPSS model score 0–100 |
| 41 | `CLTV` | Numeric | Customer Lifetime Value |
| 42 | `Churn Label` | **TARGET** | Yes = churned this quarter |
| 43 | `Churn Category` | Categorical | *(dropped — only known after churn)* |
| 44 | `Churn Reason` | Categorical | *(dropped — only known after churn)* |

</details>

---

## 🔍 Exploratory Data Analysis

### 🧹 Data Quality Check

```
Null values   →  0   ✅
NaN values    →  0   ✅
Duplicates    →  0   ✅
```

The dataset was perfectly clean — **no imputation required**, which is rare and allowed us to focus entirely on feature engineering and modeling.

---

### 📉 Columns Dropped & Why

| Column | Reason for Removal |
|--------|--------------------|
| `Customer ID` | Unique identifier — zero predictive signal |
| `Churn Category` | Only available **after** the customer has already churned — data leakage |
| `Churn Reason` | Only available **after** the customer has already churned — data leakage |
| `Offer` | Too many null/missing values to be reliable |
| `Internet Type` | Too many null/missing values to be reliable |
| `Customer Status` | Directly encodes the target label — severe data leakage |
| `Country` | Single unique value across all rows — zero variance |
| `State` | Single unique value across all rows — zero variance |
| `Quarter` | Single unique value across all rows — zero variance |

---

### 📊 Visualizations Produced in Notebook

<table>
<tr>
<th align="center">Visual</th>
<th align="center">What It Shows</th>
<th align="center">Key Insight</th>
</tr>
<tr>
<td>📊 <b>Count Plots</b></td>
<td>Every categorical feature split by Churn Label (Yes/No)</td>
<td>Month-to-Month contract = highest churn rate by far</td>
</tr>
<tr>
<td>🔥 <b>Correlation Heatmap</b></td>
<td>15×15 Pearson correlation matrix for all numeric features</td>
<td>Total Charges & Tenure are highly correlated (~0.83)</td>
</tr>
<tr>
<td>📈 <b>Distribution Histograms</b></td>
<td>KDE + histogram for every numeric column</td>
<td>High variance across features → StandardScaler is essential</td>
</tr>
<tr>
<td>🥧 <b>Pie Charts</b></td>
<td>Proportional share for every categorical feature</td>
<td>~68% of customers are on Month-to-Month contracts</td>
</tr>
<tr>
<td>📦 <b>Box Plots</b></td>
<td>Outlier detection per column</td>
<td>Financial features contain significant outliers → IQR clipping needed</td>
</tr>
<tr>
<td>🤖 <b>AutoViz Suite</b></td>
<td>Automated multi-chart cross-feature analysis</td>
<td>Confirmed patterns between CLTV, Churn Score and churn outcome</td>
</tr>
<tr>
<td>📉 <b>ROC Curves</b></td>
<td>TPR vs FPR curve per model with AUC score</td>
<td>LGBM dominates with highest AUC across all thresholds</td>
</tr>
<tr>
<td>🏆 <b>Feature Importance</b></td>
<td>Top 10 churn drivers from LightGBM</td>
<td>CLTV (150) and Churn Score (128) are by far the strongest signals</td>
</tr>
<tr>
<td>🧩 <b>Confusion Matrix</b></td>
<td>Normalized confusion matrix with % annotations for Voting Classifier</td>
<td>Strong precision and recall balance — minimal false negatives</td>
</tr>
</table>

---

### 🔑 Key EDA Findings

```
✦  CLTV is the #1 predictor — high-value customers are paradoxically more likely to churn
✦  Churn Score (IBM SPSS) is the #2 signal — confirms it encodes real behavior patterns
✦  Month-to-Month contracts have 3× higher churn rate than Two Year contracts
✦  Fiber Optic internet users churn more — likely due to price sensitivity
✦  No Online Security + No Tech Support = strongest service-combo churn signal
✦  Customers with high Monthly Charge and low Tenure are the highest-risk segment
✦  Target class is imbalanced (≈74% stay / 26% churn) → SMOTE is necessary
✦  Geo-coordinates (Latitude/Longitude) surprisingly ranked in top 10 importance
```

---

## 🛠️ Data Preprocessing

### Complete Step-by-Step Pipeline

```
STEP 1 → Drop leakage + low-value columns        (9 columns removed)
         └─ Customer ID, Country, State, Quarter, Customer Status,
            Churn Category, Churn Reason, Offer, Internet Type

STEP 2 → IQR Outlier Clipping on all numeric columns
         └─ Clip values to [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
         └─ Preserves all 7,043 rows while bounding extremes

STEP 3 → pd.get_dummies() on all categorical columns
         └─ Creates binary columns for every category
         └─ drop_first=True to avoid multicollinearity

STEP 4 → LabelEncoder on target
         └─ "Yes" → 1  (churned)
         └─ "No"  → 0  (stayed)

STEP 5 → Train / Test Split  (80% / 20%, stratified by target)
         └─ Stratified ensures both splits reflect the 26% churn ratio

STEP 6 → StandardScaler  (fit ONLY on train, transform both)
         └─ Prevents data leakage from test statistics into training

STEP 7 → SMOTE oversampling  (applied ONLY to training data)
         └─ Balances class distribution synthetically
         └─ Test set kept imbalanced for honest evaluation
```

> **Why IQR clipping over removal?** Dropping outlier rows would lose legitimate customer data. Clipping bounds extreme values while keeping all 7,043 records intact.

> **Why SMOTE only on train?** Applying SMOTE to the test set would make evaluation unrealistically optimistic — the test set must mirror real-world class distribution.

---

## 🤖 Modeling Pipeline

### Phase 1 — AutoML Benchmark with LazyPredict

```python
LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
.fit(X_train_resampled, X_test_scaled, y_train_resampled, y_test)
```

- Automatically benchmarks **40+ scikit-learn classifiers** in one call
- Ranks models by: Accuracy, Balanced Accuracy, ROC-AUC, F1-Score, Training Time
- Requires **zero manual configuration** — perfect for model shortlisting

> **Top 3 selected for manual tuning:**

| Rank | Model | Selection Reason |
|------|-------|-----------------|
| 🥇 | `LGBMClassifier` | Highest F1 + fastest training time |
| 🥈 | `AdaBoostClassifier` | Strong performance on imbalanced class problems |
| 🥉 | `DecisionTreeClassifier` | Human-interpretable, solid baseline |

---

### Phase 2 — Manual Hyperparameter Tuning with HalvingGridSearchCV

`HalvingGridSearchCV` is a **resource-efficient alternative** to standard GridSearchCV. It starts with a small subset of data and progressively allocates more resources only to the most promising parameter combinations — cutting tuning time dramatically.

Combined with **5-fold Stratified K-Fold** to maintain class balance in every fold:

<table>
<tr>
<td>

**🌿 LightGBM Grid**
```python
n_estimators:  [100, 200, 250, 299, 400]
learning_rate: [0.001, 0.01, 0.1, 1.0]
num_leaves:    [20, 31, 50, 70, 80]
```

</td>
<td>

**🚀 AdaBoost Grid**
```python
n_estimators:  [50, 100, 200, 300, 400, 450]
learning_rate: [0.001, 0.01, 0.1, 1.0]
```

</td>
<td>

**🌳 Decision Tree Grid**
```python
max_depth:         [None, 10, 20, 30, 40, 50]
min_samples_split: [2, 5, 10, 15, 20]
```

</td>
</tr>
</table>

---

### Phase 3 — Soft Voting Ensemble

```python
VotingClassifier(
    estimators=[
        ('LGBM',        best_lgbm),
        ('AdaBoost',    best_ada),
        ('DecisionTree',best_dt)
    ],
    voting='soft'   # uses class probabilities — more nuanced than hard majority vote
)
```

**Why Soft Voting?**
Hard voting counts raw predictions (1 vote each). Soft voting **averages class probabilities**, giving more confident predictions more influence. When models agree strongly, the ensemble amplifies that signal. When they disagree, it naturally hedges — reducing variance.

---

## 🏆 Results & Performance

### F1-Score Comparison (Test Set)

```
LightGBM      ████████████████████████████████████  ~93%  🥇
Voting        ███████████████████████████████████░  ~93%  🥈
AdaBoost      ███████████████████████████████░░░░░  ~89%  🥉
Decision Tree █████████████████████████████░░░░░░░  ~85%
```

### ROC-AUC Comparison

```
LightGBM      → Highest AUC — best true positive / false positive tradeoff
AdaBoost      → Strong AUC — reliable across all decision thresholds  
Decision Tree → Moderate AUC — interpretable but limited complexity
Voting        → Marginal gain over LGBM — most stable across edge cases
```

---

### 🏅 Top 10 Churn Drivers — LightGBM Feature Importance

| Rank | Feature | Score | Visual |
|------|---------|-------|--------|
| 🥇 1 | `CLTV` | 150 | `██████████████████████████` |
| 🥈 2 | `Churn Score` | 128 | `█████████████████████░░░░░` |
| 🥉 3 | `Age` | 113 | `███████████████████░░░░░░░` |
| 4 | `Population` | 108 | `██████████████████░░░░░░░░` |
| 5 | `Avg Monthly Long Distance Charges` | 106 | `█████████████████░░░░░░░░░` |
| 6 | `Monthly Charge` | 95 | `████████████████░░░░░░░░░░` |
| 7 | `Longitude` | 95 | `████████████████░░░░░░░░░░` |
| 8 | `Latitude` | 91 | `███████████████░░░░░░░░░░░` |
| 9 | `Total Long Distance Charges` | 90 | `███████████████░░░░░░░░░░░` |
| 10 | `Total Charges` | 90 | `███████████████░░░░░░░░░░░` |

---

## 🌐 Streamlit App

The production dashboard replicates the **exact training pipeline** for every prediction:

| Feature | Details |
|---------|---------|
| 🎨 **UI Design** | Dark cyberpunk theme — animated glow, gradient accents, Space Mono + Syne fonts |
| 📐 **Layout** | 3-column responsive grid — Financial, Profile, Services |
| 🔄 **Pipeline** | `get_dummies` → `reindex(model_columns)` → `scaler.transform` → `model.predict_proba` |
| 🎯 **Risk Bands** | Low (0–30%) · Medium (30–60%) · High (60–100%) with color-coded animated result |
| 📊 **Results** | Animated probability gauge + churn/stay probability chips + action recommendation |
| 📈 **Sidebar** | Embedded feature importance mini-chart from actual LGBM results |
| 🔍 **Debug Mode** | Auto-expands column diff on prediction failure — shows missing vs extra columns |
| ⚡ **Performance** | Assets cached with `@st.cache_resource` — loads once, predicts instantly |

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/customer-churn-intelligence.git
cd customer-churn-intelligence

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit app
streamlit run APP.py
```

> Make sure `lgbm_model.pkl`, `scaler.pkl`, and `model_columns.pkl` are in the same folder as `APP.py`.

**To regenerate the `.pkl` files** from your notebook:

```python
import joblib

joblib.dump(best_estimators['LGBM'],          "lgbm_model.pkl")
joblib.dump(scaler,                            "scaler.pkl")
joblib.dump(X_train_full.columns.tolist(),     "model_columns.pkl")
```

---

## ☁️ Streamlit Cloud Deployment

```
Step 1 → Push all files (including .pkl files) to your GitHub repo
Step 2 → Go to share.streamlit.io → New App
Step 3 → Connect your GitHub repo
Step 4 → Set Main file path → APP.py
Step 5 → Click Deploy ✅
```

> If `.pkl` files exceed GitHub's 100MB limit, use Git LFS:
> ```bash
> git lfs track "*.pkl"
> git add .gitattributes
> git add *.pkl && git commit -m "add model assets" && git push
> ```

---

## 📦 Dependencies

```txt
streamlit>=1.32.0       → Web app framework
pandas>=2.0.0           → Data manipulation
numpy>=1.24.0           → Numerical computing
scikit-learn>=1.3.0     → Preprocessing, metrics, ensemble
lightgbm>=4.0.0         → Gradient boosting model
joblib>=1.3.0           → Model serialization
imbalanced-learn>=0.11.0 → SMOTE oversampling
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

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

</div>
