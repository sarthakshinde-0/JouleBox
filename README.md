# ⚡ JouleBox

**Smart Home Energy Management System: Load Forecasting & Electricity Theft Detection using Machine Learning**

JouleBox analyzes smart home energy data to do two things: **predict how much electricity a household will use** (for smart grid load balancing) and **flag unusual consumption readings** that could point to electricity theft or equipment faults. It combines a full ML pipeline with an interactive Streamlit dashboard, built as a Python for Data Science mini project.

🔗 **Live App:** [smart-home-energy-analytics.streamlit.app](https://smart-home-energy-analytics-jfnxjyklhugtv7bkz46nhx.streamlit.app/)

---

## Table of Contents

1. [Why this project](#why-this-project)
2. [What it does](#what-it-does)
3. [Dashboard pages](#dashboard-pages)
4. [How it works](#how-it-works)
5. [The dataset](#the-dataset)
6. [Models](#models)
7. [Anomaly detection](#anomaly-detection)
8. [Repo structure](#repo-structure)
9. [Tech stack](#tech-stack)
10. [Running locally](#running-locally)
11. [Notes on the data](#notes-on-the-data)
12. [Limitations](#limitations)
13. [Future work](#future-work)
14. [Acknowledgements](#acknowledgements)

---

## Why this project

Two real problems in the power sector inspired it:

- **Load balancing.** Utilities must match supply to demand at every moment. If they can forecast household demand from the hour of day and the weather, they can plan generation and avoid overloading the grid.
- **Electricity theft and faults.** Meter tampering and faulty appliances show up as abnormal readings. Finding them automatically is far cheaper than inspecting homes one by one.

JouleBox tackles both on real smart meter data.

---

## What it does

Both tasks run on the [HomeC dataset](https://www.kaggle.com/datasets/taranvee/smart-home-dataset-with-weather-information): roughly 500,000 rows of 1-minute smart home energy readings paired with weather data.

### Task 1: Smart Grid Load Balancing (Regression)

Predicts household consumption (`use [kW]`) from environmental and time-based features: temperature, humidity, hour of day, month and day of week. Four models are trained and compared:

- Linear Regression
- Ridge Regression
- Random Forest
- XGBoost, with an optional **GridSearchCV** step that auto-tunes its hyperparameters

### Task 2: Electricity Theft Detection (Anomaly Detection)

Flags abnormal readings with the **IQR (Interquartile Range)** method, surfacing records that fall far outside normal usage bounds: possible meter tampering, theft or equipment malfunction.

---

## Dashboard pages

The Streamlit app (`app.py`) has six pages, chosen from the sidebar.

| Page | What you can do |
|---|---|
| **Home** | See the project overview, the two tasks, dataset metrics, a sample of the data and column descriptions. |
| **EDA** | Explore the consumption distribution, a correlation heatmap with a ranked correlation table, and hourly and monthly consumption trends. |
| **Model Training** | Pick an algorithm, tune hyperparameters with sliders (or auto-tune XGBoost with GridSearchCV), train, and see MAE, RMSE and R². Random Forest and XGBoost also show feature importance. |
| **Anomaly Detection** | Run IQR detection, see the anomaly count and rate, a scatter plot with anomalies highlighted, and a table of flagged records. |
| **Model Comparison** | Train all four models on the same split and compare their R² scores side by side, with a winner announcement. |
| **Live Prediction** | Enter temperature, humidity and hour of day, and get an instant predicted load in kW with a usage advisory. |

The sidebar also has an optional auto-refresh control.

---

## How it works

```
HomeC_sample.csv
      |
      v
Clean + parse timestamps (Unix seconds -> datetime)
      |
      v
Feature engineering: hour, month, dayofweek
      |
      +--------------------------+
      |                          |
      v                          v
Regression models           IQR anomaly detection
(Linear, Ridge, RF, XGB)    (Q1, Q3, 1.5 x IQR bounds)
      |                          |
      v                          v
MAE / RMSE / R2             Flagged records + anomaly rate
      |
      v
Live prediction (Random Forest)
```

1. **Load and clean.** The CSV is read, numeric columns are converted, and rows with missing values are dropped.
2. **Parse time.** `time` is converted from Unix seconds to a datetime, and `hour`, `month` and `dayofweek` are extracted.
3. **Split.** Regression pages use an 80/20 train-test split with `random_state=42`.
4. **Train and evaluate.** Models are scored with MAE, RMSE and R² on the held-out 20%.
5. **Detect anomalies.** The IQR bounds are computed on `use [kW]` and any reading outside them is flagged.

---

## The dataset

**HomeC: Smart Home Dataset with Weather Information** (Kaggle).

| Column | Meaning |
|---|---|
| `time` | Timestamp of the reading (Unix seconds in the raw data) |
| `use [kW]` | Household energy consumption. This is the prediction **target**. |
| `gen [kW]` | Energy generated (solar) |
| `temperature` | Outdoor temperature |
| `humidity` | Relative humidity |
| `hour`, `month`, `dayofweek` | Engineered from `time` (`dayofweek`: 0 = Monday, 6 = Sunday) |
| `net_consumption` | `use [kW]` minus `gen [kW]` |

The full dataset is large, so the app runs on `HomeC_sample.csv`, a smaller sample created by `create_sample.py`.

---

## Models

| Model | Why it is included |
|---|---|
| **Linear Regression** | Simple baseline. Shows how far straight-line relationships go. |
| **Ridge Regression** | Linear model with L2 regularization. Handles correlated features better. |
| **Random Forest** | Averages many decision trees. Captures non-linear patterns such as AC use rising with heat. |
| **XGBoost** | Gradient boosting: each new tree corrects the errors of the previous ones. Often the strongest on tabular data. |

**Metrics used**

| Metric | Reads as |
|---|---|
| **MAE** | Average size of the prediction error, in kW. Lower is better. |
| **RMSE** | Like MAE but penalizes large misses more. Lower is better. |
| **R²** | Share of the variation in consumption the model explains. Closer to 1 is better. |

**GridSearchCV (XGBoost).** Tries every combination of the chosen `n_estimators`, `max_depth` and `learning_rate` with 3-fold cross-validation and keeps the best by R². To stay fast on free-tier hosting, the search runs on a sample of at most 50,000 training rows. The winning settings are then refit on the full training set for the final scores.

---

## Anomaly detection

JouleBox uses the **IQR method**:

1. Compute Q1 (25th percentile) and Q3 (75th percentile) of `use [kW]`.
2. IQR = Q3 − Q1.
3. Lower bound = Q1 − 1.5 × IQR. Upper bound = Q3 + 1.5 × IQR.
4. Any reading outside the bounds is flagged as an anomaly.

It needs no labels and no training, which makes it a good first-pass screen. Flagged records are meant to be **investigated**, not treated as confirmed theft.

---

## Repo structure

```
.
├── .devcontainer/        # Dev container config for a consistent environment
├── .streamlit/
│   └── config.toml       # Dark theme settings for the dashboard
├── .gitignore
├── HomeC_sample.csv      # Smaller sample of HomeC, used by app.py
├── HomeC_Test.csv        # Smaller test split of the dataset
├── app.ipynb             # Notebook: full ML pipeline, EDA, model training
├── app.py                # Streamlit dashboard (multi-page)
├── create_sample.py      # Generates the sample/test CSVs from the full dataset
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Tech stack

- **Language:** Python
- **ML:** scikit-learn (Linear Regression, Ridge, Random Forest), XGBoost
- **Data:** Pandas, NumPy
- **Visualization:** Plotly, Matplotlib, Seaborn
- **App and deployment:** Streamlit, Streamlit Community Cloud

---

## Running locally

```bash
git clone <repo-url>
cd joulebox
pip install -r requirements.txt
streamlit run app.py
```

Keep `HomeC_sample.csv` in the same folder as `app.py`. The app loads it by filename. It opens at `http://localhost:8501`.

---

## Notes on the data

- **Timestamps are Unix seconds.** Parse them with `pd.to_datetime(df['time'], unit='s')`. Without `unit='s'`, pandas assumes nanoseconds and silently maps every row to 1970, which corrupts `hour`, `month` and `dayofweek` for every model.
- **Outlier handling differs by task.** For regression, outliers are IQR-capped to improve model accuracy. For anomaly detection they are kept, because abnormal readings are exactly what that task looks for.
- **GridSearchCV runs on a capped sample** of the training data so the search finishes in reasonable time on free-tier compute.

---

## Limitations

- **Random train-test split.** The data is a minute-by-minute time series, and a shuffled split puts neighbouring minutes in both the training and test sets. This can make R² look better than it would on genuinely future data. A chronological split would be a stricter test.
- **IQR finds unusual, not guilty.** A high reading can be a heater, an AC or a party. The method flags candidates for inspection; it cannot prove theft.
- **One household.** HomeC describes a single home, so the models learn that household's behaviour and may not transfer to others.
- **Live Prediction uses the server clock** for the month and day of week, and only hour, temperature and humidity are user inputs.
- **Sample data.** The deployed app uses a sample of the full dataset, so numbers can differ slightly from the full-data notebook.

---

## Future work

- Chronological (time-based) train-test split and lag features such as the previous hour's consumption
- Cost-savings view using time-of-use tariffs, to recommend when to run heavy appliances
- Solar analysis using `gen [kW]` and `net_consumption`
- A second anomaly method such as Isolation Forest, with downloadable flagged records
- Actual vs predicted chart on the Model Training page

---

## Acknowledgements

Built for the Python for Data Science mini project, SY ECE (AIML), DES Pune University, under the guidance of Dr. Prajakta Khadakikar and Dr. Mosam Sangole.

Dataset: [HomeC: Smart Home Dataset with Weather Information](https://www.kaggle.com/datasets/taranvee/smart-home-dataset-with-weather-information) (Kaggle).
