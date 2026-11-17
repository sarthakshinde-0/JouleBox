# ═══════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Smart Home Energy Analytics", page_icon="⚡", layout="wide")

# ═══════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS  (night-grid palette: ink background, amber = energy, teal = solar/off-peak)
# ═══════════════════════════════════════════════════════════════════════════
INK = "#0c1220"
PANEL = "#131b2e"
LINE = "#25314d"
TEXT = "#eef1f8"
MUTED = "#9ba7c2"
AMBER = "#ffb020"
TEAL = "#2dd4bf"
RED = "#ff5d73"

# ═══════════════════════════════════════════════════════════════════════════
# CSS STYLING
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #0c1220;
        --panel: #131b2e;
        --panel-2: #18223a;
        --line: #25314d;
        --text: #eef1f8;
        --muted: #9ba7c2;
        --amber: #ffb020;
        --teal: #2dd4bf;
        --red: #ff5d73;
    }

    /* ---------- Base: dark canvas + readable text everywhere ---------- */
    .stApp {
        background: var(--ink);
        color: var(--text);
        font-family: 'Manrope', 'Segoe UI', system-ui, sans-serif;
    }
    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .block-container { padding-top: 2.2rem; max-width: 1280px; }

    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: var(--text) !important;
        font-family: inherit;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] li,
    .stApp [data-testid="stMarkdownContainer"] td,
    .stApp [data-testid="stMarkdownContainer"] th { color: var(--text); }
    .stApp [data-testid="stMarkdownContainer"] strong { color: #ffffff; }
    .stApp label,
    .stApp [data-testid="stWidgetLabel"] p { color: var(--text) !important; font-weight: 600; }
    .stApp [data-testid="stCaptionContainer"],
    .stApp .stCaption { color: var(--muted) !important; }
    .stApp hr { border-color: var(--line) !important; opacity: 1; }

    /* ---------- Hero ---------- */
    .stApp .hero {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 2rem;
        padding-bottom: 1.4rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid var(--line);
    }
    .stApp .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        line-height: 1.1;
        margin: 0 0 0.6rem 0;
        padding: 0;
        color: #ffffff !important;
    }
    .stApp .hero-sub {
        font-size: 1.05rem;
        color: var(--muted) !important;
        margin: 0;
        max-width: 38rem;
        line-height: 1.5;
    }
    .stApp .hero-curve { width: 240px; height: 70px; flex-shrink: 0; }
    @media (max-width: 760px) {
        .stApp .hero-curve { display: none; }
        .stApp .hero-title { font-size: 2rem; }
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: #0f1626;
        border-right: 1px solid var(--line);
    }

    /* ---------- Metrics ---------- */
    .stApp [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 1rem 1.2rem;
    }
    .stApp [data-testid="stMetricLabel"] p { color: var(--muted) !important; font-weight: 600; }
    .stApp [data-testid="stMetricValue"] { color: #ffffff; font-weight: 800; }

    /* ---------- Buttons ---------- */
    .stApp .stButton > button {
        background: var(--amber);
        border: 0;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 700;
        transition: background 0.15s ease;
    }
    .stApp .stButton > button p,
    .stApp .stButton > button span { color: #1a1200 !important; font-weight: 700; }
    .stApp .stButton > button:hover { background: #ffc24d; }
    .stApp .stButton > button:focus-visible { outline: 2px solid var(--teal); outline-offset: 2px; }

    /* ---------- Tabs ---------- */
    .stApp .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
    .stApp .stTabs [data-baseweb="tab"] p { color: var(--muted) !important; font-weight: 600; }
    .stApp .stTabs [aria-selected="true"] p { color: #ffffff !important; }
    .stApp .stTabs [data-baseweb="tab-highlight"] { background-color: var(--amber); }

    /* ---------- Expanders, alerts, tables ---------- */
    .stApp [data-testid="stExpander"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 10px;
    }
    .stApp [data-testid="stExpander"] summary p { font-weight: 600; }
    .stApp [data-testid="stAlert"] { border-radius: 10px; }
    .stApp [data-testid="stMarkdownContainer"] table { border-collapse: collapse; width: 100%; }
    .stApp [data-testid="stMarkdownContainer"] th,
    .stApp [data-testid="stMarkdownContainer"] td {
        border: 1px solid var(--line) !important;
        padding: 0.5rem 0.8rem;
    }
    .stApp [data-testid="stMarkdownContainer"] th { background: var(--panel-2); }

    /* ---------- Task cards (Home) ---------- */
    .stApp .task-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-left-width: 4px;
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        margin: 0.5rem 0 1rem 0;
    }
    .stApp .task-card.amber { border-left-color: var(--amber); }
    .stApp .task-card.red { border-left-color: var(--red); }
    .stApp .task-card h4 { font-size: 1.2rem; margin: 0 0 0.9rem 0; }
    .stApp .task-card p { font-size: 0.95rem; margin: 0.45rem 0; line-height: 1.6; color: var(--muted); }
    .stApp .task-card p strong { color: var(--text); }

    /* ---------- Callouts (observations, insights, results) ---------- */
    .stApp .callout {
        background: var(--panel);
        border: 1px solid var(--line);
        border-left-width: 4px;
        border-radius: 10px;
        padding: 1rem 1.3rem;
        margin: 1rem 0;
        line-height: 1.7;
    }
    .stApp .callout.teal { border-left-color: var(--teal); }
    .stApp .callout.amber { border-left-color: var(--amber); }
    .stApp .callout.red { border-left-color: var(--red); }
    .stApp .callout h3 { margin: 0 0 0.4rem 0; }
    .stApp .callout p { margin: 0.2rem 0; }

    /* ---------- Live prediction result ---------- */
    .stApp .result {
        background: var(--panel);
        border: 1px solid var(--line);
        border-top: 4px solid var(--amber);
        border-radius: 10px;
        text-align: center;
        padding: 2rem 1rem;
        margin: 1.2rem 0;
    }
    .stApp .result-value { font-size: 3.4rem; font-weight: 800; color: var(--amber); letter-spacing: -0.02em; line-height: 1; }
    .stApp .result-label { font-size: 1.05rem; color: var(--muted); margin-top: 0.6rem; }

    /* ---------- Footer ---------- */
    .stApp .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1.5rem;
        color: var(--muted);
        font-size: 0.85rem;
        border-top: 1px solid var(--line);
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# CHART HELPERS  (keeps every Plotly chart on the same dark theme)
# ═══════════════════════════════════════════════════════════════════════════
def style_fig(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, Segoe UI, sans-serif", color=TEXT),
        title_font=dict(size=18, color="#ffffff"),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.07)", zeroline=False, linecolor=LINE)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)", zeroline=False, linecolor=LINE)
    return fig


def show_chart(fig):
    st.plotly_chart(style_fig(fig), width='stretch', theme=None)


# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div>
        <h1 class="hero-title">Smart Home Energy Analytics</h1>
        <p class="hero-sub">AI-Powered Load Forecasting &amp; Conservation Advisor for Smart Grid Optimization</p>
    </div>
    <svg class="hero-curve" viewBox="0 0 240 70" fill="none" aria-hidden="true">
        <path d="M0,54 C18,52 30,50 46,40 S72,20 92,30 S122,50 142,43 S166,12 186,10 S222,40 240,46 L240,70 L0,70 Z" fill="#ffb020" fill-opacity="0.10"/>
        <path d="M0,54 C18,52 30,50 46,40 S72,20 92,30 S122,50 142,43 S166,12 186,10 S222,40 240,46" stroke="#ffb020" stroke-width="2.5" stroke-linecap="round"/>
        <circle cx="186" cy="10" r="4" fill="#ffb020"/>
    </svg>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.write("### 🧭 Navigation")
    st.markdown("---")
    page = st.radio("Select Page", ["Home", "EDA", "Model Training", "Anomaly Detection", "Model Comparison", "Live Prediction"], index=0)
    st.markdown("---")
    st.write("### 📋 Project Info")
    st.write("- **Subject:** Python for Data Science")
    st.write("- **Team:** SY ECE A1")
    st.write("- **Dataset:** HomeC (Kaggle)")
    st.write("- **Models:** LR, Ridge, RF, XGBoost")
    st.write("### 🔄 Auto-Refresh")
    auto_refresh = st.checkbox("Enable Auto-Refresh", value=False)
    refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 30)
    if auto_refresh:
        st.info(f"🔄 Auto-refreshing every {refresh_interval} seconds")
        time.sleep(refresh_interval)
        st.rerun()
    st.markdown("---")
    st.caption("v2.2 | 2026")

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('HomeC_sample.csv', low_memory=False)
        df = df.rename(columns={'use_kW': 'use_kW', 'gen_kW': 'gen_kW', 'Temperature': 'temperature', 'Humidity': 'humidity'})
        if 'time' in df.columns:
            # NOTE: HomeC timestamps are Unix seconds, not nanoseconds.
            # pd.to_datetime defaults to nanoseconds, which silently mis-parses
            # every row to the year 1970 and corrupts hour/month/dayofweek.
            # unit='s' fixes this -> correct dates from 2016 onward.
            df['time'] = pd.to_datetime(df['time'], unit='s', errors='coerce')
            df['hour'] = df['time'].dt.hour
            df['month'] = df['time'].dt.month
            df['dayofweek'] = df['time'].dt.dayofweek
        for col in ['use [kW]', 'gen [kW]', 'temperature', 'humidity']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['use [kW]', 'gen [kW]', 'temperature', 'humidity', 'time'])
        if 'use [kW]' in df.columns and 'gen [kW]' in df.columns:
            df['net_consumption'] = df['use [kW]'] - df['gen [kW]']
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

df = load_data()

if df is None:
    st.error("Data failed to load. Check HomeC_sample.csv exists.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# PAGE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def show_home():
    """Home Page - Task cards + Metrics + Info"""
    st.markdown("---")

    # Welcome Message
    st.write("### 👋 Welcome to Smart Home Energy Analytics")
    st.write("""
    This application helps utility companies and homeowners:
    - **Predict** energy consumption using machine learning
    - **Detect** anomalies that may indicate electricity theft
    - **Optimize** energy usage for cost savings
    - **Balance** grid load for better efficiency
    """)

    st.markdown("---")

    # Task Cards
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="task-card amber">
            <h4>🔋 Task 1: Smart Grid Load Balancing</h4>
            <p><strong>Type:</strong> Supervised Regression</p>
            <p><strong>Goal:</strong> Predict energy consumption based on environmental and usage factors</p>
            <p><strong>Models:</strong> Linear Regression, Ridge, Random Forest, XGBoost</p>
            <p><strong>Metrics:</strong> R² Score, MAE, RMSE</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="task-card red">
            <h4>🚨 Task 2: Electricity Theft Detection</h4>
            <p><strong>Type:</strong> Unsupervised Anomaly Detection</p>
            <p><strong>Goal:</strong> Identify abnormal consumption patterns</p>
            <p><strong>Method:</strong> IQR (Interquartile Range) Statistical Method</p>
            <p><strong>Output:</strong> Flagged records for investigation</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick Statistics
    st.write("### 📊 Dataset Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📁 Total Records", f"{len(df):,}")
    c2.metric("⚡ Avg Consumption", f"{df['use [kW]'].mean():.2f} kW")
    c3.metric("🌡️ Avg Temperature", f"{df['temperature'].mean():.1f}°C")
    c4.metric("💧 Avg Humidity", f"{df['humidity'].mean():.1f}%")
    c5.metric("📋 Features", f"{len(df.columns)}")

    st.markdown("---")

    # Dataset Sample
    with st.expander("📄 View Dataset Sample (First 10 Records)"):
        st.dataframe(df.head(10), width='stretch')

    # Column Information
    with st.expander("📋 View Column Descriptions"):
        st.write("""
        | Column | Description | Type |
        |--------|-------------|------|
        | time | Timestamp of measurement | DateTime |
        | use [kW] | Energy consumption in kilowatts | Numeric |
        | gen [kW] | Energy generated (solar) in kilowatts | Numeric |
        | temperature | Outdoor temperature in Celsius | Numeric |
        | humidity | Relative humidity percentage | Numeric |
        | hour | Hour of day (0-23) | Numeric |
        | month | Month of year (1-12) | Numeric |
        | dayofweek | Day of week (0=Monday, 6=Sunday) | Numeric |
        """)

def show_eda():
    """EDA Page - Mix of Charts, Text, Tables, Insights"""
    st.markdown("---")

    # Introduction
    st.write("### 📊 Exploratory Data Analysis")
    st.write("""
    EDA helps us understand data patterns before building models. We analyze:
    - **Distribution** of energy consumption values
    - **Correlations** between features
    - **Temporal trends** (hourly, monthly patterns)
    """)

    st.markdown("---")

    # Key Statistics
    st.write("### 📈 Key Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Consumption", f"{df['use [kW]'].mean():.2f} kW")
    c2.metric("Median Consumption", f"{df['use [kW]'].median():.2f} kW")
    c3.metric("Std Deviation", f"{df['use [kW]'].std():.2f} kW")
    c4.metric("Range", f"{df['use [kW]'].min():.1f} - {df['use [kW]'].max():.1f} kW")

    st.markdown("---")

    # Distribution Chart
    st.write("### 📉 Consumption Distribution")
    fig1 = px.histogram(df, x='use [kW]', nbins=50, title="Energy Consumption Distribution",
                        color_discrete_sequence=[AMBER])
    show_chart(fig1)

    st.markdown("""
    <div class="callout teal">
        <strong>Observation:</strong> Most consumption values cluster between 0.5-2.0 kW (typical household usage).
        Some high-consumption outliers exist (appliance spikes, AC usage).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Correlation Heatmap
    st.write("### 🔗 Feature Correlation")

    key_features = ['use [kW]', 'temperature', 'humidity', 'hour', 'month']
    key_features = [f for f in key_features if f in df.columns]

    if len(key_features) >= 2:
        corr_df = df[key_features]
        corr_matrix = corr_df.corr()

        fig = px.imshow(
            corr_matrix, text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
            color_continuous_scale=[[0, TEAL], [0.5, PANEL], [1, AMBER]],
            title="Feature Correlation Matrix"
        )
        fig.update_layout(height=480)
        show_chart(fig)

        # Correlation Insights (TEXT + TABLE)
        st.write("#### 🔝 Top Correlations with Consumption")
        correlations = corr_matrix['use [kW]'].abs().sort_values(ascending=False)
        corr_table = pd.DataFrame({
            'Feature': correlations.index,
            'Correlation': correlations.values.round(2)
        })
        st.dataframe(corr_table.head(5), width='stretch')

        if len(correlations) > 1:
            st.success(f"💡 **Strongest Predictor:** {correlations.index[1]} (correlation: {correlations.iloc[1]:.2f})")

    st.markdown("---")

    # Time Trends (TABS)
    st.write("### 📅 Time-Based Patterns")

    tab1, tab2 = st.tabs(["Hourly Pattern", "Monthly Pattern"])

    with tab1:
        st.write("#### Consumption by Hour of Day")
        hourly = df.groupby('hour')['use [kW]'].mean().reset_index()
        fig = px.bar(hourly, x='hour', y='use [kW]', title="Average Consumption by Hour",
                     color_discrete_sequence=[AMBER])
        show_chart(fig)

        st.markdown("""
        <div class="callout amber">
            <strong>Peak Hours:</strong> 6-9 AM (morning) and 5-10 PM (evening)<br>
            <strong>Low Hours:</strong> 11 PM - 5 AM (night)<br>
            <strong>Recommendation:</strong> Shift high-energy appliances to off-peak hours
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.write("#### Consumption by Month")
        monthly = df.groupby('month')['use [kW]'].mean().reset_index()
        fig = px.line(monthly, x='month', y='use [kW]', markers=True, title="Average Consumption by Month",
                      color_discrete_sequence=[AMBER])
        show_chart(fig)

        st.markdown("""
        <div class="callout amber">
            <strong>Summer:</strong> Higher consumption (AC usage)<br>
            <strong>Winter:</strong> Elevated consumption (heating)<br>
            <strong>Spring/Fall:</strong> Lower consumption (moderate weather)
        </div>
        """, unsafe_allow_html=True)

def show_model_training():
    """Model Training Page - Interactive + Metrics + Results"""
    st.markdown("---")

    st.write("### 🤖 Machine Learning Model Training")
    st.write("""
    Train and evaluate regression models to predict energy consumption.
    Select a model, adjust parameters, and see performance metrics.
    """)

    st.markdown("---")

    # Model Selection
    st.write("### ⚙️ Model Configuration")

    features = ['temperature', 'humidity', 'hour', 'month', 'dayofweek']
    features = [f for f in features if f in df.columns]
    target = 'use [kW]'

    if target in df.columns and len(features) >= 2:
        model_choice = st.selectbox("Select Algorithm",
                                    ["Linear Regression", "Ridge Regression", "Random Forest", "XGBoost"])

        # Model-specific parameters
        use_gridsearch = False
        if model_choice == "Ridge Regression":
            alpha = st.slider("Regularization Strength (Alpha)", 0.01, 10.0, 1.0, 0.1)
        elif model_choice == "Random Forest":
            n_trees = st.slider("Number of Trees", 10, 200, 100, 10)
        elif model_choice == "XGBoost":
            tuning_mode = st.radio(
                "How do you want to set hyperparameters?",
                ["Manual sliders", "Auto-tune with GridSearchCV"],
                horizontal=True
            )
            use_gridsearch = (tuning_mode == "Auto-tune with GridSearchCV")
            if not use_gridsearch:
                xgb_n_estimators = st.slider("Number of Trees (n_estimators)", 10, 300, 100, 10)
                xgb_max_depth = st.slider("Max Tree Depth", 2, 10, 5, 1)
                xgb_lr = st.select_slider("Learning Rate", options=[0.01, 0.05, 0.1, 0.2, 0.3], value=0.1)
            else:
                st.info(
                    "🔍 **GridSearchCV** will try every combination of hyperparameters below using "
                    "3-fold cross-validation and pick the combo with the best average R² score."
                )
                colg1, colg2, colg3 = st.columns(3)
                with colg1:
                    grid_n_estimators = st.multiselect("n_estimators options", [50, 100, 150, 200], default=[50, 100])
                with colg2:
                    grid_max_depth = st.multiselect("max_depth options", [3, 5, 7], default=[3, 5])
                with colg3:
                    grid_lr = st.multiselect("learning_rate options", [0.05, 0.1, 0.2], default=[0.05, 0.1])

                n_combos = max(len(grid_n_estimators), 1) * max(len(grid_max_depth), 1) * max(len(grid_lr), 1)
                st.caption(f"⏱️ This will fit {n_combos} combinations × 3 folds = **{n_combos * 3} models**. "
                           f"To keep this fast in a live demo, GridSearch trains on a random sample of the data "
                           f"(max 50,000 rows) rather than the full dataset.")

        # Train button
        if st.button("🚀 Train Model"):
            with st.spinner("Training model..."):
                X = df[features]
                y = df[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                best_params_found = None
                cv_results_df = None

                if model_choice == "Linear Regression":
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                elif model_choice == "Ridge Regression":
                    model = Ridge(alpha=alpha)
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                elif model_choice == "Random Forest":
                    model = RandomForestRegressor(n_estimators=n_trees, random_state=42)
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                else:  # XGBoost
                    if not use_gridsearch:
                        model = XGBRegressor(
                            n_estimators=xgb_n_estimators,
                            max_depth=xgb_max_depth,
                            learning_rate=xgb_lr,
                            random_state=42
                        )
                        model.fit(X_train, y_train)
                        pred = model.predict(X_test)
                    else:
                        # Cap rows fed into GridSearch so it stays demo-fast even on the full ~500K row dataset
                        if len(X_train) > 50000:
                            sample_idx = X_train.sample(n=50000, random_state=42).index
                            X_grid, y_grid = X_train.loc[sample_idx], y_train.loc[sample_idx]
                        else:
                            X_grid, y_grid = X_train, y_train

                        param_grid = {
                            'n_estimators': grid_n_estimators if grid_n_estimators else [100],
                            'max_depth': grid_max_depth if grid_max_depth else [5],
                            'learning_rate': grid_lr if grid_lr else [0.1],
                        }
                        gs = GridSearchCV(
                            XGBRegressor(random_state=42),
                            param_grid,
                            cv=3,
                            scoring='r2',
                            n_jobs=-1
                        )
                        gs.fit(X_grid, y_grid)
                        model = gs.best_estimator_
                        best_params_found = gs.best_params_
                        cv_results_df = pd.DataFrame(gs.cv_results_)[
                            ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
                        ].sort_values('rank_test_score')
                        # Refit best params on the FULL training set (not just the grid sample) for the final model
                        model.fit(X_train, y_train)
                        pred = model.predict(X_test)

                # Metrics
                mae = mean_absolute_error(y_test, pred)
                rmse = np.sqrt(mean_squared_error(y_test, pred))
                r2 = r2_score(y_test, pred)

                st.markdown("---")
                st.write("### 📏 Model Performance")

                c1, c2, c3 = st.columns(3)
                c1.metric("MAE (Error)", f"{mae:.4f}")
                c2.metric("RMSE (Error)", f"{rmse:.4f}")
                c3.metric("R² (Accuracy)", f"{r2:.4f}")

                # GridSearch-specific output
                if best_params_found is not None:
                    st.markdown("---")
                    st.write("### 🔍 GridSearchCV Results")
                    st.success(f"✅ **Best hyperparameters found:** {best_params_found}")
                    st.write("Top combinations tried (ranked by cross-validation R²):")
                    st.dataframe(cv_results_df.head(10), width='stretch')
                    st.caption(
                        "Note: search was run on a 50,000-row sample for speed; the winning hyperparameters "
                        "were then refit on the full training set to produce the metrics above."
                    )

                # Interpretation
                st.markdown("---")
                st.write("### 📊 Performance Interpretation")

                if r2 > 0.7:
                    st.success(f"✅ **Excellent!** R² = {r2:.4f} indicates strong predictive power")
                elif r2 > 0.5:
                    st.info(f"👍 **Good!** R² = {r2:.4f} indicates moderate predictive power")
                else:
                    st.warning(f"⚠️ **Fair** R² = {r2:.4f} - consider adding more features")

                # Feature Importance (for RF and XGBoost)
                if model_choice in ("Random Forest", "XGBoost"):
                    st.markdown("---")
                    st.write("### 🎯 Feature Importance")
                    fi = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
                    fi = fi.sort_values('Importance', ascending=True)
                    fig = px.bar(fi, x='Importance', y='Feature', orientation='h',
                                 title="Which Features Matter Most?",
                                 color_discrete_sequence=[TEAL])
                    show_chart(fig)

                st.success(f"✅ {model_choice} trained successfully!")
    else:
        st.error("Missing required columns for training")

def show_anomaly():
    """Anomaly Detection Page - Metrics + Scatter + Table"""
    st.markdown("---")

    st.write("### 🔍 Anomaly Detection")
    st.write("""
    Identify unusual consumption patterns that may indicate:
    - **Electricity theft** (tampering with meters)
    - **Equipment malfunction** (faulty appliances)
    - **Energy leakage** (wastage)
    """)

    st.markdown("---")

    if 'use [kW]' in df.columns:
        # IQR Calculation
        Q1 = df['use [kW]'].quantile(0.25)
        Q3 = df['use [kW]'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df['anomaly'] = ((df['use [kW]'] < lower) | (df['use [kW]'] > upper))
        anomalies = df[df['anomaly'] == True]
        normal = df[df['anomaly'] == False]

        # Metrics
        st.write("### 📊 Detection Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("📁 Total Records", f"{len(df):,}")
        c2.metric("🚨 Anomalies Detected", f"{len(anomalies):,}")
        c3.metric("✅ Normal Records", f"{len(normal):,}")

        # Percentage
        anomaly_pct = (len(anomalies) / len(df)) * 100
        st.write(f"**Anomaly Rate:** {anomaly_pct:.2f}%")

        if anomaly_pct > 5:
            st.warning("⚠️ High anomaly rate - recommend investigation")
        else:
            st.success("✅ Normal anomaly rate")

        st.markdown("---")

        # Method Explanation
        with st.expander("📖 How IQR Method Works"):
            st.write("""
            **Step 1:** Calculate Q1 (25th percentile) and Q3 (75th percentile)

            **Step 2:** Calculate IQR = Q3 - Q1

            **Step 3:** Define bounds:
            - Lower Bound = Q1 - 1.5 × IQR
            - Upper Bound = Q3 + 1.5 × IQR

            **Step 4:** Flag values outside bounds as anomalies
            """)

        st.markdown("---")

        # Visualization
        st.write("### 📈 Anomaly Visualization")
        plot_df = df.head(1000).reset_index(drop=True)
        fig = px.scatter(plot_df, y='use [kW]', color='anomaly',
                         color_discrete_map={False: TEAL, True: RED, "False": TEAL, "True": RED},
                         title="Energy Consumption (Red = Anomaly)")
        show_chart(fig)

        st.markdown("---")

        # Sample Anomalous Records
        st.write("### 📋 Sample Anomalous Records")
        if len(anomalies) > 0:
            st.dataframe(anomalies[['time', 'use [kW]', 'hour', 'temperature']].head(10), width='stretch')
            st.warning("⚠️ These records should be investigated for potential theft or equipment issues")
        else:
            st.success("✅ No anomalies found in sample")
    else:
        st.error("Column 'use [kW]' not found")

def show_comparison():
    """Model Comparison Page - Table + Chart + Winner"""
    st.markdown("---")

    st.write("### 📊 Model Comparison")
    st.write("""
    Compare all four models on the same test set to determine the best performer.
    All models trained with 80-20 train-test split.
    """)

    st.markdown("---")

    features = ['temperature', 'humidity', 'hour', 'month']
    features = [f for f in features if f in df.columns]
    target = 'use [kW]'

    if target in df.columns and len(features) >= 2:
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results = {}

        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        results['Linear'] = r2_score(y_test, lr.predict(X_test))

        # Ridge
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        results['Ridge'] = r2_score(y_test, ridge.predict(X_test))

        # Random Forest
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X_train, y_train)
        results['RF'] = r2_score(y_test, rf.predict(X_test))

        # XGBoost (fixed reasonable defaults — fast, no tuning here; tuning lives on Model Training page)
        xgb = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        xgb.fit(X_train, y_train)
        results['XGBoost'] = r2_score(y_test, xgb.predict(X_test))

        # Results Table
        st.write("### Performance Comparison")
        results_df = pd.DataFrame(list(results.items()), columns=['Model', 'R2_Score'])
        results_df['R2_Score'] = results_df['R2_Score'].round(4)
        st.dataframe(
            results_df.style.highlight_max(
                subset=['R2_Score'],
                props='background-color: rgba(255,176,32,0.30); color: #ffffff; font-weight: 700;'
            ),
            width='stretch'
        )

        st.markdown("---")

        # Bar Chart
        st.write("### 📈 Visual Comparison")
        fig = px.bar(results_df, x='Model', y='R2_Score', color='R2_Score',
                     title="R² Score Comparison (Higher = Better)",
                     color_continuous_scale=[[0, "#3a4a72"], [1, AMBER]])
        show_chart(fig)

        st.markdown("---")

        # Winner Announcement
        best = results_df.loc[results_df['R2_Score'].idxmax(), 'Model']
        st.markdown(f"""
        <div class="callout teal">
            <h3>🏆 Best Performing Model: {best}</h3>
            <p>R² Score: {results_df.loc[results_df['Model'] == best, 'R2_Score'].values[0]:.4f}</p>
            <p><strong>Recommendation:</strong> Use this model for production deployment</p>
        </div>
        """, unsafe_allow_html=True)

        # Why This Model Won
        with st.expander("💡 Why Did This Model Win?"):
            if best == 'Linear':
                st.write("Linear Regression works well when relationships are linear and features are not highly correlated.")
            elif best == 'Ridge':
                st.write("Ridge Regression handles multicollinearity better through L2 regularization.")
            elif best == 'RF':
                st.write("Random Forest captures non-linear relationships and handles complex patterns better.")
            else:
                st.write(
                    "XGBoost builds trees sequentially, where each new tree corrects the errors of the "
                    "previous ones (gradient boosting). This often lets it capture subtle non-linear patterns "
                    "that a single Random Forest pass misses, especially once hyperparameters are tuned "
                    "(see the Model Training page for GridSearchCV tuning)."
                )
    else:
        st.error("Missing required columns")

def show_live_prediction():
    """Live Energy Prediction Calculator"""
    st.markdown("---")
    st.write("### ⚡ Live Energy Prediction")
    st.write("Enter current conditions to predict energy consumption:")

    col1, col2, col3 = st.columns(3)

    with col1:
        temp = st.slider("Temperature (°C)", 10.0, 40.0, 25.0, 0.5)
    with col2:
        humidity = st.slider("Humidity (%)", 30.0, 90.0, 60.0, 5.0)
    with col3:
        hour = st.slider("Hour of Day", 0, 23, 12)

    # Make prediction
    if st.button("🔮 Predict Now"):
        input_data = pd.DataFrame({
            'temperature': [temp],
            'humidity': [humidity],
            'hour': [hour],
            'month': [datetime.now().month],
            'dayofweek': [datetime.now().weekday()]
        })

        # Train the Random Forest on the full dataset
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        features = ['temperature', 'humidity', 'hour', 'month', 'dayofweek']
        X = df[features]
        y = df['use [kW]']
        model.fit(X, y)

        prediction = model.predict(input_data)[0]

        # Display result
        st.markdown(f"""
        <div class="result">
            <div class="result-value">{prediction:.2f} kW</div>
            <div class="result-label">Predicted energy consumption</div>
        </div>
        """, unsafe_allow_html=True)

        # Recommendation
        if prediction > 2.0:
            st.warning("⚠️ **High Consumption Alert:** Consider turning off non-essential appliances")
        elif prediction > 1.5:
            st.info("ℹ️ **Moderate Consumption:** Normal usage pattern")
        else:
            st.success("✅ **Low Consumption:** Efficient energy usage!")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE ROUTER
# ═══════════════════════════════════════════════════════════════════════════
if page == "Home":
    show_home()
elif page == "EDA":
    show_eda()
elif page == "Model Training":
    show_model_training()
elif page == "Anomaly Detection":
    show_anomaly()
elif page == "Model Comparison":
    show_comparison()
elif page == "Live Prediction":
    show_live_prediction()

# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="footer">Deployed on Streamlit Cloud | Python for Data Science Project | 2026</div>', unsafe_allow_html=True)
