
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(
    page_title="Renewable Energy Prediction",
    page_icon="☀️",
    layout="wide"
)

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

st.title("☀️ Renewable Energy Generation Prediction")
st.caption("Machine Learning-Based Predictive Analytics for Renewable Energy Generation")

st.sidebar.title("⚙️ Dashboard Settings")

@st.cache_data
def load_dataset():
    path = "data/renewable_energy.csv"

    if os.path.exists(path):
        return pd.read_csv(path)

    return None


uploaded = st.sidebar.file_uploader(
    "Upload Renewable Energy CSV",
    type=["csv"]
)

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = load_dataset()

if df is None:
    st.error("Dataset not found. Please run generate_data.py first.")
    st.stop()

df.columns = df.columns.str.strip()
df = df.replace([np.inf, -np.inf], np.nan)
df = df.drop_duplicates()

st.success("Dataset loaded successfully!")

st.sidebar.markdown("---")
st.sidebar.write("Dataset Information")
st.sidebar.write(f"Total records: {len(df)}")
st.sidebar.write(f"Total columns: {len(df.columns)}")

st.subheader("1. Dataset Overview")

c1, c2, c3 = st.columns(3)

c1.metric("Total Records", f"{len(df):,}")
c2.metric("Total Features", len(df.columns))
c3.metric("Missing Values", f"{df.isna().sum().sum():,}")

st.dataframe(df.head(10), use_container_width=True)

st.subheader("2. Select Prediction Target")

target_options = [
    col for col in df.columns
    if pd.api.types.is_numeric_dtype(df[col])
]

if not target_options:
    st.error("No numeric target columns found.")
    st.stop()

default_target = (
    target_options.index("total_generation")
    if "total_generation" in target_options
    else 0
)

target = st.selectbox(
    "Choose the energy generation target",
    target_options,
    index=default_target
)

# Prepare data

data = df.copy()

date_col = None

for col in data.columns:
    if any(word in col.lower() for word in ["date", "time", "timestamp"]):
        parsed = pd.to_datetime(data[col], errors="coerce")

        if parsed.notna().mean() > 0.5:
            data[col + "_hour"] = parsed.dt.hour
            data[col + "_day"] = parsed.dt.day
            data[col + "_month"] = parsed.dt.month
            date_col = col
            break

# Environmental features

preferred_features = [
    "temperature",
    "solar_radiation",
    "wind_speed",
    "humidity",
    "cloud_cover",
    "date_hour",
    "date_day",
    "date_month"
]

feature_columns = [
    col for col in preferred_features
    if col in data.columns and col != target
]

# Avoid using generated energy outputs as inputs

if not feature_columns:
    excluded = [
        target,
        "solar_generation",
        "wind_generation",
        "total_generation"
    ]

    feature_columns = [
        col for col in data.select_dtypes(include=np.number).columns
        if col not in excluded
    ]

if not feature_columns:
    st.error("No suitable input features found.")
    st.stop()

X = data[feature_columns].copy()
y = pd.to_numeric(data[target], errors="coerce")

valid = y.notna()

X = X.loc[valid].replace([np.inf, -np.inf], np.nan)
y = y.loc[valid]

X = X.dropna(axis=1, how="all")

if X.empty:
    st.error("No usable input features found.")
    st.stop()

# Chronological sorting

if date_col:
    dates = pd.to_datetime(data.loc[valid, date_col], errors="coerce")
    order = dates.sort_values().index
    X = X.loc[order]
    y = y.loc[order]

if len(X) < 20:
    st.error("Please provide at least 20 valid records.")
    st.stop()

split = int(len(X) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

st.subheader("3. Exploratory Data Analysis")

tab1, tab2, tab3 = st.tabs([
    "Energy Generation",
    "Correlation Analysis",
    "Feature Statistics"
])

with tab1:
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(y.iloc[:1000].values)
    ax.set_title("Renewable Energy Generation")
    ax.set_xlabel("Observations")
    ax.set_ylabel(target)

    st.pyplot(fig)
    plt.close(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(10, 6))

    correlation_data = data[feature_columns + [target]]

    sns.heatmap(
        correlation_data.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    ax.set_title("Feature Correlation")

    st.pyplot(fig)
    plt.close(fig)

with tab3:
    st.dataframe(
        data[feature_columns + [target]].describe().T,
        use_container_width=True
    )

st.subheader("4. Train Machine Learning Models")

st.write(
    "Compare four machine learning algorithms "
    "to identify the best prediction model."
)

if st.button("🚀 Train All Models", type="primary"):

    models = {
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeRegressor(
                max_depth=15,
                random_state=42
            ))
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                n_jobs=-1,
                random_state=42
            ))
        ]),

        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                max_iter=100,
                random_state=42
            ))
        ])
    }

    results = []
    predictions = {}

    progress = st.progress(0)

    for i, (name, model) in enumerate(models.items()):

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        predictions[name] = pred

        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2 Score": r2_score(y_test, pred)
        })

        joblib.dump(
            model,
            f"models/{name.replace(' ', '_')}.pkl"
        )

        progress.progress((i + 1) / len(models))

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("RMSE")

    best_name = results_df.iloc[0]["Model"]
    best_model = models[best_name]

    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(list(X.columns), "models/features.pkl")
    joblib.dump(target, "models/target.pkl")

    st.session_state["results"] = results_df
    st.session_state["predictions"] = predictions
    st.session_state["best_name"] = best_name
    st.session_state["best_model"] = best_model
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test

    st.success("All four models trained successfully!")

# Model performance

if "results" in st.session_state:

    st.subheader("5. Model Performance")

    results_df = st.session_state["results"]

    st.dataframe(
        results_df.style.format({
            "MAE": "{:.3f}",
            "RMSE": "{:.3f}",
            "R2 Score": "{:.4f}"
        }),
        use_container_width=True
    )

    best_name = st.session_state["best_name"]

    st.success(f"Best Performing Model: {best_name}")

    best_result = results_df[
        results_df["Model"] == best_name
    ].iloc[0]

    m1, m2, m3 = st.columns(3)

    m1.metric("MAE", f"{best_result['MAE']:.3f}")
    m2.metric("RMSE", f"{best_result['RMSE']:.3f}")
    m3.metric("R² Score", f"{best_result['R2 Score']:.4f}")

    # Actual vs Predicted graph

    st.subheader("6. Actual vs Predicted Energy Generation")

    actual = st.session_state["y_test"]
    pred = st.session_state["predictions"][best_name]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(actual.iloc[:500].values, label="Actual")
    ax.plot(pred[:500], label="Predicted")

    ax.set_title("Actual vs Predicted Energy")
    ax.set_xlabel("Test Observations")
    ax.set_ylabel(target)
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    # Interactive prediction

    st.subheader("7. Interactive Energy Prediction")

    model = st.session_state["best_model"]
    X_test = st.session_state["X_test"]

    sample = X_test.iloc[-1].copy()

    st.write("Enter environmental conditions to predict energy generation.")

    with st.form("prediction_form"):

        edited = {}

        columns = st.columns(2)

        for i, col in enumerate(X_test.columns):

            value = float(sample[col]) if pd.notna(sample[col]) else 0.0

            with columns[i % 2]:
                edited[col] = st.number_input(
                    col.replace("_", " ").title(),
                    value=value,
                    key=f"input_{col}"
                )

        submitted = st.form_submit_button(
            "🔮 Predict Energy Generation",
            type="primary"
        )

    if submitted:

        input_df = pd.DataFrame(
            [edited],
            columns=X_test.columns
        )

        prediction = model.predict(input_df)[0]

        st.success("Prediction completed successfully!")

        st.metric(
            "Predicted Renewable Energy",
            f"{prediction:,.3f}"
        )

        st.caption(f"Prediction target: {target}")

    # Download results

    st.subheader("8. Download Prediction Results")

    output = pd.DataFrame({
        "Actual": actual.values,
        "Predicted": pred
    })

    st.download_button(
        "📥 Download Prediction Results",
        output.to_csv(index=False),
        file_name="renewable_energy_predictions.csv",
        mime="text/csv"
    )

    st.download_button(
        "📊 Download Model Performance",
        results_df.to_csv(index=False),
        file_name="model_performance.csv",
        mime="text/csv"
    )

else:

    st.info("Click 'Train All Models' to begin machine learning training.")

st.sidebar.markdown("---")
st.sidebar.caption("Renewable Energy Prediction | Machine Learning Project")

