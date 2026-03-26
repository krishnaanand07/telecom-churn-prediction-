import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

CATEGORICAL_COLS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_MAP_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]


@st.cache_data
def load_default_data() -> pd.DataFrame:
    return pd.read_csv("Telco-Customer-Churn.csv")


def basic_clean(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    working = df.copy()

    if "customerID" in working.columns:
        working = working.drop("customerID", axis=1)

    if "gender" in working.columns:
        working["gender"] = working["gender"].replace({"Male": 1, "Female": 0})

    for col in BINARY_MAP_COLS:
        if col in working.columns:
            working[col] = working[col].replace({"Yes": 1, "No": 0})

    if "TotalCharges" in working.columns:
        working["TotalCharges"] = pd.to_numeric(working["TotalCharges"], errors="coerce")
        working["TotalCharges"] = working["TotalCharges"].fillna(working["TotalCharges"].median())

    if not include_target and "Churn" in working.columns:
        working = working.drop("Churn", axis=1)

    return working


def fit_preprocessor(train_df: pd.DataFrame):
    encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    encoded = encoder.fit_transform(train_df[CATEGORICAL_COLS])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(CATEGORICAL_COLS),
        index=train_df.index,
    )

    no_cat = train_df.drop(CATEGORICAL_COLS, axis=1)
    transformed = pd.concat([no_cat, encoded_df], axis=1)

    scaler = StandardScaler()
    transformed[NUMERICAL_COLS] = scaler.fit_transform(transformed[NUMERICAL_COLS])

    feature_names = transformed.columns.tolist()
    return encoder, scaler, feature_names


def transform_with_preprocessor(df: pd.DataFrame, encoder, scaler, feature_names):
    encoded = encoder.transform(df[CATEGORICAL_COLS])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(CATEGORICAL_COLS),
        index=df.index,
    )

    no_cat = df.drop(CATEGORICAL_COLS, axis=1)
    transformed = pd.concat([no_cat, encoded_df], axis=1)

    transformed[NUMERICAL_COLS] = scaler.transform(transformed[NUMERICAL_COLS])

    # Ensure inference data matches train-time features
    for col in feature_names:
        if col not in transformed.columns:
            transformed[col] = 0

    transformed = transformed[feature_names]
    return transformed


def create_model(name: str):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=42)
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42)
    if name == "Decision Tree (Pre-Pruned)":
        return DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
    if name == "KNN":
        return KNeighborsClassifier()
    if name == "SVM (RBF)":
        return SVC(kernel="rbf", random_state=42)
    raise ValueError("Unsupported model type")


def train_pipeline(df: pd.DataFrame, model_name: str):
    cleaned = basic_clean(df, include_target=True)

    if "Churn" not in cleaned.columns:
        raise ValueError("Dataset must include a 'Churn' column.")

    if cleaned["Churn"].isna().any():
        cleaned = cleaned.dropna(subset=["Churn"])

    y = cleaned["Churn"].astype(int)
    X_raw = cleaned.drop("Churn", axis=1)

    encoder, scaler, feature_names = fit_preprocessor(X_raw)
    X_prepared = transform_with_preprocessor(X_raw, encoder, scaler, feature_names)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_prepared, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled,
        y_resampled,
        test_size=0.2,
        random_state=42,
    )

    model = create_model(model_name)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    cm = confusion_matrix(y_test, y_pred_test)
    report = classification_report(y_test, y_pred_test, output_dict=True)

    return {
        "model": model,
        "encoder": encoder,
        "scaler": scaler,
        "feature_names": feature_names,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "confusion_matrix": cm,
        "classification_report": pd.DataFrame(report).transpose(),
        "cleaned_data": cleaned,
    }


def predict_on_input(input_df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    cleaned = basic_clean(input_df, include_target=False)
    X = transform_with_preprocessor(
        cleaned,
        artifacts["encoder"],
        artifacts["scaler"],
        artifacts["feature_names"],
    )

    preds = artifacts["model"].predict(X)
    probs = None
    if hasattr(artifacts["model"], "predict_proba"):
        probs = artifacts["model"].predict_proba(X)[:, 1]

    result = input_df.copy()
    result["PredictedChurn"] = np.where(preds == 1, "Yes", "No")
    if probs is not None:
        result["ChurnProbability"] = probs

    return result


def render_metrics(artifacts: dict):
    c1, c2 = st.columns(2)
    c1.metric("Train Accuracy", f"{artifacts['train_acc']:.4f}")
    c2.metric("Test Accuracy", f"{artifacts['test_acc']:.4f}")

    st.subheader("Confusion Matrix (Test)")
    cm = artifacts["confusion_matrix"]
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Classification Report (Test)")
    st.dataframe(artifacts["classification_report"], use_container_width=True)


def main():
    st.title("Telco Customer Churn: Streamlit App")
    st.caption("Built from your notebook workflow with model training and prediction in one app.")

    st.sidebar.header("Configuration")
    model_name = st.sidebar.selectbox(
        "Choose a model",
        [
            "Logistic Regression",
            "Decision Tree",
            "Decision Tree (Pre-Pruned)",
            "KNN",
            "SVM (RBF)",
        ],
        index=2,
    )

    uploaded = st.sidebar.file_uploader("Upload Telco CSV (optional)", type=["csv"])

    if uploaded is not None:
        data = pd.read_csv(uploaded)
        st.sidebar.success("Using uploaded dataset")
    else:
        data = load_default_data()
        st.sidebar.info("Using Telco-Customer-Churn.csv from project folder")

    st.subheader("Raw Data Preview")
    st.dataframe(data.head(10), use_container_width=True)

    if st.button("Train Model", type="primary"):
        try:
            artifacts = train_pipeline(data, model_name)
            st.session_state["artifacts"] = artifacts
            st.session_state["model_name"] = model_name
            st.success(f"{model_name} trained successfully.")
        except Exception as exc:
            st.error(f"Training failed: {exc}")

    if "artifacts" in st.session_state:
        artifacts = st.session_state["artifacts"]

        st.subheader(f"Model Results: {st.session_state.get('model_name', model_name)}")
        render_metrics(artifacts)

        st.subheader("Single Record Prediction")
        cleaned_preview = basic_clean(data, include_target=False)
        row_index = st.number_input(
            "Select row index from dataset",
            min_value=0,
            max_value=max(0, len(cleaned_preview) - 1),
            value=0,
            step=1,
        )

        if st.button("Predict Selected Row"):
            selected = data.iloc[[int(row_index)]].copy()
            pred_df = predict_on_input(selected, artifacts)
            st.dataframe(pred_df, use_container_width=True)

        st.subheader("Batch Prediction")
        pred_upload = st.file_uploader("Upload CSV for prediction", type=["csv"], key="pred_file")
        if pred_upload is not None:
            try:
                pred_data = pd.read_csv(pred_upload)
                pred_result = predict_on_input(pred_data, artifacts)
                st.dataframe(pred_result.head(20), use_container_width=True)

                buffer = io.BytesIO()
                pred_result.to_csv(buffer, index=False)
                st.download_button(
                    "Download Predictions CSV",
                    data=buffer.getvalue(),
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

        col1, col2 = st.columns(2)
        if col1.button("Save Trained Model"):
            saved = {
                "model": artifacts["model"],
                "encoder": artifacts["encoder"],
                "scaler": artifacts["scaler"],
                "feature_names": artifacts["feature_names"],
            }
            joblib.dump(saved, "streamlit_churn_artifacts.pkl")
            st.success("Saved as streamlit_churn_artifacts.pkl")

        if col2.button("Load Existing Trained Artifacts"):
            try:
                loaded = joblib.load("streamlit_churn_artifacts.pkl")
                st.session_state["artifacts"] = {
                    **loaded,
                    "train_acc": np.nan,
                    "test_acc": np.nan,
                    "confusion_matrix": np.array([[0, 0], [0, 0]]),
                    "classification_report": pd.DataFrame(),
                    "cleaned_data": pd.DataFrame(),
                }
                st.info("Loaded streamlit_churn_artifacts.pkl")
            except Exception as exc:
                st.error(f"Could not load file: {exc}")


if __name__ == "__main__":
    main()
