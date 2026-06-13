import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# =========================
# 1. Load dataset
# =========================
df = pd.read_csv("churn.csv")

# Remove target missing rows
df = df.dropna(subset=["Churn"])

# =========================
# 2. Split X and y
# =========================
X = df.drop("Churn", axis=1)
y = df["Churn"]

# =========================
# 3. Column types
# =========================
categorical_cols = X.select_dtypes(include=["object", "category"]).columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

# =========================
# 4. Preprocessing
# =========================
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# =========================
# 5. Model
# =========================
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", model)
])

# =========================
# 6. Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# =========================
# 7. MLflow tracking
# =========================
mlflow.set_experiment("Customer_Churn_Streamlit_Project")

with mlflow.start_run():

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    mlflow.log_param("model_name", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("num_imputer", "median")
    mlflow.log_param("cat_imputer", "most_frequent")
    mlflow.log_param("encoding", "OneHotEncoder")
    mlflow.log_param("scaling", "StandardScaler")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        registered_model_name="Customer_Churn_Streamlit_Model"
    )

    # Also save local model for Streamlit deployment
    mlflow.sklearn.save_model(
        sk_model=pipeline,
        path="saved_model"
    )

    print("Training completed")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("Model saved in saved_model/")