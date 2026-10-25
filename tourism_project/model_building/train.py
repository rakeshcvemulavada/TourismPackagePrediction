# ==========================================
# Model Training & MLflow Logging Script
# ==========================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
import mlflow

# Set MLflow tracking parameters
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

# Load preprocessed datasets
Xtrain = pd.read_csv("Xtrain.csv")
Xtest  = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest  = pd.read_csv("ytest.csv").squeeze()

# One-hot encode and scale numeric features
numeric_features = Xtrain.select_dtypes(include=[np.number]).columns
categorical_features = ['Designation', 'OwnCar', 'Passport', 'CityTier','MaritalStatus', 'ProductPitched','Gender','Occupation','TypeofContact']


# Intersect listed features with available dataset columns
numeric_features = [f for f in numeric_features if f in Xtrain.columns]
categorical_features = [f for f in categorical_features if f in Xtrain.columns]

# Handle high class imbalance dynamically by computing class ratio
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessing pipeline construction
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# Setup base XGBoost model using pos_weight scaling
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Hyperparameter tuning space (focused grid for workflow speed)
param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [2, 3],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

# Initialize MLflow experiment run
with mlflow.start_run():
    # Tune for 'recall' to minimize missed package purchasing prospects
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="recall", n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log intermediate parameter configurations to MLflow runs
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log optimal parameters
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_
    print("Best params identified:", grid_search.best_params_)

    # Generate evaluation metrics using optimized threshold 0.45
    classification_threshold = 0.45
    
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)
    
    print("\n--- Test Classification Report ---")
    print(classification_report(ytest, y_pred_test))

    # Log performance evaluation metrics to MLflow
    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1-score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1-score": test_report["1"]["f1-score"]
    })

    # Save local model artifact for Streamlit serving
    model_path = "tourism_project/deployment/best_tourism_model_v1.0.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model serialized and saved successfully to: {model_path}")
