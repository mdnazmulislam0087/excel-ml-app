from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
from joblib import dump, load

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from webapp.services.plotting import save_loss_plot


# -----------------------------
# Result container
# -----------------------------
@dataclass
class TrainResult:
    feature_cols: List[str]
    target_col: str
    task_type: str
    model_kind: str
    model_name: str
    n_rows: int

    epochs_trained: int | None
    final_loss: float | None

    val_mse: float | None
    val_r2: float | None
    val_accuracy: float | None

    plot_filename: str | None
    did_split: bool


# -----------------------------
# Helpers
# -----------------------------
def _infer_task(y: pd.Series) -> str:
    return "Regression" if pd.api.types.is_numeric_dtype(y) else "Classification"


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]), numeric_cols),

            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_cols),
        ]
    )


# -----------------------------
# MAIN TRAIN FUNCTION (THIS WAS MISSING)
# -----------------------------
def train_and_save(
    df: pd.DataFrame,
    model_path: Path,
    meta_path: Path,
    plot_dir: Path,
    *,
    model_kind: str,              # <<< THIS IS WHAT WAS MISSING
    hidden: Tuple[int, ...],
    max_epochs: int,
    random_state: int,
) -> TrainResult:

    if df.shape[1] < 2:
        raise ValueError("Excel must have at least 2 columns")

    df = df.dropna(how="all")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    feature_cols = X.columns.tolist()
    target_col = y.name
    n_rows = len(df)

    task_type = _infer_task(y)

    preprocessor = _build_preprocessor(X)

    # Split safely
    if n_rows >= 10:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        did_split = True
    else:
        X_train = X_val = X
        y_train = y_val = y
        did_split = False

    # -----------------------------
    # Model selection
    # -----------------------------
    if model_kind == "mlp":
        if task_type == "Regression":
            model = MLPRegressor(
                hidden_layer_sizes=hidden,
                max_iter=max_epochs,
                random_state=random_state,
                early_stopping=n_rows >= 50,
            )
            model_name = "MLPRegressor"
        else:
            model = MLPClassifier(
                hidden_layer_sizes=hidden,
                max_iter=max_epochs,
                random_state=random_state,
                early_stopping=n_rows >= 50,
            )
            model_name = "MLPClassifier"

    elif model_kind == "rf":
        if task_type == "Regression":
            model = RandomForestRegressor(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1
            )
            model_name = "RandomForestRegressor"
        else:
            model = RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1
            )
            model_name = "RandomForestClassifier"
    else:
        raise ValueError("Invalid model_kind")

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)

    # -----------------------------
    # Metrics
    # -----------------------------
    val_mse = val_r2 = val_accuracy = None

    y_pred = pipe.predict(X_val)

    if task_type == "Regression":
        val_mse = float(mean_squared_error(y_val, y_pred))
        val_r2 = float(r2_score(y_val, y_pred))
    else:
        val_accuracy = float(accuracy_score(y_val, y_pred))

    # -----------------------------
    # Training curve (MLP only)
    # -----------------------------
    epochs = loss = plot_filename = None

    if model_kind == "mlp":
        mlp = pipe.named_steps["model"]
        epochs = mlp.n_iter_
        loss = mlp.loss_

        if hasattr(mlp, "loss_curve_"):
            plot_dir.mkdir(exist_ok=True, parents=True)
            plot_filename = f"loss_{model_path.stem}.png"
            save_loss_plot(mlp.loss_curve_, plot_dir / plot_filename)

    # -----------------------------
    # Save
    # -----------------------------
    dump(pipe, model_path)

    meta = {
        "feature_cols": feature_cols,
        "target_col": target_col,
        "task_type": task_type,
        "model_kind": model_kind,
        "model_name": model_name,
        "n_rows": n_rows,
        "epochs_trained": epochs,
        "final_loss": loss,
        "val_mse": val_mse,
        "val_r2": val_r2,
        "val_accuracy": val_accuracy,
        "plot_filename": plot_filename,
        "did_split": did_split,
    }
    dump(meta, meta_path)

    return TrainResult(**meta)


# -----------------------------
# Load / Predict
# -----------------------------
def load_model(path: Path):
    return load(path)


def load_meta(path: Path) -> Dict[str, Any]:
    return load(path)


def predict_one(pipe, feature_cols, form):
    row = {c: form[c] for c in feature_cols}
    X = pd.DataFrame([row])

    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="ignore")

    return pipe.predict(X)[0]
