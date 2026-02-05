import uuid
from pathlib import Path

from flask import Blueprint, current_app, render_template, request, session, jsonify

from webapp.services.data_io import read_excel
from webapp.services.model_service import (
    train_and_save, load_model, load_meta, predict_one
)

main_bp = Blueprint("main", __name__)


def get_user_id():
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
    return session["user_id"]


def user_paths(user_id: str):
    model_dir: Path = current_app.config["MODEL_DIR"]
    return {
        "model": model_dir / f"{user_id}_model.joblib",
        "meta": model_dir / f"{user_id}_meta.joblib",
    }


@main_bp.get("/")
def upload_page():
    return render_template(
        "upload.html",
        title="Upload • Excel ML App",
        msg=None
    )


@main_bp.post("/train")
def train():
    # ----- model selection from buttons -----
    model_kind = request.form.get("model_kind", "").strip().lower()
    if model_kind not in {"mlp", "rf"}:
        return render_template(
            "upload.html",
            title="Upload • Excel ML App",
            msg="Please choose a model (MLP or Random Forest)."
        ), 400

    # ----- file validation -----
    if "file" not in request.files:
        return render_template(
            "upload.html",
            title="Upload • Excel ML App",
            msg="No file uploaded."
        ), 400

    f = request.files["file"]
    if not f.filename.lower().endswith(".xlsx"):
        return render_template(
            "upload.html",
            title="Upload • Excel ML App",
            msg="Please upload a .xlsx file."
        ), 400

    # ----- paths -----
    user_id = get_user_id()
    paths = user_paths(user_id)

    upload_dir: Path = current_app.config["UPLOAD_DIR"]
    plot_dir: Path = current_app.config["PLOT_DIR"]

    upload_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    upload_path = upload_dir / f"{user_id}_{uuid.uuid4().hex}.xlsx"
    f.save(upload_path)

    # ----- read excel -----
    try:
        df = read_excel(upload_path)
    except Exception as exc:
        return render_template(
            "upload.html",
            title="Upload • Excel ML App",
            msg=str(exc)
        ), 400

    # ----- train -----
    cfg = current_app.config
    try:
        result = train_and_save(
            df=df,
            model_path=paths["model"],
            meta_path=paths["meta"],
            plot_dir=plot_dir,
            model_kind=model_kind,
            hidden=cfg["MLP_HIDDEN"],
            max_epochs=cfg["MLP_MAX_EPOCHS"],
            random_state=cfg["RANDOM_STATE"],
        )
    except Exception as exc:
        return render_template(
            "upload.html",
            title="Upload • Excel ML App",
            msg=f"Training failed: {exc}"
        ), 400

    plot_url = f"/static/plots/{result.plot_filename}" if getattr(result, "plot_filename", None) else None

    # NEW: initialize empty values so template always has form_values
    form_values = {col: "" for col in result.feature_cols}

    return render_template(
        "predict.html",
        title="Predict • Excel ML App",
        feature_cols=result.feature_cols,
        target_col=result.target_col,
        train_info=result,
        plot_url=plot_url,
        prediction=None,
        form_values=form_values,  # <-- NEW
    )


@main_bp.post("/predict")
def predict():
    user_id = get_user_id()
    paths = user_paths(user_id)

    if not paths["model"].exists() or not paths["meta"].exists():
        return render_template(
            "upload.html",
            title="Upload • Excel ML App",
            msg="No trained model found. Train first."
        ), 400

    pipe = load_model(paths["model"])
    meta = load_meta(paths["meta"])

    # NEW: keep user-entered values so inputs don't clear after Predict
    form_values = {col: request.form.get(col, "") for col in meta["feature_cols"]}

    try:
        pred = predict_one(pipe, meta["feature_cols"], request.form)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 400

    plot_url = f"/static/plots/{meta.get('plot_filename')}" if meta.get("plot_filename") else None

    return render_template(
        "predict.html",
        title="Predict • Excel ML App",
        feature_cols=meta["feature_cols"],
        target_col=meta["target_col"],
        train_info=meta,
        plot_url=plot_url,
        prediction=pred,
        form_values=form_values,  # <-- NEW
    )
