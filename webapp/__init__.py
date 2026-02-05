from flask import Flask
from config import Config

def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(Config)

    # Ensure folders exist
    app.config["UPLOAD_DIR"].mkdir(parents=True, exist_ok=True)
    app.config["MODEL_DIR"].mkdir(parents=True, exist_ok=True)
    app.config["PLOT_DIR"].mkdir(parents=True, exist_ok=True)

    from webapp.routes.main import main_bp
    app.register_blueprint(main_bp)

    return app
