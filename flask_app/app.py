"""Flask application factory."""

from flask import Flask

from flask_app.config import Config


def create_app(config_class=Config, pipeline=None):
    """Create and configure the Flask application.

    Args:
        config_class: Configuration class to use.
        pipeline: Optional RAGPipeline instance. If None, routes will
                  return an error when trying to answer questions.

    Returns:
        Configured Flask app.
    """
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config.from_object(config_class)
    app.config["PIPELINE"] = pipeline

    from flask_app.routes import main_bp

    app.register_blueprint(main_bp)

    return app
