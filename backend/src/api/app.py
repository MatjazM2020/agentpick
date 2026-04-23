"""
Flask application factory.

Initializes the Flask app with configuration, logging, and route registration.
"""

import logging
import os
from flask import Flask
from dotenv import load_dotenv


def create_app(config_name: str = "development") -> Flask:
    """
    Create and configure Flask application.
    
    Responsibilities:
    - Load environment variables
    - Configure logging
    - Register blueprints (routes)
    - Set up error handlers
    
    Args:
        config_name: Configuration name ("development" or "production")
        
    Returns:
        Configured Flask application
    """
    
    # Load environment variables
    load_dotenv()
    
    # Create Flask app
    app = Flask(__name__)
    
    # === Configuration ===
    app.config["JSON_SORT_KEYS"] = False
    app.config["PROPAGATE_EXCEPTIONS"] = True
    
    if config_name == "production":
        app.config["DEBUG"] = False
    else:
        app.config["DEBUG"] = True
    
    # === Logging Setup ===
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Flask app initialized in {config_name} mode")
    
    # === Register Blueprints ===
    from backend.src.api.routes import api_bp
    
    app.register_blueprint(api_bp, url_prefix="/api/v1")
    logger.info("Routes registered")
    
    # === Health Check Endpoint (Root) ===
    @app.route("/", methods=["GET"])
    def root():
        """Root endpoint - basic connectivity check."""
        return {"status": "ok", "message": "Recommendation API is running"}, 200
    
    # === Error Handlers ===
    @app.errorhandler(400)
    def bad_request(error):
        """Handle 400 Bad Request."""
        return {
            "error": "bad_request",
            "message": str(error.description)
        }, 400
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 Not Found."""
        return {
            "error": "not_found",
            "message": "Endpoint not found"
        }, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 Internal Server Error."""
        logger.error(f"Internal server error: {error}")
        return {
            "error": "internal_error",
            "message": "An internal server error occurred"
        }, 500
    
    logger.info("Error handlers registered")
    
    return app


if __name__ == "__main__":
    app = create_app(config_name="development")
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
