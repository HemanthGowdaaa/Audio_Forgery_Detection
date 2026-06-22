import os
import sys
import logging
from django.apps import AppConfig

logger = logging.getLogger("backend_server")

class DetectorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "detector"

    def ready(self):
        # Load the models only when running the development or production server
        is_server = "runserver" in sys.argv or "gunicorn" in sys.argv[0] or "uwsgi" in sys.argv[0]
        
        if is_server:
            # Under development server, ready() is executed twice (once by reloader main, once by worker)
            # We only load models in the actual worker process (RUN_MAIN=True) or if reloader is disabled
            if os.environ.get("RUN_MAIN") == "true" or "--noreload" in sys.argv:
                try:
                    from detector.services.model_loader import get_model_cache
                    logger.info("🚀 Django server starting: Loading SVM and ResNet++ models into memory...")
                    get_model_cache().load_models()
                except Exception as e:
                    logger.error(f"❌ Error loading models during startup: {e}")
