import os
import sys
import logging
from django.apps import AppConfig

logger = logging.getLogger("backend_server")


class DetectorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "detector"

    def ready(self):
        """
        Called once when the Django app registry is fully populated.

        Model loading strategy:
        - In production (Render/gunicorn): load eagerly on startup.
        - In dev (runserver): load only in the actual worker process,
          not in the autoreload parent (avoids double-loading).
        - During management commands (migrate, collectstatic etc.): skip.
        """
        # Skip during management commands that don't serve requests
        argv = sys.argv
        _skip_commands = {
            "migrate", "makemigrations", "collectstatic",
            "createsuperuser", "shell", "dbshell", "check",
            "test", "inspectdb", "showmigrations", "sqlmigrate",
        }
        if argv and len(argv) > 1 and argv[1] in _skip_commands:
            return

        # In Django dev server, ready() fires twice (reloader parent + worker).
        # Only load in the worker process (RUN_MAIN=true) to avoid double-init.
        is_runserver = "runserver" in argv
        if is_runserver:
            if os.environ.get("RUN_MAIN") != "true" and "--noreload" not in argv:
                return  # Skip in the autoreloader parent process

        # All other cases (gunicorn, uvicorn, WSGI, etc.) — load immediately.
        try:
            from detector.services.model_loader import get_model_cache
            logger.info("🚀 Loading ML models into memory (with HF auto-download fallback)...")
            get_model_cache().load_models()
        except Exception as e:
            logger.error(f"❌ Error loading models during startup: {e}", exc_info=True)
            # Don't crash the server — predictions will return 503 until models load
