import os
import logging
from pathlib import Path

logger = logging.getLogger("backend_server")

def cleanup_file(path: Path):
    """
    Safely deletes a temporary file from disk if it exists.
    """
    if path and path.exists():
        try:
            os.remove(path)
            logger.info(f"Successfully removed temporary file: {path}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary file {path}: {e}")
