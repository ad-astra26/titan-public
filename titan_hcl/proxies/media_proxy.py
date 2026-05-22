"""
Media Proxy — bridge to the MediaWorker for image/audio digestion.

Lazy module: starts on first use (when media is first queued).
"""
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class MediaProxy:
    """Proxy for the Media perception module."""

    def __init__(self, bus, guardian):
        self._bus = bus
        self._guardian = guardian
        self._reply_queue = bus.subscribe("media_proxy", reply_only=True)
        self._queue_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "media_queue")
        )
        os.makedirs(self._queue_dir, exist_ok=True)

    def queue_file(self, file_path: str) -> bool:
        """
        Queue a media file for digestion.

        Copies the file to the media queue directory and ensures
        the MediaWorker is running.
        """
        try:
            src = Path(file_path)
            if not src.exists():
                logger.warning("[MediaProxy] File not found: %s", file_path)
                return False

            dst = Path(self._queue_dir) / src.name
            # Avoid name collisions
            if dst.exists():
                import time
                stem = src.stem
                dst = Path(self._queue_dir) / f"{stem}_{int(time.time())}{src.suffix}"

            shutil.copy2(str(src), str(dst))
            logger.info("[MediaProxy] Queued: %s", dst.name)

            # Ensure media worker is running
            self._guardian.ensure_running("media")
            return True

        except Exception as e:
            logger.error("[MediaProxy] Queue failed: %s", e)
            return False

    def get_status(self) -> dict:
        """Get media worker status."""
        try:
            pending = [f.name for f in Path(self._queue_dir).iterdir()
                      if not f.name.endswith(".failed")]
        except Exception:
            pending = []

        return {
            "queue_dir": self._queue_dir,
            "pending_files": len(pending),
            "pending": pending[:10],
        }
