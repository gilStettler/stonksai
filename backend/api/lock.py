import os
import time
import json
from pathlib import Path

class JobLock:
    """
    Simple cross-platform lock using atomic file creation.
    Returns False if lock exists.
    Removes stale locks older than stale_after_seconds.
    """
    def __init__(self, lock_path: Path, stale_after_seconds: int = 3600):
        self.lock_path = Path(lock_path)
        self.stale_after_seconds = stale_after_seconds
        self.fd = None

    def _cleanup_stale(self):
        if not self.lock_path.exists():
            return
        created = 0.0
        try:
            data = json.loads(self.lock_path.read_text(encoding="utf-8"))
            created = float(data.get("created_at", 0))
        except Exception:
            created = 0.0
        if created and (time.time() - created) > self.stale_after_seconds:
            try:
                self.lock_path.unlink()
            except Exception:
                pass

    def acquire(self) -> bool:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._cleanup_stale()
        try:
            self.fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            payload = {"pid": os.getpid(), "created_at": time.time()}
            os.write(self.fd, json.dumps(payload).encode("utf-8"))
            os.fsync(self.fd)
            return True
        except FileExistsError:
            return False

    def release(self):
        try:
            if self.fd is not None:
                os.close(self.fd)
        finally:
            self.fd = None
            try:
                if self.lock_path.exists():
                    self.lock_path.unlink()
            except Exception:
                pass

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc, tb):
        self.release()
