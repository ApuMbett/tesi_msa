"""
tagger.py — Python 3.9-compatible client
=========================================
Spawns tagger_worker.py under a Python 3.11 interpreter once, then forwards
tag() calls to it over stdin/stdout pipes.

Usage
-----
    from tagger_engine.tagger import Tagger          # or adjust the import path

    # as a plain object
    tagger = Tagger()
    tags = tagger.tag("./tracks/song.wav")           # ["electronic", "upbeat", …]
    tagger.close()                                   # shuts down the worker

    # recommended: use as a context manager
    with Tagger() as tagger:
        for path in my_track_list:
            print(tagger.tag(path))

Constructor arguments
---------------------
python311 : str
    Path (or name on PATH) of the Python 3.11 executable that has TinyMU's
    dependencies installed.  Default: "python3.11".
worker_script : str | Path | None
    Explicit path to tagger_worker.py.  When None (default) the file is
    located automatically next to this module.
startup_timeout : float
    Seconds to wait for the worker to signal it is ready.  Default: 120.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional


class Tagger:
    def __init__(
        self,
        python311: str = "python3.11",
        worker_script: "Optional[str | Path]" = None,
        startup_timeout: float = 120.0,
    ) -> None:
        if worker_script is None:
            worker_script = Path(__file__).parent / "tinymu_worker.py"
        worker_script = Path(worker_script).resolve()

        if not worker_script.exists():
            raise FileNotFoundError(f"Worker script not found: {worker_script}")

        self._proc = subprocess.Popen(
            [python311, str(worker_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,   # inherit parent stderr so worker logs are visible
            text=True,
            bufsize=1,     # line-buffered
            # Run from the worker's own directory so relative TinyMU paths work
            cwd=str(worker_script.parent),
        )

        self._lock = threading.Lock()  # one request at a time
        self._wait_for_ready(startup_timeout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tag(self, track_path: "str | Path") -> list[str]:
        """Return a list of tag strings for the given audio file."""
        with self._lock:
            self._send({"path": str(track_path)})
            response = self._recv()

        if "error" in response:
            raise RuntimeError(f"Worker error: {response['error']}")
        if not isinstance(response, list):
            raise RuntimeError(f"Unexpected worker response: {response!r}")
        return response

    def close(self) -> None:
        """Shut down the worker process gracefully."""
        if self._proc.poll() is None:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    # context-manager support
    def __enter__(self) -> "Tagger":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:   # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send(self, obj: dict) -> None:
        line = json.dumps(obj) + "\n"
        self._proc.stdin.write(line)
        self._proc.stdin.flush()

    def _recv(self) -> "list | dict":
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("Worker process closed stdout unexpectedly.")
        return json.loads(line)

    def _wait_for_ready(self, timeout: float) -> None:
        """Block until the worker prints its {"status": "ready"} handshake."""
        import queue, threading

        result_q: queue.Queue = queue.Queue()

        def _read():
            try:
                line = self._proc.stdout.readline()
                result_q.put(line)
            except Exception as exc:  # noqa: BLE001
                result_q.put(exc)

        t = threading.Thread(target=_read, daemon=True)
        t.start()

        try:
            item = result_q.get(timeout=timeout)
        except queue.Empty:
            self._proc.kill()
            raise TimeoutError(
                f"TinyMU worker did not become ready within {timeout}s. "
                "Check that the 3.11 environment has all dependencies installed."
            )

        if isinstance(item, Exception):
            raise item

        try:
            msg = json.loads(item)
        except json.JSONDecodeError:
            raise RuntimeError(f"Unexpected first line from worker: {item!r}")

        if msg.get("status") != "ready":
            raise RuntimeError(f"Worker startup failed: {msg}")