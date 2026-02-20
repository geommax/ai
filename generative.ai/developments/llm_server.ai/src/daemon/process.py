"""LLM Server.AI — Daemon Process.

A long-running background process that owns all heavy state:
  • InferenceEngine  (loaded model + GPU memory)
  • ModelManager     (downloads, cache management)
  • Database         (API keys)
  • ServerThread     (FastAPI/uvicorn)

Communication with the TUI happens over a Unix domain socket using
new-line-delimited JSON messages.

Run directly:
    python -m src.daemon.process          # foreground (logs to stdout)
    python -m src.daemon.process --quiet  # foreground, log to file only
"""

from __future__ import annotations

import gc
import json
import logging
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

# ── Paths ──────────────────────────────────────────────────────────────
CONFIG_DIR = Path.home() / ".config" / "llm_server_ai"
PID_FILE = CONFIG_DIR / "daemon.pid"
SOCKET_PATH = CONFIG_DIR / "daemon.sock"
LOG_FILE = CONFIG_DIR / "daemon.log"

# ── Logging ────────────────────────────────────────────────────────────
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("llm_daemon")
log.setLevel(logging.INFO)
_fh = logging.FileHandler(str(LOG_FILE))
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(_fh)


class LLMDaemon:
    """Background daemon that manages all LLM server state."""

    def __init__(self) -> None:
        from src.config import ServerConfig, DB_FILE
        from src.database import Database
        from src.llms import InferenceEngine, ModelManager

        self.config = ServerConfig.load()
        self.db = Database(str(DB_FILE))
        self.engine = InferenceEngine()
        self.mm = ModelManager(
            cache_dir=self.config.model_dir or None
        )

        self.server_thread: Any = None  # ServerThread | None
        self._running = False

        # ── Locks ──────────────────────────────────────────────────
        self._model_lock = threading.Lock()

        # ── Download state ─────────────────────────────────────────
        self._dl_lock = threading.Lock()
        self._dl_thread: Optional[threading.Thread] = None
        self._dl_state: Dict[str, Any] = self._empty_dl_state()

        # ── Loading indicator ──────────────────────────────────────
        self._loading_model: Optional[str] = None

        # ── Apply saved HF token to env ───────────────────────────
        if self.config.hf_token:
            os.environ["HF_TOKEN"] = self.config.hf_token

    # ── Public entry point ─────────────────────────────────────────
    def start(self) -> None:
        """Start the daemon: listen for clients on the Unix socket."""
        self._check_existing_daemon()
        self._write_pid()
        self._setup_signals()

        log.info("Daemon starting (PID %d)", os.getpid())

        # Auto-restore in background so socket is available immediately
        if self.config.auto_restore:
            threading.Thread(target=self._auto_restore, daemon=True).start()

        try:
            self._accept_loop()
        except KeyboardInterrupt:
            log.info("KeyboardInterrupt — shutting down")
        finally:
            self._shutdown()

    # ── Socket server ──────────────────────────────────────────────
    def _accept_loop(self) -> None:
        sock_path = str(SOCKET_PATH)

        # Remove stale socket
        if os.path.exists(sock_path):
            os.unlink(sock_path)

        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.bind(sock_path)
        self._server_sock.listen(10)
        self._server_sock.settimeout(1.0)  # for clean shutdown
        self._running = True

        log.info("Listening on %s", sock_path)

        while self._running:
            try:
                client, _ = self._server_sock.accept()
                t = threading.Thread(
                    target=self._handle_client, args=(client,), daemon=True
                )
                t.start()
            except socket.timeout:
                continue
            except OSError:
                if self._running:
                    log.exception("Socket accept error")
                break

    def _handle_client(self, client: socket.socket) -> None:
        """Process one request per connection (connect-per-command)."""
        try:
            rfile = client.makefile("r", encoding="utf-8")
            line = rfile.readline()
            if not line:
                return

            request = json.loads(line)
            cmd = request.pop("cmd", "")
            response = self._dispatch(cmd, request)

            payload = json.dumps(response, default=str) + "\n"
            client.sendall(payload.encode("utf-8"))
        except Exception:
            log.exception("Error handling client")
            try:
                err = json.dumps({"ok": False, "error": "Internal daemon error"}) + "\n"
                client.sendall(err.encode("utf-8"))
            except Exception:
                pass
        finally:
            try:
                client.close()
            except Exception:
                pass

    # ── Command dispatch ───────────────────────────────────────────
    def _dispatch(self, cmd: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            handler = getattr(self, f"_cmd_{cmd}", None)
            if handler is None:
                return {"ok": False, "error": f"Unknown command: {cmd}"}
            return handler(args)
        except Exception as exc:
            log.exception("Command %s failed", cmd)
            return {"ok": False, "error": str(exc)}

    # ═══════════════════════════════════════════════════════════════
    #  COMMAND HANDLERS
    # ═══════════════════════════════════════════════════════════════

    def _cmd_ping(self, _args: dict) -> dict:
        return {"ok": True, "data": "pong"}

    # ── Status ─────────────────────────────────────────────────────
    def _cmd_get_status(self, _args: dict) -> dict:
        return {
            "ok": True,
            "data": {
                "server_running": self.server_thread is not None
                and self.server_thread.is_running,
                "server_host": self.config.host,
                "server_port": self.config.port,
                "model_loaded": self.engine.is_loaded,
                "model_id": self.engine.model_id,
                "active_backend": self.engine.active_backend,
                "loading_model": self._loading_model,
            },
        }

    # ── Server control ─────────────────────────────────────────────
    def _cmd_start_server(self, _args: dict) -> dict:
        if self.server_thread and self.server_thread.is_running:
            return {"ok": False, "error": "Server is already running"}

        from src.apis import create_api, ServerThread

        api = create_api(self.engine, self.db, self.config)
        self.server_thread = ServerThread(
            api, host=self.config.host, port=self.config.port
        )
        self.server_thread.start()
        self.config.server_was_running = True
        self.config.save()
        log.info("API server started on %s:%d", self.config.host, self.config.port)
        return {"ok": True}

    def _cmd_stop_server(self, _args: dict) -> dict:
        if self.server_thread:
            self.server_thread.stop()
            self.server_thread = None
        self.config.server_was_running = False
        self.config.save()
        log.info("API server stopped")
        return {"ok": True}

    # ── Model management ───────────────────────────────────────────
    def _cmd_load_model(self, args: dict) -> dict:
        model_id = args.get("model_id", "")
        if not model_id:
            return {"ok": False, "error": "No model_id provided"}
        backend = args.get("backend")  # None / "auto" / "transformers" / "llama.cpp"
        self._loading_model = model_id
        try:
            with self._model_lock:
                self.engine.load_model(model_id, force_backend=backend)
                self.config.active_model = model_id
                self.config.save()
            log.info(
                "Model loaded: %s (backend=%s)",
                model_id,
                self.engine.active_backend,
            )
            return {
                "ok": True,
                "data": {"backend": self.engine.active_backend},
            }
        except Exception as exc:
            log.exception("Failed to load model %s", model_id)
            return {"ok": False, "error": str(exc)}
        finally:
            self._loading_model = None

    def _cmd_switch_backend(self, args: dict) -> dict:
        backend = args.get("backend", "")
        if backend not in ("transformers", "llama.cpp"):
            return {"ok": False, "error": f"Invalid backend: {backend!r}"}
        if not self.engine.is_loaded:
            return {"ok": False, "error": "No model loaded"}
        model_id = self.engine.model_id
        self._loading_model = model_id
        try:
            with self._model_lock:
                self.engine.reload_with_backend(backend)
            log.info(
                "Backend switched to %s for %s",
                self.engine.active_backend,
                model_id,
            )
            return {
                "ok": True,
                "data": {"backend": self.engine.active_backend, "model_id": model_id},
            }
        except Exception as exc:
            log.exception("Failed to switch backend to %s", backend)
            return {"ok": False, "error": str(exc)}
        finally:
            self._loading_model = None

    def _cmd_unload_model(self, _args: dict) -> dict:
        with self._model_lock:
            self.engine.unload_model()
            self.config.active_model = ""
            self.config.save()
        log.info("Model unloaded")
        return {"ok": True}

    def _cmd_model_status(self, _args: dict) -> dict:
        return {
            "ok": True,
            "data": {
                "is_loaded": self.engine.is_loaded,
                "model_id": self.engine.model_id,
                "loading_model": self._loading_model,
            },
        }

    # ── Model listing / search ─────────────────────────────────────
    def _cmd_list_models(self, _args: dict) -> dict:
        models = self.mm.list_downloaded_models()
        for m in models:
            m["is_loaded"] = self.engine.model_id == m["repo_id"]
            if m["is_loaded"] and self.engine.active_backend:
                m["backend"] = self.engine.active_backend
            else:
                m["backend"] = None
            if "last_modified" in m:
                m["last_modified"] = str(m["last_modified"])
        return {"ok": True, "data": models}

    def _cmd_search_models(self, args: dict) -> dict:
        query = args.get("query", "")
        if not query:
            return {"ok": False, "error": "No query provided"}
        results = self.mm.search_models(query)
        return {"ok": True, "data": results}

    def _cmd_delete_model(self, args: dict) -> dict:
        model_id = args.get("model_id", "")
        if not model_id:
            return {"ok": False, "error": "No model_id provided"}
        # Unload first if it's the active model
        if self.engine.model_id == model_id:
            with self._model_lock:
                self.engine.unload_model()
                self.config.active_model = ""
                self.config.save()
        ok = self.mm.delete_model(model_id)
        if ok:
            log.info("Deleted model %s", model_id)
        return {"ok": ok, "error": None if ok else f"Could not delete {model_id}"}

    def _cmd_cache_size(self, _args: dict) -> dict:
        return {"ok": True, "data": self.mm.total_cache_size()}

    # ── Download (async — runs in background) ──────────────────────
    def _cmd_list_repo_files(self, args: dict) -> dict:
        model_id = args.get("model_id", "")
        if not model_id:
            return {"ok": False, "error": "No model_id provided"}
        try:
            files = self.mm.list_repo_files(model_id)
            return {"ok": True, "data": files}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _cmd_download_model(self, args: dict) -> dict:
        model_id = args.get("model_id", "")
        if not model_id:
            return {"ok": False, "error": "No model_id provided"}
        filenames = args.get("filenames")  # None = all files
        # If a previous download thread is still alive (e.g. finishing
        # the current file after cancel), signal it to stop – the new
        # download thread will wait for it before proceeding.
        if self._dl_thread and self._dl_thread.is_alive():
            self.mm.cancel_download()
        self._start_download(model_id, filenames=filenames)
        return {"ok": True}

    def _cmd_download_status(self, _args: dict) -> dict:
        with self._dl_lock:
            state = dict(self._dl_state)
            if "files" in state and state["files"]:
                state["files"] = [dict(f) for f in state["files"]]
        return {"ok": True, "data": state}

    def _cmd_cancel_download(self, _args: dict) -> dict:
        self.mm.cancel_download()
        return {"ok": True}

    # ── API keys ───────────────────────────────────────────────────
    def _cmd_list_keys(self, args: dict) -> dict:
        status_filter = args.get("status_filter", "all")
        return {"ok": True, "data": self.db.get_keys(status_filter=status_filter)}

    def _cmd_add_key(self, args: dict) -> dict:
        from src.apis.keys import generate_api_key

        name = args.get("name", "unnamed")
        key = generate_api_key()
        try:
            self.db.add_key(name, key)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "data": {"key": key, "name": name}}

    def _cmd_set_key_status(self, args: dict) -> dict:
        key_id = args.get("key_id")
        status = args.get("status")
        if key_id is None or status is None:
            return {"ok": False, "error": "key_id and status required"}
        try:
            self.db.set_key_status(int(key_id), status)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True}

    # Legacy aliases (kept for backward compat)
    def _cmd_revoke_key(self, args: dict) -> dict:
        args["status"] = "revoked"
        return self._cmd_set_key_status(args)

    def _cmd_activate_key(self, args: dict) -> dict:
        args["status"] = "active"
        return self._cmd_set_key_status(args)

    def _cmd_delete_key(self, args: dict) -> dict:
        args["status"] = "deleted"
        return self._cmd_set_key_status(args)

    def _cmd_key_count(self, _args: dict) -> dict:
        return {"ok": True, "data": self.db.key_count()}

    def _cmd_key_usage(self, args: dict) -> dict:
        key_id = args.get("key_id")  # None → all keys
        if key_id is not None:
            key_id = int(key_id)
        return {"ok": True, "data": self.db.get_key_usage(key_id)}

    def _cmd_key_usage_history(self, args: dict) -> dict:
        key_id = args.get("key_id")
        if key_id is None:
            return {"ok": False, "error": "No key_id"}
        return {"ok": True, "data": self.db.get_key_usage_history(int(key_id))}

    # ── Device info ────────────────────────────────────────────────
    def _cmd_device_info(self, _args: dict) -> dict:
        return {"ok": True, "data": self.engine.device_info()}

    # ── Tuning / config ────────────────────────────────────────────
    def _cmd_get_config(self, _args: dict) -> dict:
        t = self.config.tuning
        return {
            "ok": True,
            "data": {
                "host": self.config.host,
                "port": self.config.port,
                "auto_restore": self.config.auto_restore,
                "hf_token": self.config.hf_token,
                "model_dir": self.config.model_dir,
                "theme": self.config.theme,
                "log_level": self.config.log_level,
                "temperature": t.temperature,
                "top_p": t.top_p,
                "top_k": t.top_k,
                "max_tokens": t.max_tokens,
                "repetition_penalty": t.repetition_penalty,
                "do_sample": t.do_sample,
            },
        }

    # ── App settings ─────────────────────────────────────────────
    def _cmd_set_hf_token(self, args: dict) -> dict:
        token = args.get("token", "").strip()
        if not token:
            # Clear token / logout
            self.config.hf_token = ""
            self.config.save()
            log.info("HuggingFace token cleared")
            return {"ok": True, "data": {"logged_in": False}}

        # Validate & login
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=token)
            user_info = api.whoami()
            username = user_info.get("name", "unknown")

            self.config.hf_token = token
            self.config.save()

            # Set for current process so downloads can use it
            import os
            os.environ["HF_TOKEN"] = token

            log.info("HuggingFace login successful: %s", username)
            return {
                "ok": True,
                "data": {
                    "logged_in": True,
                    "username": username,
                    "fullname": user_info.get("fullname", ""),
                },
            }
        except Exception as exc:
            log.warning("HuggingFace login failed: %s", exc)
            return {"ok": False, "error": f"Login failed: {exc}"}

    def _cmd_hf_status(self, _args: dict) -> dict:
        token = self.config.hf_token
        if not token:
            return {"ok": True, "data": {"logged_in": False}}
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=token)
            user_info = api.whoami()
            return {
                "ok": True,
                "data": {
                    "logged_in": True,
                    "username": user_info.get("name", "unknown"),
                    "fullname": user_info.get("fullname", ""),
                },
            }
        except Exception:
            return {"ok": True, "data": {"logged_in": False, "error": "Token invalid"}}

    def _cmd_set_server_port(self, args: dict) -> dict:
        port = args.get("port")
        host = args.get("host")
        if port is not None:
            port = int(port)
            if not (1024 <= port <= 65535):
                return {"ok": False, "error": "Port must be 1024–65535"}
            self.config.port = port
        if host is not None:
            self.config.host = str(host)
        self.config.save()

        restart_needed = self.server_thread and self.server_thread.is_running
        if restart_needed:
            # Restart server with new port
            self.server_thread.stop()
            self.server_thread = None
            from src.apis import create_api, ServerThread

            api = create_api(self.engine, self.db, self.config)
            self.server_thread = ServerThread(
                api, host=self.config.host, port=self.config.port
            )
            self.server_thread.start()
            log.info(
                "Server restarted on %s:%d", self.config.host, self.config.port
            )

        return {
            "ok": True,
            "data": {
                "host": self.config.host,
                "port": self.config.port,
                "restarted": bool(restart_needed),
            },
        }

    def _cmd_set_auto_restore(self, args: dict) -> dict:
        val = args.get("enabled")
        if val is not None:
            self.config.auto_restore = bool(val)
            self.config.save()
        return {"ok": True, "data": {"auto_restore": self.config.auto_restore}}

    def _cmd_set_model_dir(self, args: dict) -> dict:
        path = args.get("path", "").strip()
        if path:
            from pathlib import Path as P

            p = P(path).expanduser().resolve()
            if not p.exists():
                try:
                    p.mkdir(parents=True, exist_ok=True)
                except Exception as exc:
                    return {"ok": False, "error": f"Cannot create directory: {exc}"}
            if not p.is_dir():
                return {"ok": False, "error": "Path is not a directory"}
            self.config.model_dir = str(p)
        else:
            # Reset to default
            self.config.model_dir = ""
        self.config.save()

        # Update ModelManager cache_dir
        effective = self.config.model_dir or str(
            Path.home() / ".cache" / "huggingface"
        )
        self.mm.cache_dir = Path(effective)
        self.mm.hub_cache = self.mm.cache_dir / "hub"

        log.info("Model directory set to: %s", effective)
        return {
            "ok": True,
            "data": {"model_dir": self.config.model_dir, "effective": effective},
        }

    def _cmd_set_log_level(self, args: dict) -> dict:
        level = args.get("level", "INFO").upper()
        if level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            return {"ok": False, "error": "Invalid log level"}
        self.config.log_level = level
        self.config.save()
        log.setLevel(getattr(logging, level))
        return {"ok": True, "data": {"log_level": level}}

    def _cmd_set_tuning(self, args: dict) -> dict:
        t = self.config.tuning
        if "temperature" in args:
            t.temperature = float(args["temperature"])
        if "top_p" in args:
            t.top_p = float(args["top_p"])
        if "top_k" in args:
            t.top_k = int(args["top_k"])
        if "max_tokens" in args:
            t.max_tokens = int(args["max_tokens"])
        if "repetition_penalty" in args:
            t.repetition_penalty = float(args["repetition_penalty"])
        if "do_sample" in args:
            t.do_sample = bool(args["do_sample"])
        self.config.save()
        return {"ok": True}

    def _cmd_reset_tuning(self, _args: dict) -> dict:
        from src.tuning import TuningParams

        self.config.tuning = TuningParams()
        self.config.save()
        return {"ok": True, "data": self._cmd_get_config({})["data"]}

    # ── Generation ─────────────────────────────────────────────────
    def _cmd_generate(self, args: dict) -> dict:
        if not self.engine.is_loaded:
            return {"ok": False, "error": "No model loaded"}
        prompt = args.get("prompt", "")
        kwargs = self._tuning_kwargs()
        with self._model_lock:
            text = self.engine.generate(prompt, **kwargs)
        return {"ok": True, "data": {"text": text}}

    def _cmd_chat_generate(self, args: dict) -> dict:
        if not self.engine.is_loaded:
            return {"ok": False, "error": "No model loaded"}
        messages = args.get("messages", [])
        kwargs = self._tuning_kwargs()
        with self._model_lock:
            text = self.engine.chat_generate(messages, **kwargs)
        return {"ok": True, "data": {"text": text}}

    # ── Shutdown ───────────────────────────────────────────────────
    def _cmd_shutdown(self, _args: dict) -> dict:
        log.info("Shutdown requested")
        # Schedule shutdown after response is sent
        threading.Thread(target=self._deferred_shutdown, daemon=True).start()
        return {"ok": True}

    # ═══════════════════════════════════════════════════════════════
    #  INTERNAL HELPERS
    # ═══════════════════════════════════════════════════════════════

    def _tuning_kwargs(self) -> dict:
        t = self.config.tuning
        return {
            "max_tokens": t.max_tokens,
            "temperature": t.temperature,
            "top_p": t.top_p,
            "top_k": t.top_k,
            "repetition_penalty": t.repetition_penalty,
            "do_sample": t.do_sample,
        }

    # ── Download helpers ───────────────────────────────────────────
    @staticmethod
    def _empty_dl_state() -> dict:
        return {
            "active": False,
            "model_id": None,
            "phase": "idle",
            "files": [],
            "current_idx": 0,
            "total_files": 0,
            "bytes_done": 0,
            "bytes_total": 0,
            "progress_pct": 0.0,
            "speed": 0.0,
            "speed_str": "—",
            "error": None,
        }

    def _start_download(
        self, model_id: str, filenames: list[str] | None = None
    ) -> None:
        old_thread = self._dl_thread          # keep ref so new thread can join

        with self._dl_lock:
            self._dl_state = self._empty_dl_state()
            self._dl_state["active"] = True
            self._dl_state["model_id"] = model_id
            self._dl_state["phase"] = "preparing"

        self._dl_thread = threading.Thread(
            target=self._run_download,
            args=(model_id, old_thread, filenames),
            daemon=True,
        )
        self._dl_thread.start()

    def _run_download(
        self,
        model_id: str,
        old_thread: threading.Thread | None = None,
        filenames: list[str] | None = None,
    ) -> None:
        from src.llms.model_manager import DownloadCancelled

        # If a previous download thread is still running (finishing its
        # current file after a cancel), wait for it so we don't have two
        # threads calling hf_hub_download concurrently.
        if old_thread is not None and old_thread.is_alive():
            log.info("Waiting for previous download thread to finish …")
            old_thread.join(timeout=60)

        _last_update = [0.0]  # mutable for closure

        def on_file_list(files: list[dict]) -> None:
            with self._dl_lock:
                self._dl_state["files"] = [
                    {
                        "name": f["name"],
                        "size": f["size"],
                        "size_str": f["size_str"],
                        "status": "pending",
                    }
                    for f in files
                ]
                self._dl_state["total_files"] = len(files)
                self._dl_state["bytes_total"] = sum(f["size"] for f in files)
                self._dl_state["phase"] = "downloading"

        def on_progress(
            fname: str,
            idx: int,
            total: int,
            bytes_done: int,
            bytes_total: int,
            file_status: str,
            speed: float = 0.0,
        ) -> None:
            now = time.monotonic()
            # Throttle: update state at most ~4×/sec unless file done
            if file_status != "done" and (now - _last_update[0]) < 0.25:
                return
            _last_update[0] = now

            with self._dl_lock:
                self._dl_state["current_idx"] = idx
                self._dl_state["bytes_done"] = bytes_done
                self._dl_state["bytes_total"] = bytes_total
                pct = (bytes_done / bytes_total * 100) if bytes_total else 0
                self._dl_state["progress_pct"] = pct
                self._dl_state["speed"] = speed
                self._dl_state["speed_str"] = (
                    f"{self.mm._human_size(int(speed))}/s" if speed > 0 else "—"
                )
                # Update per-file status
                for f in self._dl_state["files"]:
                    if f["name"] == fname:
                        f["status"] = file_status
                        break

        try:
            log.info("Download started: %s", model_id)
            self.mm.download_model_with_progress(
                model_id,
                on_progress=on_progress,
                on_file_list=on_file_list,
                filenames=filenames,
            )
            with self._dl_lock:
                self._dl_state["active"] = False
                self._dl_state["phase"] = "completed"
                self._dl_state["progress_pct"] = 100.0
                # Mark any remaining pending as done
                for f in self._dl_state["files"]:
                    if f["status"] == "pending":
                        f["status"] = "done"
            log.info("Download completed: %s", model_id)

        except DownloadCancelled:
            with self._dl_lock:
                self._dl_state["active"] = False
                self._dl_state["phase"] = "cancelled"
                for f in self._dl_state["files"]:
                    if f["status"] == "pending":
                        f["status"] = "stopped"
            log.info("Download cancelled: %s", model_id)

        except Exception as exc:
            with self._dl_lock:
                self._dl_state["active"] = False
                self._dl_state["phase"] = "error"
                self._dl_state["error"] = str(exc)
            log.exception("Download failed: %s", model_id)

    # ── Auto-restore ───────────────────────────────────────────────
    def _auto_restore(self) -> None:
        if self.config.active_model:
            model_id = self.config.active_model
            log.info("Auto-restoring model: %s", model_id)
            self._loading_model = model_id
            try:
                with self._model_lock:
                    self.engine.load_model(model_id)
                log.info("Model restored: %s", model_id)
            except Exception:
                log.exception("Could not restore model %s", model_id)
                self.config.active_model = ""
                self.config.save()
            finally:
                self._loading_model = None

        if self.config.server_was_running:
            try:
                from src.apis import create_api, ServerThread

                api = create_api(self.engine, self.db, self.config)
                self.server_thread = ServerThread(
                    api, host=self.config.host, port=self.config.port
                )
                self.server_thread.start()
                log.info("API server auto-started on %s:%d", self.config.host, self.config.port)
            except Exception:
                log.exception("Could not auto-start server")

    # ── Lifecycle helpers ──────────────────────────────────────────
    def _check_existing_daemon(self) -> None:
        if PID_FILE.exists():
            try:
                pid = int(PID_FILE.read_text().strip())
                os.kill(pid, 0)
                # Process exists — try to ping
                try:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.settimeout(2)
                    s.connect(str(SOCKET_PATH))
                    s.sendall(b'{"cmd":"ping"}\n')
                    resp = s.makefile("r").readline()
                    s.close()
                    if "pong" in resp:
                        log.error("Daemon already running (PID %d)", pid)
                        print(f"Daemon already running (PID {pid})")
                        sys.exit(1)
                except Exception:
                    pass  # stale socket, continue
            except ProcessLookupError:
                pass  # stale PID, continue

    def _write_pid(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(os.getpid()))

    def _setup_signals(self) -> None:
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        log.info("Signal %d received", signum)
        self._running = False

    def _deferred_shutdown(self) -> None:
        """Give the response time to be sent, then shut down."""
        time.sleep(0.3)
        self._running = False

    def _shutdown(self) -> None:
        log.info("Daemon shutting down")
        self._running = False

        # Save state
        self.config.server_was_running = (
            self.server_thread is not None and self.server_thread.is_running
        )
        self.config.save()

        # Stop API server
        if self.server_thread:
            try:
                self.server_thread.stop()
            except Exception:
                pass
            self.server_thread = None

        # Unload model to free GPU
        if self.engine.is_loaded:
            try:
                self.engine.unload_model()
            except Exception:
                pass

        # Clean up socket and PID
        try:
            self._server_sock.close()
        except Exception:
            pass
        SOCKET_PATH.unlink(missing_ok=True)
        PID_FILE.unlink(missing_ok=True)

        log.info("Daemon stopped")


# ── Module entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    daemon = LLMDaemon()
    daemon.start()
