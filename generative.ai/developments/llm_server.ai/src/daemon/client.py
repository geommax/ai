"""Thin client for communicating with the LLM daemon over Unix socket."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any, Dict


SOCKET_PATH = Path.home() / ".config" / "llm_server_ai" / "daemon.sock"


class DaemonDisconnected(Exception):
    """Raised when the daemon is unreachable."""


class DaemonClient:
    """Send JSON commands to the daemon, one connection per call.

    This is inherently thread-safe: each ``send_command`` opens its own
    short-lived Unix socket connection, so multiple TUI worker threads
    can call the daemon concurrently without contention.
    """

    def __init__(self, socket_path: str | Path | None = None) -> None:
        self.socket_path = str(socket_path or SOCKET_PATH)

    # ── Core transport ─────────────────────────────────────────────
    def send_command(self, cmd: str, **kwargs: Any) -> Dict[str, Any]:
        """Send *cmd* with keyword arguments and return the response dict.

        Raises ``DaemonDisconnected`` if the daemon cannot be reached.
        """
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(300)  # generous timeout for load_model / generate
        try:
            sock.connect(self.socket_path)
            msg = json.dumps({"cmd": cmd, **kwargs}) + "\n"
            sock.sendall(msg.encode("utf-8"))

            rfile = sock.makefile("r", encoding="utf-8")
            line = rfile.readline()
            if not line:
                raise DaemonDisconnected("Empty response from daemon")
            return json.loads(line)
        except (ConnectionRefusedError, FileNotFoundError, OSError) as exc:
            raise DaemonDisconnected(f"Cannot reach daemon: {exc}") from exc
        finally:
            try:
                sock.close()
            except Exception:
                pass

    # ── Convenience helpers ────────────────────────────────────────
    def ping(self) -> bool:
        try:
            r = self.send_command("ping")
            return r.get("data") == "pong"
        except DaemonDisconnected:
            return False

    def get_status(self) -> dict:
        return self.send_command("get_status").get("data", {})

    def start_server(self) -> dict:
        return self.send_command("start_server")

    def stop_server(self) -> dict:
        return self.send_command("stop_server")

    def load_model(self, model_id: str, backend: str | None = None) -> dict:
        kwargs: dict = {"model_id": model_id}
        if backend and backend != "auto":
            kwargs["backend"] = backend
        return self.send_command("load_model", **kwargs)

    def switch_backend(self, backend: str) -> dict:
        return self.send_command("switch_backend", backend=backend)

    def unload_model(self) -> dict:
        return self.send_command("unload_model")

    def list_models(self) -> list:
        return self.send_command("list_models").get("data", [])

    def search_models(self, query: str) -> list:
        return self.send_command("search_models", query=query).get("data", [])

    def list_repo_files(self, model_id: str) -> list:
        return self.send_command("list_repo_files", model_id=model_id).get(
            "data", []
        )

    def download_model(
        self, model_id: str, filenames: list[str] | None = None
    ) -> dict:
        kwargs: dict = {"model_id": model_id}
        if filenames:
            kwargs["filenames"] = filenames
        return self.send_command("download_model", **kwargs)

    def download_status(self) -> dict:
        return self.send_command("download_status").get("data", {})

    def cancel_download(self) -> dict:
        return self.send_command("cancel_download")

    def delete_model(self, model_id: str) -> dict:
        return self.send_command("delete_model", model_id=model_id)

    def cache_size(self) -> str:
        return self.send_command("cache_size").get("data", "0 B")

    def device_info(self) -> dict:
        return self.send_command("device_info").get("data", {})

    def list_keys(self, status_filter: str = "all") -> list:
        return self.send_command("list_keys", status_filter=status_filter).get("data", [])

    def add_key(self, name: str) -> dict:
        return self.send_command("add_key", name=name)

    def set_key_status(self, key_id: int, status: str) -> dict:
        return self.send_command("set_key_status", key_id=key_id, status=status)

    def revoke_key(self, key_id: int) -> dict:
        return self.set_key_status(key_id, "revoked")

    def activate_key(self, key_id: int) -> dict:
        return self.set_key_status(key_id, "active")

    def delete_key(self, key_id: int) -> dict:
        return self.set_key_status(key_id, "deleted")

    def key_count(self) -> int:
        return self.send_command("key_count").get("data", 0)

    def key_usage(self, key_id: int | None = None) -> list:
        kwargs: dict[str, Any] = {}
        if key_id is not None:
            kwargs["key_id"] = key_id
        return self.send_command("key_usage", **kwargs).get("data", [])

    def key_usage_history(self, key_id: int) -> list:
        return self.send_command("key_usage_history", key_id=key_id).get("data", [])

    def get_config(self) -> dict:
        return self.send_command("get_config").get("data", {})

    def set_tuning(self, **params: Any) -> dict:
        return self.send_command("set_tuning", **params)

    def reset_tuning(self) -> dict:
        return self.send_command("reset_tuning")

    # ── App settings ───────────────────────────────────────────────
    def set_hf_token(self, token: str) -> dict:
        return self.send_command("set_hf_token", token=token)

    def hf_status(self) -> dict:
        return self.send_command("hf_status").get("data", {})

    def set_server_port(self, port: int, host: str | None = None) -> dict:
        kwargs: dict[str, Any] = {"port": port}
        if host is not None:
            kwargs["host"] = host
        return self.send_command("set_server_port", **kwargs)

    def set_auto_restore(self, enabled: bool) -> dict:
        return self.send_command("set_auto_restore", enabled=enabled)

    def set_model_dir(self, path: str) -> dict:
        return self.send_command("set_model_dir", path=path)

    def set_log_level(self, level: str) -> dict:
        return self.send_command("set_log_level", level=level)

    def generate(self, prompt: str) -> dict:
        return self.send_command("generate", prompt=prompt)

    def chat_generate(self, messages: list) -> dict:
        return self.send_command("chat_generate", messages=messages)

    def shutdown(self) -> dict:
        return self.send_command("shutdown")
