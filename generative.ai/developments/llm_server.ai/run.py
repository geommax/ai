#!/usr/bin/env python3
"""LLM Server.AI — Local LLM Server with TUI Management.

Usage
-----
    python run.py              Start daemon (if needed) + launch TUI
    python run.py --daemon     Start daemon in foreground (for systemd / tmux)
    python run.py --stop       Stop the background daemon
    python run.py --status     Show daemon status
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "llm_server_ai"
PID_FILE = CONFIG_DIR / "daemon.pid"
SOCKET_PATH = CONFIG_DIR / "daemon.sock"
LOG_FILE = CONFIG_DIR / "daemon.log"


# ── Daemon helpers ─────────────────────────────────────────────────────

def is_daemon_running() -> bool:
    """Return True if the daemon process is alive and responding."""
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)  # check process exists
    except (ProcessLookupError, ValueError, OSError):
        return False
    # Process exists — try socket ping
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect(str(SOCKET_PATH))
        s.sendall(b'{"cmd":"ping"}\n')
        resp = s.makefile("r").readline()
        s.close()
        return "pong" in resp
    except Exception:
        return False


def start_daemon_background() -> bool:
    """Launch the daemon as a detached background process."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    log_fh = open(str(LOG_FILE), "a")
    subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--daemon"],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    # Wait for daemon to become ready
    for _ in range(100):  # up to 10 s
        time.sleep(0.1)
        if is_daemon_running():
            return True
    return False


def stop_daemon() -> bool:
    """Ask the daemon to shut down gracefully."""
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(3)
        s.connect(str(SOCKET_PATH))
        s.sendall(b'{"cmd":"shutdown"}\n')
        s.makefile("r").readline()
        s.close()
    except Exception:
        pass
    # Wait briefly
    for _ in range(30):
        time.sleep(0.1)
        if not is_daemon_running():
            return True
    # Force kill
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            time.sleep(0.5)
            return True
        except Exception:
            pass
    return False


def print_status() -> None:
    """Print daemon status to stdout."""
    if is_daemon_running():
        pid = PID_FILE.read_text().strip()
        print(f"  ● Daemon is [running]  (PID {pid})")
        # Get more info
        try:
            from src.daemon.client import DaemonClient
            c = DaemonClient()
            st = c.get_status()
            srv = "ON" if st.get("server_running") else "OFF"
            mdl = st.get("model_id") or "none"
            port = st.get("server_port", "?")
            print(f"  ● Server {srv}  :{port}")
            print(f"  ● Model: {mdl}")
        except Exception:
            pass
    else:
        print("  ○ Daemon is [not running]")


# ── Entry point ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Server.AI — Local LLM Inference Server"
    )
    parser.add_argument(
        "--daemon", action="store_true",
        help="Start the daemon in the foreground (for systemd / tmux)",
    )
    parser.add_argument(
        "--stop", action="store_true",
        help="Stop the background daemon",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show daemon status",
    )
    args = parser.parse_args()

    # ── --stop ─────────────────────────────────────────────────────
    if args.stop:
        if is_daemon_running():
            if stop_daemon():
                print("Daemon stopped ✓")
            else:
                print("Failed to stop daemon")
                sys.exit(1)
        else:
            print("Daemon is not running")
        return

    # ── --status ───────────────────────────────────────────────────
    if args.status:
        print_status()
        return

    # ── --daemon (foreground) ──────────────────────────────────────
    if args.daemon:
        from src.daemon.process import LLMDaemon
        daemon = LLMDaemon()
        daemon.start()
        return

    # ── Default: ensure daemon + launch TUI ────────────────────────
    if not is_daemon_running():
        print("Starting LLM Server.AI daemon…")
        if start_daemon_background():
            print("Daemon started ✓")
        else:
            print("ERROR: Failed to start daemon. Check ~/.config/llm_server_ai/daemon.log")
            sys.exit(1)
    else:
        print("Daemon already running ✓")

    from src.ui.app import LLMServerApp
    app = LLMServerApp()
    app.run()


if __name__ == "__main__":
    main()
