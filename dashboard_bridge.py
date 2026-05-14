from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
HOST = "127.0.0.1"
APPLI_PORT = 8502
DASHBOARD_PORT = 8501
APPLI_SCRIPT = BASE_DIR / "appli.py"
DASHBOARD_SCRIPT = BASE_DIR / "dashboard.py"
RESULTS_FILE = BASE_DIR / "dashboard_state.json"

TASKS = [
    {
        "id": "01",
        "title": "Current Waveform Analysis",
        "audience_label": "Current waveform match",
        "accent": "#0f766e",
        "summary": "How closely the synthetic current follows the real welding current in time.",
    },
    {
        "id": "02",
        "title": "Power And Energy Delivery",
        "audience_label": "Power and energy",
        "accent": "#c2410c",
        "summary": "How the synthetic signal delivers power and total energy compared with the real weld.",
    },
    {
        "id": "03",
        "title": "Quantitative Match Metrics",
        "audience_label": "Fit quality metrics",
        "accent": "#475569",
        "summary": "Compact numeric scores that summarize waveform realism and overall agreement.",
    },
    {
        "id": "04",
        "title": "Frequency And Temporal Patterns",
        "audience_label": "Frequency behavior",
        "accent": "#4338ca",
        "summary": "Whether oscillation frequency content and time-memory behavior are preserved.",
    },
    {
        "id": "05",
        "title": "Heat Transfer And Welding Heat",
        "audience_label": "Heat engineering",
        "accent": "#b45309",
        "summary": "How the electrical signals translate into useful heat, losses, and heat density.",
    },
    {
        "id": "06",
        "title": "Material Response Analysis For Mild Steel",
        "audience_label": "Material response",
        "accent": "#166534",
        "summary": "What the delivered heat means for the mild-steel workpiece temperature and sufficiency.",
    },
    {
        "id": "07",
        "title": "Peak Detection And Peak-Type Classification",
        "audience_label": "Peak classification",
        "accent": "#be123c",
        "summary": "How detected peaks compare in count, strength, and category share.",
    },
]
TASK_LOOKUP = {task["id"]: task for task in TASKS}


def appli_url(task_id: str | None = None) -> str:
    url = f"http://localhost:{APPLI_PORT}"
    if task_id:
        return f"{url}/?task={task_id}"
    return url


def dashboard_url() -> str:
    return f"http://localhost:{DASHBOARD_PORT}"


def port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((HOST, port)) == 0


def launch_streamlit_app(script_path: Path, port: int, wait_seconds: float = 12.0) -> tuple[bool, str]:
    if not script_path.exists():
        return False, f"Could not find {script_path.name}."

    if port_is_open(port):
        return True, f"{script_path.name} is already running on port {port}."

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script_path),
        f"--server.port={port}",
        "--server.headless=true",
    ]

    creation_flags = (
        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        | getattr(subprocess, "DETACHED_PROCESS", 0)
    )

    try:
        subprocess.Popen(
            command,
            cwd=str(BASE_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
    except Exception as exc:
        return False, f"Unable to launch {script_path.name}: {exc}"

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if port_is_open(port):
            return True, f"{script_path.name} launched on port {port}."
        time.sleep(0.3)

    return False, f"{script_path.name} did not start on port {port}."


def load_dashboard_state() -> dict[str, Any]:
    if not RESULTS_FILE.exists():
        return {}

    try:
        return json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_dashboard_state(payload: dict[str, Any]) -> tuple[bool, str]:
    temp_path = RESULTS_FILE.with_suffix(".tmp")

    try:
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(RESULTS_FILE)
    except Exception as exc:
        return False, f"Could not save dashboard results: {exc}"

    return True, f"Saved latest results for {payload.get('source_file', 'the current run')}."
