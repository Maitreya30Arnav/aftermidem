from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from dashboard_bridge import (
    DASHBOARD_PORT,
    DASHBOARD_SCRIPT,
    TASK_LOOKUP,
    dashboard_url,
    launch_streamlit_app,
    port_is_open,
    save_dashboard_state,
)
from generator import generate_synthetic_and_metrics
from heat_utils import (
    DEFAULT_HEAT_LOSS_FRACTIONS,
    HEAT_LOSS_MECHANISMS,
    MILD_STEEL_PROPERTIES,
    PULSE_CATEGORY_ORDER,
    PULSE_CATEGORY_REFERENCE,
    analyze_welding_pulses,
    calculate_welding_heat,
    material_analysis,
)


st.set_page_config(page_title="Welding Dashboard", layout="wide")


REAL_COLOR = "#184e77"
SYN_COLOR = "#f97316"
TEXT_COLOR = "#0f172a"
PEAK_CATEGORY_ORDER = ["Class 1", "Class 2", "Class 3", "Class 4"]
PEAK_CATEGORY_REFERENCE = {
    "Class 1": {
        "band": "0.90 <= peak current / I0 <= 1.00",
        "description": "stable high-current pulse with the strongest peak power",
    },
    "Class 2": {
        "band": "0.65 <= peak current / I0 < 0.90",
        "description": "useful pulse with lower peak energy than Class 1",
    },
    "Class 3": {
        "band": "0.55 <= peak current / I0 < 0.65",
        "description": "weak or unstable pulse below the Class 2 range",
    },
    "Class 4": {
        "band": "peak current / I0 < 0.55",
        "description": "very low-current or short-circuit pulse near arc collapse",
    },
}
PEAK_CATEGORY_COLORS = {
    "Class 1": "#16a34a",
    "Class 2": "#2563eb",
    "Class 3": "#d97706",
    "Class 4": "#dc2626",
}
ANALYSIS_SECTION_THEMES = {
    "current": {
        "label": "Waveform Analysis",
        "accent": "#0f766e",
        "soft": "#2dd4bf",
        "start": "rgba(15, 118, 110, 0.16)",
        "end": "rgba(45, 212, 191, 0.05)",
    },
    "power": {
        "label": "Power And Energy",
        "accent": "#c2410c",
        "soft": "#fb923c",
        "start": "rgba(194, 65, 12, 0.15)",
        "end": "rgba(251, 146, 60, 0.05)",
    },
    "metrics": {
        "label": "Fit Quality",
        "accent": "#475569",
        "soft": "#94a3b8",
        "start": "rgba(71, 85, 105, 0.16)",
        "end": "rgba(148, 163, 184, 0.06)",
    },
    "frequency": {
        "label": "Frequency And Memory",
        "accent": "#4338ca",
        "soft": "#818cf8",
        "start": "rgba(67, 56, 202, 0.15)",
        "end": "rgba(129, 140, 248, 0.06)",
    },
    "heat": {
        "label": "Heat Transfer",
        "accent": "#b45309",
        "soft": "#f59e0b",
        "start": "rgba(180, 83, 9, 0.16)",
        "end": "rgba(245, 158, 11, 0.05)",
    },
    "material": {
        "label": "Material Response",
        "accent": "#166534",
        "soft": "#4ade80",
        "start": "rgba(22, 101, 52, 0.15)",
        "end": "rgba(74, 222, 128, 0.05)",
    },
    "peak": {
        "label": "Peak Classification",
        "accent": "#be123c",
        "soft": "#fb7185",
        "start": "rgba(190, 18, 60, 0.15)",
        "end": "rgba(251, 113, 133, 0.05)",
    },
}
PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
}


st.markdown(
    """
    <style>
    :root {
        --bg-top: #fffaf2;
        --bg-bottom: #f6fbff;
        --panel: rgba(255, 255, 255, 0.92);
        --panel-strong: #ffffff;
        --text-main: #111827;
        --text-soft: #475569;
        --line-soft: rgba(15, 23, 42, 0.08);
        --accent-dark: #0f172a;
        --accent-blue: #184e77;
        --accent-orange: #f97316;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(249, 115, 22, 0.10), transparent 24%),
            radial-gradient(circle at top right, rgba(24, 78, 119, 0.10), transparent 22%),
            linear-gradient(180deg, var(--bg-top) 0%, #ffffff 34%, var(--bg-bottom) 100%);
        color: var(--text-main);
        font-family: "Segoe UI", sans-serif;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1880px;
    }

    h1 {
        font-size: 3.9rem !important;
        font-weight: 800 !important;
        color: var(--accent-dark) !important;
        line-height: 1.08 !important;
    }

    h2 {
        font-size: 2.35rem !important;
        font-weight: 750 !important;
        color: var(--accent-dark) !important;
    }

    h3 {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: var(--accent-dark) !important;
    }

    p, li, label, .stMarkdown, .stCaption {
        font-size: 1.12rem !important;
        line-height: 1.7 !important;
        color: var(--text-main);
    }

    [data-testid="stFileUploader"] {
        background: var(--panel);
        border: 1px solid var(--line-soft);
        border-radius: 22px;
        padding: 0.7rem 0.9rem;
        box-shadow: 0 14px 35px rgba(15, 23, 42, 0.06);
    }

    [data-testid="stFileUploader"] section {
        padding: 0.45rem 0.2rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, #0f172a, #1f2937) !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 0.8rem 1rem !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] * {
        color: #f8fafc !important;
        opacity: 1 !important;
    }

    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, var(--accent-orange), #ea580c) !important;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        min-width: 160px !important;
        min-height: 2.85rem !important;
        box-shadow: 0 14px 28px rgba(249, 115, 22, 0.28) !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    [data-testid="stFileUploader"] button:hover {
        background: linear-gradient(135deg, #fb923c, #f97316) !important;
    }

    [data-testid="stFileUploader"] button * {
        color: white !important;
        fill: white !important;
        opacity: 1 !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--accent-orange), #ea580c) !important;
        color: white !important;
        border: none !important;
        border-radius: 999px !important;
        font-size: 1.08rem !important;
        font-weight: 700 !important;
        min-height: 3.2rem !important;
        padding: 0.35rem 1.6rem !important;
        box-shadow: 0 14px 28px rgba(249, 115, 22, 0.25);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 18px 30px rgba(249, 115, 22, 0.32);
    }

    [data-testid="stNumberInputContainer"] {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(249, 250, 251, 0.96));
        border: 1px solid var(--line-soft);
        border-radius: 999px;
        padding: 0.1rem 0.55rem;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.05);
    }

    [data-testid="stNumberInputContainer"] > div,
    [data-testid="stNumberInputContainer"] div[data-baseweb="input"],
    [data-testid="stNumberInputContainer"] div[data-baseweb="base-input"],
    [data-testid="stNumberInputContainer"] div[data-baseweb="input"] > div,
    [data-testid="stNumberInputContainer"] div[data-baseweb="base-input"] > div {
        background: white !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: 999px !important;
    }

    [data-testid="stNumberInputContainer"] input {
        background: white !important;
        color: var(--accent-dark) !important;
        border: none !important;
        font-size: 1.08rem !important;
        font-weight: 700 !important;
        box-shadow: none !important;
    }

    [data-testid="stNumberInputContainer"] button {
        background: white !important;
        color: var(--accent-dark) !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: 999px !important;
    }

    [data-testid="stNumberInputContainer"] button:hover {
        background: rgba(248, 250, 252, 1) !important;
        color: var(--accent-dark) !important;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(247, 250, 252, 0.96));
        border: 1px solid rgba(15, 23, 42, 0.07);
        border-radius: 20px;
        padding: 1rem 1.1rem;
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.06);
    }

    [data-testid="stMetricLabel"] {
        font-size: 1.02rem !important;
        font-weight: 650 !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
    }

    button[data-baseweb="tab"] {
        font-size: 1.08rem !important;
        font-weight: 700 !important;
        background: rgba(255, 255, 255, 0.78) !important;
        border: 1px solid rgba(15, 23, 42, 0.08) !important;
        border-radius: 999px !important;
        padding: 0.65rem 1.15rem !important;
        margin-right: 0.45rem !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent-dark), var(--accent-blue)) !important;
        color: white !important;
        border-color: transparent !important;
    }

    .hero-panel {
        background: linear-gradient(140deg, rgba(255, 255, 255, 0.98), rgba(248, 250, 252, 0.95));
        border: 1px solid rgba(15, 23, 42, 0.07);
        border-radius: 28px;
        padding: 2.15rem 2rem;
        box-shadow: 0 24px 60px rgba(15, 23, 42, 0.08);
        text-align: center;
    }

    .hero-kicker {
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-weight: 700;
        color: var(--accent-orange);
        margin-bottom: 0.65rem;
    }

    .hero-title {
        font-size: 4rem;
        line-height: 1.05;
        font-weight: 850;
        color: var(--accent-dark);
        margin-bottom: 0.8rem;
    }

    .hero-subtitle {
        font-size: 1.2rem;
        line-height: 1.75;
        color: var(--text-soft);
        max-width: 880px;
        margin: 0 auto;
    }

    .hero-pills {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 0.7rem;
        margin-top: 1.25rem;
    }

    .hero-pill {
        background: rgba(15, 23, 42, 0.05);
        border: 1px solid rgba(15, 23, 42, 0.07);
        border-radius: 999px;
        padding: 0.5rem 0.9rem;
        font-size: 0.98rem;
        font-weight: 700;
        color: var(--accent-dark);
    }

    .copy-card {
        background: var(--panel);
        border: 1px solid rgba(15, 23, 42, 0.07);
        border-radius: 24px;
        padding: 1.35rem 1.45rem;
        box-shadow: 0 18px 44px rgba(15, 23, 42, 0.06);
        margin: 0.7rem 0 1rem 0;
    }

    .copy-title {
        font-size: 1.22rem;
        font-weight: 800;
        color: var(--accent-dark);
        margin-bottom: 0.35rem;
    }

    .copy-body {
        font-size: 1.08rem;
        color: var(--text-soft);
    }

    .copy-card.accent {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(24, 78, 119, 0.96));
        color: white;
        border: none;
    }

    .copy-card.accent .copy-title,
    .copy-card.accent .copy-body {
        color: white;
    }

    .analysis-band {
        position: relative;
        overflow: hidden;
        border-radius: 28px;
        padding: 1.3rem 1.45rem 1.2rem 1.45rem;
        margin: 1.15rem 0 1rem 0;
        border: 1px solid rgba(15, 23, 42, 0.08);
        background: linear-gradient(135deg, var(--band-start), var(--band-end));
        box-shadow: 0 18px 44px rgba(15, 23, 42, 0.07);
    }

    .analysis-band::before {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at top right, rgba(255, 255, 255, 0.7), transparent 34%);
        pointer-events: none;
    }

    .analysis-band-top {
        position: relative;
        z-index: 1;
        display: flex;
        align-items: center;
        gap: 0.9rem;
        margin-bottom: 0.55rem;
    }

    .analysis-band-number {
        width: 54px;
        height: 54px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.15rem;
        font-weight: 850;
        color: white;
        background: linear-gradient(135deg, var(--band-accent), var(--band-soft));
        box-shadow: 0 14px 28px rgba(15, 23, 42, 0.12);
        flex-shrink: 0;
    }

    .analysis-band-kicker {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        font-weight: 800;
        color: var(--band-accent);
    }

    .analysis-band-title {
        position: relative;
        z-index: 1;
        font-size: 2rem;
        line-height: 1.08;
        font-weight: 850;
        color: var(--accent-dark);
        margin-bottom: 0.25rem;
    }

    .analysis-band-body {
        position: relative;
        z-index: 1;
        max-width: 980px;
        font-size: 1.04rem;
        color: var(--text-soft);
    }

    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(15, 23, 42, 0.14), transparent);
        margin: 1.15rem 0 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_copy_card(title: str, body: str, tone: str = "default") -> None:
    tone_class = " accent" if tone == "accent" else ""
    st.markdown(
        f"""
        <div class="copy-card{tone_class}">
            <div class="copy-title">{title}</div>
            <div class="copy-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_analysis_section_header(number: int, title: str, body: str, theme: str) -> None:
    theme_info = ANALYSIS_SECTION_THEMES[theme]
    st.markdown(
        f"""
        <div class="analysis-band"
             style="--band-start: {theme_info['start']};
                    --band-end: {theme_info['end']};
                    --band-accent: {theme_info['accent']};
                    --band-soft: {theme_info['soft']};">
            <div class="analysis-band-top">
                <div class="analysis-band-number">{number:02d}</div>
                <div class="analysis-band-kicker">{theme_info['label']}</div>
            </div>
            <div class="analysis-band-title">{title}</div>
            <div class="analysis-band-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def resolve_voltage_signal(dataframe, start_row: int, end_row: int, default_voltage_v: float):
    voltage_candidates = (
        "Voltage_V",
        "Voltage",
        "voltage_v",
        "voltage",
        "ArcVoltage_V",
        "ArcVoltage",
        "V",
    )
    for column_name in voltage_candidates:
        if column_name in dataframe.columns:
            voltage_signal = dataframe.iloc[start_row:end_row][column_name].to_numpy(dtype=float)
            return voltage_signal, column_name

    sample_count = max(end_row - start_row, 0)
    return np.full(sample_count, float(default_voltage_v)), f"Constant {default_voltage_v:.2f} V"


def format_metric_value(value: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def safe_json_value(value, digits: int = 6):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(numeric_value):
        return None
    return round(numeric_value, digits)


def downsample_indices(length: int, limit: int = 180):
    if length <= 0:
        return np.array([], dtype=int)
    if length <= limit:
        return np.arange(length, dtype=int)
    return np.linspace(0, length - 1, limit, dtype=int)


def downsample_pair(x_values, y_values, limit: int = 180):
    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(y_values, dtype=float)
    sample_count = min(len(x_array), len(y_array))
    if sample_count <= 0:
        return [], []

    selected_indices = downsample_indices(sample_count, limit)
    x_points = [safe_json_value(x_array[index]) for index in selected_indices]
    y_points = [safe_json_value(y_array[index]) for index in selected_indices]
    return x_points, y_points


def downsample_values(values, limit: int = 180):
    value_array = np.asarray(values, dtype=float)
    if len(value_array) <= 0:
        return []
    selected_indices = downsample_indices(len(value_array), limit)
    return [safe_json_value(value_array[index]) for index in selected_indices]


def build_dual_line_chart_payload(
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    x_real,
    y_real,
    x_syn,
    y_syn,
    real_name: str = "Real",
    synthetic_name: str = "Synthetic",
    use_log_y: bool = False,
) -> dict:
    real_x_points, real_y_points = downsample_pair(x_real, y_real)
    syn_x_points, syn_y_points = downsample_pair(x_syn, y_syn)
    return {
        "kind": "dual_line",
        "title": title,
        "xaxis_title": xaxis_title,
        "yaxis_title": yaxis_title,
        "use_log_y": use_log_y,
        "series": [
            {
                "name": real_name,
                "x": real_x_points,
                "y": real_y_points,
                "color": REAL_COLOR,
            },
            {
                "name": synthetic_name,
                "x": syn_x_points,
                "y": syn_y_points,
                "color": SYN_COLOR,
            },
        ],
    }


def build_grouped_bar_chart_payload(title: str, yaxis_title: str, labels, real_values, synthetic_values) -> dict:
    return {
        "kind": "grouped_bar",
        "title": title,
        "yaxis_title": yaxis_title,
        "labels": list(labels),
        "series": [
            {
                "name": "Real",
                "values": [safe_json_value(value) for value in real_values],
                "color": REAL_COLOR,
            },
            {
                "name": "Synthetic",
                "values": [safe_json_value(value) for value in synthetic_values],
                "color": SYN_COLOR,
            },
        ],
    }


def build_single_bar_chart_payload(title: str, yaxis_title: str, labels, values, color: str) -> dict:
    return {
        "kind": "single_bar",
        "title": title,
        "yaxis_title": yaxis_title,
        "labels": list(labels),
        "values": [safe_json_value(value) for value in values],
        "color": color,
    }


def build_task_metric(label: str, display: str) -> dict:
    return {"label": label, "display": display}


def normalize_task_id(value) -> str:
    if value is None:
        return ""

    normalized = str(value).strip()
    if not normalized:
        return ""

    if normalized.isdigit():
        normalized = f"{int(normalized):02d}"

    return normalized if normalized in TASK_LOOKUP else ""


def get_query_param_value(name: str):
    if hasattr(st, "query_params"):
        return st.query_params.get(name, "")

    query_params = st.experimental_get_query_params()
    raw_value = query_params.get(name, [""])
    if isinstance(raw_value, list):
        return raw_value[0] if raw_value else ""
    return raw_value


def queue_dashboard_open() -> None:
    ok, message = launch_streamlit_app(DASHBOARD_SCRIPT, DASHBOARD_PORT)
    st.session_state["dashboard_open_pending"] = ok
    st.session_state["dashboard_open_message"] = message
    st.session_state["dashboard_open_error"] = "" if ok else message


def render_pending_dashboard_open() -> None:
    if st.session_state.get("dashboard_open_error"):
        st.error(st.session_state["dashboard_open_error"])
        st.session_state["dashboard_open_error"] = ""

    if not st.session_state.get("dashboard_open_pending"):
        return

    launch_message = st.session_state.get("dashboard_open_message", "")
    if launch_message:
        st.success(launch_message)
    st.markdown(f"[Open presentation dashboard manually]({dashboard_url()})")
    components.html(
        f"""
        <script>
        window.open("{dashboard_url()}", "_blank");
        </script>
        """,
        height=0,
        width=0,
    )
    st.session_state["dashboard_open_pending"] = False


def render_task_anchor(task_id: str) -> None:
    st.markdown(f"<div id='task-{task_id}'></div>", unsafe_allow_html=True)


def scroll_to_requested_task(task_id: str) -> None:
    if not task_id:
        return

    task_title = TASK_LOOKUP[task_id]["title"]
    st.info(f"Opened from dashboard for Task {task_id}: {task_title}.")
    components.html(
        f"""
        <script>
        const taskAnchor = window.parent.document.getElementById("task-{task_id}");
        if (taskAnchor) {{
            taskAnchor.scrollIntoView({{behavior: "smooth", block: "start"}});
        }}
        </script>
        """,
        height=0,
        width=0,
    )


def render_report_button() -> None:
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    render_copy_card(
        "Generate report",
        "Create a PDF-style full-page report of the current dashboard from top to bottom.",
    )
    components.html(
        """
        <button id="generate-report-pdf" type="button">Generate Report PDF</button>
        <script>
        const button = document.getElementById("generate-report-pdf");
        if (window.frameElement) {
            window.frameElement.classList.add("report-print-frame");
        }

        function installPrintStyles() {
            const parentDocument = window.parent.document;
            if (parentDocument.getElementById("welding-report-print-style")) {
                return;
            }

            const style = parentDocument.createElement("style");
            style.id = "welding-report-print-style";
            style.textContent = `
                @media print {
                    @page { size: A4 portrait; margin: 10mm; }
                    html,
                    body,
                    .stApp,
                    [data-testid="stAppViewContainer"],
                    .main,
                    section.main,
                    .block-container {
                        background: #ffffff !important;
                        color: #0f172a !important;
                    }
                    .stApp * {
                        color: #0f172a !important;
                        text-shadow: none !important;
                    }
                    [data-testid="stToolbar"],
                    [data-testid="stSidebar"],
                    [data-testid="stHeader"],
                    .report-print-frame {
                        display: none !important;
                    }
                    .block-container {
                        max-width: none !important;
                        padding: 0 !important;
                    }
                    .main .block-container {
                        width: 100% !important;
                    }
                    .hero-panel,
                    .analysis-band,
                    .copy-card,
                    .metric-tile,
                    [data-testid="stMetric"],
                    [data-testid="stDataFrame"],
                    [data-testid="stNumberInputContainer"] {
                        background: #ffffff !important;
                        border: 1px solid #dbe3ea !important;
                        box-shadow: none !important;
                    }
                    .hero-title,
                    .analysis-band-title,
                    .copy-title,
                    h1, h2, h3, h4, p, li, label,
                    [data-testid="stMarkdownContainer"] {
                        color: #0f172a !important;
                    }
                    .hero-kicker,
                    .analysis-band-kicker {
                        color: #c2410c !important;
                    }
                    .analysis-band-number,
                    .analysis-band-number * {
                        background: #f97316 !important;
                        color: #ffffff !important;
                    }
                    input,
                    textarea {
                        background: #ffffff !important;
                        color: #0f172a !important;
                        -webkit-text-fill-color: #0f172a !important;
                    }
                    .js-plotly-plot,
                    .js-plotly-plot .plotly,
                    .js-plotly-plot .main-svg {
                        background: #ffffff !important;
                    }
                    .js-plotly-plot text {
                        fill: #0f172a !important;
                    }
                    button {
                        display: none !important;
                    }
                    .element-container {
                        break-inside: avoid;
                    }
                }
            `;
            parentDocument.head.appendChild(style);
        }

        button.addEventListener("click", () => {
            installPrintStyles();
            setTimeout(() => window.parent.print(), 150);
        });
        </script>
        <style>
            html, body {
                margin: 0;
                padding: 0;
                background: transparent;
                font-family: "Segoe UI", sans-serif;
            }
            #generate-report-pdf {
                width: 100%;
                min-height: 3.2rem;
                border: 0;
                border-radius: 18px;
                background: linear-gradient(135deg, #f97316, #ea580c);
                color: #ffffff;
                font-size: 1rem;
                font-weight: 800;
                cursor: pointer;
                box-shadow: 0 18px 32px rgba(249, 115, 22, 0.24);
            }
            #generate-report-pdf:hover {
                background: linear-gradient(135deg, #fb923c, #f97316);
            }
        </style>
        """,
        height=64,
        scrolling=False,
    )


def safe_corr(series_a, series_b) -> float:
    min_len = min(len(series_a), len(series_b))
    if min_len < 2:
        return 0.0
    corr = np.corrcoef(np.asarray(series_a[:min_len]), np.asarray(series_b[:min_len]))[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def absolute_difference_phrase(delta: float, unit: str) -> str:
    if not np.isfinite(delta):
        return "not available relative to"
    if abs(delta) < 1e-8:
        return "almost the same as"
    if delta > 0:
        return f"{abs(delta):.2f} {unit} higher than"
    return f"{abs(delta):.2f} {unit} lower than"


def percent_difference_phrase(delta_pct: float) -> str:
    if not np.isfinite(delta_pct):
        return "not available relative to"
    if abs(delta_pct) < 0.01:
        return "almost the same as"
    if delta_pct > 0:
        return f"{abs(delta_pct):.2f}% higher than"
    return f"{abs(delta_pct):.2f}% lower than"


def safe_percent_difference(reference_value: float, candidate_value: float) -> float:
    if not np.isfinite(reference_value) or not np.isfinite(candidate_value):
        return np.nan
    if abs(reference_value) < 1e-8:
        return 0.0 if abs(candidate_value) < 1e-8 else np.nan
    return ((candidate_value - reference_value) / abs(reference_value)) * 100.0


def similarity_text(value: float) -> str:
    if value >= 0.95:
        return "very strong"
    if value >= 0.85:
        return "strong"
    if value >= 0.70:
        return "moderate"
    return "weak"


def overall_match_text(rms_error: float, current_corr: float, acf_corr: float, energy_gap_pct: float) -> str:
    safe_energy_gap = abs(energy_gap_pct) if np.isfinite(energy_gap_pct) else np.inf
    if rms_error <= 10 and current_corr >= 0.9 and acf_corr >= 0.9 and safe_energy_gap <= 10:
        return "strong"
    if rms_error <= 20 and current_corr >= 0.7 and acf_corr >= 0.75 and safe_energy_gap <= 20:
        return "balanced"
    return "partial"


def build_line_figure(
    x_real,
    y_real,
    x_syn,
    y_syn,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    real_name: str = "Real",
    syn_name: str = "Synthetic",
    use_log_y: bool = False,
    show_range_slider: bool = False,
    fill_real_color=None,
    fill_syn_color=None,
):
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x_real,
            y=y_real,
            mode="lines",
            name=real_name,
            line=dict(color=REAL_COLOR, width=3),
            fill="tozeroy" if fill_real_color else None,
            fillcolor=fill_real_color,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x_syn,
            y=y_syn,
            mode="lines",
            name=syn_name,
            line=dict(color=SYN_COLOR, width=3),
            fill="tozeroy" if fill_syn_color else None,
            fillcolor=fill_syn_color,
        )
    )

    figure.update_layout(
        title=dict(
            text=title,
            x=0.02,
            xanchor="left",
            font=dict(size=24, family="Segoe UI Semibold", color=TEXT_COLOR),
        ),
        template="plotly_white",
        hovermode="x unified",
        font=dict(size=18, family="Segoe UI"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.06,
            xanchor="right",
            x=1.0,
            font=dict(size=15, color=TEXT_COLOR),
            bgcolor="rgba(255,255,255,0.92)",
        ),
        margin=dict(l=30, r=30, t=110, b=55),
        height=560,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.96)",
    )
    figure.update_xaxes(
        title=xaxis_title,
        showgrid=True,
        gridcolor="rgba(15, 23, 42, 0.08)",
        rangeslider=dict(visible=show_range_slider),
        title_font=dict(size=18, color=TEXT_COLOR),
        tickfont=dict(size=14, color=TEXT_COLOR),
        automargin=True,
        title_standoff=14,
        zeroline=False,
    )
    figure.update_yaxes(
        title=yaxis_title,
        showgrid=True,
        gridcolor="rgba(15, 23, 42, 0.08)",
        type="log" if use_log_y else "linear",
        title_font=dict(size=18, color=TEXT_COLOR),
        tickfont=dict(size=14, color=TEXT_COLOR),
        automargin=True,
        title_standoff=14,
        zeroline=False,
    )
    return figure


def build_bar_figure(labels, values, title: str, yaxis_title: str):
    figure = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                text=[format_metric_value(value) for value in values],
                textposition="outside",
                marker=dict(color=[REAL_COLOR, SYN_COLOR], line=dict(color="#ffffff", width=1.5)),
            )
        ]
    )
    figure.update_layout(
        title=dict(
            text=title,
            x=0.02,
            xanchor="left",
            font=dict(size=24, family="Segoe UI Semibold", color=TEXT_COLOR),
        ),
        template="plotly_white",
        font=dict(size=18, family="Segoe UI"),
        margin=dict(l=30, r=30, t=95, b=45),
        height=470,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.96)",
    )
    figure.update_xaxes(showgrid=False, tickfont=dict(size=15, color=TEXT_COLOR), automargin=True)
    figure.update_yaxes(
        title=yaxis_title,
        showgrid=True,
        gridcolor="rgba(15, 23, 42, 0.08)",
        title_font=dict(size=18, color=TEXT_COLOR),
        tickfont=dict(size=14, color=TEXT_COLOR),
        automargin=True,
        title_standoff=14,
        zeroline=False,
    )
    return figure


def build_grouped_bar_figure(labels, real_values, synthetic_values, title: str, yaxis_title: str):
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=labels,
            y=real_values,
            name="Real",
            marker=dict(color=REAL_COLOR, line=dict(color="#ffffff", width=1.5)),
            text=[format_metric_value(value) for value in real_values],
            textposition="outside",
        )
    )
    figure.add_trace(
        go.Bar(
            x=labels,
            y=synthetic_values,
            name="Synthetic",
            marker=dict(color=SYN_COLOR, line=dict(color="#ffffff", width=1.5)),
            text=[format_metric_value(value) for value in synthetic_values],
            textposition="outside",
        )
    )
    figure.update_layout(
        title=dict(
            text=title,
            x=0.02,
            xanchor="left",
            font=dict(size=24, family="Segoe UI Semibold", color=TEXT_COLOR),
        ),
        template="plotly_white",
        font=dict(size=18, family="Segoe UI"),
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="right",
            x=1.0,
            font=dict(size=15, color=TEXT_COLOR),
            bgcolor="rgba(255,255,255,0.92)",
        ),
        margin=dict(l=30, r=30, t=95, b=45),
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.96)",
    )
    figure.update_xaxes(showgrid=False, tickfont=dict(size=15, color=TEXT_COLOR), automargin=True)
    figure.update_yaxes(
        title=yaxis_title,
        showgrid=True,
        gridcolor="rgba(15, 23, 42, 0.08)",
        title_font=dict(size=18, color=TEXT_COLOR),
        tickfont=dict(size=14, color=TEXT_COLOR),
        automargin=True,
        title_standoff=14,
        zeroline=False,
    )
    return figure


def build_loss_sankey_figure(effective_heat_j: float, loss_breakdown_j: dict, title: str):
    labels = ["Total Electrical Energy", "Useful Heat To Weld"] + list(loss_breakdown_j.keys())
    losses = [loss_breakdown_j[name] for name in loss_breakdown_j]
    figure = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    label=labels,
                    pad=18,
                    thickness=18,
                    line=dict(color="rgba(15, 23, 42, 0.18)", width=1),
                    color=[
                        "rgba(15, 23, 42, 0.88)",
                        "rgba(249, 115, 22, 0.85)",
                        "rgba(24, 78, 119, 0.75)",
                        "rgba(14, 116, 144, 0.75)",
                        "rgba(2, 132, 199, 0.75)",
                        "rgba(59, 130, 246, 0.75)",
                        "rgba(100, 116, 139, 0.75)",
                    ][: len(labels)],
                ),
                link=dict(
                    source=[0] * (1 + len(losses)),
                    target=list(range(1, 2 + len(losses))),
                    value=[effective_heat_j] + losses,
                    color=[
                        "rgba(249, 115, 22, 0.32)",
                        "rgba(24, 78, 119, 0.22)",
                        "rgba(14, 116, 144, 0.22)",
                        "rgba(2, 132, 199, 0.22)",
                        "rgba(59, 130, 246, 0.22)",
                        "rgba(100, 116, 139, 0.22)",
                    ][: 1 + len(losses)],
                ),
            )
        ]
    )
    figure.update_layout(
        title=dict(
            text=title,
            x=0.02,
            xanchor="left",
            font=dict(size=22, family="Segoe UI Semibold", color=TEXT_COLOR),
        ),
        template="plotly_white",
        font=dict(size=14, family="Segoe UI", color=TEXT_COLOR),
        margin=dict(l=20, r=20, t=70, b=20),
        height=430,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.96)",
    )
    return figure


def render_welding_loss_diagram() -> None:
    weld_image_path = Path(__file__).resolve().parent / "weld.png"
    if weld_image_path.exists():
        st.image(str(weld_image_path), use_container_width=True)
    else:
        st.warning(f"Could not find `{weld_image_path}`. Add `weld.png` there to show the new heat-loss image.")
    return

    diagram_html = """
    <html>
    <head>
        <style>
            html, body {
                margin: 0;
                padding: 0;
                overflow: hidden;
                background: transparent;
                font-family: "Segoe UI", sans-serif;
            }
            .diagram-shell {
                background: linear-gradient(135deg, #fffaf2, #f8fbff);
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 24px;
                padding: 0.9rem 0.95rem 0.8rem 0.95rem;
                box-shadow: 0 16px 36px rgba(15, 23, 42, 0.06);
            }
            .diagram-title {
                font-size: 20px;
                font-weight: 800;
                color: #0f172a;
                margin: 0 0 0.45rem 0.2rem;
            }
            .diagram-subtitle {
                font-size: 13px;
                color: #475569;
                margin: 0 0 0.55rem 0.2rem;
                line-height: 1.35;
            }
            .diagram-svg {
                width: 100%;
                height: 285px;
                display: block;
            }
        </style>
    </head>
    <body>
    <div class="diagram-shell">
        <div class="diagram-title">Heat Flow Around The Welding Arc</div>
        <div class="diagram-subtitle">Orange arrows show useful heat entering the weld pool. Blue arrows show heat leaving through different engineering loss channels.</div>
        <svg class="diagram-svg" viewBox="0 0 980 340" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Welding heat loss schematic">
            <rect x="0" y="0" width="980" height="340" rx="22" fill="url(#bg)" />
            <defs>
                <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#fff7ed"/>
                    <stop offset="100%" stop-color="#eff6ff"/>
                </linearGradient>
                <linearGradient id="plate" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="#475569"/>
                    <stop offset="100%" stop-color="#334155"/>
                </linearGradient>
                <linearGradient id="torchBody" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#64748b"/>
                    <stop offset="100%" stop-color="#334155"/>
                </linearGradient>
                <linearGradient id="heatGlow" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#fef08a" stop-opacity="0.95"/>
                    <stop offset="50%" stop-color="#fb923c" stop-opacity="0.80"/>
                    <stop offset="100%" stop-color="#f97316" stop-opacity="0.30"/>
                </linearGradient>
                <marker id="arrowBlue" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
                    <polygon points="0 0, 6 3, 0 6" fill="#184e77"></polygon>
                </marker>
                <marker id="arrowOrange" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
                    <polygon points="0 0, 6 3, 0 6" fill="#f97316"></polygon>
                </marker>
                <filter id="glow">
                    <feGaussianBlur stdDeviation="6" result="coloredBlur"/>
                    <feMerge>
                        <feMergeNode in="coloredBlur"/>
                        <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                </filter>
            </defs>

            <rect x="58" y="246" width="500" height="56" rx="18" fill="url(#plate)" />
            <rect x="58" y="232" width="500" height="22" rx="12" fill="#64748b" opacity="0.16" />
            <text x="308" y="280" text-anchor="middle" font-size="18" fill="#f8fafc" font-family="Segoe UI, sans-serif">Mild-steel workpiece / base plate</text>

            <polygon points="110,72 260,116 218,206 70,164" fill="url(#torchBody)"/>
            <rect x="222" y="121" width="80" height="38" rx="14" fill="#1e293b"/>
            <rect x="292" y="132" width="24" height="16" rx="7" fill="#0f172a"/>
            <text x="185" y="42" text-anchor="middle" font-size="24" font-weight="700" fill="#0f172a" font-family="Segoe UI, sans-serif">Welding torch</text>
            <text x="186" y="64" text-anchor="middle" font-size="13" fill="#475569" font-family="Segoe UI, sans-serif">electrode, nozzle, contact system</text>

            <ellipse cx="362" cy="214" rx="54" ry="24" fill="url(#heatGlow)" filter="url(#glow)"/>
            <circle cx="328" cy="196" r="7" fill="#fde68a"/>
            <circle cx="347" cy="204" r="5" fill="#fdba74"/>
            <path d="M306 150 C322 170, 340 190, 356 212" stroke="#f59e0b" stroke-width="14" stroke-linecap="round" fill="none"/>
            <path d="M317 145 C334 167, 350 187, 367 209" stroke="#fde68a" stroke-width="6" stroke-linecap="round" fill="none"/>
            <text x="314" y="130" font-size="20" font-weight="700" fill="#c2410c" font-family="Segoe UI, sans-serif">Arc</text>
            <text x="350" y="233" font-size="15" font-weight="700" fill="#7c2d12" font-family="Segoe UI, sans-serif">Weld pool</text>

            <rect x="600" y="22" width="328" height="42" rx="14" fill="rgba(255,255,255,0.86)" stroke="rgba(249,115,22,0.30)" />
            <rect x="600" y="70" width="328" height="42" rx="14" fill="rgba(255,255,255,0.86)" stroke="rgba(15,118,110,0.26)" />
            <rect x="600" y="118" width="328" height="42" rx="14" fill="rgba(255,255,255,0.86)" stroke="rgba(2,132,199,0.26)" />
            <rect x="600" y="166" width="328" height="42" rx="14" fill="rgba(255,255,255,0.86)" stroke="rgba(24,78,119,0.26)" />
            <rect x="600" y="214" width="328" height="42" rx="14" fill="rgba(255,255,255,0.86)" stroke="rgba(29,78,216,0.26)" />
            <rect x="600" y="262" width="328" height="42" rx="14" fill="rgba(255,255,255,0.86)" stroke="rgba(100,116,139,0.26)" />

            <text x="620" y="40" font-size="15" font-weight="700" fill="#c2410c" font-family="Segoe UI, sans-serif">Useful heat to weld pool</text>
            <text x="620" y="56" font-size="12" fill="#7c2d12" font-family="Segoe UI, sans-serif">melting, penetration, and bead formation</text>

            <text x="620" y="88" font-size="15" font-weight="700" fill="#0f766e" font-family="Segoe UI, sans-serif">Conduction to surrounding plate</text>
            <text x="620" y="104" font-size="12" fill="#0f766e" font-family="Segoe UI, sans-serif">heat diffuses into colder base metal</text>

            <text x="620" y="136" font-size="15" font-weight="700" fill="#0369a1" font-family="Segoe UI, sans-serif">Convection to surroundings</text>
            <text x="620" y="152" font-size="12" fill="#0369a1" font-family="Segoe UI, sans-serif">hot gas and shielding flow carry heat away</text>

            <text x="620" y="184" font-size="15" font-weight="700" fill="#184e77" font-family="Segoe UI, sans-serif">Radiation loss</text>
            <text x="620" y="200" font-size="12" fill="#184e77" font-family="Segoe UI, sans-serif">arc and hot metal emit thermal radiation</text>

            <text x="620" y="232" font-size="15" font-weight="700" fill="#1d4ed8" font-family="Segoe UI, sans-serif">Spatter / fume loss</text>
            <text x="620" y="248" font-size="12" fill="#1d4ed8" font-family="Segoe UI, sans-serif">hot droplets and vapor leave the weld zone</text>

            <text x="620" y="280" font-size="15" font-weight="700" fill="#475569" font-family="Segoe UI, sans-serif">Torch, electrode, fixture losses</text>
            <text x="620" y="296" font-size="12" fill="#475569" font-family="Segoe UI, sans-serif">energy absorbed by consumables and hardware</text>

            <line x1="386" y1="214" x2="600" y2="43" stroke="#f97316" stroke-width="8" marker-end="url(#arrowOrange)" stroke-linecap="round"/>
            <line x1="360" y1="226" x2="600" y2="91" stroke="#0f766e" stroke-width="5" marker-end="url(#arrowBlue)" stroke-linecap="round"/>
            <line x1="385" y1="194" x2="600" y2="139" stroke="#0284c7" stroke-width="5" marker-end="url(#arrowBlue)" stroke-linecap="round"/>
            <line x1="340" y1="176" x2="600" y2="187" stroke="#184e77" stroke-width="5" marker-end="url(#arrowBlue)" stroke-linecap="round"/>
            <line x1="388" y1="186" x2="600" y2="235" stroke="#1d4ed8" stroke-width="5" marker-end="url(#arrowBlue)" stroke-linecap="round"/>
            <line x1="275" y1="138" x2="600" y2="283" stroke="#64748b" stroke-width="5" marker-end="url(#arrowBlue)" stroke-linecap="round"/>

            <circle cx="460" cy="174" r="4" fill="#1d4ed8"/>
            <circle cx="486" cy="184" r="3.5" fill="#1d4ed8"/>
            <circle cx="512" cy="194" r="3" fill="#1d4ed8"/>
        </svg>
    </div>
    </body>
    </html>
    """
    components.html(diagram_html, height=365, scrolling=False)


def render_heat_loss_mechanism_cards() -> None:
    st.markdown("#### Loss Mechanisms Explained")
    st.markdown(
        "These notes explain the physical meaning of each heat path, so the energy-distribution graphs are easier to justify in a viva."
    )

    mechanism_names = [
        "Useful heat to weld pool",
        "Conduction to surrounding plate",
        "Convection to surroundings",
        "Radiation loss",
        "Spatter and fume loss",
        "Torch, electrode, and fixture loss",
    ]

    for start_index in range(0, len(mechanism_names), 2):
        columns = st.columns(2, gap="large")
        for column, mechanism_name in zip(columns, mechanism_names[start_index : start_index + 2]):
            mechanism_info = HEAT_LOSS_MECHANISMS[mechanism_name]
            with column:
                render_copy_card(
                    f"{mechanism_name} ({mechanism_info['share_label']})",
                    f"<strong>Mechanism:</strong> {mechanism_info['mechanism']}<br><br>"
                    f"<strong>Engineering effect:</strong> {mechanism_info['effect']}",
                )


def metric_summary_text(rms_error: float, acf_corr: float, current_corr: float, energy_gap_pct: float) -> str:
    safe_energy_score = 0.0 if not np.isfinite(energy_gap_pct) else max(0.0, 100.0 - abs(energy_gap_pct))
    metric_scores = {
        "amplitude match": max(0.0, 100.0 - rms_error),
        "temporal memory": max(0.0, acf_corr * 100.0),
        "waveform tracking": max(0.0, current_corr * 100.0),
        "energy agreement": safe_energy_score,
    }
    best_metric = max(metric_scores, key=metric_scores.get)
    return (
        "Lower RMS error is better, while current correlation and ACF correlation should stay close to 1. "
        f"In this run, the strongest agreement is in {best_metric}."
    )


def bar_ready(values):
    return [0.0 if not np.isfinite(value) else float(value) for value in values]


def dominant_pulse_category(summary_dict) -> str:
    dominant_category = "None"
    dominant_count = 0

    for category_name in PULSE_CATEGORY_ORDER:
        category_count = int(summary_dict.get(category_name, {}).get("pulse_count", 0))
        if category_count > dominant_count:
            dominant_category = category_name
            dominant_count = category_count

    return dominant_category if dominant_count > 0 else "None"


def build_pulse_category_dataframe(real_analysis: dict, synthetic_analysis: dict) -> pd.DataFrame:
    rows = []
    for category_name in PULSE_CATEGORY_ORDER:
        real_summary = real_analysis["summary"][category_name]
        synthetic_summary = synthetic_analysis["summary"][category_name]
        rows.append(
            {
                "Pulse Type": category_name,
                "Current Rule": PULSE_CATEGORY_REFERENCE[category_name]["band"],
                "Engineering Meaning": PULSE_CATEGORY_REFERENCE[category_name]["description"],
                "Real Pulse Count": real_summary["pulse_count"],
                "Synthetic Pulse Count": synthetic_summary["pulse_count"],
                "Real Avg Useful Heat (J)": real_summary["avg_useful_heat_j"],
                "Synthetic Avg Useful Heat (J)": synthetic_summary["avg_useful_heat_j"],
                "Real Avg Temp Rise (C)": real_summary["avg_temperature_rise_c"],
                "Synthetic Avg Temp Rise (C)": synthetic_summary["avg_temperature_rise_c"],
            }
        )

    pulse_category_df = pd.DataFrame(rows)
    numeric_columns = [
        "Real Avg Useful Heat (J)",
        "Synthetic Avg Useful Heat (J)",
        "Real Avg Temp Rise (C)",
        "Synthetic Avg Temp Rise (C)",
    ]
    pulse_category_df[numeric_columns] = pulse_category_df[numeric_columns].round(3)
    return pulse_category_df


def build_pulse_detail_dataframe(pulses) -> pd.DataFrame:
    pulse_df = pd.DataFrame(pulses)
    if pulse_df.empty:
        return pd.DataFrame(
            columns=[
                "Pulse ID",
                "Start (ms)",
                "End (ms)",
                "Duration (ms)",
                "Charging Point (ms)",
                "Discharging Point (ms)",
                "Pulse Type",
                "Mean Current (A)",
                "Peak Current (A)",
                "Mean Voltage (V)",
                "Peak Current / I0",
                "Electrical Energy (J)",
                "Useful Heat (J)",
                "Pulse Length (mm)",
                "Pulse Volume (mm^3)",
                "Pulse Mass (kg)",
                "Temp Rise (C)",
                "Final Temp (C)",
            ]
        )

    pulse_df = pulse_df.rename(
        columns={
            "pulse_id": "Pulse ID",
            "start_time_ms": "Start (ms)",
            "end_time_ms": "End (ms)",
            "duration_ms": "Duration (ms)",
            "charging_point_time_ms": "Charging Point (ms)",
            "discharging_point_time_ms": "Discharging Point (ms)",
            "pulse_category": "Pulse Type",
            "mean_current_a": "Mean Current (A)",
            "peak_current_a": "Peak Current (A)",
            "mean_voltage_v": "Mean Voltage (V)",
            "current_ratio_i0": "Peak Current / I0",
            "electrical_energy_j": "Electrical Energy (J)",
            "useful_heat_j": "Useful Heat (J)",
            "pulse_length_mm": "Pulse Length (mm)",
            "pulse_volume_mm3": "Pulse Volume (mm^3)",
            "pulse_mass_kg": "Pulse Mass (kg)",
            "temperature_rise_c": "Temp Rise (C)",
            "final_temperature_c": "Final Temp (C)",
        }
    )

    display_columns = [
        "Pulse ID",
        "Start (ms)",
        "End (ms)",
        "Duration (ms)",
        "Charging Point (ms)",
        "Discharging Point (ms)",
        "Pulse Type",
        "Mean Current (A)",
        "Peak Current (A)",
        "Mean Voltage (V)",
        "Peak Current / I0",
        "Electrical Energy (J)",
        "Useful Heat (J)",
        "Pulse Length (mm)",
        "Pulse Volume (mm^3)",
        "Pulse Mass (kg)",
        "Temp Rise (C)",
        "Final Temp (C)",
    ]
    pulse_df = pulse_df[display_columns]

    numeric_columns = [
        "Start (ms)",
        "End (ms)",
        "Duration (ms)",
        "Charging Point (ms)",
        "Discharging Point (ms)",
        "Mean Current (A)",
        "Peak Current (A)",
        "Mean Voltage (V)",
        "Peak Current / I0",
        "Electrical Energy (J)",
        "Useful Heat (J)",
        "Pulse Length (mm)",
        "Pulse Volume (mm^3)",
        "Pulse Mass (kg)",
        "Temp Rise (C)",
        "Final Temp (C)",
    ]
    pulse_df[numeric_columns] = pulse_df[numeric_columns].round(4)
    return pulse_df


def safe_average(values) -> float:
    finite_values = [float(value) for value in values if np.isfinite(value)]
    if not finite_values:
        return np.nan
    return float(np.mean(finite_values))


def smooth_signal(values, window_samples: int):
    values = np.asarray(values, dtype=float).reshape(-1)
    window_samples = max(1, int(window_samples))
    if window_samples % 2 == 0:
        window_samples += 1
    if window_samples <= 1 or values.size < 3:
        return values.copy()

    kernel = np.ones(window_samples, dtype=float)
    finite_mask = np.isfinite(values)
    sample_counts = np.convolve(finite_mask.astype(float), kernel, mode="same")
    sample_sums = np.convolve(np.where(finite_mask, values, 0.0), kernel, mode="same")
    return np.divide(
        sample_sums,
        sample_counts,
        out=np.full(values.shape, np.nan, dtype=float),
        where=sample_counts > 0,
    )


def median_sample_spacing_ms(time_ms) -> float:
    time_ms = np.asarray(time_ms, dtype=float).reshape(-1)
    if time_ms.size < 2:
        return np.nan
    diffs = np.diff(time_ms)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return np.nan
    return float(np.median(diffs))


def find_local_maxima(signal):
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if signal.size < 3:
        return np.array([], dtype=int)

    peaks = []
    index = 1
    while index < signal.size - 1:
        current_value = signal[index]
        if not np.isfinite(current_value):
            index += 1
            continue

        left_value = signal[index - 1]
        right_value = signal[index + 1]
        if not np.isfinite(left_value) or not np.isfinite(right_value):
            index += 1
            continue

        if current_value == right_value:
            plateau_start = index
            plateau_end = index
            while (
                plateau_end + 1 < signal.size
                and np.isfinite(signal[plateau_end + 1])
                and signal[plateau_end + 1] == current_value
            ):
                plateau_end += 1

            if plateau_start - 1 >= 0 and plateau_end + 1 < signal.size:
                plateau_left = signal[plateau_start - 1]
                plateau_right = signal[plateau_end + 1]
                if (
                    np.isfinite(plateau_left)
                    and np.isfinite(plateau_right)
                    and current_value > plateau_left
                    and current_value > plateau_right
                ):
                    peaks.append((plateau_start + plateau_end) // 2)
            index = plateau_end + 1
            continue

        if current_value > left_value and current_value >= right_value:
            peaks.append(index)
        index += 1

    return np.asarray(peaks, dtype=int)


def estimate_peak_prominence(signal, peak_index: int) -> float:
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if peak_index < 0 or peak_index >= signal.size or not np.isfinite(signal[peak_index]):
        return np.nan

    peak_value = float(signal[peak_index])
    left_min = peak_value
    right_min = peak_value

    left_index = peak_index
    while left_index > 0:
        left_index -= 1
        if not np.isfinite(signal[left_index]):
            continue
        left_min = min(left_min, float(signal[left_index]))
        if signal[left_index] > peak_value:
            break

    right_index = peak_index
    while right_index < signal.size - 1:
        right_index += 1
        if not np.isfinite(signal[right_index]):
            continue
        right_min = min(right_min, float(signal[right_index]))
        if signal[right_index] > peak_value:
            break

    contour_level = max(left_min, right_min)
    return float(max(0.0, peak_value - contour_level))


def select_peaks_by_distance(candidates, min_distance_samples: int):
    if not candidates:
        return []
    min_distance_samples = max(1, int(min_distance_samples))
    if min_distance_samples <= 1:
        return sorted(candidates, key=lambda peak: peak["peak_index"])

    selected = []
    kept_indices = []
    ranked_candidates = sorted(
        candidates,
        key=lambda peak: (
            peak["peak_current_a"] if np.isfinite(peak["peak_current_a"]) else -np.inf,
            peak["peak_prominence_a"] if np.isfinite(peak["peak_prominence_a"]) else -np.inf,
        ),
        reverse=True,
    )

    for candidate in ranked_candidates:
        if all(abs(candidate["peak_index"] - kept_index) >= min_distance_samples for kept_index in kept_indices):
            selected.append(candidate)
            kept_indices.append(candidate["peak_index"])

    return sorted(selected, key=lambda peak: peak["peak_index"])


def classify_peak_ratio(current_ratio: float) -> str:
    if not np.isfinite(current_ratio):
        return "Unclassified"
    if current_ratio >= 0.90:
        return "Class 1"
    if current_ratio >= 0.65:
        return "Class 2"
    if current_ratio >= 0.55:
        return "Class 3"
    return "Class 4"


def dominant_peak_category(summary_dict) -> str:
    dominant_category = "None"
    dominant_count = 0

    for category_name in PEAK_CATEGORY_ORDER:
        category_count = int(summary_dict.get(category_name, {}).get("peak_count", 0))
        if category_count > dominant_count:
            dominant_category = category_name
            dominant_count = category_count

    return dominant_category if dominant_count > 0 else "None"


def analyze_peak_power_categories(
    time_ms,
    current_a,
    resistance_ohm: float,
    min_height_ratio: float = 0.35,
    min_distance_ms: float = 0.0,
    prominence_ratio: float = 0.04,
    smoothing_window_samples: int = 7,
):
    time_ms = np.asarray(time_ms, dtype=float).reshape(-1)
    current_a = np.asarray(current_a, dtype=float).reshape(-1)

    if time_ms.size == 0:
        raise ValueError("time_ms must contain at least one sample.")
    if current_a.size != time_ms.size:
        raise ValueError("current_a must have the same number of samples as time_ms.")

    smoothing_window_samples = max(1, int(smoothing_window_samples))
    if smoothing_window_samples % 2 == 0:
        smoothing_window_samples += 1

    smoothed_current = smooth_signal(current_a, smoothing_window_samples)
    finite_current = current_a[np.isfinite(current_a)]
    signal_max_current_a = float(np.nanmax(finite_current)) if finite_current.size else np.nan

    min_height_ratio = float(max(0.0, min(1.0, min_height_ratio)))
    prominence_ratio = float(max(0.0, min(1.0, prominence_ratio)))
    min_height_a = (
        min_height_ratio * signal_max_current_a
        if np.isfinite(signal_max_current_a) and signal_max_current_a > 0
        else np.nan
    )
    prominence_threshold_a = (
        prominence_ratio * signal_max_current_a
        if np.isfinite(signal_max_current_a) and signal_max_current_a > 0
        else np.nan
    )

    sample_spacing_ms = median_sample_spacing_ms(time_ms)
    duration_ms = float(time_ms[-1] - time_ms[0]) if time_ms.size > 1 else 0.0
    auto_min_distance_ms = (
        max(3.0 * sample_spacing_ms, duration_ms / 80.0)
        if np.isfinite(sample_spacing_ms) and sample_spacing_ms > 0
        else 0.0
    )
    distance_ms_used = float(min_distance_ms) if float(min_distance_ms) > 0 else auto_min_distance_ms
    min_distance_samples = (
        max(1, int(np.ceil(distance_ms_used / sample_spacing_ms)))
        if np.isfinite(sample_spacing_ms) and sample_spacing_ms > 0 and distance_ms_used > 0
        else 1
    )

    candidate_indices = find_local_maxima(smoothed_current)
    candidate_peaks = []
    for peak_index in candidate_indices:
        peak_current_a = float(current_a[peak_index]) if np.isfinite(current_a[peak_index]) else float(smoothed_current[peak_index])
        peak_prominence_a = estimate_peak_prominence(smoothed_current, int(peak_index))

        if np.isfinite(min_height_a) and np.isfinite(peak_current_a) and peak_current_a < min_height_a:
            continue
        if np.isfinite(prominence_threshold_a) and np.isfinite(peak_prominence_a) and peak_prominence_a < prominence_threshold_a:
            continue

        candidate_peaks.append(
            {
                "peak_index": int(peak_index),
                "time_ms": float(time_ms[peak_index]),
                "peak_current_a": peak_current_a,
                "peak_prominence_a": peak_prominence_a,
            }
        )

    selected_peaks = select_peaks_by_distance(candidate_peaks, min_distance_samples)
    reference_current_a = (
        float(max(peak["peak_current_a"] for peak in selected_peaks if np.isfinite(peak["peak_current_a"])))
        if selected_peaks
        else signal_max_current_a
    )

    processed_peaks = []
    for peak_id, peak in enumerate(selected_peaks, start=1):
        current_ratio_i0 = (
            peak["peak_current_a"] / reference_current_a
            if np.isfinite(peak["peak_current_a"]) and np.isfinite(reference_current_a) and reference_current_a > 0
            else np.nan
        )
        peak_power_w = (
            (peak["peak_current_a"] ** 2) * float(resistance_ohm)
            if np.isfinite(peak["peak_current_a"]) and np.isfinite(resistance_ohm)
            else np.nan
        )
        peak_category = classify_peak_ratio(current_ratio_i0)

        processed_peaks.append(
            {
                "peak_id": peak_id,
                **peak,
                "current_ratio_i0": current_ratio_i0,
                "peak_power_w": peak_power_w,
                "peak_category": peak_category,
            }
        )

    total_peaks = len(processed_peaks)
    summary = {}
    for category_name in PEAK_CATEGORY_ORDER:
        category_peaks = [peak for peak in processed_peaks if peak["peak_category"] == category_name]
        summary[category_name] = {
            "peak_count": len(category_peaks),
            "peak_percentage": (len(category_peaks) / total_peaks * 100.0) if total_peaks else 0.0,
            "avg_peak_current_a": safe_average([peak["peak_current_a"] for peak in category_peaks]),
            "avg_peak_power_w": safe_average([peak["peak_power_w"] for peak in category_peaks]),
            "band": PEAK_CATEGORY_REFERENCE[category_name]["band"],
            "description": PEAK_CATEGORY_REFERENCE[category_name]["description"],
        }

    return {
        "peaks": processed_peaks,
        "total_peaks": total_peaks,
        "reference_current_a": reference_current_a,
        "signal_max_current_a": signal_max_current_a,
        "min_height_a": min_height_a,
        "min_height_ratio": min_height_ratio,
        "prominence_threshold_a": prominence_threshold_a,
        "prominence_ratio": prominence_ratio,
        "min_distance_ms": distance_ms_used,
        "min_distance_samples": min_distance_samples,
        "sample_spacing_ms": sample_spacing_ms,
        "smoothing_window_samples": smoothing_window_samples,
        "avg_peak_power_w": safe_average([peak["peak_power_w"] for peak in processed_peaks]),
        "avg_peak_current_a": safe_average([peak["peak_current_a"] for peak in processed_peaks]),
        "dominant_category": dominant_peak_category(summary),
        "summary": summary,
        "smoothed_current_a": smoothed_current,
    }


def build_peak_detail_dataframe(peak_analysis: dict) -> pd.DataFrame:
    peak_df = pd.DataFrame(peak_analysis.get("peaks", []))
    if peak_df.empty:
        return pd.DataFrame(
            columns=[
                "Peak ID",
                "Peak Index",
                "Time (ms)",
                "Peak Current (A)",
                "Peak Power (W)",
                "Peak Current / I0",
                "Prominence (A)",
                "Category",
            ]
        )

    peak_df = peak_df.rename(
        columns={
            "peak_id": "Peak ID",
            "peak_index": "Peak Index",
            "time_ms": "Time (ms)",
            "peak_current_a": "Peak Current (A)",
            "peak_power_w": "Peak Power (W)",
            "current_ratio_i0": "Peak Current / I0",
            "peak_prominence_a": "Prominence (A)",
            "peak_category": "Category",
        }
    )
    display_columns = [
        "Peak ID",
        "Peak Index",
        "Time (ms)",
        "Peak Current (A)",
        "Peak Power (W)",
        "Peak Current / I0",
        "Prominence (A)",
        "Category",
    ]
    peak_df = peak_df[display_columns]

    numeric_columns = [
        "Time (ms)",
        "Peak Current (A)",
        "Peak Power (W)",
        "Peak Current / I0",
        "Prominence (A)",
    ]
    peak_df[numeric_columns] = peak_df[numeric_columns].round(4)
    return peak_df


def build_peak_category_summary_dataframe(real_analysis: dict, synthetic_analysis: dict) -> pd.DataFrame:
    rows = []
    for category_name in PEAK_CATEGORY_ORDER:
        real_summary = real_analysis["summary"][category_name]
        synthetic_summary = synthetic_analysis["summary"][category_name]
        rows.append(
            {
                "Category": category_name,
                "Current Rule": PEAK_CATEGORY_REFERENCE[category_name]["band"],
                "Meaning": PEAK_CATEGORY_REFERENCE[category_name]["description"],
                "Real Count": real_summary["peak_count"],
                "Synthetic Count": synthetic_summary["peak_count"],
                "Real Share (%)": real_summary["peak_percentage"],
                "Synthetic Share (%)": synthetic_summary["peak_percentage"],
                "Real Avg Peak Power (W)": real_summary["avg_peak_power_w"],
                "Synthetic Avg Peak Power (W)": synthetic_summary["avg_peak_power_w"],
            }
        )

    summary_df = pd.DataFrame(rows)
    numeric_columns = [
        "Real Share (%)",
        "Synthetic Share (%)",
        "Real Avg Peak Power (W)",
        "Synthetic Avg Peak Power (W)",
    ]
    summary_df[numeric_columns] = summary_df[numeric_columns].round(3)
    return summary_df


def build_peak_detection_figure(
    time_ms,
    current_a,
    peak_analysis: dict,
    title: str,
    signal_name: str,
    signal_color: str,
):
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=time_ms,
            y=current_a,
            mode="lines",
            name=signal_name,
            line=dict(color=signal_color, width=3),
        )
    )

    for category_name in PEAK_CATEGORY_ORDER:
        category_peaks = [peak for peak in peak_analysis.get("peaks", []) if peak["peak_category"] == category_name]
        if not category_peaks:
            continue

        customdata = np.asarray(
            [
                [
                    peak["peak_power_w"],
                    peak["current_ratio_i0"],
                    peak["peak_index"],
                ]
                for peak in category_peaks
            ],
            dtype=float,
        )
        figure.add_trace(
            go.Scatter(
                x=[peak["time_ms"] for peak in category_peaks],
                y=[peak["peak_current_a"] for peak in category_peaks],
                mode="markers",
                name=category_name,
                marker=dict(
                    size=11,
                    color=PEAK_CATEGORY_COLORS[category_name],
                    line=dict(color="#ffffff", width=1.5),
                ),
                customdata=customdata,
                hovertemplate=(
                    "Time: %{x:.3f} ms<br>"
                    "Peak current: %{y:.3f} A<br>"
                    "Peak power: %{customdata[0]:.3f} W<br>"
                    "Peak current / I0: %{customdata[1]:.3f}<br>"
                    "Peak index: %{customdata[2]:.0f}<extra>"
                    + category_name
                    + "</extra>"
                ),
            )
        )

    figure.update_layout(
        title=dict(
            text=title,
            x=0.02,
            xanchor="left",
            font=dict(size=22, family="Segoe UI Semibold", color=TEXT_COLOR),
        ),
        template="plotly_white",
        hovermode="closest",
        font=dict(size=17, family="Segoe UI"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="right",
            x=1.0,
            font=dict(size=14, color=TEXT_COLOR),
            bgcolor="rgba(255,255,255,0.92)",
        ),
        margin=dict(l=30, r=30, t=95, b=45),
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.96)",
    )
    figure.update_xaxes(
        title="Time (ms)",
        showgrid=True,
        gridcolor="rgba(15, 23, 42, 0.08)",
        title_font=dict(size=18, color=TEXT_COLOR),
        tickfont=dict(size=14, color=TEXT_COLOR),
        automargin=True,
        title_standoff=14,
        zeroline=False,
    )
    figure.update_yaxes(
        title="Current (A)",
        showgrid=True,
        gridcolor="rgba(15, 23, 42, 0.08)",
        title_font=dict(size=18, color=TEXT_COLOR),
        tickfont=dict(size=14, color=TEXT_COLOR),
        automargin=True,
        title_standoff=14,
        zeroline=False,
    )
    return figure


requested_task_id = normalize_task_id(get_query_param_value("task"))


header_left, header_center, header_right = st.columns([1.6, 5.2, 4.2], gap="medium")

with header_left:
    st.markdown("<div style='height: 74px;'></div>", unsafe_allow_html=True)
    st.image("iit_logo.png", width=230)

with header_center:
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-kicker">Indian Institute of Technology Ropar</div>
            <div class="hero-title">Welding Signal Synthetic Evaluation Dashboard</div>
            <div class="hero-subtitle">
                Explore how closely synthetic welding signals reproduce the real process across
                current behavior, power delivery, cumulative energy, frequency content, and
                temporal memory. Every plot below includes a short interpretation so the dashboard
                reads like an analysis report instead of only a chart wall.
            </div>
            <div class="hero-pills">
                <span class="hero-pill">Time-domain current tracking</span>
                <span class="hero-pill">Power and energy behavior</span>
                <span class="hero-pill">Frequency-domain PSD comparison</span>
                <span class="hero-pill">Temporal ACF similarity</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown("<div style='height: 86px;'></div>", unsafe_allow_html=True)
    st.image("mech_logo.png", width=980)

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

dashboard_status_col, dashboard_action_col = st.columns([1.35, 1], gap="large")
dashboard_running = port_is_open(DASHBOARD_PORT)

with dashboard_status_col:
    render_copy_card(
        "Linked presentation dashboard",
        "Use `dashboard.py` as the professor-facing summary screen while this page stays your full working application for inputs and outputs. "
        + (
            f"The presentation dashboard is already running on port {DASHBOARD_PORT}."
            if dashboard_running
            else f"The presentation dashboard is not running yet. Use the button on the right to start it."
        ),
    )

with dashboard_action_col:
    st.write("")
    if st.button("Open Presentation Dashboard", key="open_presentation_dashboard", use_container_width=True):
        queue_dashboard_open()

render_pending_dashboard_open()

render_copy_card(
    "How to use this page",
    "Upload a CSV file with Relative_ms and Current_A columns, select the row range you want to analyze, and generate the plots. "
    "The charts support hover, zoom, and pan so you can inspect local pulse behavior and overall waveform trends.",
)

if requested_task_id:
    render_copy_card(
        "Task requested from dashboard",
        f"This app was opened from the audience dashboard for Task {requested_task_id}: {TASK_LOOKUP[requested_task_id]['title']}. "
        "After you generate the analysis, this page will jump to that section automatically.",
    )

file = st.file_uploader("Upload welding CSV file", type=["csv"])

if file:
    df = pd.read_csv(file)
    detected_voltage_column = next(
        (
            column_name
            for column_name in (
                "Voltage_V",
                "Voltage",
                "voltage_v",
                "voltage",
                "ArcVoltage_V",
                "ArcVoltage",
                "V",
            )
            if column_name in df.columns
        ),
        None,
    )

    default_start_row = 0
    default_end_row = min(1000, len(df))
    if "row_prefill_start" not in st.session_state or int(st.session_state["row_prefill_start"]) < 0 or int(st.session_state["row_prefill_start"]) >= len(df):
        st.session_state["row_prefill_start"] = default_start_row
    if "row_prefill_end" not in st.session_state or int(st.session_state["row_prefill_end"]) <= int(st.session_state["row_prefill_start"]) or int(st.session_state["row_prefill_end"]) > len(df):
        st.session_state["row_prefill_end"] = default_end_row

    control_left, control_mid, control_right = st.columns([1, 1, 1.35], gap="large")

    with control_left:
        start = st.number_input("Start Row", min_value=0, max_value=len(df) - 1, step=1, key="row_prefill_start")

    with control_mid:
        end = st.number_input(
            "End Row",
            min_value=1,
            max_value=len(df),
            step=1,
            key="row_prefill_end",
        )

    with control_right:
        render_copy_card(
            "Selection guide",
            "Choose a window that captures the welding behavior you want to evaluate. Wider windows show the global trend, while shorter windows make local switching and pulse transitions easier to inspect.",
        )

    heat_input_defaults = {
        "efficiency": 0.80,
        "welding_speed_mm_per_s": 5.00,
        "weld_area_mm2": 12.00,
        "weld_length_mm": 100.00,
        "default_voltage_v": 25.00,
    }

    def input_needs_default(input_key: str, value, allow_zero: bool = False) -> bool:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return True
        if input_key == "efficiency":
            return numeric_value <= 0.0 or numeric_value > 1.0
        if input_key in {"min_height_ratio", "prominence_ratio"}:
            return numeric_value <= 0.0 or numeric_value > 1.0
        return numeric_value < 0.0 if allow_zero else numeric_value <= 0.0

    for heat_key, heat_default in heat_input_defaults.items():
        applied_key = f"heat_calc_value_{heat_key}"
        if not st.session_state.get("heat_values_applied", False):
            st.session_state[applied_key] = heat_default
        elif applied_key not in st.session_state or input_needs_default(heat_key, st.session_state[applied_key]):
            st.session_state[applied_key] = heat_default

    efficiency = float(st.session_state["heat_calc_value_efficiency"])
    welding_speed_mm_per_s = float(st.session_state["heat_calc_value_welding_speed_mm_per_s"])
    weld_area_mm2 = float(st.session_state["heat_calc_value_weld_area_mm2"])
    weld_length_mm = float(st.session_state["heat_calc_value_weld_length_mm"])
    default_voltage_v = float(st.session_state["heat_calc_value_default_voltage_v"])

    if end <= start:
        st.error("End Row must be greater than Start Row.")
        st.stop()

    selected_time_window = df.iloc[int(start):int(end)]["Relative_ms"].to_numpy(dtype=float)
    selected_time_start_ms = float(np.nanmin(selected_time_window)) if selected_time_window.size else 0.0
    selected_time_end_ms = float(np.nanmax(selected_time_window)) if selected_time_window.size else 0.0
    peak_time_step_ms = max(0.1, (selected_time_end_ms - selected_time_start_ms) / 200.0)

    peak_input_defaults = {
        "resistance_ohm": 0.25,
        "min_height_ratio": 0.35,
        "min_distance_ms": 5.0,
        "prominence_ratio": 0.04,
        "smoothing_window_samples": 7,
    }
    for peak_key, peak_default in peak_input_defaults.items():
        applied_key = f"active_peak_{peak_key}"
        if not st.session_state.get("peak_values_applied", False):
            st.session_state[applied_key] = peak_default
        elif applied_key not in st.session_state or input_needs_default(peak_key, st.session_state[applied_key]):
            st.session_state[applied_key] = peak_default

    peak_time_defaults = {
        "analysis_start_ms": selected_time_start_ms,
        "analysis_end_ms": selected_time_end_ms,
    }
    for peak_key, peak_default in peak_time_defaults.items():
        applied_key = f"active_peak_{peak_key}"
        if not st.session_state.get("peak_values_applied", False):
            st.session_state[applied_key] = peak_default
        else:
            st.session_state.setdefault(applied_key, peak_default)
            state_value = float(st.session_state[applied_key])
            if state_value < selected_time_start_ms or state_value > selected_time_end_ms:
                st.session_state[applied_key] = peak_default

    if float(st.session_state["active_peak_analysis_end_ms"]) <= float(st.session_state["active_peak_analysis_start_ms"]):
        st.session_state["active_peak_analysis_start_ms"] = selected_time_start_ms
        st.session_state["active_peak_analysis_end_ms"] = selected_time_end_ms

    peak_resistance_ohm = float(st.session_state["active_peak_resistance_ohm"])
    peak_min_height_ratio = float(st.session_state["active_peak_min_height_ratio"])
    peak_min_distance_ms = float(st.session_state["active_peak_min_distance_ms"])
    peak_prominence_ratio = float(st.session_state["active_peak_prominence_ratio"])
    peak_smoothing_window_samples = int(st.session_state["active_peak_smoothing_window_samples"])
    peak_analysis_start_ms = float(st.session_state["active_peak_analysis_start_ms"])
    peak_analysis_end_ms = float(st.session_state["active_peak_analysis_end_ms"])

    if peak_analysis_end_ms <= peak_analysis_start_ms:
        st.error("Peak Analysis End Time must be greater than Peak Analysis Start Time.")
        st.stop()

    current_file_token = (
        getattr(file, "name", ""),
        getattr(file, "size", 0),
    )
    if st.session_state.get("analysis_file_token") != current_file_token:
        st.session_state["analysis_file_token"] = current_file_token
        st.session_state["analysis_ready"] = False
        st.session_state["heat_values_applied"] = False
        st.session_state["peak_values_applied"] = False

    if st.button("Generate And Evaluate", use_container_width=True):
        st.session_state["analysis_ready"] = True

    if st.session_state.get("analysis_ready", False):
        result = generate_synthetic_and_metrics(df, int(start), int(end))

        if len(result) != 12:
            st.error("Generator output is incomplete.")
            st.stop()

        (
            x_real,
            syn,
            t_real,
            f_real,
            psd_real,
            f_syn,
            psd_syn,
            acf_real,
            acf_syn,
            _power_real_generated,
            _power_syn_generated,
            metrics,
        ) = result

        if len(t_real) > 1:
            duration_ms = float(t_real[-1] - t_real[0])
        else:
            duration_ms = 0.0

        voltage_real, voltage_source = resolve_voltage_signal(df, int(start), int(end), float(default_voltage_v))

        real_heat = calculate_welding_heat(
            time_ms=t_real,
            current_a=x_real,
            voltage_v=voltage_real,
            efficiency=efficiency,
            welding_speed_mm_per_s=welding_speed_mm_per_s,
            weld_area_mm2=weld_area_mm2,
            weld_length_mm=weld_length_mm,
        )
        synthetic_heat = calculate_welding_heat(
            time_ms=t_real,
            current_a=syn,
            voltage_v=voltage_real,
            efficiency=efficiency,
            welding_speed_mm_per_s=welding_speed_mm_per_s,
            weld_area_mm2=weld_area_mm2,
            weld_length_mm=weld_length_mm,
        )

        power_real = real_heat["power_w"]
        power_syn = synthetic_heat["power_w"]
        cumulative_power_real = np.cumsum(power_real)
        cumulative_power_syn = np.cumsum(power_syn)
        energy_real = real_heat["cumulative_energy_j"]
        energy_syn = synthetic_heat["cumulative_energy_j"]
        cumulative_heat_real = real_heat["cumulative_heat_j"]
        cumulative_heat_syn = synthetic_heat["cumulative_heat_j"]

        avg_current_real = float(np.mean(x_real))
        avg_current_syn = float(np.mean(syn))
        avg_power_real = float(np.mean(power_real))
        avg_power_syn = float(np.mean(power_syn))
        peak_power_real = float(np.max(power_real))
        peak_power_syn = float(np.max(power_syn))
        final_cumulative_power_real = float(cumulative_power_real[-1]) if len(cumulative_power_real) else 0.0
        final_cumulative_power_syn = float(cumulative_power_syn[-1]) if len(cumulative_power_syn) else 0.0
        final_energy_real = float(real_heat["total_energy_j"])
        final_energy_syn = float(synthetic_heat["total_energy_j"])
        effective_heat_real = float(real_heat["effective_heat_j"])
        effective_heat_syn = float(synthetic_heat["effective_heat_j"])
        total_loss_real = float(real_heat["total_loss_j"])
        total_loss_syn = float(synthetic_heat["total_loss_j"])
        loss_breakdown_real = real_heat["loss_breakdown_j"]
        loss_breakdown_syn = synthetic_heat["loss_breakdown_j"]
        heat_input_per_length_real = float(real_heat["heat_input_per_length_j_per_mm"])
        heat_input_per_length_syn = float(synthetic_heat["heat_input_per_length_j_per_mm"])
        heat_density_real = float(real_heat["heat_density_j_per_mm3"])
        heat_density_syn = float(synthetic_heat["heat_density_j_per_mm3"])
        instantaneous_heat_input_real = real_heat["instantaneous_heat_input_j_per_mm"]
        instantaneous_heat_input_syn = synthetic_heat["instantaneous_heat_input_j_per_mm"]
        real_material = material_analysis(effective_heat_real, weld_area_mm2, weld_length_mm)
        synthetic_material = material_analysis(effective_heat_syn, weld_area_mm2, weld_length_mm)
        peak_time_mask = (t_real >= float(peak_analysis_start_ms)) & (t_real <= float(peak_analysis_end_ms))
        peak_time_real = t_real[peak_time_mask]
        peak_current_real = x_real[peak_time_mask]
        peak_current_syn = syn[peak_time_mask]

        if len(peak_time_real) < 3:
            st.error("The selected peak-analysis time window must contain at least 3 samples.")
            st.stop()

        real_peak_analysis = analyze_peak_power_categories(
            time_ms=peak_time_real,
            current_a=peak_current_real,
            resistance_ohm=peak_resistance_ohm,
            min_height_ratio=peak_min_height_ratio,
            min_distance_ms=peak_min_distance_ms,
            prominence_ratio=peak_prominence_ratio,
            smoothing_window_samples=peak_smoothing_window_samples,
        )
        synthetic_peak_analysis = analyze_peak_power_categories(
            time_ms=peak_time_real,
            current_a=peak_current_syn,
            resistance_ohm=peak_resistance_ohm,
            min_height_ratio=peak_min_height_ratio,
            min_distance_ms=peak_min_distance_ms,
            prominence_ratio=peak_prominence_ratio,
            smoothing_window_samples=peak_smoothing_window_samples,
        )
        real_peak_df = build_peak_detail_dataframe(real_peak_analysis)
        synthetic_peak_df = build_peak_detail_dataframe(synthetic_peak_analysis)
        peak_category_summary_df = build_peak_category_summary_dataframe(real_peak_analysis, synthetic_peak_analysis)
        peak_reference_df = pd.DataFrame(
            [
                {
                    "Category": category_name,
                    "Current Rule": PEAK_CATEGORY_REFERENCE[category_name]["band"],
                    "Meaning": PEAK_CATEGORY_REFERENCE[category_name]["description"],
                }
                for category_name in PEAK_CATEGORY_ORDER
            ]
        )
        real_peak_count = int(real_peak_analysis["total_peaks"])
        synthetic_peak_count = int(synthetic_peak_analysis["total_peaks"])
        real_avg_detected_peak_power = float(real_peak_analysis["avg_peak_power_w"])
        synthetic_avg_detected_peak_power = float(synthetic_peak_analysis["avg_peak_power_w"])
        real_peak_reference_current = float(real_peak_analysis["reference_current_a"])
        synthetic_peak_reference_current = float(synthetic_peak_analysis["reference_current_a"])
        real_dominant_peak_category = real_peak_analysis["dominant_category"]
        synthetic_dominant_peak_category = synthetic_peak_analysis["dominant_category"]
        peak_count_gap_pct = abs(safe_percent_difference(real_peak_count, synthetic_peak_count))
        avg_detected_peak_power_gap_pct = abs(
            safe_percent_difference(real_avg_detected_peak_power, synthetic_avg_detected_peak_power)
        )
        effective_peak_distance_ms = float(real_peak_analysis["min_distance_ms"])
        effective_peak_height_a_real = float(real_peak_analysis["min_height_a"])
        effective_peak_height_a_syn = float(synthetic_peak_analysis["min_height_a"])
        effective_peak_prominence_a_real = float(real_peak_analysis["prominence_threshold_a"])
        effective_peak_prominence_a_syn = float(synthetic_peak_analysis["prominence_threshold_a"])

        rms_error = float(metrics.get("RMS % Error", 0.0))
        psd_error = float(metrics.get("PSD Peak % Difference", 0.0))
        acf_corr = float(metrics.get("ACF Correlation", 0.0))
        kurt_error = float(metrics.get("Kurtosis % Error", 0.0))

        current_corr = safe_corr(x_real, syn)
        power_corr = safe_corr(power_real, power_syn)
        avg_power_delta_pct = safe_percent_difference(avg_power_real, avg_power_syn)
        cumulative_power_delta_pct = safe_percent_difference(final_cumulative_power_real, final_cumulative_power_syn)
        energy_delta_pct = safe_percent_difference(final_energy_real, final_energy_syn)
        effective_heat_delta_pct = safe_percent_difference(effective_heat_real, effective_heat_syn)
        heat_density_delta_pct = safe_percent_difference(heat_density_real, heat_density_syn)
        total_loss_delta_pct = safe_percent_difference(total_loss_real, total_loss_syn)
        temperature_difference_pct = abs(
            safe_percent_difference(real_material["final_temperature_c"], synthetic_material["final_temperature_c"])
        )
        heat_sufficiency_difference_pct = abs(
            safe_percent_difference(
                real_material["heat_sufficiency_pct"],
                synthetic_material["heat_sufficiency_pct"],
            )
        )
        heat_gap_pct = abs(safe_percent_difference(effective_heat_real, effective_heat_syn))
        real_peak_freq = float(f_real[int(np.argmax(psd_real))]) if len(f_real) else 0.0
        syn_peak_freq = float(f_syn[int(np.argmax(psd_syn))]) if len(f_syn) else 0.0
        overall_match = overall_match_text(rms_error, current_corr, acf_corr, energy_delta_pct)

        current_trend = (
            f"The synthetic current shows a {similarity_text(current_corr)} waveform match to the real signal "
            f"(correlation {current_corr:.3f}). Its average current is {avg_current_syn:.2f} A, which is "
            f"{absolute_difference_phrase(avg_current_syn - avg_current_real, 'A')} the real average of {avg_current_real:.2f} A."
        )
        power_trend = (
            f"Power bursts remain {similarity_text(power_corr)} aligned between the two traces "
            f"(correlation {power_corr:.3f}). The synthetic average power is {avg_power_syn:.2f} W, "
            f"{percent_difference_phrase(avg_power_delta_pct)} the real average."
        )
        avg_power_trend = (
            f"The bar gap shows whether the synthetic process is delivering too much or too little heat on average. "
            f"Here the synthetic mean is {percent_difference_phrase(avg_power_delta_pct)} the real mean."
        )
        cumulative_power_trend = (
            f"This running-sum curve shows whether the synthetic signal accumulates power at the same pace as the real weld. "
            f"By the end of the window, cumulative power is {percent_difference_phrase(cumulative_power_delta_pct)} the real trace."
        )
        energy_trend = (
            f"Both cumulative energy curves should rise steadily if the welding process remains active. "
            f"The synthetic signal finishes at {final_energy_syn:.2f} J versus {final_energy_real:.2f} J, "
            f"so total delivered energy is {percent_difference_phrase(energy_delta_pct)} the real case."
        )
        heat_trend = (
            f"With efficiency set to {efficiency:.2f}, the synthetic signal produces {percent_difference_phrase(effective_heat_delta_pct)} "
            f"the real effective heat, while total heat losses are {percent_difference_phrase(total_loss_delta_pct)} the real case."
        )
        peak_trend = (
            f"The peak detector found {real_peak_count} real peaks and {synthetic_peak_count} synthetic peaks. "
            f"The dominant category is {real_dominant_peak_category} for the real signal and "
            f"{synthetic_dominant_peak_category} for the synthetic signal. "
            f"Average detected peak power is {format_metric_value(real_avg_detected_peak_power)} W for the real data and "
            f"{format_metric_value(synthetic_avg_detected_peak_power)} W for the synthetic data."
        )
        peak_settings_trend = (
            f"Peak power uses R = {peak_resistance_ohm:.2f} Ohm. Minimum peak height is set to {peak_min_height_ratio:.2f} of the signal maximum, "
            f"minimum prominence is {peak_prominence_ratio:.2f} of the signal maximum, and the active peak spacing rule is "
            f"{format_metric_value(effective_peak_distance_ms, 3)} ms with a smoothing window of "
            f"{int(real_peak_analysis['smoothing_window_samples'])} samples."
        )
        peak_time_window_text = (
            f"Peak analysis is evaluated only from {format_metric_value(peak_analysis_start_ms, 3)} ms "
            f"to {format_metric_value(peak_analysis_end_ms, 3)} ms."
        )
        material_trend = (
            f"For the mild-steel weld volume defined by area {weld_area_mm2:.2f} mm^2 and length {weld_length_mm:.2f} mm, "
            f"the synthetic case shows a temperature difference of {format_metric_value(temperature_difference_pct)}% "
            f"and a heat-sufficiency difference of {format_metric_value(heat_sufficiency_difference_pct)}% relative to the real case."
        )
        psd_trend = (
            f"The dominant spectral peak appears near {real_peak_freq:.2f} Hz for the real signal and "
            f"{syn_peak_freq:.2f} Hz for the synthetic signal. A small gap here indicates that the main switching "
            f"and oscillation frequencies are being preserved."
        )
        acf_trend = (
            f"The ACF comparison measures how well repeated temporal patterns survive in the synthetic signal. "
            f"The current ACF correlation is {acf_corr:.3f}, which suggests {similarity_text(acf_corr)} preservation "
            f"of lag-based structure."
        )
        metric_score_labels = [
            "Amplitude Match",
            "Waveform Tracking",
            "Temporal Memory",
            "Energy Agreement",
        ]
        metric_score_values = [
            max(0.0, 100.0 - rms_error),
            max(0.0, current_corr * 100.0),
            max(0.0, acf_corr * 100.0),
            0.0 if not np.isfinite(energy_delta_pct) else max(0.0, 100.0 - abs(energy_delta_pct)),
        ]
        peak_category_labels = list(PEAK_CATEGORY_ORDER)
        real_peak_percentages = [real_peak_analysis["summary"][label]["peak_percentage"] for label in peak_category_labels]
        synthetic_peak_percentages = [
            synthetic_peak_analysis["summary"][label]["peak_percentage"] for label in peak_category_labels
        ]
        dashboard_payload = {
            "generated_at": datetime.now().strftime("%d %b %Y, %I:%M:%S %p"),
            "source_file": getattr(file, "name", "Uploaded CSV"),
            "overall_match": overall_match.title(),
            "summary_metrics": [
                build_task_metric("Samples Analyzed", f"{len(x_real):,}"),
                build_task_metric("Window Duration (ms)", f"{duration_ms:.1f}"),
                build_task_metric("Current Correlation", f"{current_corr:.3f}"),
                build_task_metric("Energy Gap (%)", format_metric_value(abs(energy_delta_pct))),
            ],
            "controls": {
                "start_row": int(start),
                "end_row": int(end),
                "efficiency": safe_json_value(efficiency),
                "welding_speed_mm_per_s": safe_json_value(welding_speed_mm_per_s),
                "weld_area_mm2": safe_json_value(weld_area_mm2),
                "weld_length_mm": safe_json_value(weld_length_mm),
                "default_voltage_v": safe_json_value(default_voltage_v),
                "peak_resistance_ohm": safe_json_value(peak_resistance_ohm),
                "peak_min_height_ratio": safe_json_value(peak_min_height_ratio),
                "peak_min_distance_ms": safe_json_value(peak_min_distance_ms),
                "peak_prominence_ratio": safe_json_value(peak_prominence_ratio),
                "peak_smoothing_window_samples": int(peak_smoothing_window_samples),
                "peak_analysis_start_ms": safe_json_value(peak_analysis_start_ms),
                "peak_analysis_end_ms": safe_json_value(peak_analysis_end_ms),
            },
            "sections": {
                "01": {
                    "title": "Current Waveform Analysis",
                    "narrative": current_trend,
                    "metrics": [
                        build_task_metric("Real Avg Current (A)", format_metric_value(avg_current_real)),
                        build_task_metric("Synthetic Avg Current (A)", format_metric_value(avg_current_syn)),
                        build_task_metric("Correlation", f"{current_corr:.3f}"),
                        build_task_metric("Peak Power Gap (%)", format_metric_value(abs(safe_percent_difference(peak_power_real, peak_power_syn)))),
                    ],
                    "chart": build_dual_line_chart_payload(
                        "Current vs Time",
                        "Time (ms)",
                        "Current (A)",
                        t_real,
                        x_real,
                        t_real,
                        syn,
                        real_name="Real Current",
                        synthetic_name="Synthetic Current",
                    ),
                },
                "02": {
                    "title": "Power And Energy Delivery Analysis",
                    "narrative": f"{power_trend} {energy_trend}",
                    "metrics": [
                        build_task_metric("Real Avg Power (W)", format_metric_value(avg_power_real)),
                        build_task_metric("Synthetic Avg Power (W)", format_metric_value(avg_power_syn)),
                        build_task_metric("Real Energy (J)", format_metric_value(final_energy_real)),
                        build_task_metric("Synthetic Energy (J)", format_metric_value(final_energy_syn)),
                    ],
                    "chart": build_dual_line_chart_payload(
                        "Cumulative Energy vs Time",
                        "Time (ms)",
                        "Energy (J)",
                        t_real,
                        energy_real,
                        t_real,
                        energy_syn,
                        real_name="Real Energy",
                        synthetic_name="Synthetic Energy",
                    ),
                },
                "03": {
                    "title": "Quantitative Match Metrics",
                    "narrative": metric_summary_text(rms_error, acf_corr, current_corr, energy_delta_pct),
                    "metrics": [
                        build_task_metric("RMS Error (%)", format_metric_value(rms_error)),
                        build_task_metric("Current Correlation", f"{current_corr:.3f}"),
                        build_task_metric("ACF Correlation", f"{acf_corr:.3f}"),
                        build_task_metric("Energy Gap (%)", format_metric_value(abs(energy_delta_pct))),
                    ],
                    "chart": build_single_bar_chart_payload(
                        "Metric Scoreboard",
                        "Score (%)",
                        metric_score_labels,
                        metric_score_values,
                        ANALYSIS_SECTION_THEMES["metrics"]["accent"],
                    ),
                },
                "04": {
                    "title": "Frequency And Temporal Pattern Analysis",
                    "narrative": f"{psd_trend} {acf_trend}",
                    "metrics": [
                        build_task_metric("Real Dominant Freq (Hz)", format_metric_value(real_peak_freq)),
                        build_task_metric("Synthetic Dominant Freq (Hz)", format_metric_value(syn_peak_freq)),
                        build_task_metric("PSD Peak Error (%)", format_metric_value(abs(psd_error))),
                        build_task_metric("ACF Correlation", f"{acf_corr:.3f}"),
                    ],
                    "chart": build_dual_line_chart_payload(
                        "PSD Comparison",
                        "Frequency (Hz)",
                        "Power Spectral Density",
                        f_real,
                        psd_real,
                        f_syn,
                        psd_syn,
                        real_name="Real PSD",
                        synthetic_name="Synthetic PSD",
                        use_log_y=True,
                    ),
                },
                "05": {
                    "title": "Heat Transfer And Welding Heat Analysis",
                    "narrative": f"{heat_trend} Voltage source used for both cases: {voltage_source}.",
                    "metrics": [
                        build_task_metric("Real Useful Heat (J)", format_metric_value(effective_heat_real)),
                        build_task_metric("Synthetic Useful Heat (J)", format_metric_value(effective_heat_syn)),
                        build_task_metric("Real Heat Density (J/mm^3)", format_metric_value(heat_density_real)),
                        build_task_metric("Synthetic Heat Density (J/mm^3)", format_metric_value(heat_density_syn)),
                    ],
                    "chart": build_dual_line_chart_payload(
                        "Cumulative Heat vs Time",
                        "Time (ms)",
                        "Effective Heat (J)",
                        t_real,
                        cumulative_heat_real,
                        t_real,
                        cumulative_heat_syn,
                        real_name="Real Heat",
                        synthetic_name="Synthetic Heat",
                    ),
                },
                "06": {
                    "title": "Material Response Analysis For Mild Steel",
                    "narrative": f"{material_trend} The effective-heat gap between real and synthetic cases is {format_metric_value(heat_gap_pct)}%.",
                    "metrics": [
                        build_task_metric("Real Final Temperature (C)", format_metric_value(real_material["final_temperature_c"])),
                        build_task_metric("Synthetic Final Temperature (C)", format_metric_value(synthetic_material["final_temperature_c"])),
                        build_task_metric("Real Sufficiency (%)", format_metric_value(real_material["heat_sufficiency_pct"])),
                        build_task_metric("Synthetic Sufficiency (%)", format_metric_value(synthetic_material["heat_sufficiency_pct"])),
                    ],
                    "chart": build_grouped_bar_chart_payload(
                        "Required vs Supplied Heat",
                        "Heat (J)",
                        ["Required Heat", "Supplied Heat"],
                        [real_material["required_heat_j"], real_material["supplied_heat_j"]],
                        [synthetic_material["required_heat_j"], synthetic_material["supplied_heat_j"]],
                    ),
                },
                "07": {
                    "title": "Peak Detection And Peak-Type Classification",
                    "narrative": peak_trend,
                    "metrics": [
                        build_task_metric("Real Peak Count", f"{real_peak_count}"),
                        build_task_metric("Synthetic Peak Count", f"{synthetic_peak_count}"),
                        build_task_metric("Real Dominant Peak Type", real_dominant_peak_category),
                        build_task_metric("Synthetic Dominant Peak Type", synthetic_dominant_peak_category),
                    ],
                    "chart": build_grouped_bar_chart_payload(
                        "Peak Category Share Comparison",
                        "Share Of Detected Peaks (%)",
                        peak_category_labels,
                        real_peak_percentages,
                        synthetic_peak_percentages,
                    ),
                },
            },
        }
        dashboard_save_ok, dashboard_save_message = save_dashboard_state(dashboard_payload)

        if dashboard_save_ok:
            st.success(f"{dashboard_save_message} The professor dashboard can now present this run.")
        else:
            st.warning(dashboard_save_message)

        render_copy_card(
            "Evaluation summary",
            f"This run covers {len(x_real):,} samples over {duration_ms:.1f} ms. The overall match is {overall_match}, "
            f"with RMS error {rms_error:.2f}%, current correlation {current_corr:.3f}, "
            f"ACF correlation {acf_corr:.3f}, and energy gap {format_metric_value(abs(energy_delta_pct))}%.",
            tone="accent",
        )

        overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
        overview_col1.metric("Samples Analyzed", f"{len(x_real):,}")
        overview_col2.metric("Window Duration (ms)", f"{duration_ms:.1f}")
        overview_col3.metric("Current Correlation", f"{current_corr:.3f}")
        overview_col4.metric("Energy Gap (%)", format_metric_value(abs(energy_delta_pct)))

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        render_task_anchor("01")
        render_analysis_section_header(
            1,
            "Current Waveform Analysis",
            "This block compares the real and synthetic welding current directly in the time domain so you can inspect pulse height, switching rhythm, and overall waveform shape.",
            "current",
        )
        st.subheader("Current Signal Comparison")
        st.markdown(
            "This chart compares real and synthetic welding current over the selected time window. "
            "Use it to inspect pulse heights, switching transitions, and whether the synthetic trace follows the same envelope."
        )
        render_copy_card("General trend", current_trend)

        current_figure = build_line_figure(
            t_real,
            x_real,
            t_real,
            syn,
            title="Current vs Time",
            xaxis_title="Time (ms)",
            yaxis_title="Current (A)",
            real_name="Real Current",
            syn_name="Synthetic Current",
            show_range_slider=True,
        )
        st.plotly_chart(current_figure, use_container_width=True, config=PLOT_CONFIG)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        render_task_anchor("02")
        render_analysis_section_header(
            2,
            "Power And Energy Delivery Analysis",
            "This group converts current into electrical power and cumulative energy so you can see whether the synthetic signal is depositing heat at the same intensity and pace as the real weld.",
            "power",
        )
        st.subheader("Power Comparison")
        st.markdown(
            "Power is computed from the current signal and helps show how heat input changes with time. "
            "If the synthetic signal is realistic, its power bursts should line up with the real process."
        )
        render_copy_card("General trend", power_trend)

        power_figure = build_line_figure(
            t_real,
            power_real,
            t_real,
            power_syn,
            title="Power vs Time",
            xaxis_title="Time (ms)",
            yaxis_title="Power (W)",
            real_name="Real Power",
            syn_name="Synthetic Power",
            show_range_slider=True,
        )
        st.plotly_chart(power_figure, use_container_width=True, config=PLOT_CONFIG)

        st.subheader("Cumulative Power vs Time")
        st.markdown(
            "This graph shows the running sum of power over the selected time window. "
            "It helps you see whether the synthetic signal is accumulating power too quickly, too slowly, or at nearly the same pace as the real signal."
        )
        render_copy_card("General trend", cumulative_power_trend)

        cumulative_power_figure = build_line_figure(
            t_real,
            cumulative_power_real,
            t_real,
            cumulative_power_syn,
            title="Cumulative Power vs Time",
            xaxis_title="Time (ms)",
            yaxis_title="Cumulative Power",
            real_name="Real Cumulative Power",
            syn_name="Synthetic Cumulative Power",
        )
        st.plotly_chart(cumulative_power_figure, use_container_width=True, config=PLOT_CONFIG)

        st.subheader("Average Power Comparison")
        st.markdown(
            "This bar chart compresses the full power signal into a single mean value for each series. "
            "It gives a quick view of whether the synthetic signal is overestimating or underestimating the overall heat input."
        )
        render_copy_card("General trend", avg_power_trend)

        avg_power_figure = build_bar_figure(
            ["Real Average", "Synthetic Average"],
            [avg_power_real, avg_power_syn],
            title="Average Power Comparison",
            yaxis_title="Average Power (W)",
        )
        st.plotly_chart(avg_power_figure, use_container_width=True, config=PLOT_CONFIG)

        stat_col1, stat_col2 = st.columns(2)
        stat_col1.metric("Real Peak Power (W)", f"{peak_power_real:.2f}")
        stat_col2.metric("Synthetic Peak Power (W)", f"{peak_power_syn:.2f}")

        st.subheader("Cumulative Energy")
        st.markdown(
            "Cumulative energy integrates power over time, so it reflects the total heat delivered across the selected welding segment. "
            "This is useful when average power is close but the long-run energy delivery may still drift."
        )
        render_copy_card("General trend", energy_trend)

        energy_figure = build_line_figure(
            t_real,
            energy_real,
            t_real,
            energy_syn,
            title="Cumulative Energy vs Time",
            xaxis_title="Time (ms)",
            yaxis_title="Energy (J)",
            real_name="Real Energy",
            syn_name="Synthetic Energy",
            fill_real_color="rgba(24, 78, 119, 0.12)",
            fill_syn_color="rgba(249, 115, 22, 0.12)",
        )
        st.plotly_chart(energy_figure, use_container_width=True, config=PLOT_CONFIG)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        render_task_anchor("03")
        render_analysis_section_header(
            3,
            "Quantitative Match Metrics",
            "These summary metrics compress the real-vs-synthetic comparison into easy-to-read scores, making it faster to judge overall fit quality at a glance.",
            "metrics",
        )
        st.subheader("Quantitative Metrics")
        st.markdown(
            "These metrics summarize how close the synthetic signal is to the real welding behavior. "
            "Lower RMS error is better, while current correlation and ACF correlation should stay high."
        )
        render_copy_card(
            "How to read the metrics",
            metric_summary_text(rms_error, acf_corr, current_corr, energy_delta_pct),
        )

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("RMS Error (%)", f"{rms_error:.2f}")
        metric_col2.metric("Current Correlation", f"{current_corr:.3f}")
        metric_col3.metric("ACF Correlation", f"{acf_corr:.3f}")
        metric_col4.metric("Energy Gap (%)", format_metric_value(abs(energy_delta_pct)))

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        render_task_anchor("04")
        render_analysis_section_header(
            4,
            "Frequency And Temporal Pattern Analysis",
            "This chapter checks whether the synthetic signal preserves the same dominant oscillation bands and lag-based memory patterns as the real welding process.",
            "frequency",
        )
        st.subheader("Power Spectral Density")
        st.markdown(
            "PSD shows how signal power is distributed across frequencies. "
            "This tells you whether the synthetic signal preserves the same dominant oscillations and switching rhythms as the real process."
        )
        render_copy_card("General trend", psd_trend)

        psd_figure = build_line_figure(
            f_real,
            psd_real,
            f_syn,
            psd_syn,
            title="PSD Comparison",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Power Spectral Density",
            real_name="Real PSD",
            syn_name="Synthetic PSD",
            use_log_y=True,
        )
        st.plotly_chart(psd_figure, use_container_width=True, config=PLOT_CONFIG)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        st.subheader("Autocorrelation Function")
        st.markdown(
            "The ACF chart measures how strongly each signal resembles itself over increasing time lags. "
            "If the synthetic process keeps the same temporal memory as the real one, the two ACF curves should decay in a similar way."
        )
        render_copy_card("General trend", acf_trend)

        lags_real = np.arange(1, len(acf_real) + 1)
        lags_syn = np.arange(1, len(acf_syn) + 1)

        acf_figure = build_line_figure(
            lags_real,
            acf_real,
            lags_syn,
            acf_syn,
            title="ACF Comparison",
            xaxis_title="Lag",
            yaxis_title="Correlation",
            real_name="Real ACF",
            syn_name="Synthetic ACF",
        )
        st.plotly_chart(acf_figure, use_container_width=True, config=PLOT_CONFIG)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        render_task_anchor("05")
        render_analysis_section_header(
            5,
            "Heat Transfer And Welding Heat Analysis",
            "This section translates the electrical signals into engineering heat quantities, loss channels, and heat-input measures that are easier to justify in welding terms.",
            "heat",
        )
        if detected_voltage_column:
            render_copy_card(
                "Heat model inputs",
                f"Heat calculations use the `{detected_voltage_column}` column from the CSV. "
                "The synthetic current is evaluated against the same voltage trace so the heat comparison isolates the effect of current realism.",
            )
        else:
            render_copy_card(
                "Heat model inputs",
                "No voltage column was detected in the CSV, so the fallback voltage below is used for power and heat calculations.",
            )

        heat_input_col1, heat_input_col2, heat_input_col3, heat_input_col4, heat_input_col5 = st.columns(
            [1, 1, 1, 1, 1.1],
            gap="large",
        )

        with heat_input_col1:
            st.number_input(
                "Efficiency (eta)",
                min_value=0.01,
                max_value=1.0,
                value=heat_input_defaults["efficiency"],
                step=0.05,
                key="heat_box_default_v2_efficiency",
            )

        with heat_input_col2:
            st.number_input(
                "Welding Speed (mm/s)",
                min_value=0.001,
                value=heat_input_defaults["welding_speed_mm_per_s"],
                step=0.50,
                key="heat_box_default_v2_welding_speed_mm_per_s",
            )

        with heat_input_col3:
            st.number_input(
                "Weld Area (mm^2)",
                min_value=0.001,
                value=heat_input_defaults["weld_area_mm2"],
                step=0.50,
                key="heat_box_default_v2_weld_area_mm2",
            )

        with heat_input_col4:
            st.number_input(
                "Weld Length (mm)",
                min_value=0.001,
                value=heat_input_defaults["weld_length_mm"],
                step=1.00,
                key="heat_box_default_v2_weld_length_mm",
            )

        with heat_input_col5:
            st.number_input(
                "Default Voltage (V)",
                min_value=0.1,
                value=heat_input_defaults["default_voltage_v"],
                step=0.5,
                key="heat_box_default_v2_default_voltage_v",
            )

        if st.button("Generate Heat Results", key="generate_heat_results", use_container_width=True):
            for heat_key in heat_input_defaults:
                input_key = f"heat_box_default_v2_{heat_key}"
                if input_needs_default(heat_key, st.session_state[input_key]):
                    st.session_state[input_key] = heat_input_defaults[heat_key]
                st.session_state[f"heat_calc_value_{heat_key}"] = st.session_state[input_key]
            st.session_state["heat_values_applied"] = True
            st.session_state["analysis_ready"] = True
            st.rerun()

        st.subheader("Heat Input And Heat Density")
        st.markdown(
            "This section converts electrical power into engineering heat quantities using welding efficiency, travel speed, weld area, and weld length. "
            "Real and synthetic heat are compared using the same voltage reference so the difference mainly reflects how the current signals behave."
        )
        render_copy_card(
            "General trend",
            f"{heat_trend} Voltage source used for both cases: {voltage_source}.",
        )

        equation_col, steps_col = st.columns([1.1, 1], gap="large")

        with equation_col:
            st.markdown("#### Engineering Equations")
            st.latex(r"P(t) = V(t)\,I(t)")
            st.latex(r"E_{total} = \sum_i P_i \,\Delta t")
            st.latex(r"Q_{loss} = Q_{cond} + Q_{conv} + Q_{rad} + Q_{spatter} + Q_{tool}")
            st.latex(r"Q_{net} = E_{total} - Q_{loss} = \eta E_{total}")
            st.latex(r"H(t) = \frac{Q_{net}(t)}{v} = \frac{\eta V(t) I(t)}{v}")
            st.latex(r"q = \frac{Q_{net}}{A\,L}")

        with steps_col:
            st.markdown("#### Simple Calculation Steps")
            st.markdown(
                "1. Compute instantaneous power from voltage and current.\n"
                "2. Convert the sampling interval from milliseconds to seconds.\n"
                "3. Sum `P(t) x dt` to get total electrical energy `E_total`.\n"
                "4. Split the lost portion into conduction, convection, radiation, spatter/fume, and tool losses.\n"
                "5. Subtract losses to get net useful heat `Q_net = eta x E_total`.\n"
                "6. Divide by welding travel distance to get heat input per unit length.\n"
                "7. Divide by weld volume `A x L` to get heat density."
            )

        render_copy_card(
            "Assumptions used in this run",
            f"Efficiency eta = {efficiency:.2f}, welding speed = {welding_speed_mm_per_s:.2f} mm/s, "
            f"weld area = {weld_area_mm2:.2f} mm^2, weld length = {weld_length_mm:.2f} mm. "
            "The dashboard reports heat per unit length as effective heat divided by the travel distance across the selected signal window.",
        )

        render_copy_card(
            "Loss model used",
            "The lost heat portion is distributed into engineering loss channels using a fixed explanatory model: "
            + ", ".join(
                f"{name} = {fraction * 100:.0f}% of total losses"
                for name, fraction in DEFAULT_HEAT_LOSS_FRACTIONS.items()
            )
            + ". The net useful heat is what remains available for the weld pool after these losses.",
        )

        render_welding_loss_diagram()
        render_heat_loss_mechanism_cards()

        sankey_col1, sankey_col2 = st.columns(2, gap="large")
        with sankey_col1:
            real_loss_sankey = build_loss_sankey_figure(
                effective_heat_j=effective_heat_real,
                loss_breakdown_j=loss_breakdown_real,
                title="Real Energy Distribution",
            )
            st.plotly_chart(real_loss_sankey, use_container_width=True, config=PLOT_CONFIG)
        with sankey_col2:
            synthetic_loss_sankey = build_loss_sankey_figure(
                effective_heat_j=effective_heat_syn,
                loss_breakdown_j=loss_breakdown_syn,
                title="Synthetic Energy Distribution",
            )
            st.plotly_chart(synthetic_loss_sankey, use_container_width=True, config=PLOT_CONFIG)

        st.markdown("#### Real Heat Results")
        real_heat_col1, real_heat_col2, real_heat_col3, real_heat_col4, real_heat_col5 = st.columns(5)
        real_heat_col1.metric("Total Energy (J)", format_metric_value(final_energy_real))
        real_heat_col2.metric("Total Losses (J)", format_metric_value(total_loss_real))
        real_heat_col3.metric("Useful Heat (J)", format_metric_value(effective_heat_real))
        real_heat_col4.metric("Heat Per Length (J/mm)", format_metric_value(heat_input_per_length_real))
        real_heat_col5.metric("Heat Density (J/mm^3)", format_metric_value(heat_density_real))

        st.markdown("#### Synthetic Heat Results")
        synthetic_heat_col1, synthetic_heat_col2, synthetic_heat_col3, synthetic_heat_col4, synthetic_heat_col5 = st.columns(5)
        synthetic_heat_col1.metric("Total Energy (J)", format_metric_value(final_energy_syn))
        synthetic_heat_col2.metric("Total Losses (J)", format_metric_value(total_loss_syn))
        synthetic_heat_col3.metric("Useful Heat (J)", format_metric_value(effective_heat_syn))
        synthetic_heat_col4.metric("Heat Per Length (J/mm)", format_metric_value(heat_input_per_length_syn))
        synthetic_heat_col5.metric("Heat Density (J/mm^3)", format_metric_value(heat_density_syn))

        heat_compare_col1, heat_compare_col2, heat_compare_col3 = st.columns(3)
        heat_compare_col1.metric("Effective Heat Gap (%)", format_metric_value(abs(effective_heat_delta_pct)))
        heat_compare_col2.metric("Total Loss Gap (%)", format_metric_value(abs(total_loss_delta_pct)))
        heat_compare_col3.metric("Heat Density Gap (%)", format_metric_value(abs(heat_density_delta_pct)))

        st.subheader("Cumulative Heat vs Time")
        st.markdown(
            "Cumulative heat shows the running effective heat delivered into the weld after applying efficiency. "
            "This lets you compare how quickly real and synthetic signals deposit usable heat during the selected segment."
        )

        cumulative_heat_figure = build_line_figure(
            t_real,
            cumulative_heat_real,
            t_real,
            cumulative_heat_syn,
            title="Cumulative Heat vs Time",
            xaxis_title="Time (ms)",
            yaxis_title="Effective Heat (J)",
            real_name="Real Heat",
            syn_name="Synthetic Heat",
            fill_real_color="rgba(24, 78, 119, 0.12)",
            fill_syn_color="rgba(249, 115, 22, 0.12)",
        )
        st.plotly_chart(cumulative_heat_figure, use_container_width=True, config=PLOT_CONFIG)

        st.subheader("Instantaneous Heat Input Per Unit Length")
        st.markdown(
            "This graph applies the engineering relation `H(t) = eta x V(t) x I(t) / v` sample by sample. "
            "It is useful in a viva because it connects the welding power signal directly to heat input per unit length."
        )

        instantaneous_heat_figure = build_line_figure(
            t_real,
            instantaneous_heat_input_real,
            t_real,
            instantaneous_heat_input_syn,
            title="Heat Input Per Unit Length vs Time",
            xaxis_title="Time (ms)",
            yaxis_title="Heat Input Per Length (J/mm)",
            real_name="Real Heat Input",
            syn_name="Synthetic Heat Input",
        )
        st.plotly_chart(instantaneous_heat_figure, use_container_width=True, config=PLOT_CONFIG)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        render_task_anchor("06")
        render_analysis_section_header(
            6,
            "Material Response Analysis For Mild Steel",
            "This block checks what the delivered heat means for the actual workpiece by estimating final temperature, melting sufficiency, and the gap between real and synthetic material response.",
            "material",
        )
        st.subheader("Material Analysis - Mild Steel")
        st.markdown(
            "This thermodynamic section checks whether the supplied effective heat is enough to raise mild steel from room temperature to melting and supply latent heat of fusion. "
            "Because geometry and material stay the same, the difference between real and synthetic results comes directly from the heat delivered by each signal."
        )
        render_copy_card(
            "General trend",
            f"{material_trend} The effective-heat gap between real and synthetic cases is {format_metric_value(heat_gap_pct)}%.",
        )

        material_constants_col, material_equations_col = st.columns([1, 1.1], gap="large")

        with material_constants_col:
            st.markdown("#### Mild Steel Constants")
            st.markdown(
                f"- Density, rho = {MILD_STEEL_PROPERTIES['density_kg_per_m3']:.0f} kg/m^3\n"
                f"- Specific heat, c = {MILD_STEEL_PROPERTIES['specific_heat_j_per_kgk']:.0f} J/kg.K\n"
                f"- Melting temperature, Tm = {MILD_STEEL_PROPERTIES['melting_temperature_c']:.0f} C\n"
                f"- Initial temperature, T0 = {MILD_STEEL_PROPERTIES['initial_temperature_c']:.0f} C\n"
                f"- Latent heat of fusion, Lf = {MILD_STEEL_PROPERTIES['latent_heat_fusion_j_per_kg']:.0f} J/kg"
            )

        with material_equations_col:
            st.markdown("#### Thermodynamic Equations")
            st.latex(r"V = A \times L")
            st.latex(r"m = \rho V")
            st.latex(r"Q_{net} = E_{total} - (Q_{cond} + Q_{conv} + Q_{rad} + Q_{spatter} + Q_{tool})")
            st.latex(r"Q_{required} = m c (T_m - T_0) + m L_f")
            st.latex(r"\Delta T = \frac{Q_{net}}{m c}")
            st.latex(r"T_{final} = T_0 + \Delta T")
            st.latex(r"\%\,Sufficiency = \frac{Q_{net}}{Q_{required}} \times 100")

        render_copy_card(
            "Calculation flow used in this run",
            f"Volume = area x length = {format_metric_value(real_material['volume_mm3'])} mm^3, which is "
            f"{format_metric_value(real_material['volume_m3'], 9)} m^3. The resulting mass is "
            f"{format_metric_value(real_material['mass_kg'], 6)} kg. The thermodynamic model uses net useful heat after losses, "
            "so the melting check is based on the heat that actually reaches the weld pool.",
        )

        st.markdown("#### Real Data")
        real_material_row1_col1, real_material_row1_col2, real_material_row1_col3, real_material_row1_col4 = st.columns(4)
        real_material_row1_col1.metric("Mass (kg)", format_metric_value(real_material["mass_kg"], 6))
        real_material_row1_col2.metric("Required Heat (J)", format_metric_value(real_material["required_heat_j"]))
        real_material_row1_col3.metric("Total Losses (J)", format_metric_value(total_loss_real))
        real_material_row1_col4.metric("Net Heat After Losses (J)", format_metric_value(real_material["supplied_heat_j"]))

        real_material_row2_col1, real_material_row2_col2, real_material_row2_col3 = st.columns(3)
        real_material_row2_col1.metric("Final Temperature (C)", format_metric_value(real_material["final_temperature_c"]))
        real_material_row2_col2.metric("Heat Sufficiency (%)", format_metric_value(real_material["heat_sufficiency_pct"]))
        real_material_row2_col3.metric(
            "Melting Status",
            "[OK] Achieved" if real_material["melting_achieved"] else "[X] Not Achieved",
        )

        st.markdown("#### Synthetic Data")
        synthetic_material_row1_col1, synthetic_material_row1_col2, synthetic_material_row1_col3, synthetic_material_row1_col4 = st.columns(4)
        synthetic_material_row1_col1.metric("Mass (kg)", format_metric_value(synthetic_material["mass_kg"], 6))
        synthetic_material_row1_col2.metric("Required Heat (J)", format_metric_value(synthetic_material["required_heat_j"]))
        synthetic_material_row1_col3.metric("Total Losses (J)", format_metric_value(total_loss_syn))
        synthetic_material_row1_col4.metric(
            "Net Heat After Losses (J)",
            format_metric_value(synthetic_material["supplied_heat_j"]),
        )

        synthetic_material_row2_col1, synthetic_material_row2_col2, synthetic_material_row2_col3 = st.columns(3)
        synthetic_material_row2_col1.metric(
            "Final Temperature (C)",
            format_metric_value(synthetic_material["final_temperature_c"]),
        )
        synthetic_material_row2_col2.metric(
            "Heat Sufficiency (%)",
            format_metric_value(synthetic_material["heat_sufficiency_pct"]),
        )
        synthetic_material_row2_col3.metric(
            "Melting Status",
            "[OK] Achieved" if synthetic_material["melting_achieved"] else "[X] Not Achieved",
        )

        st.markdown("#### Comparison")
        comparison_col1, comparison_col2, comparison_col3 = st.columns(3)
        comparison_col1.metric("Heat Gap (%)", format_metric_value(heat_gap_pct))
        comparison_col2.metric("Temperature Difference (%)", format_metric_value(temperature_difference_pct))
        comparison_col3.metric(
            "Heat Sufficiency Difference (%)",
            format_metric_value(heat_sufficiency_difference_pct),
        )

        required_vs_supplied_figure = build_grouped_bar_figure(
            ["Required Heat", "Supplied Heat"],
            [real_material["required_heat_j"], real_material["supplied_heat_j"]],
            [synthetic_material["required_heat_j"], synthetic_material["supplied_heat_j"]],
            title="Required Heat vs Supplied Heat",
            yaxis_title="Heat (J)",
        )
        st.plotly_chart(required_vs_supplied_figure, use_container_width=True, config=PLOT_CONFIG)

        temperature_comparison_figure = build_bar_figure(
            ["Real Final Temperature", "Synthetic Final Temperature"],
            [real_material["final_temperature_c"], synthetic_material["final_temperature_c"]],
            title="Final Temperature Comparison",
            yaxis_title="Temperature (C)",
        )
        st.plotly_chart(temperature_comparison_figure, use_container_width=True, config=PLOT_CONFIG)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        render_task_anchor("07")
        render_analysis_section_header(
            7,
            "Peak Detection And Peak-Type Classification",
            "This final chapter isolates the selected peak-analysis window, finds filtered local maxima, computes peak power, and compares how the real and synthetic signals distribute their peak types.",
            "peak",
        )
        render_copy_card(
            "Peak analysis settings",
            "The peak section uses smoothed local-maxima detection with user-controlled filtering. "
            "Minimum height and prominence are entered as fractions of the current maximum in each signal, the same distance rule is used for real and synthetic signals, and peak power is computed from `P = I^2 R`.",
        )

        peak_input_col1, peak_input_col2, peak_input_col3, peak_input_col4, peak_input_col5 = st.columns(
            [1, 1, 1, 1, 1],
            gap="large",
        )

        with peak_input_col1:
            st.number_input(
                "Peak Resistance (Ohm)",
                min_value=0.01,
                value=peak_input_defaults["resistance_ohm"],
                step=0.05,
                key="peak_box_default_v2_resistance_ohm",
            )

        with peak_input_col2:
            st.number_input(
                "Peak Min Height Ratio",
                min_value=0.01,
                max_value=1.0,
                value=peak_input_defaults["min_height_ratio"],
                step=0.05,
                key="peak_box_default_v2_min_height_ratio",
            )

        with peak_input_col3:
            st.number_input(
                "Peak Min Distance (ms)",
                min_value=0.01,
                value=peak_input_defaults["min_distance_ms"],
                step=0.05,
                key="peak_box_default_v2_min_distance_ms",
            )

        with peak_input_col4:
            st.number_input(
                "Peak Prominence Ratio",
                min_value=0.01,
                max_value=1.0,
                value=peak_input_defaults["prominence_ratio"],
                step=0.01,
                key="peak_box_default_v2_prominence_ratio",
            )

        with peak_input_col5:
            st.number_input(
                "Peak Smoothing Window",
                min_value=1,
                max_value=101,
                value=peak_input_defaults["smoothing_window_samples"],
                step=2,
                key="peak_box_default_v2_smoothing_window_samples",
            )

        render_copy_card(
            "Peak analysis time window",
            "Set the peak-detection time window below before regenerating the peak results. "
            "The main dashboard charts still use the full selected row window.",
        )

        peak_time_col1, peak_time_col2 = st.columns(2, gap="large")
        with peak_time_col1:
            st.number_input(
                "Peak Analysis Start Time (ms)",
                min_value=selected_time_start_ms,
                max_value=selected_time_end_ms,
                value=selected_time_start_ms,
                step=peak_time_step_ms,
                key="peak_box_default_v2_analysis_start_ms",
            )

        with peak_time_col2:
            st.number_input(
                "Peak Analysis End Time (ms)",
                min_value=selected_time_start_ms,
                max_value=selected_time_end_ms,
                value=selected_time_end_ms,
                step=peak_time_step_ms,
                key="peak_box_default_v2_analysis_end_ms",
            )

        if st.button("Generate Results", key="generate_peak_results", use_container_width=True):
            if float(st.session_state["peak_box_default_v2_analysis_end_ms"]) <= float(st.session_state["peak_box_default_v2_analysis_start_ms"]):
                st.error("Peak Analysis End Time must be greater than Peak Analysis Start Time.")
            else:
                for peak_key in peak_input_defaults:
                    input_key = f"peak_box_default_v2_{peak_key}"
                    if input_needs_default(peak_key, st.session_state[input_key]):
                        st.session_state[input_key] = peak_input_defaults[peak_key]
                    st.session_state[f"active_peak_{peak_key}"] = st.session_state[input_key]
                for peak_key in peak_time_defaults:
                    st.session_state[f"active_peak_{peak_key}"] = st.session_state[f"peak_box_default_v2_{peak_key}"]
                st.session_state["peak_values_applied"] = True
                st.session_state["analysis_ready"] = True
                st.rerun()

        peak_category_labels = list(PEAK_CATEGORY_ORDER)
        real_peak_percentages = [real_peak_analysis["summary"][label]["peak_percentage"] for label in peak_category_labels]
        synthetic_peak_percentages = [synthetic_peak_analysis["summary"][label]["peak_percentage"] for label in peak_category_labels]
        real_avg_peak_power_by_category = bar_ready(
            [real_peak_analysis["summary"][label]["avg_peak_power_w"] for label in peak_category_labels]
        )
        synthetic_avg_peak_power_by_category = bar_ready(
            [synthetic_peak_analysis["summary"][label]["avg_peak_power_w"] for label in peak_category_labels]
        )

        st.subheader("Peak Detection, Peak Power, And Peak Classification")
        st.markdown(
            "This end section detects local maxima in the welding current, filters them with height, distance, and prominence rules, "
            "computes peak power from `P = I^2 R`, and classifies every accepted peak for both the real and synthetic signals. "
            "The lines shown below are the same original current traces used earlier in the dashboard, now clipped to the peak-analysis time window and marked with detected peaks."
        )
        render_copy_card("General trend", peak_trend)

        peak_intro_col1, peak_intro_col2 = st.columns([1.15, 1], gap="large")
        with peak_intro_col1:
            render_copy_card(
                "Detection settings used",
                f"{peak_settings_trend} The resulting minimum height thresholds are "
                f"{format_metric_value(effective_peak_height_a_real)} A for the real signal and "
                f"{format_metric_value(effective_peak_height_a_syn)} A for the synthetic signal. "
                f"Prominence thresholds are {format_metric_value(effective_peak_prominence_a_real)} A and "
                f"{format_metric_value(effective_peak_prominence_a_syn)} A respectively. {peak_time_window_text}",
            )
        with peak_intro_col2:
            st.markdown("#### Peak Classification Rules Used")
            st.dataframe(peak_reference_df, use_container_width=True)

        peak_metric_row1_col1, peak_metric_row1_col2, peak_metric_row1_col3, peak_metric_row1_col4 = st.columns(4)
        peak_metric_row1_col1.metric("Real Peak Count", f"{real_peak_count}")
        peak_metric_row1_col2.metric("Synthetic Peak Count", f"{synthetic_peak_count}")
        peak_metric_row1_col3.metric("Peak Count Gap (%)", format_metric_value(peak_count_gap_pct))
        peak_metric_row1_col4.metric("Peak Resistance (Ohm)", format_metric_value(peak_resistance_ohm))

        peak_metric_row2_col1, peak_metric_row2_col2, peak_metric_row2_col3, peak_metric_row2_col4 = st.columns(4)
        peak_metric_row2_col1.metric("Real Reference I0 (A)", format_metric_value(real_peak_reference_current))
        peak_metric_row2_col2.metric("Synthetic Reference I0 (A)", format_metric_value(synthetic_peak_reference_current))
        peak_metric_row2_col3.metric("Real Avg Peak Power (W)", format_metric_value(real_avg_detected_peak_power))
        peak_metric_row2_col4.metric("Synthetic Avg Peak Power (W)", format_metric_value(synthetic_avg_detected_peak_power))

        peak_metric_row3_col1, peak_metric_row3_col2, peak_metric_row3_col3 = st.columns(3)
        peak_metric_row3_col1.metric("Avg Peak Power Gap (%)", format_metric_value(avg_detected_peak_power_gap_pct))
        peak_metric_row3_col2.metric("Real Dominant Peak Type", real_dominant_peak_category)
        peak_metric_row3_col3.metric("Synthetic Dominant Peak Type", synthetic_dominant_peak_category)

        peak_window_col1, peak_window_col2 = st.columns(2, gap="large")
        peak_window_col1.metric("Peak Window Start (ms)", format_metric_value(peak_analysis_start_ms, 3))
        peak_window_col2.metric("Peak Window End (ms)", format_metric_value(peak_analysis_end_ms, 3))
        st.caption("Adjust these values with the Peak Analysis Start Time and Peak Analysis End Time controls above.")

        peak_plot_col1, peak_plot_col2 = st.columns(2, gap="large")
        with peak_plot_col1:
            st.markdown("#### Real Current With Detected Peaks")
            real_peak_figure = build_peak_detection_figure(
                peak_time_real,
                peak_current_real,
                real_peak_analysis,
                title="Real Current Peak Detection",
                signal_name="Real Current",
                signal_color=REAL_COLOR,
            )
            st.plotly_chart(real_peak_figure, use_container_width=True, config=PLOT_CONFIG)

        with peak_plot_col2:
            st.markdown("#### Synthetic Current With Detected Peaks")
            synthetic_peak_figure = build_peak_detection_figure(
                peak_time_real,
                peak_current_syn,
                synthetic_peak_analysis,
                title="Synthetic Current Peak Detection",
                signal_name="Synthetic Current",
                signal_color=SYN_COLOR,
            )
            st.plotly_chart(synthetic_peak_figure, use_container_width=True, config=PLOT_CONFIG)

        peak_table_col1, peak_table_col2 = st.columns(2, gap="large")
        with peak_table_col1:
            st.markdown("#### Real Peak Table")
            st.dataframe(real_peak_df, use_container_width=True, height=360)

        with peak_table_col2:
            st.markdown("#### Synthetic Peak Table")
            st.dataframe(synthetic_peak_df, use_container_width=True, height=360)

        st.subheader("Peak Category Share Comparison")
        st.markdown(
            "This comparison shows what percentage of detected peaks fall into each category for the real and synthetic signals. "
            "It helps you check whether the synthetic waveform preserves the same distribution of strong, useful, weak, and short peaks."
        )
        peak_share_figure = build_grouped_bar_figure(
            peak_category_labels,
            real_peak_percentages,
            synthetic_peak_percentages,
            title="Peak Category Share Comparison",
            yaxis_title="Share Of Detected Peaks (%)",
        )
        st.plotly_chart(peak_share_figure, use_container_width=True, config=PLOT_CONFIG)

        st.subheader("Average Peak Power By Category")
        st.markdown(
            "This chart compares the average peak power inside each current-based category. "
            "It shows whether synthetic peaks only match the count distribution or also preserve the peak-energy strength of each class."
        )
        peak_power_category_figure = build_grouped_bar_figure(
            peak_category_labels,
            real_avg_peak_power_by_category,
            synthetic_avg_peak_power_by_category,
            title="Average Peak Power By Category",
            yaxis_title="Average Peak Power (W)",
        )
        st.plotly_chart(peak_power_category_figure, use_container_width=True, config=PLOT_CONFIG)

        st.subheader("Peak Category Comparison Table")
        st.markdown(
            "The table below combines the category rules with the side-by-side real and synthetic counts, percentages, and average peak power values."
        )
        st.dataframe(peak_category_summary_df, use_container_width=True)
        render_report_button()
        scroll_to_requested_task(requested_task_id)

else:
    render_copy_card(
        "Ready for analysis",
        "Upload a CSV file to unlock the interactive charts, trend explanations, and quantitative comparison metrics for the welding signal.",
    )
