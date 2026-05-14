from __future__ import annotations

from html import escape

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from dashboard_bridge import (
    APPLI_PORT,
    APPLI_SCRIPT,
    TASKS,
    appli_url,
    launch_streamlit_app,
    load_dashboard_state,
    port_is_open,
)


def queue_appli_open(task_id: str | None = None) -> None:
    ok, message = launch_streamlit_app(APPLI_SCRIPT, APPLI_PORT)
    st.session_state["appli_open_pending"] = ok
    st.session_state["appli_open_url"] = appli_url(task_id)
    st.session_state["appli_open_message"] = message
    st.session_state["appli_open_error"] = "" if ok else message


def render_pending_appli_open() -> None:
    if st.session_state.get("appli_open_error"):
        st.error(st.session_state["appli_open_error"])
        st.session_state["appli_open_error"] = ""

    if not st.session_state.get("appli_open_pending"):
        return

    launch_message = st.session_state.get("appli_open_message", "")
    open_url = st.session_state.get("appli_open_url", appli_url())
    if launch_message:
        st.success(launch_message)
    st.markdown(f"[Open appli.py manually]({open_url})")
    components.html(
        f"""
        <script>
        window.open("{open_url}", "_blank");
        </script>
        """,
        height=0,
        width=0,
    )
    st.session_state["appli_open_pending"] = False


def build_chart_from_payload(chart_payload: dict | None):
    if not chart_payload:
        return None

    kind = chart_payload.get("kind")
    figure = go.Figure()

    if kind == "dual_line":
        for series in chart_payload.get("series", []):
            figure.add_trace(
                go.Scatter(
                    x=series.get("x", []),
                    y=series.get("y", []),
                    mode="lines",
                    name=series.get("name", ""),
                    line=dict(color=series.get("color", "#0f172a"), width=2.6),
                )
            )
        figure.update_layout(hovermode="x unified")
        if chart_payload.get("use_log_y"):
            figure.update_yaxes(type="log")

    elif kind == "grouped_bar":
        labels = chart_payload.get("labels", [])
        for series in chart_payload.get("series", []):
            figure.add_trace(
                go.Bar(
                    x=labels,
                    y=series.get("values", []),
                    name=series.get("name", ""),
                    marker_color=series.get("color", "#0f172a"),
                )
            )
        figure.update_layout(barmode="group")

    elif kind == "single_bar":
        figure.add_trace(
            go.Bar(
                x=chart_payload.get("labels", []),
                y=chart_payload.get("values", []),
                marker_color=chart_payload.get("color", "#0f172a"),
            )
        )

    else:
        return None

    figure.update_layout(
        title=dict(
            text=chart_payload.get("title", ""),
            x=0.02,
            xanchor="left",
            font=dict(size=11, family="Segoe UI Semibold", color="#0f172a"),
        ),
        template="plotly_white",
        font=dict(size=8, family="Segoe UI"),
        margin=dict(l=14, r=8, t=30, b=14),
        height=105,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.95)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="right",
            x=1.0,
            font=dict(size=7),
        ),
    )
    figure.update_xaxes(
        title=chart_payload.get("xaxis_title", ""),
        showgrid=True,
        gridcolor="rgba(15, 23, 42, 0.08)",
        tickfont=dict(size=7, color="#334155"),
        title_font=dict(size=7, color="#0f172a"),
    )
    figure.update_yaxes(
        title=chart_payload.get("yaxis_title", ""),
        showgrid=True,
        gridcolor="rgba(15, 23, 42, 0.08)",
        tickfont=dict(size=7, color="#334155"),
        title_font=dict(size=7, color="#0f172a"),
    )
    return figure


def render_metric_tiles(metrics: list[dict]) -> None:
    if not metrics:
        return

    for start_index in range(0, len(metrics), 2):
        row_metrics = metrics[start_index:start_index + 2]
        columns = st.columns(2, gap="small")
        for column, metric in zip(columns, row_metrics):
            with column:
                st.markdown(
                    f"""
                    <div class="metric-tile">
                        <div class="metric-label">{metric.get("label", "")}</div>
                        <div class="metric-value">{metric.get("display", "N/A")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_status_grid(tiles: list[dict]) -> None:
    if not tiles:
        return

    columns = st.columns(len(tiles), gap="small")
    for column, tile in zip(columns, tiles):
        with column:
            st.markdown(
                (
                    '<div class="status-tile">'
                    f'<div class="status-label">{escape(str(tile.get("label", "")))}</div>'
                    f'<div class="status-value">{escape(str(tile.get("value", "")))}</div>'
                    f'<div class="status-subvalue">{escape(str(tile.get("subvalue", "")))}</div>'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


def render_task_panel(task: dict, section_data: dict | None) -> None:
    st.markdown(
        f"""
        <div class="task-header" style="--task-accent: {task['accent']};">
            <div class="task-kicker">Task {task['id']} - {task['audience_label']}</div>
            <div class="task-title">{task['title']}</div>
            <div class="task-summary">{task['summary']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if section_data:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-label">Latest takeaway</div>
                <div class="insight-body">{section_data.get('narrative', 'No summary available.')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_metric_tiles(section_data.get("metrics", []))
        task_chart = build_chart_from_payload(section_data.get("chart"))
        if task_chart is not None:
            st.plotly_chart(
                task_chart,
                use_container_width=True,
                config={"displaylogo": False, "responsive": True},
            )
    else:
        st.info("Run `Generate And Evaluate` in `appli.py` to populate this dashboard task.")

st.set_page_config(page_title="Welding Signal Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 220, 180, 0.55), transparent 26%),
            radial-gradient(circle at top right, rgba(176, 220, 255, 0.42), transparent 30%),
            linear-gradient(180deg, #fff8ef 0%, #f7fbff 100%);
    }

    .block-container {
        max-width: 1860px;
        padding-top: 1.6rem;
        padding-bottom: 0.8rem;
    }

    .block-container h3 {
        font-size: 1.05rem !important;
        line-height: 1.15 !important;
        margin: 0.4rem 0 0.45rem 0 !important;
        color: #0f172a !important;
    }

    .hero-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98), rgba(247, 250, 252, 0.95));
        border: 1px solid rgba(15, 23, 42, 0.07);
        border-radius: 16px;
        padding: 0.8rem 1rem;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
        margin-bottom: 0.45rem;
    }

    .hero-kicker {
        color: #c2410c;
        font-size: 0.66rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.18rem;
    }

    .hero-title {
        color: #0f172a;
        font-size: 2rem;
        line-height: 1.05;
        font-weight: 850;
        margin-bottom: 0.25rem;
    }

    .hero-body {
        color: #475569;
        font-size: 0.82rem;
        line-height: 1.25;
        max-width: 940px;
    }

    .hero-note {
        background: rgba(15, 23, 42, 0.05);
        border: 1px solid rgba(15, 23, 42, 0.07);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        color: #334155;
        font-size: 0.98rem;
        margin-top: 1.15rem;
    }

    .summary-card {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(15, 23, 42, 0.07);
        border-radius: 12px;
        padding: 0.5rem 0.62rem;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.045);
        margin-bottom: 0.4rem;
        min-height: 62px;
    }

    .summary-label {
        color: #64748b;
        font-size: 0.58rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.18rem;
    }

    .summary-value {
        color: #0f172a;
        font-size: 0.94rem;
        font-weight: 850;
        line-height: 1.15;
    }

    .summary-subvalue {
        color: #475569;
        font-size: 0.64rem;
        margin-top: 0.16rem;
        line-height: 1.15;
    }

    .status-grid {
        display: grid;
        grid-template-columns: repeat(9, minmax(0, 1fr));
        gap: 0.42rem;
        margin: 0.35rem 0 0.28rem 0;
    }

    .status-tile {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(15, 23, 42, 0.07);
        border-radius: 10px;
        padding: 0.38rem 0.45rem;
        min-height: 54px;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.04);
    }

    .status-label {
        color: #64748b;
        font-size: 0.47rem;
        font-weight: 750;
        letter-spacing: 0.05em;
        line-height: 1.1;
        text-transform: uppercase;
        margin-bottom: 0.12rem;
    }

    .status-value {
        color: #0f172a;
        font-size: 0.74rem;
        font-weight: 850;
        line-height: 1.05;
    }

    .status-subvalue {
        color: #475569;
        font-size: 0.48rem;
        line-height: 1.1;
        margin-top: 0.12rem;
    }

    .task-header {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 12px;
        padding: 0.48rem 0.58rem 0.46rem 0.58rem;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.045);
        margin: 0.15rem 0 0.32rem 0;
        border-left: 5px solid var(--task-accent);
    }

    .task-kicker {
        color: var(--task-accent);
        font-size: 0.52rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        margin-bottom: 0.18rem;
    }

    .task-title {
        color: #0f172a;
        font-size: 0.82rem;
        line-height: 1.1;
        font-weight: 850;
        margin-bottom: 0.2rem;
    }

    .task-summary {
        color: #475569;
        font-size: 0.6rem;
        line-height: 1.2;
    }

    .insight-card {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(15, 23, 42, 0.07);
        border-radius: 10px;
        padding: 0.42rem 0.5rem;
        margin-bottom: 0.28rem;
    }

    .insight-label {
        color: #0f172a;
        font-size: 0.52rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.16rem;
    }

    .insight-body {
        color: #475569;
        font-size: 0.58rem;
        line-height: 1.2;
    }

    .metric-tile {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(15, 23, 42, 0.07);
        border-radius: 10px;
        padding: 0.36rem 0.42rem;
        box-shadow: 0 6px 14px rgba(15, 23, 42, 0.035);
        margin-bottom: 0.28rem;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.5rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.14rem;
    }

    .metric-value {
        color: #0f172a;
        font-size: 0.72rem;
        font-weight: 800;
        line-height: 1.2;
    }

    div[data-testid="stButton"] > button {
        border-radius: 10px;
        border: 0;
        background: linear-gradient(135deg, #0f766e, #0f4c81);
        color: white;
        font-weight: 700;
        min-height: 1.9rem;
        font-size: 0.62rem;
        padding: 0.2rem 0.45rem;
    }

    div[data-testid="stButton"] > button:hover {
        color: white;
        border: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

dashboard_state = load_dashboard_state()
section_state = dashboard_state.get("sections", {}) if dashboard_state else {}
appli_running = port_is_open(APPLI_PORT)

header_left, header_center, header_right = st.columns([0.9, 5.8, 2.3], gap="small")

with header_left:
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.image("iit_logo.png", width=82)

with header_center:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Virtual Instrumentation of Arc Welding</div>
            <div class="hero-title">Dashboard</div>
            <div class="hero-body">
                Synthetic Signal Generation Using Adversarial and Probabilistic Model for Advanced Weld Monitoring.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.image("mech_logo.png", width=270)

summary_metrics = dashboard_state.get("summary_metrics", [])
if summary_metrics:
    status_tiles = [
        {
            "label": "Source File",
            "value": dashboard_state.get("source_file", "Waiting for analysis"),
            "subvalue": "Latest file used in appli.py",
        },
        {
            "label": "Last Generated",
            "value": dashboard_state.get("generated_at", "Not generated yet"),
            "subvalue": "Refresh after a new run",
        },
        {
            "label": "Overall Match",
            "value": dashboard_state.get("overall_match", "Pending"),
            "subvalue": "Latest run summary",
        },
        {
            "label": "Working App Status",
            "value": "Running" if appli_running else "Not Running",
            "subvalue": f"appli.py on port {APPLI_PORT}",
        },
        {
            "label": "Dashboard Control",
            "value": "Latest Analysis",
            "subvalue": "Refresh results or open appli.py",
        },
    ]
    status_tiles.extend(
        {
            "label": metric.get("label", ""),
            "value": metric.get("display", "N/A"),
            "subvalue": "Latest run snapshot",
        }
        for metric in summary_metrics
    )
    render_status_grid(status_tiles)
else:
    st.info("No saved dashboard data yet. Run `Generate And Evaluate` in `appli.py`, then refresh this page.")

action_col1, action_col2 = st.columns(2, gap="small")
with action_col1:
    st.button("Refresh Latest Results", key="refresh_dashboard", use_container_width=True)
with action_col2:
    if st.button("Open Full appli.py", key="open_full_appli", use_container_width=True):
        queue_appli_open()

render_pending_appli_open()

st.markdown("### Seven Task Results")

tasks_per_row = 4
for row_start in range(0, len(TASKS), tasks_per_row):
    row_tasks = TASKS[row_start:row_start + tasks_per_row]
    columns = st.columns(tasks_per_row, gap="small")
    for index, task in enumerate(row_tasks):
        with columns[index]:
            render_task_panel(task, section_state.get(task["id"]))
    if len(row_tasks) < tasks_per_row:
        for empty_index in range(len(row_tasks), tasks_per_row):
            with columns[empty_index]:
                st.empty()
