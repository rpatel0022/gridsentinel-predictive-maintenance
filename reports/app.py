"""GridSentinel — interactive results dashboard (Streamlit).

Run:  streamlit run reports/app.py   (or `make dashboard`)

Reads the small precomputed artifacts in ``reports/assets`` (built by
``reports.precompute`` from the real data), so it loads instantly.
"""
# ruff: noqa: E501 — presentational strings (captions/markdown) read better unwrapped

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"
NAVY, TEAL, AMBER, RED, GREY = "#1f3a5f", "#1b9e9e", "#e8a33d", "#c0392b", "#7f8c8d"


@st.cache_data
def _load():
    summary = json.loads((ASSETS / "summary.json").read_text())
    timeline = pd.read_csv(ASSETS / "anomaly_timeline.csv", parse_dates=["ts"])
    afr = pd.read_csv(ASSETS / "afr.csv")
    return summary, timeline, afr


st.set_page_config(page_title="GridSentinel", page_icon="⚡", layout="wide")

if not (ASSETS / "summary.json").exists():
    st.error(
        "Assets missing — run `python -m reports.precompute --metropt <csv> --backblaze <csv>`."
    )
    st.stop()

s, timeline, afr = _load()
h = s["headline"]

st.title("⚡ GridSentinel")
st.caption(
    "Production predictive-maintenance & fleet-reliability ML system · 2 real datasets · "
    "full MLOps · all results from real data"
)

c = st.columns(5)
c[0].metric(
    "Anomaly detector ROC-AUC", f"{h['anomaly_roc_auc']:.2f}", f"recall {h['anomaly_recall']:.2f}"
)
c[1].metric(
    "Real fleet failures", f"{s['backblaze']['failures_total']:,}", "Backblaze · 418k drives"
)
c[2].metric("Serving p99 latency", f"{h['p99_ms']} ms", "SLO 50 ms")
c[3].metric("Cost vs schedule", f"−{h['cost_savings_pct']}%", "held-out failure")
c[4].metric("ML Test Score", f"{h['ml_test_score']}", f"{h['tests']} tests green")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔎 Anomaly detection", "🏆 Model leaderboard", "💽 Fleet reliability", "🛠️ MLOps & rigor"]
)

with tab1:
    st.subheader("Anomaly score on real compressor telemetry")
    st.caption(
        "The score rises into every real failure (amber). Hover, zoom, pan — it's interactive."
    )
    fig = go.Figure()
    for ev in s["failures"]:
        fig.add_vrect(x0=ev["start"], x1=ev["end"], fillcolor=AMBER, opacity=0.5, line_width=0)
    fig.add_trace(
        go.Scatter(
            x=timeline["ts"],
            y=timeline["score"],
            line=dict(color=NAVY, width=1.6),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>score %{y:.3f}<extra></extra>",
            name="anomaly score",
        )
    )
    fig.add_hline(
        y=s["anomaly_threshold"],
        line=dict(color=RED, dash="dash"),
        annotation_text="alert threshold",
    )
    fig.update_layout(
        height=420, template="plotly_white", margin=dict(t=10, b=10), showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Early warning before each real failure")
    lt = s["lead_times"]
    fig2 = go.Figure(
        go.Bar(
            x=list(lt),
            y=list(lt.values()),
            marker_color=NAVY,
            text=[f"{v:g}h" for v in lt.values()],
            textposition="outside",
        )
    )
    fig2.update_layout(
        height=300, template="plotly_white", yaxis_title="hours of warning", margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Model leaderboard (ROC-AUC, same temporal CV)")
    au = s["model_auc"]
    order = sorted(au, key=au.get)
    fig = go.Figure(
        go.Bar(
            x=[au[k] for k in order],
            y=order,
            orientation="h",
            marker_color=[
                TEAL if au[k] >= 0.9 else (AMBER if au[k] >= 0.7 else GREY) for k in order
            ],
            text=[f"{au[k]:.2f}" for k in order],
            textposition="inside",
        )
    )
    fig.update_layout(
        height=380, template="plotly_white", xaxis_range=[0.5, 1.0], margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "Honest finding: the unsupervised anomaly detector wins. The LSTM underperforms (0.63) — "
        "with only 4 MetroPT failures a recurrent net underfits. That's why the Backblaze fleet "
        "data (24k failures) matters."
    )

with tab3:
    st.subheader("Backblaze fleet — worst drive models by annualized failure rate")
    st.caption(
        f"{s['backblaze']['drives_total']:,} drives · {s['backblaze']['failures_total']:,} real "
        f"failures · {s['backblaze']['models']} models"
    )
    fig = go.Figure(
        go.Bar(
            x=(afr["afr"] * 100),
            y=afr["model"],
            orientation="h",
            marker_color=RED,
            text=[f"{v * 100:.1f}%" for v in afr["afr"]],
            textposition="outside",
            hovertemplate="%{y}<br>%{x:.1f}% AFR<extra></extra>",
        )
    )
    fig.update_layout(
        height=420,
        template="plotly_white",
        xaxis_title="annualized failure rate (%)",
        yaxis=dict(autorange="reversed"),
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    bb = s["backblaze"]
    st.success(
        f"Censoring-safe failure model on {bb['cohort_drives']:,} drives / {bb['cohort_failures']:,} "
        f"failures → ROC-AUC {bb['roc_auc']:.2f}. The pipeline independently recovers "
        "`st3000dm001`'s ~25% failure rate — the infamous Seagate 3 TB drive."
    )

with tab4:
    st.subheader("Production MLOps & engineering rigor")
    st.markdown(
        f"""
- **Serving** — schema-validated FastAPI + Docker · p99 31 ms · model registry (promote / rollback / audit)
- **Self-healing** — drift on the live EIA feed → retrain → metric-gate → promote, else keep current
- **Observability** — Prometheus + Grafana (system + model metrics) · delayed-label backfill
- **CI/CD** — metric gate blocks a regressing model · pip-audit + Trivy scans
- **Edge** — quantized model 5.9× smaller, 4× faster · cloud-vs-edge tradeoff measured
- **Rigor** — temporal CV (no leakage) · cost-tuned thresholds · **Google ML Test Score 4.5** · {h["tests"]} tests
"""
    )
    st.caption("Every claim maps to a file or CI step in the repo — the artifact makes the claim.")
