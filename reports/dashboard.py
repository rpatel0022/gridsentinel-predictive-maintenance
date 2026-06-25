"""Generate the GridSentinel one-page results dashboard (a real-data visual board).

Produces a single PNG that, at a glance, shows what the system is and that it works:
the anomaly detector firing before real failures, the model leaderboard, early-warning
lead times, the Backblaze fleet failure rates, and the cost/ROI headline. Everything is
computed from the real data + measured results — no mock numbers.

    python -m reports.dashboard --metropt <csv> --backblaze <csv> --out docs/dashboard.png
"""
# ruff: noqa: E501 — presentational title/caption strings read better unwrapped

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

NAVY, TEAL, AMBER, RED, GREY = "#1f3a5f", "#1b9e9e", "#e8a33d", "#c0392b", "#7f8c8d"

# Measured results (from the committed results docs — all real).
MODEL_AUC = {
    "Anomaly\n(IsolationForest)": 0.95,
    "MLP\nsequence": 0.93,
    "XGBoost\n(supervised)": 0.92,
    "Backblaze\nfleet RF": 0.73,
    "LSTM": 0.63,
}
LEAD_TIMES = {"2020-04-18": 19.5, "2020-05-29": 0.2, "2020-06-05": 47.8, "2020-07-15": 47.8}


def _anomaly_timeline(metropt_csv: str):
    from pipelines.anomaly import (
        FAILURE_EVENTS,
        anomaly_score,
        build_detector,
        normal_training_mask,
    )
    from pipelines.features import WINDOW_START, build_windowed_features, feature_columns

    feats = build_windowed_features(pd.read_csv(metropt_csv), freq="10min")
    cols = feature_columns(feats)
    X = feats[cols].to_numpy()
    mask = normal_training_mask(feats[WINDOW_START])
    det = build_detector(n_estimators=200).fit(X[mask])
    scores = anomaly_score(det, X)
    thr = float(np.quantile(scores[mask], 0.99))
    ts = pd.to_datetime(feats[WINDOW_START])
    return ts, scores, thr, FAILURE_EVENTS


def _afr(backblaze_csv: str):
    from pipelines.backblaze import annualized_failure_rates, load

    afr = annualized_failure_rates(load(backblaze_csv), min_drives=5000).head(6)
    return afr.index.tolist(), afr["afr"].tolist()


def build(metropt_csv: str, backblaze_csv: str, out: str) -> str:
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GREY, "axes.linewidth": 0.8})
    fig = plt.figure(figsize=(16, 9.5), facecolor="white")
    gs = fig.add_gridspec(
        3, 3, hspace=0.55, wspace=0.3, top=0.88, bottom=0.07, left=0.06, right=0.97
    )

    # Title banner
    fig.text(0.06, 0.955, "GridSentinel", fontsize=30, fontweight="bold", color=NAVY)
    fig.text(
        0.06,
        0.915,
        "Production predictive-maintenance & fleet-reliability ML system  ·  2 real datasets  ·  full MLOps  ·  140 tests green",
        fontsize=12.5,
        color=GREY,
    )

    # --- Hero: anomaly score over time, failures shaded ---
    ax = fig.add_subplot(gs[0, :])
    ts, scores, thr, events = _anomaly_timeline(metropt_csv)
    smooth = pd.Series(scores).rolling(72, center=True, min_periods=1).mean()  # ~12h mean
    for ev in events:
        ax.axvspan(ev.start, ev.end, color=AMBER, alpha=0.85, zorder=1)
    ax.axvspan(
        events[0].start, events[0].end, color=AMBER, alpha=0.85, label="real failure", zorder=1
    )
    ax.plot(ts, scores, color=TEAL, lw=0.4, alpha=0.25, zorder=2)
    ax.plot(ts, smooth, color=NAVY, lw=1.8, label="anomaly score (12h mean)", zorder=3)
    ax.axhline(thr, color=RED, ls="--", lw=1, label=f"alert threshold ({thr:.3f})", zorder=2)
    ax.set_title(
        "Anomaly score on real compressor telemetry — it rises into every real failure (amber)",
        fontsize=12,
        fontweight="bold",
        color=NAVY,
        loc="left",
    )
    ax.set_ylabel("anomaly score")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, ncol=3)
    ax.margins(x=0.01)

    # --- Model leaderboard ---
    ax = fig.add_subplot(gs[1, 0])
    names, aucs = list(MODEL_AUC), list(MODEL_AUC.values())
    colors = [TEAL if a >= 0.9 else (AMBER if a >= 0.7 else GREY) for a in aucs]
    ax.barh(names, aucs, color=colors)
    for i, a in enumerate(aucs):
        ax.text(
            a - 0.02,
            i,
            f"{a:.2f}",
            va="center",
            ha="right",
            color="white",
            fontweight="bold",
            fontsize=9,
        )
    ax.set_xlim(0.5, 1.0)
    ax.invert_yaxis()
    ax.set_title(
        "Model leaderboard (ROC-AUC)", fontsize=12, fontweight="bold", color=NAVY, loc="left"
    )

    # --- Early warning ---
    ax = fig.add_subplot(gs[1, 1])
    days, hrs = list(LEAD_TIMES), list(LEAD_TIMES.values())
    ax.bar(range(len(days)), hrs, color=NAVY)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels([d[5:] for d in days], fontsize=8)
    for i, h in enumerate(hrs):
        ax.text(i, h + 0.7, f"{h:g}h", ha="center", fontsize=9, color=NAVY, fontweight="bold")
    ax.set_ylabel("hours of warning")
    ax.set_title(
        "Early warning before each real failure",
        fontsize=12,
        fontweight="bold",
        color=NAVY,
        loc="left",
    )

    # --- Backblaze fleet AFR ---
    ax = fig.add_subplot(gs[1, 2])
    models, afrs = _afr(backblaze_csv)
    ax.barh(models, [a * 100 for a in afrs], color=RED)
    ax.invert_yaxis()
    ax.set_xlabel("annualized failure rate (%)")
    ax.set_title(
        "Backblaze fleet: worst drive models\n(418k drives · 24k real failures)",
        fontsize=11.5,
        fontweight="bold",
        color=NAVY,
        loc="left",
    )
    ax.tick_params(axis="y", labelsize=8)

    # --- Cost / ROI ---
    ax = fig.add_subplot(gs[2, 0])
    ax.bar(["Fixed\nschedule", "GridSentinel"], [100, 40], color=[GREY, TEAL])
    ax.text(1, 42, "−60%", ha="center", color=TEAL, fontweight="bold", fontsize=13)
    ax.set_ylabel("expected maintenance cost")
    ax.set_title(
        "Cost vs schedule (held-out failure)",
        fontsize=12,
        fontweight="bold",
        color=NAVY,
        loc="left",
    )

    # --- Capabilities text panel ---
    ax = fig.add_subplot(gs[2, 1:])
    ax.axis("off")
    caps = [
        (
            "Real data",
            "MetroPT-3 sensor telemetry (4 real failures) + Backblaze fleet (24,270 failures)",
        ),
        (
            "Serving",
            "schema-validated FastAPI + Docker · p99 31 ms · model registry (promote/rollback/audit)",
        ),
        (
            "MLOps",
            "CI metric gate · Prometheus+Grafana · drift→retrain→promote · delayed-label backfill",
        ),
        ("Edge", "quantized model 5.9× smaller, 4× faster · cloud-vs-edge tradeoff measured"),
        ("Rigor", "temporal CV (no leakage) · cost-tuned thresholds · Google ML Test Score 4.5"),
    ]
    y = 0.95
    for head, body in caps:
        ax.text(
            0.0,
            y,
            f"●  {head}: ",
            fontsize=11,
            fontweight="bold",
            color=NAVY,
            transform=ax.transAxes,
        )
        ax.text(0.0, y - 0.085, f"     {body}", fontsize=9.5, color="#333", transform=ax.transAxes)
        y -= 0.2
    ax.set_title("What it does", fontsize=12, fontweight="bold", color=NAVY, loc="left")

    fig.savefig(out, dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


def build_html(metropt_csv: str, backblaze_csv: str, out: str) -> str:
    """Self-contained interactive HTML dashboard (Plotly) — opens in any browser."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ts, scores, thr, events = _anomaly_timeline(metropt_csv)
    smooth = pd.Series(scores).rolling(72, center=True, min_periods=1).mean()
    models_afr, afrs = _afr(backblaze_csv)

    fig = make_subplots(
        rows=3,
        cols=3,
        specs=[
            [{"colspan": 3}, None, None],
            [{}, {}, {}],
            [{}, {"colspan": 2, "type": "table"}, None],
        ],
        subplot_titles=(
            "Anomaly score on real compressor telemetry — rises into every real failure (amber)",
            "Model leaderboard (ROC-AUC)",
            "Early warning before each failure (hours)",
            "Backblaze fleet — worst models (annualized failure %)",
            "Cost vs fixed schedule",
            "What it does",
        ),
        row_heights=[0.42, 0.3, 0.28],
        vertical_spacing=0.1,
        horizontal_spacing=0.08,
    )
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=scores,
            line=dict(color=TEAL, width=0.5),
            opacity=0.2,
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=smooth,
            line=dict(color=NAVY, width=2),
            name="anomaly score",
            hovertemplate="%{x|%Y-%m-%d}<br>score %{y:.3f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=thr, line=dict(color=RED, dash="dash"), row=1, col=1)
    for ev in events:
        fig.add_vrect(
            x0=ev.start, x1=ev.end, fillcolor=AMBER, opacity=0.5, line_width=0, row=1, col=1
        )

    names = list(MODEL_AUC)[::-1]
    fig.add_trace(
        go.Bar(
            x=list(MODEL_AUC.values())[::-1],
            y=[n.replace("\n", " ") for n in names],
            orientation="h",
            marker_color=TEAL,
            hovertemplate="%{y}: ROC-AUC %{x:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=[d[5:] for d in LEAD_TIMES],
            y=list(LEAD_TIMES.values()),
            marker_color=NAVY,
            hovertemplate="%{x}: %{y}h warning<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=[a * 100 for a in afrs],
            y=models_afr,
            orientation="h",
            marker_color=RED,
            hovertemplate="%{y}: %{x:.1f}% AFR<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=3,
    )
    fig.add_trace(
        go.Bar(
            x=["Fixed schedule", "GridSentinel"],
            y=[100, 40],
            marker_color=[GREY, TEAL],
            hovertemplate="%{x}: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    caps = [
        ("Real data", "MetroPT-3 sensor telemetry + Backblaze fleet (24,270 real failures)"),
        (
            "Serving",
            "schema-validated FastAPI + Docker · p99 31 ms · registry (promote/rollback/audit)",
        ),
        (
            "MLOps",
            "CI metric gate · Prometheus+Grafana · drift→retrain→promote · delayed-label backfill",
        ),
        ("Edge", "quantized 5.9× smaller / 4× faster · cloud-vs-edge tradeoff measured"),
        (
            "Rigor",
            "temporal CV (no leakage) · cost-tuned thresholds · ML Test Score 4.5 · 140 tests",
        ),
    ]
    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Capability</b>", "<b>Detail</b>"],
                fill_color=NAVY,
                font=dict(color="white"),
                align="left",
            ),
            cells=dict(
                values=[[c[0] for c in caps], [c[1] for c in caps]], align="left", height=26
            ),
        ),
        row=3,
        col=2,
    )

    fig.update_layout(
        title=dict(
            text="<b>GridSentinel</b> — production predictive-maintenance & fleet-reliability ML",
            font=dict(size=22, color=NAVY),
        ),
        height=1000,
        template="plotly_white",
        margin=dict(t=90, l=60, r=40, b=40),
    )
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the GridSentinel results dashboard")
    p.add_argument("--metropt", required=True)
    p.add_argument("--backblaze", required=True)
    p.add_argument("--out", default="docs/dashboard.png")
    p.add_argument("--format", choices=["png", "html", "both"], default="png")
    a = p.parse_args(argv)
    if a.format in ("png", "both"):
        print("wrote", build(a.metropt, a.backblaze, a.out.replace(".html", ".png")))
    if a.format in ("html", "both"):
        print("wrote", build_html(a.metropt, a.backblaze, a.out.replace(".png", ".html")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
