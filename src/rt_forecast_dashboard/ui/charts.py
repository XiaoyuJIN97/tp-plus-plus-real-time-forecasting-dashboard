from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from rt_forecast_dashboard.ui.analytics import MODEL_FAMILY_ORDER


def _add_brussels_delivery_time(frame: pd.DataFrame) -> pd.DataFrame:
    plot = frame.copy()
    plot["delivery_time_brussels"] = (
        pd.to_datetime(plot["timestamp"], utc=True)
        .dt.tz_convert("Europe/Brussels")
        .dt.tz_localize(None)
    )
    return plot


def forecast_line_chart(frame: pd.DataFrame) -> go.Figure:
    plot = _add_brussels_delivery_time(frame).sort_values(["target", "zone", "model_label", "run_date", "timestamp"])
    plot["line_id"] = plot["zone"].astype(str) + "|" + plot["target"].astype(str) + "|" + plot["model_label"].astype(str)
    fig = px.line(
        plot,
        x="delivery_time_brussels",
        y="forecast_mw",
        color="model_label",
        line_dash="target",
        line_group="line_id",
        facet_row="target",
        labels={"forecast_mw": "Forecast (MW)", "delivery_time_brussels": "Delivery time (Europe/Brussels)", "model_label": "Model"},
        height=680,
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=105),
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5),
        hovermode="x unified",
    )
    fig.update_yaxes(matches=None)
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dash", spikecolor="#111827", spikethickness=1)
    fig.update_traces(connectgaps=True)
    return fig


def zone_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby(["zone", "target", "model_label"], as_index=False)
        .agg(avg_forecast_mw=("forecast_mw", "mean"), peak_forecast_mw=("forecast_mw", "max"), first_delivery=("timestamp", "min"))
        .sort_values(["zone", "target", "model_label"])
    )


def deterministic_forecast_chart(frame: pd.DataFrame, title: str) -> go.Figure:
    plot = _add_brussels_delivery_time(frame).sort_values(["zone", "model_label", "run_date", "timestamp"])
    if "actual_mw" in plot.columns and plot["actual_mw"].notna().any():
        forecast = plot[["zone", "run_date", "timestamp", "delivery_time_brussels", "model_label", "forecast_mw"]].rename(columns={"forecast_mw": "value"})
        forecast["series"] = forecast["model_label"]
        actual = (
            plot[["zone", "timestamp", "delivery_time_brussels", "actual_mw"]]
            .dropna()
            .drop_duplicates(["zone", "timestamp"])
            .rename(columns={"actual_mw": "value"})
        )
        actual["run_date"] = "actual"
        actual["series"] = "Actual"
        plot_long = pd.concat(
            [
                forecast[["zone", "run_date", "timestamp", "delivery_time_brussels", "series", "value"]],
                actual[["zone", "run_date", "timestamp", "delivery_time_brussels", "series", "value"]],
            ],
            ignore_index=True,
        )
    else:
        plot_long = plot.rename(columns={"forecast_mw": "value"}).copy()
        plot_long["series"] = plot_long["model_label"]
    plot_long["line_id"] = plot_long["zone"].astype(str) + "|" + plot_long["series"].astype(str)
    fig = px.line(
        plot_long,
        x="delivery_time_brussels",
        y="value",
        color="series",
        line_group="line_id",
        facet_row="zone",
        labels={"value": "MW", "delivery_time_brussels": "Delivery time (Europe/Brussels)", "series": "Model"},
        title=title,
        height=max(420, 260 * max(1, plot_long["zone"].nunique())),
    )
    fig.update_yaxes(matches=None)
    fig.update_layout(
        margin=dict(l=20, r=20, t=60, b=115),
        legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="center", x=0.5),
        hovermode="x unified",
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dash", spikecolor="#111827", spikethickness=1)
    fig.update_traces(connectgaps=True)
    return fig


def accuracy_summary_chart(frame: pd.DataFrame, metric: str = "RMSE", title: str | None = None, show_legend: bool = True) -> go.Figure:
    plot = frame.dropna(subset=[metric]).copy()
    x_col = "display_family" if "display_family" in plot.columns else "display_model"
    color_col = "display_family" if "display_family" in plot.columns else "base_model"
    present_order = [family for family in MODEL_FAMILY_ORDER if family in set(plot[x_col].dropna())]
    present_color_order = [family for family in MODEL_FAMILY_ORDER if family in set(plot[color_col].dropna())]
    fig = px.bar(
        plot,
        x=x_col,
        y=metric,
        color=color_col,
        facet_col="country",
        category_orders={x_col: present_order, color_col: present_color_order},
        labels={x_col: "Model", metric: metric, color_col: "Family"},
        title=title or metric,
        height=420,
    )
    fig.update_xaxes(tickangle=35)
    fig.update_layout(
        margin=dict(l=20, r=20, t=60, b=120),
        showlegend=show_legend,
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
    )
    return fig


def scatter_diagnostics_chart(frame: pd.DataFrame, title: str) -> go.Figure:
    plot = frame.dropna(subset=["actual_mw", "forecast_mw"]).copy()
    fig = px.scatter(
        plot,
        x="actual_mw",
        y="forecast_mw",
        color="model_label",
        facet_col="zone",
        opacity=0.62,
        labels={"actual_mw": "Actual (MW)", "forecast_mw": "Forecast (MW)", "model_label": "Model"},
        title=title,
        height=460,
    )
    if not plot.empty:
        lo = float(min(plot["actual_mw"].min(), plot["forecast_mw"].min()))
        hi = float(max(plot["actual_mw"].max(), plot["forecast_mw"].max()))
        for idx in range(plot["zone"].nunique()):
            axis_suffix = "" if idx == 0 else str(idx + 1)
            fig.add_shape(
                type="line",
                x0=lo,
                y0=lo,
                x1=hi,
                y1=hi,
                xref=f"x{axis_suffix}",
                yref=f"y{axis_suffix}",
                line=dict(color="#1f2937", dash="dash", width=1),
            )
        fig.update_xaxes(range=[lo, hi])
        fig.update_yaxes(range=[lo, hi], scaleanchor=None)
    fig.update_layout(
        margin=dict(l=20, r=20, t=60, b=105),
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5),
        hovermode="closest",
    )
    return fig
