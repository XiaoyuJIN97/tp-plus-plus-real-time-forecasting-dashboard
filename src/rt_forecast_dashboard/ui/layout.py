from __future__ import annotations

import pandas as pd
import streamlit as st

from rt_forecast_dashboard.config import features, zones
from rt_forecast_dashboard.storage import ForecastStore
from rt_forecast_dashboard.ui.actuals import attach_display_actuals
from rt_forecast_dashboard.ui.analytics import (
    filter_last_n_days,
    online_forecast_accuracy,
    valid_online_forecasts,
)
from rt_forecast_dashboard.ui.charts import accuracy_summary_chart, deterministic_forecast_chart, scatter_diagnostics_chart


TARGET_ORDER = ["load", "solar", "wind_onshore", "wind_offshore"]


@st.cache_data(ttl=300, show_spinner=False)
def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    store = ForecastStore()
    return store.read_forecasts(), store.read_issues()


@st.cache_data(ttl=300, show_spinner=False)
def _prepared_online_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    return attach_display_actuals(valid_online_forecasts(forecasts))


def _target_label(target: str) -> str:
    return features()[target].get("label", target)


def _render_target_section(target: str, prepared: pd.DataFrame, countries: list[str]) -> None:
    if not countries:
        st.info("Select at least one bidding zone.")
        return
    st.header(_target_label(target))
    actual_errors = prepared.attrs.get("actual_fetch_errors", [])
    current = prepared[(prepared["target"] == target) & (prepared["zone"].isin(countries))].copy()

    if current.empty:
        st.info("No stored online forecast rows for this target yet.")
        return
    available_zones = sorted(current["zone"].dropna().unique())

    view = st.radio(
        "View",
        ["Deterministic forecast analysis", "Scatter diagnostics", "Model accuracy summary", "Run history"],
        horizontal=True,
        key=f"{target}_view",
    )
    control_cols = st.columns([1.2, 1.0, 1.0])
    model_options = sorted(current["model_label"].dropna().unique())
    selected_models = control_cols[0].multiselect(
        "Model family",
        model_options,
        default=model_options,
        key=f"{target}_online_models",
    )
    last_n_days = control_cols[1].slider("Plot last N days", 1, 60, min(14, max(1, current["run_date"].nunique())), key=f"{target}_online_last_n")
    selected_zone = control_cols[2].selectbox("Displayed zone", available_zones, key=f"{target}_displayed_zone")

    if not selected_models:
        st.warning("Select at least one model.")
        return

    current = current[current["model_label"].isin(selected_models) & current["zone"].eq(selected_zone)].copy()
    current = filter_last_n_days(current, last_n_days)
    if current.empty:
        st.warning("No forecast rows match the selected models and time window.")
        return

    meta_cols = st.columns(4)
    meta_cols[0].metric("Latest run", current["run_date"].max())
    latest_primary = current.copy()
    if not latest_primary.empty:
        latest_primary = latest_primary[latest_primary["run_date"].eq(latest_primary["run_date"].max())]
    horizon = latest_primary["horizon"].nunique() if not latest_primary.empty else current["horizon"].nunique()
    meta_cols[1].metric("Day-ahead horizon", f"{int(horizon)} hours")
    meta_cols[2].metric("Context", f"{int(current['context_hours'].max()):,} hours")
    meta_cols[3].metric("Models", current["model_label"].nunique())
    if actual_errors and current["timestamp"].lt(pd.Timestamp.now(tz="UTC")).any():
        st.warning("Actual line unavailable: " + "; ".join(actual_errors[:3]))

    if view == "Deterministic forecast analysis":
        st.plotly_chart(
            deterministic_forecast_chart(current, f"{_target_label(target)} deterministic forecast analysis"),
            width="stretch",
        )
        detail_cols = [
            "zone",
            "model_label",
            "covariate_case",
            "context_hours",
            "context_start",
            "context_end",
            "timestamp",
            "horizon",
            "forecast_mw",
            "actual_mw",
            "tso_forecast_mw",
        ]
        details = current[[c for c in detail_cols if c in current.columns]].sort_values(["zone", "model_label", "timestamp"])
        st.dataframe(details, width="stretch", hide_index=True)
    elif view == "Scatter diagnostics":
        _render_scatter_diagnostics(target, current)
    elif view == "Model accuracy summary":
        _render_accuracy_section(target, current)
    else:
        _render_run_history(current)


def _render_accuracy_section(target: str, forecasts: pd.DataFrame) -> None:
    accuracy = online_forecast_accuracy(forecasts)
    st.subheader(f"{_target_label(target)} model accuracy summary")
    if accuracy.empty:
        st.info("No realized actual values are attached to this target in the selected window yet.")
        return
    metrics = ["MAE", "RMSE", "MAPE", "R2"] if target == "load" else ["MAE", "R2"]
    available_metrics = [metric for metric in metrics if metric in accuracy.columns and accuracy[metric].notna().any()]
    if not available_metrics:
        st.info("No accuracy metrics are available for the selected rows yet.")
        return
    chart_cols = st.columns(2)
    for idx, metric in enumerate(available_metrics):
        with chart_cols[idx % 2]:
            st.plotly_chart(
                accuracy_summary_chart(accuracy, metric, title=metric, show_legend=idx == 0),
                width="stretch",
            )
    display_cols = ["country", "display_family", "display_model", "case", "MAE", "RMSE", "MAPE", "R2", "n"]
    table = accuracy.sort_values(["country", "family_rank"])
    st.dataframe(
        table[[c for c in display_cols if c in table.columns]],
        width="stretch",
        hide_index=True,
    )


def _render_scatter_diagnostics(target: str, forecasts: pd.DataFrame) -> None:
    st.subheader("Scatter diagnostics")
    current = forecasts.dropna(subset=["actual_mw", "forecast_mw"]).copy()
    if current.empty:
        st.info("No realized actual values are attached to this target in the selected window yet.")
        return

    st.plotly_chart(scatter_diagnostics_chart(current, f"{_target_label(target)} actual vs forecast scatter"), width="stretch")


def _render_run_history(forecasts: pd.DataFrame) -> None:
    st.subheader("Run history")
    history = (
        forecasts.groupby(["run_date", "zone", "target", "model_label", "covariate_case", "context_hours"], as_index=False)
        .agg(rows=("forecast_mw", "size"), run_at=("run_at", "max"))
        .sort_values(["run_date", "zone", "target"], ascending=[False, True, True])
    )
    st.dataframe(history, width="stretch", hide_index=True)


def render_app() -> None:
    st.set_page_config(page_title="Real-Time Energy Forecasting", page_icon="chart_with_upwards_trend", layout="wide")
    st.title("Real-Time Load and Renewables Forecasting")
    st.caption("Daily 18:00 Europe/Brussels forecasts with latest 3-month context, selected 4-point weather covariates, and TSO forecast inputs.")

    forecasts, issues = _load_data()
    if forecasts.empty:
        st.info("No stored forecasts yet. The scheduled daily forecast has not populated the dashboard data store.")
        return

    forecasts["timestamp"] = pd.to_datetime(forecasts["timestamp"], utc=True)
    prepared = _prepared_online_forecasts(forecasts)
    actual_errors = prepared.attrs.get("actual_fetch_errors", [])
    target_config = features()
    all_zones = list(zones().keys())
    filter_cols = st.columns([1.2, 1.8])
    target_filter = filter_cols[0].multiselect(
        "Forecasting tasks",
        TARGET_ORDER,
        default=TARGET_ORDER,
        format_func=lambda value: target_config[value].get("label", value),
    )
    zone_filter = filter_cols[1].multiselect("Available bidding zones", all_zones, default=all_zones)

    filtered = prepared.copy()
    filtered = filtered[filtered["zone"].isin(zone_filter) & filtered["target"].isin(target_filter)].copy()
    latest_run = filtered["run_date"].max() if not filtered.empty else None

    metric_cols = st.columns(4)
    metric_cols[0].metric("Latest run", latest_run or "n/a")
    metric_cols[1].metric("Forecast rows", f"{len(filtered):,}")
    metric_cols[2].metric("Zones", filtered["zone"].nunique() if not filtered.empty else 0)
    metric_cols[3].metric("Actual rows", int(filtered["actual_mw"].notna().sum()) if "actual_mw" in filtered else 0)
    if actual_errors and not filtered.empty and filtered["timestamp"].lt(pd.Timestamp.now(tz="UTC")).any():
        st.warning("ENTSO-E realized actuals were not loaded: " + "; ".join(actual_errors[:3]))

    for target in TARGET_ORDER:
        if target in target_filter:
            _render_target_section(target, filtered, zone_filter)
            st.divider()

    with st.expander("Issues", expanded=False):
        if issues.empty:
            st.success("No issues have been logged.")
        else:
            st.dataframe(issues.sort_values("logged_at", ascending=False), width="stretch", hide_index=True)
