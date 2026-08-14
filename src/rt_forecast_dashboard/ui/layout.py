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
HIDDEN_MODEL_KEYS = {"tabpfn_online"}


@st.cache_data(ttl=300, show_spinner=False)
def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    store = ForecastStore()
    return store.read_forecasts(), store.read_issues(), store.read_backfill_history()


@st.cache_data(ttl=300, show_spinner=False)
def _prepared_online_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    prepared = valid_online_forecasts(forecasts)
    if "model" in prepared.columns:
        prepared = prepared[~prepared["model"].isin(HIDDEN_MODEL_KEYS)].copy()
    return attach_display_actuals(prepared)


def _target_label(target: str) -> str:
    return features()[target].get("label", target)


def _add_brussels_delivery_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    delivery = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert("Europe/Brussels")
    frame["delivery_time_brussels"] = delivery.dt.strftime("%Y-%m-%d %H:%M %Z")
    frame["delivery_hour_brussels"] = delivery.dt.strftime("%H:%M")
    return frame


def _format_brussels_timestamp(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return pd.Timestamp(value).tz_convert("Europe/Brussels").strftime("%Y-%m-%d %H:%M %Z")


def _actual_status_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "actual_mw" not in frame.columns:
        return pd.DataFrame()
    actual = frame.dropna(subset=["actual_mw"]).copy()
    if actual.empty:
        return pd.DataFrame()
    actual = actual.drop_duplicates(["zone", "target", "timestamp"])
    status = (
        actual.groupby(["zone", "target"], as_index=False)
        .agg(actual_through=("timestamp", "max"), actual_points=("timestamp", "size"))
        .sort_values(["zone", "target"])
    )
    status["Task"] = status["target"].map(_target_label)
    status["Actual through"] = status["actual_through"].map(_format_brussels_timestamp)
    status = status.rename(columns={"zone": "Zone", "actual_points": "Hourly actual points"})
    return status[["Zone", "Task", "Actual through", "Hourly actual points"]]


def _render_timeline_and_inputs(frame: pd.DataFrame) -> None:
    with st.expander("Daily update timeline and data inputs", expanded=False):
        st.markdown(
            """
            The forecast run is scheduled after the 18:00 Europe/Brussels publication point. Timestamps are stored in UTC, but dashboard plots use Europe/Brussels delivery time. A 24-hour run from 18:00 therefore has hourly delivery timestamps from 18:00 through 17:00; the 17:00 point is the 17:00-18:00 delivery hour.
            """
        )
        timeline = pd.DataFrame(
            [
                ("Before 18:00 Brussels", "Open-Meteo collector updates four-point weather forecast archive."),
                ("18:00 Brussels", "ENTSO-E TP TSO forecasts for the next delivery window should be available."),
                ("18:02-18:27 Brussels", "ENTSO-E realtime-data collector snapshots TP forecast and realized-value files."),
                ("18:30+ Brussels", "Dashboard workflow reads the latest ENTSO-E/Open-Meteo archives, runs forecasts, commits forecast CSVs."),
                ("After realization", "Dashboard fetches realized ENTSO-E actuals for completed delivery hours and updates diagnostics."),
            ],
            columns=["Time", "Step"],
        )
        st.dataframe(timeline, width="stretch", hide_index=True)
        actual_status = _actual_status_table(frame)
        st.markdown("#### Current actual data display")
        if actual_status.empty:
            st.info("No realized actual values are currently attached for the selected tasks and zones.")
        else:
            st.dataframe(actual_status, width="stretch", hide_index=True)
        inputs = pd.DataFrame(
            [
                ("ENTSO-E realtime-data", "load", "forecast_load + actual_load", "TSO covariate, TSO benchmark, realized load actuals"),
                ("ENTSO-E realtime-data", "solar", "forecast_solar_generation + actual_solar_generation", "TSO covariate, TSO benchmark, realized solar actuals"),
                ("ENTSO-E realtime-data", "onshore wind", "forecast_onshore_wind_generation + actual_onshore_wind_generation", "TSO covariate, TSO benchmark, realized onshore wind actuals"),
                ("ENTSO-E realtime-data", "offshore wind", "forecast_offshore_wind_generation + actual_offshore_wind_generation", "TSO covariate, TSO benchmark, realized offshore wind actuals"),
                ("Open-Meteo realtime-data", "load", "temperature_2m, relative_humidity_2m, shortwave_radiation at four selected points; degree proxy derived in dashboard", "Optimal load engineering covariates by country/model"),
                ("Open-Meteo realtime-data", "solar", "shortwave_radiation + temperature_2m at four selected points", "Solar weather covariates"),
                ("Open-Meteo realtime-data", "onshore/offshore wind", "wind_speed_100m_ms + wind_dir_sin + wind_dir_cos at four selected points", "Wind weather covariates"),
            ],
            columns=["Resource", "Task", "Included data", "Dashboard use"],
        )
        st.dataframe(inputs, width="stretch", hide_index=True)


def _render_target_section(target: str, prepared: pd.DataFrame, countries: list[str]) -> None:
    if not countries:
        st.info("Select at least one bidding zone.")
        return
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

    meta_cols = st.columns(5)
    meta_cols[0].metric("Latest run", current["run_date"].max())
    latest_primary = current.copy()
    if not latest_primary.empty:
        latest_primary = latest_primary[latest_primary["run_date"].eq(latest_primary["run_date"].max())]
    horizon = latest_primary["horizon"].nunique() if not latest_primary.empty else current["horizon"].nunique()
    meta_cols[1].metric("Day-ahead horizon", f"{int(horizon)} hours")
    meta_cols[2].metric("Context", f"{int(current['context_hours'].max()):,} hours")
    latest_actual = current.loc[current["actual_mw"].notna(), "timestamp"].max() if "actual_mw" in current else None
    meta_cols[3].metric("Actual through", _format_brussels_timestamp(latest_actual))
    meta_cols[4].metric("Models", current["model_label"].nunique())
    if actual_errors and current["timestamp"].lt(pd.Timestamp.now(tz="UTC")).any():
        st.warning("Actual line unavailable: " + "; ".join(actual_errors[:3]))

    if view == "Deterministic forecast analysis":
        st.plotly_chart(
            deterministic_forecast_chart(current, f"{_target_label(target)} deterministic forecast analysis"),
            width="stretch",
        )
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


def _render_failure_backfill_history(issues: pd.DataFrame, backfills: pd.DataFrame) -> None:
    st.subheader("Failure and backfill history")
    metric_cols = st.columns(4)
    issue_status = issues["status"].astype(str) if "status" in issues.columns else pd.Series("", index=issues.index)
    backfill_status = backfills["status"].astype(str) if "status" in backfills.columns else pd.Series("", index=backfills.index)
    open_issues = issues[issue_status.eq("open")] if not issues.empty else pd.DataFrame()
    failed_backfills = backfills[backfill_status.ne("ok")] if not backfills.empty else pd.DataFrame()
    metric_cols[0].metric("Logged issues", len(issues))
    metric_cols[1].metric("Open issues", len(open_issues))
    metric_cols[2].metric("Backfill rows", len(backfills))
    metric_cols[3].metric("Failed backfills", len(failed_backfills))

    with st.expander("Recent failure / backfill records", expanded=False):
        issue_cols = ["logged_at", "run_date", "zone", "target", "stage", "message", "status"]
        st.markdown("#### Recent failures")
        if issues.empty:
            st.success("No failures have been logged.")
        else:
            recent_issues = issues.sort_values("logged_at", ascending=False).head(30).copy()
            st.dataframe(recent_issues[[c for c in issue_cols if c in recent_issues.columns]], width="stretch", hide_index=True)

        backfill_cols = ["report", "run_date", "rows", "seconds", "status", "message"]
        st.markdown("#### Backfill reports")
        if backfills.empty:
            st.info("No backfill reports have been stored yet.")
        else:
            recent_backfills = backfills.sort_values(["report", "run_date"], ascending=[False, False]).head(30).copy()
            st.dataframe(recent_backfills[[c for c in backfill_cols if c in recent_backfills.columns]], width="stretch", hide_index=True)


def render_app() -> None:
    st.set_page_config(page_title="Real-Time Energy Forecasting", page_icon="chart_with_upwards_trend", layout="wide")
    st.title("Real-Time Load and Renewables Forecasting")
    st.caption("Daily 18:00 Europe/Brussels forecasts with latest 3-month context, selected 4-point weather covariates, and TSO forecast inputs.")

    forecasts, issues, backfills = _load_data()
    if forecasts.empty:
        st.info("No stored forecasts yet. The scheduled daily forecast has not populated the dashboard data store.")
        return

    forecasts["timestamp"] = pd.to_datetime(forecasts["timestamp"], utc=True)
    prepared = _prepared_online_forecasts(forecasts)
    actual_errors = prepared.attrs.get("actual_fetch_errors", [])
    all_zones = list(zones().keys())
    filtered = prepared.copy()
    latest_run = filtered["run_date"].max() if not filtered.empty else None

    metric_cols = st.columns(4)
    metric_cols[0].metric("Latest run", latest_run or "n/a")
    metric_cols[1].metric("Forecast rows", f"{len(filtered):,}")
    metric_cols[2].metric("Zones", filtered["zone"].nunique() if not filtered.empty else 0)
    metric_cols[3].metric("Actual rows", int(filtered["actual_mw"].notna().sum()) if "actual_mw" in filtered else 0)
    if actual_errors and not filtered.empty and filtered["timestamp"].lt(pd.Timestamp.now(tz="UTC")).any():
        st.warning("ENTSO-E realized actuals were not loaded: " + "; ".join(actual_errors[:3]))
    _render_timeline_and_inputs(filtered)

    for target in TARGET_ORDER:
        with st.expander(_target_label(target), expanded=target == "load"):
            _render_target_section(target, filtered, all_zones)

    _render_failure_backfill_history(issues, backfills)
