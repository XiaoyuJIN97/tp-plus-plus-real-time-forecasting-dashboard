from __future__ import annotations

import pandas as pd
import streamlit as st

from rt_forecast_dashboard.config import features, zones
from rt_forecast_dashboard.storage import ForecastStore
from rt_forecast_dashboard.ui.actuals import attach_display_actuals
from rt_forecast_dashboard.ui.analytics import (
    MODEL_FAMILY_ORDER,
    filter_last_n_days,
    online_forecast_accuracy,
    online_rmae_leaderboard,
    valid_online_forecasts,
)
from rt_forecast_dashboard.ui.charts import (
    accuracy_summary_chart,
    deterministic_forecast_chart,
)


TARGET_ORDER = ["load", "solar", "wind_onshore", "wind_offshore"]
HIDDEN_MODEL_KEYS = {"tabpfn_online"}


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .task-section-title {
            font-size: 2.1rem;
            font-weight: 800;
            line-height: 1.2;
            margin: 1.75rem 0 0.35rem 0;
            color: #111827;
        }
        .ops-section-title {
            font-size: 1.15rem;
            font-weight: 650;
            line-height: 1.25;
            margin: 3.25rem 0 0.45rem 0;
            color: #64748b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _data_version(store: ForecastStore) -> tuple[int, int]:
    path = store.dashboard_forecast_path()
    if not path.exists():
        return (0, 0)
    stat = path.stat()
    return (stat.st_mtime_ns, stat.st_size)


@st.cache_data(show_spinner=False)
def _load_auxiliary_data(version: tuple[int, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    store = ForecastStore()
    return store.read_issues(), store.read_backfill_history()


@st.cache_data(ttl=3600, show_spinner=False)
def _prepared_online_forecasts(version: tuple[int, int]) -> pd.DataFrame:
    prepared = valid_online_forecasts(ForecastStore().read_dashboard_forecasts())
    if "model" in prepared.columns:
        prepared = prepared[~prepared["model"].isin(HIDDEN_MODEL_KEYS)].copy()
    return attach_display_actuals(prepared)


def _target_label(target: str) -> str:
    return features()[target].get("label", target)


def _task_expander_label(target: str) -> str:
    return f"Open / close {_target_label(target)} display"


def _render_task_title(target: str) -> None:
    st.markdown(f'<div class="task-section-title">{_target_label(target)}</div>', unsafe_allow_html=True)


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
        ["Deterministic forecast analysis", "Model accuracy summary"],
        horizontal=True,
        key=f"{target}_view",
    )
    control_cols = st.columns([1.0, 1.0])
    selected_zone = control_cols[0].selectbox("Displayed zone", available_zones, key=f"{target}_displayed_zone")
    available_run_days = max(1, current["run_date"].nunique())
    last_n_days = control_cols[1].slider(
        "Plot last N days",
        1,
        60,
        min(14, available_run_days),
        key=f"{target}_online_last_n",
    )

    current = current[current["zone"].eq(selected_zone)].copy()
    current = filter_last_n_days(current, last_n_days)
    if current.empty:
        st.warning("No forecast rows match the selected zone and time window.")
        return

    if actual_errors and current["timestamp"].lt(pd.Timestamp.now(tz="UTC")).any():
        st.warning("Actual line unavailable: " + "; ".join(actual_errors[:3]))

    if view == "Deterministic forecast analysis":
        st.plotly_chart(
            deterministic_forecast_chart(current, f"{_target_label(target)} deterministic forecast analysis"),
            width="stretch",
        )
    elif view == "Model accuracy summary":
        _render_accuracy_section(target, current)


def _render_accuracy_section(target: str, forecasts: pd.DataFrame) -> None:
    accuracy = online_forecast_accuracy(forecasts)
    st.subheader(f"{_target_label(target)} model accuracy summary")
    if accuracy.empty:
        st.info("No realized actual values are attached to this target in the selected window yet.")
        return
    metrics = ["MAE", "RMSE", "R2"] if target == "load" else ["MAE", "R2"]
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
    display_cols = ["country", "display_family", "display_model", "case", "MAE", "RMSE", "R2", "n"]
    table = accuracy.sort_values(["country", "family_rank"])
    st.dataframe(
        table[[c for c in display_cols if c in table.columns]],
        width="stretch",
        hide_index=True,
    )


def _render_failure_backfill_history(issues: pd.DataFrame, backfills: pd.DataFrame) -> None:
    st.markdown('<div class="ops-section-title">Failure and backfill history</div>', unsafe_allow_html=True)
    with st.expander("Open / close records", expanded=False):
        metric_cols = st.columns(4)
        issue_status = issues["status"].astype(str) if "status" in issues.columns else pd.Series("", index=issues.index)
        backfill_status = backfills["status"].astype(str) if "status" in backfills.columns else pd.Series("", index=backfills.index)
        open_issues = issues[issue_status.eq("open")] if not issues.empty else pd.DataFrame()
        failed_backfills = backfills[backfill_status.ne("ok")] if not backfills.empty else pd.DataFrame()
        metric_cols[0].metric("Logged issues", len(issues))
        metric_cols[1].metric("Open issues", len(open_issues))
        metric_cols[2].metric("Backfill rows", len(backfills))
        metric_cols[3].metric("Failed backfills", len(failed_backfills))

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


def _render_rmae_landing(frame: pd.DataFrame) -> None:
    detail, _ = online_rmae_leaderboard(frame)
    st.markdown('<div class="task-section-title">Leaderboard</div>', unsafe_allow_html=True)
    st.caption(
        "rMAE = model MAE / TSO forecast MAE over the same fully realized runs."
    )
    if detail.empty:
        st.info("The leaderboard will appear after complete realized runs are available for every compared model.")
        return
    selected_target = st.selectbox(
        "Forecast task",
        TARGET_ORDER,
        index=0,
        format_func=_target_label,
        key="leaderboard_target",
    )
    target_detail = detail[detail["target"].eq(selected_target)].copy()
    if target_detail.empty:
        st.info("No complete comparisons are available for this forecasting task yet.")
        return

    family_order = [family for family in MODEL_FAMILY_ORDER if family in set(target_detail["display_family"])]
    zone_order = sorted(target_detail["zone"].unique())
    rmae = (
        target_detail.pivot(index="display_family", columns="zone", values="rMAE")
        .reindex(index=family_order, columns=zone_order)
        .reset_index()
        .rename(columns={"display_family": "Model Family"})
    )
    comparisons = target_detail[~target_detail["display_family"].eq("TSO forecast")].copy()
    win_rate = (
        comparisons.assign(
            win_score=comparisons["rMAE"].lt(1.0).astype(float)
            + 0.5 * comparisons["rMAE"].eq(1.0).astype(float)
        )
        .groupby("display_family", as_index=False)
        .agg(WinScore=("win_score", "sum"), Compared=("zone", "nunique"))
        .rename(columns={"display_family": "Model Family"})
    )
    win_rate["Win rate vs TSO"] = 100 * win_rate["WinScore"] / win_rate["Compared"]
    win_rate["family_rank"] = win_rate["Model Family"].map(
        {family: rank for rank, family in enumerate(MODEL_FAMILY_ORDER)}
    )
    win_rate = win_rate.sort_values("family_rank").drop(columns="family_rank")

    rmae_col, wins_col = st.columns([1.55, 1.0])
    with rmae_col:
        st.markdown("#### rMAE by bidding zone")
        st.dataframe(
            rmae,
            width="stretch",
            hide_index=True,
            column_config={zone: st.column_config.NumberColumn(format="%.2f") for zone in zone_order},
        )
    with wins_col:
        st.markdown("#### Win rate vs TSO")
        st.caption(
            "Percentage of bidding-zone tasks where the model has lower error than TSO; ties count as half-wins. "
            "Above 50% means the model is more accurate than TSO on average."
        )
        st.dataframe(
            win_rate[["Model Family", "Compared", "Win rate vs TSO"]],
            width="stretch",
            hide_index=True,
            column_config={
                "Win rate vs TSO": st.column_config.ProgressColumn(
                    format="%.0f%%",
                    min_value=0.0,
                    max_value=100.0,
                )
            },
        )


def render_app() -> None:
    st.set_page_config(page_title="Transparency++", page_icon="chart_with_upwards_trend", layout="wide")
    _inject_styles()
    st.title("Transparency++")
    st.subheader("Real-Time Load and Renewables Forecasting")

    store = ForecastStore()
    data_version = _data_version(store)
    issues, backfills = _load_auxiliary_data(data_version)
    prepared = _prepared_online_forecasts(data_version)
    if "zone" in prepared.columns:
        prepared = prepared[prepared["zone"].isin(zones())].copy()
    if prepared.empty:
        st.info("No stored forecasts yet. The scheduled daily forecast has not populated the dashboard data store.")
        return

    actual_errors = prepared.attrs.get("actual_fetch_errors", [])
    filtered = prepared.copy()
    artifact_age = prepared.attrs.get("actual_artifact_age_minutes")
    if actual_errors and not filtered.empty and filtered["timestamp"].lt(pd.Timestamp.now(tz="UTC")).any():
        st.warning("ENTSO-E realized actuals were not loaded: " + "; ".join(actual_errors[:3]))
    elif artifact_age is not None and artifact_age > 60:
        st.warning(
            f"ENTSO-E comparison data is delayed: the static actuals snapshot is {artifact_age:.0f} minutes old."
        )
    _render_rmae_landing(filtered)
    st.markdown('<div class="task-section-title">Explore forecast details</div>', unsafe_allow_html=True)
    selected_target = st.selectbox(
        "Forecast target",
        TARGET_ORDER,
        format_func=_target_label,
        index=0,
    )
    available_zones = sorted(filtered.loc[filtered["target"].eq(selected_target), "zone"].dropna().unique())
    _render_task_title(selected_target)
    _render_target_section(selected_target, filtered, available_zones)

    _render_timeline_and_inputs(filtered)

    _render_failure_backfill_history(issues, backfills)
