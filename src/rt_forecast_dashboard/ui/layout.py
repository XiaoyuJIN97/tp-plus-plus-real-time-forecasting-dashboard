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
    online_win_rate_by_zone,
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


def _render_task_title(target: str) -> None:
    st.markdown(f'<div class="task-section-title">{_target_label(target)}</div>', unsafe_allow_html=True)


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

        issue_history = pd.DataFrame()
        if not issues.empty:
            issue_history = issues.rename(
                columns={
                    "logged_at": "Recorded at",
                    "run_date": "Run date",
                    "zone": "Zone",
                    "target": "Task",
                    "stage": "Stage",
                    "message": "Message",
                    "status": "Status",
                }
            )
            issue_history["Type"] = "Failure"
            issue_history["Rows"] = pd.NA
            issue_history["Seconds"] = pd.NA
            issue_history["Report"] = pd.NA

        backfill_history = pd.DataFrame()
        if not backfills.empty:
            backfill_history = backfills.rename(
                columns={
                    "checked_at": "Recorded at",
                    "run_date": "Run date",
                    "rows": "Rows",
                    "seconds": "Seconds",
                    "message": "Message",
                    "status": "Status",
                    "report": "Report",
                }
            )
            backfill_history["Type"] = "Backfill"
            backfill_history["Zone"] = pd.NA
            backfill_history["Task"] = pd.NA
            backfill_history["Stage"] = "backfill"

        history = pd.concat([issue_history, backfill_history], ignore_index=True)
        if history.empty:
            st.success("No failures or backfills have been recorded.")
            return
        history["Recorded at"] = pd.to_datetime(history["Recorded at"], utc=True, errors="coerce")
        history = history.sort_values("Recorded at", ascending=False).head(60)
        columns = ["Recorded at", "Type", "Run date", "Zone", "Task", "Stage", "Rows", "Seconds", "Status", "Message", "Report"]
        st.dataframe(history[[column for column in columns if column in history]], width="stretch", hide_index=True)


def _render_rmae_landing(frame: pd.DataFrame) -> None:
    detail, _ = online_rmae_leaderboard(frame)
    win_rate_detail = online_win_rate_by_zone(frame)
    st.markdown('<div class="task-section-title">Leaderboard</div>', unsafe_allow_html=True)
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
    target_win_rates = win_rate_detail[win_rate_detail["target"].eq(selected_target)].copy()
    win_rate = (
        target_win_rates.pivot(index="display_family", columns="zone", values="win_rate")
        .reindex(index=[family for family in family_order if family != "TSO forecast"], columns=zone_order)
        .reset_index()
        .rename(columns={"display_family": "Model Family"})
    )

    rmae_col, wins_col = st.columns([1.55, 1.0])
    with rmae_col:
        st.markdown("#### rMAE by bidding zone")
        st.caption("rMAE = model MAE / TSO forecast MAE over the same fully realized runs.")
        st.dataframe(
            rmae,
            width="stretch",
            hide_index=True,
            column_config={zone: st.column_config.NumberColumn(format="%.2f") for zone in zone_order},
        )
    with wins_col:
        st.markdown("#### Win rate vs TSO")
        st.caption(
            "Win rate = lower-error runs + ½ tied runs; above 50% beats TSO on average."
        )
        st.dataframe(
            win_rate,
            width="stretch",
            hide_index=True,
            column_config={
                zone: st.column_config.ProgressColumn(
                    format="%.0f%%",
                    min_value=0.0,
                    max_value=100.0,
                )
                for zone in zone_order
            },
        )


def _render_header() -> None:
    _inject_styles()
    st.title("Transparency++")
    st.subheader("Real-Time Load and Renewables Forecasting")


def _load_dashboard() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    store = ForecastStore()
    data_version = _data_version(store)
    issues, backfills = _load_auxiliary_data(data_version)
    prepared = _prepared_online_forecasts(data_version)
    if "zone" in prepared.columns:
        prepared = prepared[prepared["zone"].isin(zones())].copy()
    return prepared, issues, backfills


def _render_data_warning(prepared: pd.DataFrame) -> None:
    actual_errors = prepared.attrs.get("actual_fetch_errors", [])
    artifact_age = prepared.attrs.get("actual_artifact_age_minutes")
    if actual_errors and not prepared.empty and prepared["timestamp"].lt(pd.Timestamp.now(tz="UTC")).any():
        st.warning("ENTSO-E realized actuals were not loaded: " + "; ".join(actual_errors[:3]))
    elif artifact_age is not None and artifact_age > 60:
        st.warning(f"ENTSO-E comparison data is delayed: the static actuals snapshot is {artifact_age:.0f} minutes old.")


def _render_leaderboard_page(details_page: st.Page) -> None:
    _render_header()
    prepared, _, _ = _load_dashboard()
    if prepared.empty:
        st.info("No stored forecasts yet. The scheduled daily forecast has not populated the dashboard data store.")
        return
    _render_data_warning(prepared)
    _render_rmae_landing(prepared)
    st.page_link(details_page, label="Explore forecast details", icon="📈")


def _render_forecast_details_page() -> None:
    _render_header()
    prepared, issues, backfills = _load_dashboard()
    if prepared.empty:
        st.info("No stored forecasts yet. The scheduled daily forecast has not populated the dashboard data store.")
        return
    _render_data_warning(prepared)
    st.markdown('<div class="task-section-title">Explore forecast details</div>', unsafe_allow_html=True)
    selected_target = st.selectbox("Forecast target", TARGET_ORDER, format_func=_target_label, index=0)
    available_zones = sorted(prepared.loc[prepared["target"].eq(selected_target), "zone"].dropna().unique())
    _render_task_title(selected_target)
    _render_target_section(selected_target, prepared, available_zones)
    _render_failure_backfill_history(issues, backfills)


def render_app() -> None:
    st.set_page_config(page_title="Transparency++", page_icon="chart_with_upwards_trend", layout="wide")
    details_page = st.Page(_render_forecast_details_page, title="Explore forecast details", icon="📈")
    leaderboard_page = st.Page(
        lambda: _render_leaderboard_page(details_page),
        title="Leaderboard",
        icon="🏆",
        default=True,
    )
    navigation = st.navigation([leaderboard_page, details_page])
    navigation.run()
