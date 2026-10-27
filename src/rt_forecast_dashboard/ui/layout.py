from __future__ import annotations

import pandas as pd
import streamlit as st

from rt_forecast_dashboard.config import features, zones
from rt_forecast_dashboard.conformal import conformal_intervals, interval_leaderboard
from rt_forecast_dashboard.storage import ForecastStore
from rt_forecast_dashboard.ui.actuals import attach_display_actuals
from rt_forecast_dashboard.ui.analytics import (
    MODEL_FAMILY_ORDER,
    filter_last_n_days,
    online_rmae_leaderboard,
    online_win_rate_by_zone,
    valid_online_forecasts,
    _normal_model_family,
)
from rt_forecast_dashboard.ui.charts import (
    deterministic_forecast_chart,
    conformal_forecast_chart,
)


TARGET_ORDER = ["wind_offshore", "wind_onshore", "solar", "load"]
HIDDEN_MODEL_KEYS = {"tabpfn_online"}
FORECAST_VISUALIZATION_VIEW = "Forecast visualization and analysis"


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
        div[data-testid="stPageLink"] a {
            display: inline-flex;
            width: fit-content;
            align-items: center;
            gap: 0.45rem;
            margin-top: 1rem;
            padding: 0.2rem 0;
            border: 0;
            background: transparent;
            color: #2563eb;
            font-size: 1.3rem;
            font-weight: 700;
            text-decoration: none;
            box-shadow: none;
            transition: color 120ms ease, transform 120ms ease;
        }
        div[data-testid="stPageLink"] a:hover {
            background: transparent;
            color: #1e40af;
            text-decoration: underline;
            transform: translateX(2px);
        }
        div[data-testid="stPageLink"] a p {
            color: inherit;
            font-size: 1.55rem !important;
            font-weight: 700 !important;
            line-height: 1.25;
        }
        div[data-testid="stPageLink"] a span {
            font-size: 1.45rem !important;
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


@st.cache_data(max_entries=1, show_spinner=False)
def _load_auxiliary_data(version: tuple[int, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    store = ForecastStore()
    return store.read_issues(), store.read_backfill_history()


@st.cache_data(ttl=3600, max_entries=1, show_spinner=False)
def _prepared_online_forecasts(version: tuple[int, int]) -> pd.DataFrame:
    store = ForecastStore()
    path = store.dashboard_forecast_path()
    if path.exists():
        import pyarrow.parquet as pq
        needed = ["run_date", "zone", "target", "model", "model_label", "timestamp", "forecast_mw",
                  "actual_mw", "source", "context_hours", "context_end"]
        names = pq.read_schema(path).names
        forecasts = pd.read_parquet(path, columns=[c for c in needed if c in names])
    else:
        forecasts = store.read_dashboard_forecasts()
    prepared = valid_online_forecasts(forecasts)
    if "model" in prepared.columns:
        prepared = prepared[~prepared["model"].isin(HIDDEN_MODEL_KEYS)].copy()
    prepared = attach_display_actuals(prepared)
    keep = ["run_date", "zone", "target", "model", "model_label", "timestamp", "forecast_mw",
            "actual_mw", "source", "horizon", "context_hours", "context_end"]
    prepared = prepared[[c for c in keep if c in prepared]].copy()
    for column in ["run_date", "zone", "target", "model", "model_label", "source"]:
        if column in prepared:
            prepared[column] = prepared[column].astype("category")
    return prepared


@st.cache_data(ttl=3600, max_entries=1, show_spinner=False)
def _leaderboard_summary(version: tuple[int, int]) -> pd.DataFrame:
    """Load only coverage endpoints; the landing page uses precomputed scores."""
    path = ForecastStore().dashboard_forecast_path()
    if not path.exists():
        return _prepared_online_forecasts(version)
    frame = pd.read_parquet(path, columns=["target", "timestamp", "actual_mw", "run_date", "source"])
    return _coverage_endpoints(frame)


def _coverage_endpoints(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    realized = frame.dropna(subset=["actual_mw"])
    indices = [pd.to_datetime(frame["run_date"]).idxmax()]
    if not realized.empty:
        grouped = realized.groupby("target")["timestamp"]
        indices.extend(grouped.idxmin().tolist())
        indices.extend(grouped.idxmax().tolist())
    historical = frame[frame["source"].astype(str).str.startswith("historical_mock")]
    if not historical.empty:
        indices.append(historical.index[0])
    return frame.loc[list(dict.fromkeys(indices))].copy()


def _target_label(target: str) -> str:
    return features()[target].get("label", target)


def _render_task_title(target: str) -> None:
    st.markdown(f'<div class="task-section-title">{_target_label(target)}</div>', unsafe_allow_html=True)


def _analysis_window_sentence(frame: pd.DataFrame) -> str | None:
    if frame.empty or "timestamp" not in frame.columns or "actual_mw" not in frame.columns:
        return None
    realized = frame.dropna(subset=["actual_mw"]).copy()
    if realized.empty:
        return None
    timestamps = pd.to_datetime(realized["timestamp"], utc=True, errors="coerce").dropna()
    if timestamps.empty:
        return None
    start = timestamps.min().tz_convert("Europe/Brussels").date().isoformat()
    end = timestamps.max().tz_convert("Europe/Brussels").date().isoformat()
    return f"These results were compiled with data from ENTSO-E TP from {start} to {end}."


def _render_analysis_window(frame: pd.DataFrame) -> None:
    sentence = _analysis_window_sentence(frame)
    if sentence:
        st.caption(sentence)


def _render_target_section(target: str, prepared: pd.DataFrame, countries: list[str]) -> None:
    if not countries:
        st.info("Select at least one bidding zone.")
        return
    current = prepared[(prepared["target"] == target) & (prepared["zone"].isin(countries))].copy()

    if current.empty:
        st.info("No stored online forecast rows for this target yet.")
        return
    available_zones = sorted(current["zone"].dropna().unique())

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

    series = current[current["zone"].eq(selected_zone)].copy()
    current = filter_last_n_days(series, last_n_days)
    if current.empty:
        st.warning("No forecast rows match the selected zone and time window.")
        return

    point_tab, conformal_tab = st.tabs(["Point forecasts", "Probabilistic forecasts"])
    with point_tab:
        st.plotly_chart(
            deterministic_forecast_chart(current, f"{_target_label(target)} forecast visualization and analysis"),
            width="stretch",
        )
    with conformal_tab:
        _render_conformal_section(target, selected_zone, current)


@st.cache_data(max_entries=2, show_spinner=False)
def _conformal_series(version: tuple, target: str, zone: str) -> pd.DataFrame:
    path = ForecastStore().dashboard_dir() / "conformal_forecasts.parquet"
    if path.exists():
        return pd.read_parquet(path, filters=[("target", "=", target), ("zone", "=", zone)])
    # Calibration must see the full history, before the plotting window is cut.
    prepared = _prepared_online_forecasts(version[:2])
    selected = prepared[prepared.target.eq(target) & prepared.zone.eq(zone)]
    return conformal_intervals(selected)


def _render_conformal_section(target: str, zone: str, current: pd.DataFrame) -> None:
    labels = dict(zip(current.model.astype(str), current.model_label.astype(str)))
    def rank(key):
        family = _normal_model_family(labels[key])
        return (MODEL_FAMILY_ORDER.index(family) if family in MODEL_FAMILY_ORDER else len(MODEL_FAMILY_ORDER), labels[key])
    options = sorted(labels, key=rank)
    controls = st.columns([3, 1])
    defaults = [key for key in options if key in {'ensemble_selected','tso_reference'}]
    selected = controls[0].multiselect("Show model plots", options, default=defaults,
                                      format_func=lambda key: labels[key], key=f"{target}_{zone}_cp_models_v2")
    levels = [80]
    columns = controls[1].selectbox("Plots per row", [2, 1], key=f"{target}_{zone}_cp_columns")
    if not selected:
        st.info("Select one or more models to show their plots.")
        return
    store = ForecastStore()
    path = store.dashboard_dir() / "conformal_forecasts.parquet"
    stat = path.stat() if path.exists() else None
    version = (*_data_version(store), stat.st_mtime_ns if stat else 0, stat.st_size if stat else 0)
    with st.spinner("Loading conformal bands…"):
        bands = _conformal_series(version, target, zone)
    # Match the displayed timestamps without reducing the calibration history.
    bands = bands[bands.timestamp.between(current.timestamp.min(), current.timestamp.max()) & bands.model.isin(selected)].copy()
    for first in range(0, len(selected), columns):
        containers = st.columns(columns)
        for container, model in zip(containers, selected[first:first+columns]):
            with container:
                part = bands[bands.model.eq(model)]
                if part.empty:
                    st.info(f"{labels[model]}: no forecast rows in this window.")
                    continue
                st.plotly_chart(conformal_forecast_chart(part, levels, labels[model]), width="stretch",
                                key=f"{target}_{zone}_{model}_conformal")
                missing = part[f"lower_{min(levels)}"].isna().sum()
                if missing:
                    st.caption(f"{missing} forecast hours have insufficient prior calibration; the point forecast is shown without a band.")



@st.cache_data(max_entries=1, show_spinner=False)
def _fallback_daily_leaderboard(day: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    store = ForecastStore()
    frame = _prepared_online_forecasts(_data_version(store))
    frame = frame[frame.zone.isin(zones())]
    detail, _ = online_rmae_leaderboard(frame)
    return detail, online_win_rate_by_zone(frame)


@st.cache_data(max_entries=4, show_spinner=False)
def _probabilistic_leaderboard(version: tuple, target: str) -> pd.DataFrame:
    path = ForecastStore().dashboard_dir()/'conformal_forecasts.parquet'
    if path.exists():
        bands = pd.read_parquet(path,filters=[('target','=',target)])
    else:
        frame = _prepared_online_forecasts(version[:2])
        bands = conformal_intervals(frame[frame.target.eq(target)])
    bands = bands[bands.zone.isin(zones()) & ~bands.model.isin(HIDDEN_MODEL_KEYS)]
    return interval_leaderboard(bands, .80)


def _render_probabilistic_leaderboard(target: str) -> None:
    st.markdown('#### Probabilistic forecast leaderboard')
    store = ForecastStore()
    path = store.dashboard_dir()/'conformal_forecasts.parquet'
    stat = path.stat() if path.exists() else None
    version = (*_data_version(store),stat.st_mtime_ns if stat else 0,stat.st_size if stat else 0)
    with st.spinner('Calculating interval coverage and width…'):
        scores = _probabilistic_leaderboard(version,target)
    if scores.empty:
        st.info('Interval comparisons require shared, fully observed runs with calibrated intervals for every compared model.')
        return
    scores['Model Family'] = scores.model_label.map(_normal_model_family)
    families = list(dict.fromkeys([f for f in MODEL_FAMILY_ORDER if f in set(scores['Model Family'])]
                                 +sorted(set(scores['Model Family'])-set(MODEL_FAMILY_ORDER))))
    st.caption('80% intervals, evaluated on the same complete observed runs within each zone. Coverage target: 80%. Compare widths alongside coverage.')
    scores['_model_order'] = scores['Model Family'].map({family:index for index,family in enumerate(families)})
    table = scores.sort_values(['zone','_model_order'])[
        ['zone','Model Family','coverage_pct','mean_interval_width_mw']
    ].rename(columns={'zone':'Zone','coverage_pct':'Coverage (%)',
                      'mean_interval_width_mw':'Mean interval width (MW)'}).reset_index(drop=True)
    st.dataframe(table,width='stretch',hide_index=True,column_config={
        'Coverage (%)':st.column_config.NumberColumn(format='%.1f%%'),
        'Mean interval width (MW)':st.column_config.NumberColumn(format='%.2f'),
    })


def _render_rmae_landing(frame: pd.DataFrame) -> None:
    snapshot = ForecastStore().dashboard_dir() / "leaderboard_daily.parquet"
    if snapshot.exists():
        detail = pd.read_parquet(snapshot)
        win_rate_detail = detail
    else:
        detail, win_rate_detail = _fallback_daily_leaderboard(
            pd.Timestamp.now(tz="Europe/Brussels").date().isoformat()
        )
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
    mae = (
        target_detail.pivot(index="display_family", columns="zone", values="MAE")
        .reindex(index=family_order, columns=zone_order)
        .reset_index().rename(columns={"display_family": "Model Family"})
    )
    target_win_rates = win_rate_detail[win_rate_detail["target"].eq(selected_target)].copy()
    win_rate = (
        target_win_rates.pivot(index="display_family", columns="zone", values="win_rate")
        .reindex(index=[family for family in family_order if family != "TSO forecast"], columns=zone_order)
        .reset_index()
        .rename(columns={"display_family": "Model Family"})
    )

    st.markdown("#### rMAE vs TSO")
    st.caption("rMAE = model MAE / TSO forecast MAE over the same fully realized runs.")
    st.caption("A value below 1 means the model is more accurate than the TSO model on average. Lower is better.")
    st.dataframe(
        rmae,
        width="stretch",
        hide_index=True,
        column_config={zone: st.column_config.NumberColumn(format="%.2f") for zone in zone_order},
    )
    st.markdown("#### Win rate vs TSO")
    st.caption(
        "Win rate: Percentage of tasks where Model 1 achieves lower error than Model 2 (ties count as half-wins). A value above 50% means Model 1 is more accurate than Model 2 on average."
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
    st.markdown("#### MAE (MW)")
    st.caption("Mean absolute error on the same fully realized runs for all compared models. Lower is better.")
    st.dataframe(
        mae, width="stretch", hide_index=True,
        column_config={zone: st.column_config.NumberColumn(format="%.2f") for zone in zone_order},
    )
    _render_analysis_window(frame[frame["target"].eq(selected_target)])
    _render_probabilistic_leaderboard(selected_target)


def _render_header() -> None:
    _inject_styles()
    st.title("Transparency++")
    st.subheader("Day-ahead Load and Renewables Forecasting")


def _load_dashboard() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    store = ForecastStore()
    data_version = _data_version(store)
    issues, backfills = _load_auxiliary_data(data_version)
    prepared = _prepared_online_forecasts(data_version)
    if "zone" in prepared.columns:
        prepared = prepared[prepared["zone"].isin(zones())].copy()
    return prepared, issues, backfills


def _render_data_warning(prepared: pd.DataFrame) -> None:
    if prepared.empty:
        return
    actual_times = pd.to_datetime(
        prepared.loc[prepared["actual_mw"].notna(), "timestamp"], utc=True, errors="coerce"
    )
    if actual_times.empty or actual_times.dropna().empty:
        st.warning("Observed ENTSO-E outcomes are unavailable; accuracy results cannot advance yet.")
    elif pd.Timestamp.now(tz="UTC") - actual_times.max() > pd.Timedelta(days=2):
        latest_actual = actual_times.max().tz_convert("Europe/Brussels").date().isoformat()
        st.warning(
            f"ENTSO-E observed outcomes have only reached {latest_actual}. "
            "Newer forecasts are available, but their accuracy cannot be scored until outcomes arrive."
        )


def _render_leaderboard_page(details_page: st.Page) -> None:
    _render_header()
    prepared = _leaderboard_summary(_data_version(ForecastStore()))
    if prepared.empty:
        st.info("No stored forecasts yet. The scheduled daily forecast has not populated the dashboard data store.")
        return
    _render_data_warning(prepared)
    _render_rmae_landing(prepared)
    st.page_link(details_page, label="More detailed insights →", icon="📈")


def _render_forecast_details_page() -> None:
    _render_header()
    prepared, _, _ = _load_dashboard()
    if prepared.empty:
        st.info("No stored forecasts yet. The scheduled daily forecast has not populated the dashboard data store.")
        return
    _render_data_warning(prepared)
    st.markdown('<div class="task-section-title">More detailed insights</div>', unsafe_allow_html=True)
    selected_target = st.selectbox("Forecast target", TARGET_ORDER, format_func=_target_label, index=0)
    available_zones = sorted(prepared.loc[prepared["target"].eq(selected_target), "zone"].dropna().unique())
    _render_task_title(selected_target)
    _render_target_section(selected_target, prepared, available_zones)


def render_app() -> None:
    st.set_page_config(page_title="Transparency++", page_icon="chart_with_upwards_trend", layout="wide")
    details_page = st.Page(_render_forecast_details_page, title="More detailed insights", icon="📈")
    leaderboard_page = st.Page(
        lambda: _render_leaderboard_page(details_page),
        title="Leaderboard",
        icon="🏆",
        default=True,
    )
    navigation = st.navigation([leaderboard_page, details_page])
    navigation.run()
