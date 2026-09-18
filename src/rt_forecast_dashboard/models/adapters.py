from __future__ import annotations

from pathlib import Path
from typing import Protocol
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_CHRONOS2_PIPELINE = None
_TIMESFM3_PIPELINE = None


class ForecastAdapter(Protocol):
    def predict(self, features: pd.DataFrame, target: str, context: pd.DataFrame | None = None, covariates: list[str] | None = None) -> np.ndarray:
        ...


class TsoReferenceAdapter:
    def predict(self, features: pd.DataFrame, target: str, context: pd.DataFrame | None = None, covariates: list[str] | None = None) -> np.ndarray:
        return features["tso_forecast_mw"].to_numpy(dtype=float)


class PersistenceAdapter:
    def predict(self, features: pd.DataFrame, target: str, context: pd.DataFrame | None = None, covariates: list[str] | None = None) -> np.ndarray:
        if context is None or context.empty or "actual_mw" not in context.columns:
            return features["tso_forecast_mw"].to_numpy(dtype=float)
        lag = pd.Timedelta(days=7 if target == "load" else 1)
        hist = context[["timestamp", "actual_mw"]].dropna().copy()
        hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True)
        hist = hist.drop_duplicates("timestamp", keep="last").set_index("timestamp")["actual_mw"].sort_index()
        values = []
        fallback = float(hist.tail(24).mean()) if not hist.empty else 0.0
        for ts in pd.to_datetime(features["timestamp"], utc=True):
            lookup_ts = ts - lag
            values.append(float(hist.get(lookup_ts, fallback)))
        return np.maximum(np.asarray(values, dtype=float), 0.0)


class RidgeContextAdapter:
    def predict(self, features: pd.DataFrame, target: str, context: pd.DataFrame | None = None, covariates: list[str] | None = None) -> np.ndarray:
        if context is None or context.empty or "actual_mw" not in context.columns:
            return features["tso_forecast_mw"].to_numpy(dtype=float)
        covariates = covariates or ["tso_forecast_mw"]
        feature_cols = [c for c in covariates + ["hour", "dayofweek", "month", "is_weekend"] if c in context.columns and c in features.columns]
        train = context.dropna(subset=feature_cols + ["actual_mw"]).copy()
        if len(train) < 7 * 24:
            return features["tso_forecast_mw"].to_numpy(dtype=float)
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(train[feature_cols], train["actual_mw"])
        pred = model.predict(features[feature_cols])
        if "tso_forecast_mw" in features:
            # Keep the online model anchored to the new TSO forecast, while allowing the
            # selected weather covariates and calendar effects to correct the TSO shape.
            tso = features["tso_forecast_mw"].to_numpy(dtype=float)
            pred = 0.65 * np.asarray(pred, dtype=float) + 0.35 * tso
        return np.maximum(np.asarray(pred, dtype=float), 0.0)


class TabPFNContextAdapter:
    def __init__(self, max_train_rows: int = 4096, model_path: str | None = None) -> None:
        self.max_train_rows = max_train_rows
        self.model_path = model_path

    def predict(self, features: pd.DataFrame, target: str, context: pd.DataFrame | None = None, covariates: list[str] | None = None) -> np.ndarray:
        if context is None or context.empty or "actual_mw" not in context.columns:
            return features["tso_forecast_mw"].to_numpy(dtype=float)
        os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
        from tabpfn import TabPFNRegressor

        covariates = covariates or ["tso_forecast_mw"]
        feature_cols = [c for c in covariates + ["hour", "dayofweek", "month", "is_weekend"] if c in context.columns and c in features.columns]
        train = context.dropna(subset=feature_cols + ["actual_mw"]).tail(self.max_train_rows).copy()
        if len(train) < 7 * 24:
            return features["tso_forecast_mw"].to_numpy(dtype=float)
        model = TabPFNRegressor(
            model_path=self.model_path or "auto",
            n_estimators=4,
            device="cpu",
            random_state=0,
            n_preprocessing_jobs=1,
            ignore_pretraining_limits=True,
        )
        model.fit(train[feature_cols].to_numpy(dtype=np.float32), train["actual_mw"].to_numpy(dtype=np.float32))
        pred = model.predict(features[feature_cols].to_numpy(dtype=np.float32))
        return np.maximum(np.asarray(pred, dtype=float), 0.0)


class XGBoostContextAdapter:
    def __init__(self, max_train_rows: int = 12000, n_estimators: int = 80, max_depth: int = 3, n_jobs: int = 1) -> None:
        self.max_train_rows = max_train_rows
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs

    def predict(self, features: pd.DataFrame, target: str, context: pd.DataFrame | None = None, covariates: list[str] | None = None) -> np.ndarray:
        if context is None or context.empty or "actual_mw" not in context.columns:
            return features["tso_forecast_mw"].to_numpy(dtype=float)
        from xgboost import XGBRegressor

        covariates = covariates or ["tso_forecast_mw"]
        feature_cols = [c for c in covariates + ["hour", "dayofweek", "month", "is_weekend"] if c in context.columns and c in features.columns]
        train = context.dropna(subset=feature_cols + ["actual_mw"]).tail(self.max_train_rows).copy()
        if len(train) < 7 * 24:
            return features["tso_forecast_mw"].to_numpy(dtype=float)
        model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            n_jobs=self.n_jobs,
            random_state=0,
        )
        model.fit(train[feature_cols], train["actual_mw"])
        pred = model.predict(features[feature_cols])
        return np.maximum(np.asarray(pred, dtype=float), 0.0)


class Chronos2OnlineAdapter:
    def __init__(self, model_id: str = "s3://autogluon/chronos-2/", device_map: str = "cpu") -> None:
        self.model_id = model_id
        self.device_map = device_map

    def _pipeline(self):
        global _CHRONOS2_PIPELINE
        if _CHRONOS2_PIPELINE is None:
            from chronos import BaseChronosPipeline

            _CHRONOS2_PIPELINE = BaseChronosPipeline.from_pretrained(self.model_id, device_map=self.device_map)
        return _CHRONOS2_PIPELINE

    def predict(self, features: pd.DataFrame, target: str, context: pd.DataFrame | None = None, covariates: list[str] | None = None) -> np.ndarray:
        if context is None or context.empty or "actual_mw" not in context.columns:
            return features["tso_forecast_mw"].to_numpy(dtype=float)
        covariates = covariates or ["tso_forecast_mw"]
        covariates = [c for c in covariates if c in context.columns and c in features.columns]
        context_in = context[["timestamp", "actual_mw", *covariates]].copy()
        context_in["id"] = target
        context_in = context_in.rename(columns={"actual_mw": "target"})
        context_in = context_in.dropna(subset=["timestamp", "target"]).sort_values("timestamp")
        if len(context_in) < 7 * 24:
            return features["tso_forecast_mw"].to_numpy(dtype=float)
        for column in covariates:
            context_in[column] = context_in[column].interpolate(limit_direction="both").ffill().bfill()

        future = features[["timestamp", *covariates]].copy()
        future["id"] = target
        for column in covariates:
            future[column] = future[column].interpolate(limit_direction="both").ffill().bfill()

        pred = self._pipeline().predict_df(
            context_in[["id", "timestamp", "target", *covariates]],
            future_df=future[["id", "timestamp", *covariates]] if covariates else None,
            prediction_length=len(features),
            id_column="id",
            timestamp_column="timestamp",
            target="target",
            quantile_levels=[0.1, 0.5, 0.9],
            freq="h",
            validate_inputs=False,
        )
        if "0.5" in pred.columns:
            values = pred["0.5"].to_numpy(dtype=float)
        elif "median" in pred.columns:
            values = pred["median"].to_numpy(dtype=float)
        else:
            numeric = pred.select_dtypes(include=[np.number])
            values = numeric.iloc[:, -1].to_numpy(dtype=float)
        return np.maximum(values[: len(features)], 0.0)


class TimesFM3OnlineAdapter:
    def __init__(
        self,
        checkpoint_path: str = "google/timesfm-3.0-pytorch",
        device: str = "cpu",
        standardize_covariates: bool = False,
        local_files_only: bool = False,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.standardize_covariates = standardize_covariates
        self.local_files_only = local_files_only

    def _pipeline(self):
        global _TIMESFM3_PIPELINE
        if _TIMESFM3_PIPELINE is None:
            try:
                import timesfm
            except ImportError as exc:
                raise ImportError(
                    "TimesFM3 is not installed. Install `timesfm==3.0.2` in the forecast environment."
                ) from exc

            _TIMESFM3_PIPELINE = timesfm.TimesFM3Forecaster(
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                local_files_only=self.local_files_only,
            )
        return _TIMESFM3_PIPELINE

    @staticmethod
    def _clean_covariates(context: pd.DataFrame, features: pd.DataFrame, covariates: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        valid = [column for column in covariates if column in context.columns and column in features.columns]
        context = context.copy()
        features = features.copy()
        for column in valid:
            context[column] = pd.to_numeric(context[column], errors="coerce").interpolate(limit_direction="both").ffill().bfill()
            features[column] = pd.to_numeric(features[column], errors="coerce").interpolate(limit_direction="both").ffill().bfill()
        return context, features, valid

    def _covariate_array(self, context: pd.DataFrame, features: pd.DataFrame, covariates: list[str]) -> np.ndarray | None:
        if not covariates:
            return None
        combined = pd.concat([context[covariates], features[covariates]], ignore_index=True).astype(float)
        if self.standardize_covariates:
            mean = combined.iloc[: len(context)].mean(axis=0)
            std = combined.iloc[: len(context)].std(axis=0).replace(0, 1.0)
            combined = (combined - mean) / std
        return combined.to_numpy(dtype=np.float32).T

    @staticmethod
    def _extract_forecast(result: object, horizon: int) -> np.ndarray:
        values = None
        for attr in ("forecast", "mean", "median", "predictions"):
            if hasattr(result, attr):
                values = getattr(result, attr)
                break
        if values is None and isinstance(result, dict):
            for key in ("forecast", "mean", "median", "predictions"):
                if key in result:
                    values = result[key]
                    break
        if values is None:
            values = result
        array = np.asarray(values, dtype=float).reshape(-1)[:horizon]
        if len(array) != horizon:
            raise RuntimeError(f"Could not extract a {horizon}-step TimesFM3 forecast from output shape {np.asarray(values).shape}.")
        return np.maximum(array, 0.0)

    def predict(self, features: pd.DataFrame, target: str, context: pd.DataFrame | None = None, covariates: list[str] | None = None) -> np.ndarray:
        if context is None or context.empty or "actual_mw" not in context.columns:
            return features["tso_forecast_mw"].to_numpy(dtype=float)
        covariates = covariates or ["tso_forecast_mw"]
        context_in = context.dropna(subset=["timestamp", "actual_mw"]).sort_values("timestamp").copy()
        features_in = features.sort_values("timestamp").copy()
        if len(context_in) < 7 * 24:
            return features["tso_forecast_mw"].to_numpy(dtype=float)
        context_in, features_in, covariates = self._clean_covariates(context_in, features_in, covariates)
        y_context = context_in["actual_mw"].to_numpy(dtype=np.float32)
        cov_array = self._covariate_array(context_in, features_in, covariates)

        result = self._pipeline().predict(
            context=y_context,
            horizon=len(features_in),
            past_future_covariates=cov_array,
            ts_id=f"{target}:{features_in['timestamp'].iloc[0]}",
            make_positive=True,
            use_znorm=True,
        )
        return self._extract_forecast(result, len(features_in))


class ArtifactModelAdapter:
    def __init__(self, artifact_path: str | dict[str, str] | None = None) -> None:
        self.artifact_path = artifact_path
        self._models: dict[str, object] = {}

    def _path_for(self, target: str) -> Path | None:
        if isinstance(self.artifact_path, dict):
            value = self.artifact_path.get(target)
        else:
            value = self.artifact_path
        return Path(value) if value else None

    def predict(self, features: pd.DataFrame, target: str, context: pd.DataFrame | None = None, covariates: list[str] | None = None) -> np.ndarray:
        path = self._path_for(target)
        if path is None or not path.exists():
            raise FileNotFoundError(f"No artifact configured for {target}")
        if target not in self._models:
            self._models[target] = joblib.load(path)
        model = self._models[target]
        model_features = features[covariates] if covariates else features.drop(columns=["timestamp"], errors="ignore")
        return np.asarray(model.predict(model_features), dtype=float)


def make_adapter(config: dict) -> ForecastAdapter:
    adapter = config["adapter"]
    if adapter == "tso_reference":
        return TsoReferenceAdapter()
    if adapter == "persistence":
        return PersistenceAdapter()
    if adapter == "ridge_context":
        return RidgeContextAdapter()
    if adapter == "tabpfn_context":
        return TabPFNContextAdapter(max_train_rows=int(config.get("max_train_rows", 4096)), model_path=config.get("model_path"))
    if adapter == "xgboost_context":
        return XGBoostContextAdapter(
            max_train_rows=int(config.get("max_train_rows", 12000)),
            n_estimators=int(config.get("n_estimators", 80)),
            max_depth=int(config.get("max_depth", 3)),
            n_jobs=int(config.get("n_jobs", 1)),
        )
    if adapter == "chronos2_online":
        return Chronos2OnlineAdapter(model_id=config.get("model_id", "s3://autogluon/chronos-2/"), device_map=config.get("device_map", "cpu"))
    if adapter == "timesfm3_online":
        return TimesFM3OnlineAdapter(
            checkpoint_path=config.get("checkpoint_path", "google/timesfm-3.0-pytorch"),
            device=config.get("device", "cpu"),
            standardize_covariates=bool(config.get("standardize_covariates", False)),
            local_files_only=bool(config.get("local_files_only", False)),
        )
    if adapter == "artifact_model":
        return ArtifactModelAdapter(config.get("artifact_path"))
    raise ValueError(f"Unknown model adapter: {adapter}")
