"""
Feature schema validation for training/live parity.

This module is intentionally strict. A live trading system should fail closed
before inference when the feature matrix no longer matches the model artifact.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from config.settings import config
from utils.logger import get_logger

logger = get_logger()


SCHEMA_VERSION = 1
RAW_COLUMNS = ["time", "open", "high", "low", "close", "volume", "spread"]
MODEL_EXCLUDE_COLUMNS = {"time", "label", "label_name", "close"}
LABEL_MAPPING = {0: "NO_TRADE", 1: "BUY", 2: "SELL"}


class FeatureSchemaError(ValueError):
    """Raised when live features do not match the trained model schema."""


@dataclass(frozen=True)
class FeatureSchema:
    """Serializable feature schema stored with every model artifact."""

    feature_names: List[str]
    feature_count: int
    raw_columns: List[str] = field(default_factory=lambda: RAW_COLUMNS.copy())
    engineered_feature_names: List[str] = field(default_factory=list)
    label_mapping: Dict[int, str] = field(default_factory=lambda: LABEL_MAPPING.copy())
    symbol: str = ""
    timeframe: str = ""
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_feature_names(
        cls,
        feature_names: Iterable[str],
        engineered_feature_names: Optional[Iterable[str]] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> "FeatureSchema":
        names = list(feature_names)
        engineered = list(engineered_feature_names or [])
        return cls(
            feature_names=names,
            feature_count=len(names),
            engineered_feature_names=engineered,
            symbol=symbol or config.symbol.symbol,
            timeframe=timeframe or config.symbol.timeframe,
        )

    @classmethod
    def from_artifact(cls, artifact: Dict[str, Any]) -> "FeatureSchema":
        feature_names = list(artifact.get("feature_names") or [])
        if not feature_names:
            raise FeatureSchemaError("Model artifact does not contain feature_names")

        schema_data = artifact.get("feature_schema") or {}
        return cls(
            feature_names=feature_names,
            feature_count=int(schema_data.get("feature_count", artifact.get("feature_count", len(feature_names)))),
            raw_columns=list(schema_data.get("raw_columns", artifact.get("raw_columns", RAW_COLUMNS.copy()))),
            engineered_feature_names=list(
                schema_data.get(
                    "engineered_feature_names",
                    artifact.get("engineered_feature_names", []),
                )
            ),
            label_mapping=dict(schema_data.get("label_mapping", artifact.get("label_mapping", LABEL_MAPPING.copy()))),
            symbol=str(schema_data.get("symbol", artifact.get("symbol", ""))),
            timeframe=str(schema_data.get("timeframe", artifact.get("timeframe", ""))),
            schema_version=int(schema_data.get("schema_version", artifact.get("schema_version", SCHEMA_VERSION))),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FeatureValidator:
    """Validate and align feature dataframes before model inference."""

    @staticmethod
    def duplicate_columns(df: pd.DataFrame) -> List[str]:
        return df.columns[df.columns.duplicated()].tolist()

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, schema: FeatureSchema, *, strict_extra: bool = True) -> None:
        duplicates = FeatureValidator.duplicate_columns(df)
        if duplicates:
            logger.error(f"Duplicate feature dataframe columns detected: {duplicates}")
            raise FeatureSchemaError(f"Duplicate dataframe columns: {duplicates}")

        expected = list(schema.feature_names)
        expected_dupes = sorted({c for c in expected if expected.count(c) > 1})
        if expected_dupes:
            raise FeatureSchemaError(f"Model schema contains duplicate feature names: {expected_dupes}")

        if len(expected) != schema.feature_count:
            raise FeatureSchemaError(
                f"Feature schema count mismatch inside artifact: "
                f"feature_count={schema.feature_count}, names={len(expected)}"
            )

        missing = [c for c in expected if c not in df.columns]
        if missing:
            logger.error(f"Missing model features: {missing}")
            raise FeatureSchemaError(f"Missing model features: {missing}")

        if strict_extra:
            allowed_non_features = MODEL_EXCLUDE_COLUMNS
            extras = [c for c in df.columns if c not in expected and c not in allowed_non_features]
            if extras:
                logger.error(f"Extra feature columns not present in model schema: {extras}")
                raise FeatureSchemaError(f"Extra feature columns not present in model schema: {extras}")

        observed_order = [c for c in df.columns if c in expected]
        if observed_order != expected:
            raise FeatureSchemaError("Feature dataframe column order does not match model schema")

        matrix = df[expected]
        if list(matrix.columns) != expected:
            raise FeatureSchemaError("Feature matrix order does not match model schema")

        non_numeric = [
            c for c in expected
            if not pd.api.types.is_numeric_dtype(matrix[c])
        ]
        if non_numeric:
            logger.error(f"Non-numeric model features: {non_numeric}")
            raise FeatureSchemaError(f"Non-numeric model features: {non_numeric}")

        nan_counts = matrix.isna().sum()
        nan_cols = nan_counts[nan_counts > 0].to_dict()
        if nan_cols:
            logger.error(f"NaN values detected in model features: {nan_cols}")
            raise FeatureSchemaError(f"NaN values detected in model features: {nan_cols}")

        arr = matrix.to_numpy(dtype=np.float64, copy=False)
        inf_mask = np.isinf(arr)
        if inf_mask.any():
            inf_cols = {
                expected[idx]: int(inf_mask[:, idx].sum())
                for idx in np.where(inf_mask.any(axis=0))[0]
            }
            logger.error(f"Infinite values detected in model features: {inf_cols}")
            raise FeatureSchemaError(f"Infinite values detected in model features: {inf_cols}")

        logger.info(
            f"Feature schema validated: rows={len(df)}, "
            f"features={len(expected)}, symbol={schema.symbol or 'unknown'}, "
            f"timeframe={schema.timeframe or 'unknown'}"
        )

    @staticmethod
    def align_matrix(df: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
        FeatureValidator.validate_dataframe(df, schema)
        return df[list(schema.feature_names)].astype(np.float32)
