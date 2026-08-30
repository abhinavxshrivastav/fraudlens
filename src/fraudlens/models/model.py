"""Model training, calibration and persistence.

Two decisions here carry most of the weight.

**Class imbalance is handled by weighting, not resampling.** ``scale_pos_weight``
reweights the loss so the minority class matters proportionally more, without
inventing data. SMOTE and friends are not used, for reasons set out in
``docs/EVALUATION.md`` -- briefly: they interpolate across the temporal boundary
the embargo exists to protect, they destroy calibration, and they fabricate
behaviourally impossible feature combinations.

**Outputs are calibrated.** A gradient-boosted margin is a ranking score, not a
probability. Isotonic regression fitted on the validation fold maps it onto
something that can be read as "87% of transactions scoring this high are fraud".
Cost-based thresholding and the policy bands both depend on that being true.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from fraudlens.features.pipeline import FeaturePipeline

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

MODEL_FILENAME: Final = "model.joblib"
METADATA_FILENAME: Final = "metadata.json"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyperparameters and training policy.

    Defaults are conservative rather than tuned: shallow trees and strong
    regularisation, because with a 0.5% positive rate a deep forest memorises
    the handful of frauds in the training fold and generalises poorly.
    """

    n_estimators: int = 600
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = 6
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
    random_state: int = 42
    #: Isotonic handles the sigmoid-shaped miscalibration of boosted trees
    #: better than Platt scaling, and there is enough validation data here to
    #: fit it without overfitting.
    calibration_method: str = "isotonic"

    def to_lightgbm_params(self, scale_pos_weight: float) -> dict[str, Any]:
        return {
            "objective": "binary",
            "metric": "average_precision",
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "subsample_freq": 1,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "scale_pos_weight": scale_pos_weight,
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbose": -1,
        }


@dataclass(slots=True)
class ModelMetadata:
    """Everything needed to identify and audit a trained artefact."""

    version: str
    model_type: str
    trained_at: str
    feature_names: tuple[str, ...]
    n_train_rows: int
    n_train_frauds: int
    train_fraud_rate: float
    scale_pos_weight: float
    calibration_method: str
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    data_window: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "model_type": self.model_type,
            "trained_at": self.trained_at,
            "feature_names": list(self.feature_names),
            "n_train_rows": self.n_train_rows,
            "n_train_frauds": self.n_train_frauds,
            "train_fraud_rate": self.train_fraud_rate,
            "scale_pos_weight": self.scale_pos_weight,
            "calibration_method": self.calibration_method,
            "config": self.config,
            "metrics": self.metrics,
            "thresholds": self.thresholds,
            "data_window": self.data_window,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ModelMetadata:
        return cls(
            version=str(raw["version"]),
            model_type=str(raw["model_type"]),
            trained_at=str(raw["trained_at"]),
            feature_names=tuple(raw["feature_names"]),
            n_train_rows=int(raw["n_train_rows"]),
            n_train_frauds=int(raw["n_train_frauds"]),
            train_fraud_rate=float(raw["train_fraud_rate"]),
            scale_pos_weight=float(raw["scale_pos_weight"]),
            calibration_method=str(raw["calibration_method"]),
            config=dict(raw.get("config", {})),
            metrics=dict(raw.get("metrics", {})),
            thresholds=dict(raw.get("thresholds", {})),
            data_window=dict(raw.get("data_window", {})),
        )


class FeatureContractError(ValueError):
    """Raised when features presented at scoring time do not match training."""


class FraudModel:
    """A trained, calibrated fraud classifier.

    Satisfies the :class:`~fraudlens.scoring.decision.ScoringModel` protocol, so
    the decision engine depends only on ``version`` and ``predict_proba``.
    """

    def __init__(
        self,
        booster: Any,
        calibrator: Any | None,
        metadata: ModelMetadata,
    ) -> None:
        self._booster = booster
        self._calibrator = calibrator
        self._metadata = metadata

    @property
    def version(self) -> str:
        return self._metadata.version

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._metadata.feature_names

    @property
    def is_calibrated(self) -> bool:
        return self._calibrator is not None

    def predict_proba(self, features: Mapping[str, float]) -> float:
        """Calibrated probability of fraud for one transaction."""
        vector = self._to_frame([features])
        return float(self.predict_proba_batch(vector)[0])

    def predict_proba_batch(self, features: pd.DataFrame) -> np.ndarray:
        """Calibrated probabilities for a batch.

        The feature contract is checked here rather than trusted: a column
        ordering change between training and serving is silent and disastrous.
        """
        ordered = self._align(features)
        raw = self._booster.predict_proba(ordered)[:, 1]
        if self._calibrator is None:
            return np.asarray(raw, dtype=np.float64)
        return np.asarray(self._calibrator.predict(raw), dtype=np.float64)

    def predict_raw_batch(self, features: pd.DataFrame) -> np.ndarray:
        """Uncalibrated scores, for calibration diagnostics."""
        return np.asarray(self._booster.predict_proba(self._align(features))[:, 1], np.float64)

    def _align(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_names) - set(features.columns)
        if missing:
            msg = (
                f"features missing at scoring time: {sorted(missing)}. "
                f"The model was trained on a different feature set."
            )
            raise FeatureContractError(msg)
        return features[list(self.feature_names)]

    def _to_frame(self, rows: Sequence[Mapping[str, float]]) -> pd.DataFrame:
        return pd.DataFrame(list(rows), columns=list(self.feature_names))

    # -- persistence -------------------------------------------------------

    def save(self, directory: Path) -> Path:
        """Persist the model, calibrator and metadata."""
        import joblib

        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"booster": self._booster, "calibrator": self._calibrator},
            directory / MODEL_FILENAME,
        )
        (directory / METADATA_FILENAME).write_text(
            json.dumps(self._metadata.to_dict(), indent=2), encoding="utf-8"
        )
        logger.info("Saved model %s to %s", self.version, directory)
        return directory

    @classmethod
    def load(cls, directory: Path) -> FraudModel:
        import joblib

        payload = joblib.load(directory / MODEL_FILENAME)
        metadata = ModelMetadata.from_dict(
            json.loads((directory / METADATA_FILENAME).read_text(encoding="utf-8"))
        )
        return cls(payload["booster"], payload["calibrator"], metadata)


def compute_scale_pos_weight(y: np.ndarray | pd.Series) -> float:
    """Ratio of negatives to positives.

    This is the whole of the imbalance strategy. Setting it to ``n_neg/n_pos``
    makes the total gradient contribution of each class equal, which is what
    resampling achieves without inventing rows.
    """
    labels = np.asarray(y).ravel()
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0:
        msg = "cannot train: the training fold contains no fraud examples"
        raise ValueError(msg)
    return negatives / positives


def train_lightgbm(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    config: TrainingConfig | None = None,
    *,
    version: str | None = None,
    data_window: Mapping[str, str] | None = None,
) -> FraudModel:
    """Train a LightGBM classifier and calibrate it on the validation fold.

    ``x_valid`` is used for early stopping *and* calibration. Both are legitimate
    uses of a validation fold; what must not happen is either touching test.
    """
    import lightgbm as lgb

    config = config or TrainingConfig()
    weight = compute_scale_pos_weight(y_train)
    logger.info(
        "Training on %d rows (%d fraud, %.4f%%), scale_pos_weight=%.1f",
        len(x_train),
        int(y_train.sum()),
        100 * float(y_train.mean()),
        weight,
    )

    booster = lgb.LGBMClassifier(**config.to_lightgbm_params(weight))

    # LightGBM 4.6 renamed `eval_set` to `eval_X`/`eval_y` and deprecated the
    # old spelling. Pick whichever the installed version prefers so the project
    # is quiet on new releases without raising its own minimum.
    import inspect

    fit_params = inspect.signature(lgb.LGBMClassifier.fit).parameters
    holdout: dict[str, Any] = (
        {"eval_X": x_valid, "eval_y": y_valid}
        if "eval_X" in fit_params
        else {"eval_set": [(x_valid, y_valid)]}
    )

    booster.fit(
        x_train,
        y_train,
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
        **holdout,
    )

    calibrator = _fit_calibrator(booster, x_valid, y_valid, config.calibration_method)

    metadata = ModelMetadata(
        version=version or _default_version("lightgbm"),
        model_type="lightgbm",
        trained_at=datetime.now(UTC).isoformat(),
        feature_names=tuple(x_train.columns),
        n_train_rows=len(x_train),
        n_train_frauds=int(y_train.sum()),
        train_fraud_rate=float(y_train.mean()),
        scale_pos_weight=weight,
        calibration_method=config.calibration_method,
        config=config.to_lightgbm_params(weight),
        data_window=dict(data_window or {}),
    )
    return FraudModel(booster, calibrator, metadata)


def train_logistic_baseline(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    version: str | None = None,
) -> FraudModel:
    """An interpretable linear baseline on the same features.

    Separates "the features carry signal" from "the gradient boosting is doing
    the work". If the linear model is close, the complexity is not earning its
    keep.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=42,
                ),
            ),
        ]
    )
    prepared = x_train.fillna(0.0)
    pipeline.fit(prepared, y_train)

    calibrator = _fit_calibrator(pipeline, x_valid.fillna(0.0), y_valid, "isotonic")
    metadata = ModelMetadata(
        version=version or _default_version("logistic"),
        model_type="logistic_regression",
        trained_at=datetime.now(UTC).isoformat(),
        feature_names=tuple(x_train.columns),
        n_train_rows=len(x_train),
        n_train_frauds=int(y_train.sum()),
        train_fraud_rate=float(y_train.mean()),
        scale_pos_weight=compute_scale_pos_weight(y_train),
        calibration_method="isotonic",
    )
    return FraudModel(pipeline, calibrator, metadata)


def _fit_calibrator(
    booster: Any,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    method: str,
) -> Any | None:
    """Fit a probability calibrator on held-out validation scores."""
    if method == "none":
        return None

    raw = booster.predict_proba(x_valid)[:, 1]

    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(raw, y_valid)
        return calibrator

    if method == "sigmoid":
        from sklearn.linear_model import LogisticRegression

        platt = LogisticRegression()
        platt.fit(raw.reshape(-1, 1), y_valid)

        class _PlattWrapper:
            def __init__(self, model: Any) -> None:
                self._model = model

            def predict(self, scores: np.ndarray) -> np.ndarray:
                return np.asarray(
                    self._model.predict_proba(np.asarray(scores).reshape(-1, 1))[:, 1]
                )

        return _PlattWrapper(platt)

    msg = f"unknown calibration method {method!r}; use 'isotonic', 'sigmoid' or 'none'"
    raise ValueError(msg)


def _default_version(model_type: str) -> str:
    return f"{model_type}-{datetime.now(UTC).strftime('%Y%m%d.%H%M%S')}"


def build_training_matrix(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    feature_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Select and order feature columns to match the model contract."""
    names = list(feature_names or FeaturePipeline.FEATURE_NAMES)
    missing = set(names) - set(features.columns)
    if missing:
        msg = f"feature frame is missing columns: {sorted(missing)}"
        raise FeatureContractError(msg)
    return features[names], target.loc[features.index]
