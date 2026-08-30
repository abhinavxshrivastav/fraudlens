"""Train, evaluate and persist the FraudLens champion model.

The order of operations here is the methodology, and it is deliberate:

1. Split chronologically with an embargo.
2. Compute features by replaying transactions through the state machine, warm
   state carried forward across folds in time order (never backwards).
3. Fit every candidate on **train**.
4. Select thresholds, calibrate and early-stop on **validation**.
5. Score **test** exactly once, at the end, with everything frozen.

Baselines are evaluated alongside the champion so the uplift is quantified
rather than asserted.

Usage::

    python scripts/train.py                    # full processed dataset
    python scripts/train.py --sample           # the committed 5k sample (fast)
    python scripts/train.py --no-cache         # recompute features
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from fraudlens.config import constants as C
from fraudlens.config import get_settings
from fraudlens.data.loader import load_sample
from fraudlens.data.splits import Split, split_frame
from fraudlens.data.synthetic import TYPOLOGY_COL
from fraudlens.evaluation.metrics import (
    CostModel,
    brier_score,
    evaluate_at_threshold,
    min_cost_threshold,
    pr_auc,
    precision_at_k,
    recall_at_fpr,
    roc_auc,
)
from fraudlens.features.pipeline import FeaturePipeline
from fraudlens.models.model import (
    TrainingConfig,
    train_lightgbm,
    train_logistic_baseline,
)
from fraudlens.rules.engine import RuleEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("train")

ROOT = Path(__file__).resolve().parents[1]
FEATURE_CACHE = "features_{split}.parquet"
KEEP = (C.TARGET_COL, C.AMOUNT_COL, C.TIMESTAMP_COL, C.TRANSACTION_ID_COL)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_dataset(use_sample: bool) -> pd.DataFrame:
    settings = get_settings()
    if use_sample:
        logger.info("Using the committed sample")
        return load_sample()

    path = settings.processed_dir / "transactions.parquet"
    if not path.exists():
        msg = f"{path} not found. Run scripts/prepare_data.py first."
        raise FileNotFoundError(msg)
    logger.info("Loading %s", path)
    return pd.read_parquet(path)


def build_features(df: pd.DataFrame, *, use_cache: bool, tag: str) -> dict[Split, pd.DataFrame]:
    """Compute per-fold features, replaying folds in chronological order."""
    settings = get_settings()
    cache_dir = settings.processed_dir / f"features_{tag}"
    keep = (*KEEP, TYPOLOGY_COL) if TYPOLOGY_COL in df.columns else KEEP

    if use_cache and cache_dir.exists():
        cached = {s: cache_dir / FEATURE_CACHE.format(split=s) for s in Split}
        if all(p.exists() for p in cached.values()):
            logger.info("Reusing cached features from %s", cache_dir)
            return {s: pd.read_parquet(p) for s, p in cached.items()}

    frames, report = split_frame(df)
    logger.info("Temporal split:\n%s", report.to_markdown())

    pipeline = FeaturePipeline()
    # Warm state carries forward train -> validation -> test, mirroring how a
    # live system accumulates history. It only ever moves forward in time.
    states: dict[int, object] = {}
    out: dict[Split, pd.DataFrame] = {}
    for split in Split:
        started = time.perf_counter()
        out[split] = pipeline.transform_frame(
            frames[split],
            states=states,  # type: ignore[arg-type]
            keep_columns=keep,
        )
        elapsed = time.perf_counter() - started
        logger.info(
            "  %-11s %s rows in %.1fs (%s rows/s)",
            split,
            f"{len(out[split]):,}",
            elapsed,
            f"{len(out[split]) / max(elapsed, 1e-9):,.0f}",
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    for split, frame in out.items():
        frame.to_parquet(cache_dir / FEATURE_CACHE.format(split=split), index=False)
    logger.info("Cached features -> %s", cache_dir)
    return out


def xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return frame[list(FeaturePipeline.FEATURE_NAMES)], frame[C.TARGET_COL]


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def rules_only_scores(frame: pd.DataFrame) -> np.ndarray:
    """Score using the deterministic rule set alone.

    This is the honest comparator for the claim that ML reduces false positives:
    it is what a traditional system achieves on the same data.
    """
    engine = RuleEngine.from_yaml(ROOT / "config" / "rules.yaml")
    severity_score = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
    scores = np.zeros(len(frame), dtype=np.float64)
    records = frame[list(FeaturePipeline.FEATURE_NAMES)].to_dict("records")
    for i, features in enumerate(records):
        result = engine.evaluate(features)
        if result.blocked:
            scores[i] = 1.0
        elif result.hits:
            scores[i] = max(severity_score[str(h.severity)] for h in result.hits)
    return scores


def amount_only_scores(frame: pd.DataFrame) -> np.ndarray:
    """Rank by transaction value. A trivially available signal."""
    return frame[C.AMOUNT_COL].to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    name: str,
    y_true: pd.Series,
    scores: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    *,
    calibrated: bool,
) -> dict[str, object]:
    point = evaluate_at_threshold(y_true, scores, threshold, amounts=amounts)
    row: dict[str, object] = {
        "model": name,
        "pr_auc": pr_auc(y_true, scores),
        "roc_auc": roc_auc(y_true, scores),
        "brier": brier_score(y_true, scores) if calibrated else float("nan"),
        "precision": point.precision,
        "recall": point.recall,
        "alert_rate": point.alert_rate,
        "fpr": point.false_positive_rate,
        "value_detection": point.value_detection_rate,
        "expected_cost": point.expected_cost,
        "precision_at_k": precision_at_k(y_true, scores, C.ANALYST_DAILY_CAPACITY),
        "threshold": threshold,
    }
    for target in C.REPORTED_FPRS:
        recall, _ = recall_at_fpr(y_true, scores, target)
        row[f"recall_at_fpr_{target}"] = recall
    return row


def results_table(rows: list[dict[str, object]]) -> str:
    header = (
        "| Model | PR-AUC | ROC-AUC | Recall@1%FPR | Precision | Recall | "
        "Alert rate | P@100 | Expected cost |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} "
            f"| {r['recall_at_fpr_0.01']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} "
            f"| {r['alert_rate']:.3%} | {r['precision_at_k']:.3f} "
            f"| {r['expected_cost']:,.0f} |"
        )
    return "\n".join(lines)


def typology_recall(
    frame: pd.DataFrame, y_true: pd.Series, scores: np.ndarray, threshold: float
) -> str:
    if TYPOLOGY_COL not in frame.columns:
        return ""
    flagged = scores >= threshold
    typ = frame[TYPOLOGY_COL].to_numpy()
    truth = y_true.to_numpy()
    lines = ["| Typology | Fraud rows | Recall |", "|---|---:|---:|"]
    for name in sorted({t for t, y in zip(typ, truth, strict=True) if y == 1}):
        mask = (typ == name) & (truth == 1)
        if mask.sum():
            lines.append(f"| {name} | {int(mask.sum()):,} | {flagged[mask].mean():.3f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true", help="use the committed 5k sample")
    parser.add_argument("--no-cache", action="store_true", help="recompute features")
    parser.add_argument("--skip-xgboost", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    tag = "sample" if args.sample else "full"
    # Artefact names carry the dataset tag. A quick `--sample` smoke run must
    # not be able to overwrite a champion trained on the full dataset -- that
    # silently swaps a 3k-row model into serving and nothing complains.
    model_name = "champion" if not args.sample else "champion-sample"

    df = load_dataset(args.sample)
    features = build_features(df, use_cache=not args.no_cache, tag=tag)

    x_train, y_train = xy(features[Split.TRAIN])
    x_valid, y_valid = xy(features[Split.VALIDATION])
    x_test, y_test = xy(features[Split.TEST])
    amt_valid = features[Split.VALIDATION][C.AMOUNT_COL].to_numpy()
    amt_test = features[Split.TEST][C.AMOUNT_COL].to_numpy()
    costs = CostModel()

    logger.info(
        "train %d (%d fraud) | validation %d (%d fraud) | test %d (%d fraud)",
        len(x_train),
        int(y_train.sum()),
        len(x_valid),
        int(y_valid.sum()),
        len(x_test),
        int(y_test.sum()),
    )

    rows: list[dict[str, object]] = []

    # -- baselines ---------------------------------------------------------
    logger.info("Baseline: amount only")
    amt_scores_valid = amount_only_scores(features[Split.VALIDATION])
    amt_thr = min_cost_threshold(
        y_valid, amt_scores_valid, amounts=amt_valid, cost_model=costs
    ).threshold
    rows.append(
        evaluate(
            "Amount only (heuristic)",
            y_test,
            amount_only_scores(features[Split.TEST]),
            amt_test,
            amt_thr,
            calibrated=False,
        )
    )

    logger.info("Baseline: rules only")
    rule_scores_valid = rules_only_scores(features[Split.VALIDATION])
    rule_thr = min_cost_threshold(
        y_valid, rule_scores_valid, amounts=amt_valid, cost_model=costs
    ).threshold
    rule_scores_test = rules_only_scores(features[Split.TEST])
    rows.append(
        evaluate("Rules only", y_test, rule_scores_test, amt_test, rule_thr, calibrated=False)
    )

    logger.info("Baseline: logistic regression")
    logistic = train_logistic_baseline(x_train, y_train, x_valid, y_valid)
    log_valid = logistic.predict_proba_batch(x_valid.fillna(0.0))
    log_thr = min_cost_threshold(y_valid, log_valid, amounts=amt_valid, cost_model=costs).threshold
    rows.append(
        evaluate(
            "Logistic regression",
            y_test,
            logistic.predict_proba_batch(x_test.fillna(0.0)),
            amt_test,
            log_thr,
            calibrated=True,
        )
    )

    # -- champion ----------------------------------------------------------
    logger.info("Champion: LightGBM")
    started = time.perf_counter()
    model = train_lightgbm(
        x_train,
        y_train,
        x_valid,
        y_valid,
        TrainingConfig(),
        data_window={
            "train": f"{C.TRAIN_START} to {C.TRAIN_END}",
            "validation": f"{C.VALID_START} to {C.VALID_END}",
            "test": f"{C.TEST_START} to {C.TEST_END}",
        },
    )
    logger.info("  trained in %.1fs", time.perf_counter() - started)

    valid_scores = model.predict_proba_batch(x_valid)
    operating = min_cost_threshold(y_valid, valid_scores, amounts=amt_valid, cost_model=costs)
    _, thr_1pct = recall_at_fpr(y_valid, valid_scores, 0.01)
    logger.info(
        "  validation: PR-AUC %.4f | min-cost threshold %.4f (alert rate %.3f%%)",
        pr_auc(y_valid, valid_scores),
        operating.threshold,
        100 * operating.alert_rate,
    )

    # Calibration quality, before and after.
    raw_valid = model.predict_raw_batch(x_valid)
    logger.info(
        "  Brier: raw %.6f -> calibrated %.6f",
        brier_score(y_valid, raw_valid),
        brier_score(y_valid, valid_scores),
    )

    test_scores = model.predict_proba_batch(x_test)
    rows.append(
        evaluate(
            "LightGBM (champion)",
            y_test,
            test_scores,
            amt_test,
            operating.threshold,
            calibrated=True,
        )
    )

    if not args.skip_xgboost:
        logger.info("Challenger: XGBoost")
        rows.append(
            _train_xgboost(
                x_train, y_train, x_valid, y_valid, x_test, y_test, amt_valid, amt_test, costs
            )
        )

    # -- report ------------------------------------------------------------
    table = results_table(rows)
    typ_table = typology_recall(features[Split.TEST], y_test, test_scores, operating.threshold)

    print("\n" + "=" * 100)
    print("TEST RESULTS (scored once; every threshold selected on validation)")
    print("=" * 100)
    print(table)
    if typ_table:
        print("\nRecall by fraud typology (champion):")
        print(typ_table)

    settings.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = settings.report_dir / f"results-{tag}.md"
    report_path.write_text(
        _render_report(rows, table, typ_table, model, len(x_train), len(x_test)),
        encoding="utf-8",
    )
    (settings.report_dir / f"results-{tag}.json").write_text(
        json.dumps(rows, indent=2, default=float), encoding="utf-8"
    )
    logger.info("Wrote %s", report_path)

    model.metadata.metrics = {
        k: float(v)
        for k, v in rows[-1 if args.skip_xgboost else -2].items()
        if isinstance(v, (int, float))
    }
    model.metadata.thresholds = {
        "min_expected_cost": operating.threshold,
        "recall_at_1pct_fpr": thr_1pct,
    }
    model.save(settings.model_dir / model_name)
    logger.info("Saved model -> %s", settings.model_dir / model_name)
    return 0


def _train_xgboost(x_train, y_train, x_valid, y_valid, x_test, y_test, amt_valid, amt_test, costs):
    import xgboost as xgb

    from fraudlens.models.model import compute_scale_pos_weight

    clf = xgb.XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=compute_scale_pos_weight(y_train),
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    valid = clf.predict_proba(x_valid)[:, 1]
    thr = min_cost_threshold(y_valid, valid, amounts=amt_valid, cost_model=costs).threshold
    return evaluate(
        "XGBoost (challenger)",
        y_test,
        clf.predict_proba(x_test)[:, 1],
        amt_test,
        thr,
        calibrated=False,
    )


def _render_report(
    rows: list[dict[str, object]],
    table: str,
    typ_table: str,
    model: object,
    n_train: int,
    n_test: int,
) -> str:
    meta = model.metadata  # type: ignore[attr-defined]
    parts = [
        "# Results",
        "",
        "Generated by `scripts/train.py`. Every threshold was selected on the",
        "validation fold; the test fold was scored once, at the end.",
        "",
        f"- Champion: `{meta.version}`",
        f"- Training rows: {n_train:,} ({meta.n_train_frauds:,} fraud, "
        f"{meta.train_fraud_rate:.4%})",
        f"- Test rows: {n_test:,}",
        f"- Imbalance handling: `scale_pos_weight={meta.scale_pos_weight:.1f}` (no resampling)",
        f"- Calibration: {meta.calibration_method}",
        "",
        "## Headline",
        "",
        table,
        "",
        "Expected cost uses the model in `docs/EVALUATION.md`: £4.00 per alert",
        "investigated, £18.00 friction per false positive, and the full",
        "transaction amount for every missed fraud. Lower is better.",
        "",
    ]
    if typ_table:
        parts += [
            "## Recall by fraud typology",
            "",
            typ_table,
            "",
            "The `subtle` typology is fraud constructed to be indistinguishable",
            "from ordinary spending: a familiar merchant, a normal amount, in the",
            "afternoon, with no velocity anomaly. Near-zero recall on it is the",
            "expected and correct result -- catching it needs signals outside the",
            "transaction record. It is included precisely so that headline recall",
            "is not flattered by an artificially separable problem.",
            "",
        ]
    return "\n".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
