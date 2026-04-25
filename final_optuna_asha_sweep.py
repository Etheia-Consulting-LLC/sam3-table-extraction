"""Final-stage Optuna + ASHA sweep for SAM3 LoRA training.

This script runs Optuna with an ASHA-style pruner (Successive Halving):
- Promotion fraction (eta): 1/5  -> reduction_factor=5
- Progressive resource schedule per trial:
    - sample_percent increases by rung
    - num_epochs increases more aggressively near the final rung

Usage:
    modal run final_optuna_asha_sweep.py

Notes:
    - Requires `optuna` to be installed in your environment/image.
    - Uses the existing `train_sam3` Modal function and reads each run's
      `val_stats.json` from artifacts volume to score trials.
    - Pruning remains loss-based for speed, then an optional final rerank
      prefers quality metrics (IoU/F1) when metric files are present.
"""

from __future__ import annotations

import copy
import json
import sqlite3
import time
from pathlib import Path

import modal
import optuna

from sam3_table.cstone_train_sam3 import (
    MODAL_ARTIFACTS_DIR,
    app,
    artifacts_vol,
    train_sam3,
)
from sam3_table.training_config import SAM3LoRAConfig
from sweep import _deep_merge

# ASHA: keep the best 1/5 each rung.
PROMOTION_FRACTION = 0.20
REDUCTION_FACTOR = int(round(1 / PROMOTION_FRACTION))

# 500k-image regime: aggressively cheap early rungs, full budget only at the end.
SAMPLE_SCHEDULE = [1.0, 3.0, 10.0, 30.0, 100.0]
EPOCH_RATIO_SCHEDULE = [1.0 / 14.0, 2.0 / 14.0, 4.0 / 14.0, 8.0 / 14.0, 1.0]
NUM_RUNG_STAGES = len(SAMPLE_SCHEDULE)


def _build_stage_schedule(max_epochs: int) -> list[dict]:
    """Create progressively larger sample/epoch budgets by rung."""
    if len(SAMPLE_SCHEDULE) != len(EPOCH_RATIO_SCHEDULE):
        raise ValueError("SAMPLE_SCHEDULE and EPOCH_RATIO_SCHEDULE must have same length")

    stages: list[dict] = []
    prev_epochs = 0
    for sample_percent, ratio in zip(SAMPLE_SCHEDULE, EPOCH_RATIO_SCHEDULE):
        epoch_budget = max(1, int(round(max_epochs * ratio)))
        epoch_budget = max(epoch_budget, prev_epochs + 1)
        stages.append(
            {
                "sample_percent": sample_percent,
                "num_epochs": epoch_budget,
            }
        )
        prev_epochs = epoch_budget
    return stages


def _suggest_trial_overrides(trial: optuna.trial.Trial) -> tuple[dict, int]:
    """Suggest a rich final-run hyperparameter set."""
    lora_rank = trial.suggest_categorical("lora_rank", [8, 16, 24, 32, 48, 64])
    lora_alpha = trial.suggest_categorical("lora_alpha", [lora_rank, lora_rank * 2, lora_rank * 4])

    max_epochs = 14

    overrides = {
        "lora": {
            "rank": lora_rank,
            "alpha": lora_alpha,
            "dropout": trial.suggest_float("lora_dropout", 0.0, 0.25),
            # Include all encoder/decoder apply switches for structural search.
            "apply_to_vision_encoder": trial.suggest_categorical("apply_to_vision_encoder", [True, False]),
            "apply_to_text_encoder": trial.suggest_categorical("apply_to_text_encoder", [True, False]),
            "apply_to_geometry_encoder": trial.suggest_categorical("apply_to_geometry_encoder", [True, False]),
            "apply_to_detr_encoder": trial.suggest_categorical("apply_to_detr_encoder", [True, False]),
            "apply_to_detr_decoder": trial.suggest_categorical("apply_to_detr_decoder", [True, False]),
            "apply_to_mask_decoder": trial.suggest_categorical("apply_to_mask_decoder", [True, False]),
        },
        "training": {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 0.1, log=True),
            "adam_beta1": trial.suggest_float("adam_beta1", 0.85, 0.95),
            "adam_beta2": trial.suggest_float("adam_beta2", 0.97, 0.9999),
            "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-9, 1e-7, log=True),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.5, 2.0),
            "warmup_steps": trial.suggest_int("warmup_steps", 50, 800, step=50),
            "lr_scheduler": trial.suggest_categorical(
                "lr_scheduler",
                ["cosine", "linear", "constant_with_warmup"],
            ),
            "batch_size": trial.suggest_categorical("batch_size", [8]),
            "gradient_accumulation_steps": trial.suggest_categorical(
                "gradient_accumulation_steps",
                [1, 2, 4],
            ),
            "num_workers": trial.suggest_categorical("num_workers", [4, 8]),
            "mixed_precision": trial.suggest_categorical("mixed_precision", ["bf16"]),
            # Keep seed fixed during search; evaluate top configs across seeds separately.
            "seed": 42,
        },
        "output": {"output_dir": "outputs/final_optuna_asha"},
    }

    return overrides, max_epochs


@app.function(
    volumes={MODAL_ARTIFACTS_DIR: artifacts_vol},
    timeout=60 * 10,
)
def read_latest_val_loss(output_dir: str) -> float:
    """Read the latest val loss written by trainer to val_stats.json."""
    stats_path = Path(output_dir) / "val_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing validation stats file: {stats_path}")

    last_val_loss = None
    for raw_line in stats_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        record = json.loads(line)
        if "val_loss" in record:
            last_val_loss = float(record["val_loss"])

    if last_val_loss is None:
        raise ValueError(f"No val_loss entries found in {stats_path}")

    return last_val_loss


@app.function(
    volumes={MODAL_ARTIFACTS_DIR: artifacts_vol},
    timeout=60 * 10,
)
def read_stage_quality_metrics(output_dir: str) -> dict[str, float | None]:
    """Read best-effort IoU/F1 metrics from common evaluation artifact files."""
    output_path = Path(output_dir)
    metric_files = [
        "eval_metrics.json",
        "metrics.json",
        "validation_metrics.json",
        "val_metrics.json",
        "eval_results.json",
        "full_eval.json",
        "val_stats.json",
    ]
    iou_keys = ("mean_iou", "iou", "val_iou", "mask_iou")
    f1_keys = ("f1", "cgf1", "cg_f1", "val_f1", "mask_f1")

    def _extract_value(record: dict, candidates: tuple[str, ...]) -> float | None:
        for key in candidates:
            value = record.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    iou_value = None
    f1_value = None

    for filename in metric_files:
        file_path = output_path / filename
        if not file_path.exists():
            continue

        text = file_path.read_text().strip()
        if not text:
            continue

        records: list[dict] = []
        try:
            # val_stats.json is JSONL in this project, even though extension is .json.
            if file_path.name == "val_stats.json":
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    maybe = json.loads(line)
                    if isinstance(maybe, dict):
                        records.append(maybe)
            else:
                payload = json.loads(text)
                if isinstance(payload, dict):
                    records = [payload]
                elif isinstance(payload, list):
                    records = [item for item in payload if isinstance(item, dict)]
        except Exception:
            # Best-effort metrics reader: skip malformed/non-standard files.
            continue

        for record in records:
            maybe_iou = _extract_value(record, iou_keys)
            maybe_f1 = _extract_value(record, f1_keys)
            if maybe_iou is not None:
                iou_value = maybe_iou
            if maybe_f1 is not None:
                f1_value = maybe_f1

    composite = None
    if iou_value is not None and f1_value is not None:
        composite = 0.5 * iou_value + 0.5 * f1_value

    return {"iou": iou_value, "f1": f1_value, "composite": composite}


def _build_stage_config(base_config: SAM3LoRAConfig, trial_overrides: dict, stage_cfg: dict, trial_number: int, stage_idx: int) -> dict:
    config_dict = copy.deepcopy(base_config.model_dump(mode="json"))
    _deep_merge(config_dict, trial_overrides)
    _deep_merge(
        config_dict,
        {
            "training": {
                "data": {"sample_percent": stage_cfg["sample_percent"]},
                "num_epochs": stage_cfg["num_epochs"],
            },
            "output": {
                "output_dir": f"outputs/final_optuna_asha/trial_{trial_number:04d}",
            },
        },
    )
    return config_dict


def _objective_factory(
    base_config: SAM3LoRAConfig,
    objective_mode: str,
    cost_penalty_per_gpu_hour: float,
):
    def objective(trial: optuna.trial.Trial) -> float:
        trial_overrides, max_epochs = _suggest_trial_overrides(trial)
        stage_schedule = _build_stage_schedule(max_epochs=max_epochs)
        normalized_mode = objective_mode.strip().lower()
        trial_start_time = time.perf_counter()
        cumulative_stage_runtime_hours = 0.0

        best_stage_val_loss = float("inf")
        final_stage_val_loss = None
        resume_output_dir = None
        for stage_idx, stage_cfg in enumerate(stage_schedule, start=1):
            stage_config = _build_stage_config(
                base_config=base_config,
                trial_overrides=trial_overrides,
                stage_cfg=stage_cfg,
                trial_number=trial.number,
                stage_idx=stage_idx,
            )

            stage_start_time = time.perf_counter()
            run_result = train_sam3.remote(
                stage_config,
                fresh_run=(stage_idx == 1),
                resume_output_dir=resume_output_dir,
            )
            stage_runtime_hours = (time.perf_counter() - stage_start_time) / 3600.0
            cumulative_stage_runtime_hours += stage_runtime_hours
            resume_output_dir = run_result["output_dir"]
            stage_val_loss = read_latest_val_loss.remote(run_result["output_dir"])
            best_stage_val_loss = min(best_stage_val_loss, stage_val_loss)
            final_stage_val_loss = stage_val_loss

            trial.set_user_attr(f"stage_{stage_idx}_output_dir", run_result["output_dir"])
            trial.set_user_attr(f"stage_{stage_idx}_sample_percent", stage_cfg["sample_percent"])
            trial.set_user_attr(f"stage_{stage_idx}_num_epochs", stage_cfg["num_epochs"])
            trial.set_user_attr(f"stage_{stage_idx}_val_loss", stage_val_loss)
            trial.set_user_attr(f"stage_{stage_idx}_runtime_hours", stage_runtime_hours)
            try:
                stage_metrics = read_stage_quality_metrics.remote(run_result["output_dir"])
                metric_f1 = stage_metrics.get("f1")
                metric_quality = stage_metrics.get("composite")
                if isinstance(metric_f1, (int, float)):
                    trial.set_user_attr(f"stage_{stage_idx}_f1", float(metric_f1))
                if isinstance(metric_quality, (int, float)):
                    trial.set_user_attr(f"stage_{stage_idx}_quality_score", float(metric_quality))
            except Exception as exc:
                trial.set_user_attr(f"stage_{stage_idx}_metrics_error", str(exc))

            stage_score = stage_val_loss
            if normalized_mode == "cost_aware":
                stage_score = stage_val_loss + cost_penalty_per_gpu_hour * cumulative_stage_runtime_hours
            trial.set_user_attr(f"stage_{stage_idx}_objective_score", stage_score)
            trial.report(stage_score, step=stage_idx)
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"Pruned at stage {stage_idx}: val_loss={stage_val_loss:.6f}, "
                    f"objective={stage_score:.6f} "
                    f"(sample={stage_cfg['sample_percent']}%, epochs={stage_cfg['num_epochs']})"
                )

        trial_runtime_hours = (time.perf_counter() - trial_start_time) / 3600.0
        trial.set_user_attr("trial_runtime_hours", trial_runtime_hours)
        trial.set_user_attr("objective_mode", normalized_mode)
        trial.set_user_attr("cost_penalty_per_gpu_hour", cost_penalty_per_gpu_hour)

        if final_stage_val_loss is None:
            final_stage_val_loss = best_stage_val_loss

        trial.set_user_attr("final_stage_val_loss", float(final_stage_val_loss))

        if normalized_mode == "cost_aware":
            objective_score = float(final_stage_val_loss) + cost_penalty_per_gpu_hour * trial_runtime_hours
            trial.set_user_attr("objective_score", objective_score)
            return objective_score

        trial.set_user_attr("objective_score", float(final_stage_val_loss))
        return float(final_stage_val_loss)

    return objective


def _print_study_results(
    study: optuna.study.Study,
    rerank_top_fraction: float,
    objective_mode: str,
) -> None:
    """Print best-trial details plus optional IoU/F1 rerank summary."""
    normalized_mode = objective_mode.strip().lower()
    score_label = "objective_score" if normalized_mode == "cost_aware" else "val_loss"

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed_trials:
        sorted_by_score = sorted(completed_trials, key=lambda t: float(t.value))
        top_k = max(1, int(round(len(sorted_by_score) * max(0.01, min(rerank_top_fraction, 1.0)))))
        rerank_pool = sorted_by_score[:top_k]

        print("\nHybrid rerank (IoU/F1) over top-score trials:")
        print(f"  completed trials: {len(sorted_by_score)}")
        print(
            f"  rerank pool size: {top_k} "
            f"(top {max(0.01, min(rerank_top_fraction, 1.0)):.2f} by {score_label})"
        )

        rerank_rows = []
        for trial in rerank_pool:
            stage_output_dir = trial.user_attrs.get(f"stage_{NUM_RUNG_STAGES}_output_dir")
            if not isinstance(stage_output_dir, str):
                stage_keys = sorted(
                    [k for k in trial.user_attrs.keys() if k.startswith("stage_") and k.endswith("_output_dir")]
                )
                if stage_keys:
                    stage_output_dir = trial.user_attrs[stage_keys[-1]]

            if not isinstance(stage_output_dir, str):
                continue

            metrics = read_stage_quality_metrics.remote(stage_output_dir)
            rerank_rows.append(
                {
                    "trial_number": trial.number,
                    "objective_score": float(trial.value),
                    "val_loss": float(trial.user_attrs.get("final_stage_val_loss", trial.value)),
                    "trial_runtime_hours": trial.user_attrs.get("trial_runtime_hours"),
                    "output_dir": stage_output_dir,
                    "iou": metrics.get("iou"),
                    "f1": metrics.get("f1"),
                    "composite": metrics.get("composite"),
                }
            )

        scored_rows = [row for row in rerank_rows if isinstance(row["composite"], float)]
        if scored_rows:
            scored_rows.sort(key=lambda row: row["composite"], reverse=True)
            best_hybrid = scored_rows[0]
            print("  best hybrid trial:")
            print(f"    - trial_number: {best_hybrid['trial_number']}")
            print(f"    - composite(0.5*IoU+0.5*F1): {best_hybrid['composite']:.6f}")
            print(f"    - iou: {best_hybrid['iou']:.6f}")
            print(f"    - f1: {best_hybrid['f1']:.6f}")
            print(f"    - objective_score: {best_hybrid['objective_score']:.6f}")
            print(f"    - val_loss: {best_hybrid['val_loss']:.6f}")
            if isinstance(best_hybrid["trial_runtime_hours"], (int, float)):
                print(f"    - runtime_hours: {best_hybrid['trial_runtime_hours']:.3f}")
            print(f"    - output_dir: {best_hybrid['output_dir']}")
        else:
            print("  no IoU/F1 metrics found in rerank pool artifacts; defaulting to objective ranking.")

    if not completed_trials:
        print("\nNo completed trials found (all pruned/failed).")
        return

    print("\nBest trial:")
    print(f"  number: {study.best_trial.number}")
    print(f"  value ({score_label}): {study.best_trial.value:.6f}")
    best_val_loss = study.best_trial.user_attrs.get("final_stage_val_loss")
    best_runtime_hours = study.best_trial.user_attrs.get("trial_runtime_hours")
    if isinstance(best_val_loss, (int, float)):
        print(f"  final_stage_val_loss: {float(best_val_loss):.6f}")
    if isinstance(best_runtime_hours, (int, float)):
        print(f"  trial_runtime_hours: {float(best_runtime_hours):.3f}")
    print("  params:")
    for key, value in study.best_trial.params.items():
        print(f"    - {key}: {value}")
    print("  attrs:")
    for key, value in sorted(study.best_trial.user_attrs.items()):
        print(f"    - {key}: {value}")


@app.function(
    volumes={MODAL_ARTIFACTS_DIR: artifacts_vol},
    timeout=60 * 60 * 24 * 14,
)
def run_optuna_study(
    n_trials: int = 120,
    timeout_hours: float = 96,
    study_name: str = "sam3-final-optuna-asha",
    rerank_top_fraction: float = 0.25,
    parallel_workers: int = 10,
    sqlite_lock_timeout_sec: int = 60,
    pruner_type: str = "hyperband",
    objective_mode: str = "cost_aware",
    cost_penalty_per_gpu_hour: float = 0.03,
):
    """Run final-stage Optuna sweep (SHA/Hyperband) using shared SQLite on artifacts volume."""
    config_path = (
        Path(__file__).resolve().parent
        / "sam3_table"
        / "testSamples"
        / "full_lora_config.yaml"
    )
    base_config = SAM3LoRAConfig.from_yaml(config_path)
    parallel_workers = max(1, int(parallel_workers))
    sqlite_lock_timeout_sec = max(1, int(sqlite_lock_timeout_sec))
    cost_penalty_per_gpu_hour = max(0.0, float(cost_penalty_per_gpu_hour))
    normalized_objective_mode = objective_mode.strip().lower()
    if normalized_objective_mode not in {"loss_only", "cost_aware"}:
        raise ValueError(
            f"Unsupported objective_mode='{objective_mode}'. Use 'loss_only' or 'cost_aware'."
        )

    optuna_dir = Path(MODAL_ARTIFACTS_DIR) / "optuna"
    optuna_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = optuna_dir / f"{study_name}.db"

    with sqlite3.connect(str(sqlite_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.commit()

    storage_url = f"sqlite:///{sqlite_path.as_posix()}?timeout={sqlite_lock_timeout_sec}"

    sampler = optuna.samplers.TPESampler(n_startup_trials=24, multivariate=True, seed=42)
    normalized_pruner = pruner_type.strip().lower()
    if normalized_pruner == "sha":
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=REDUCTION_FACTOR,
            min_early_stopping_rate=0,
        )
    elif normalized_pruner == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=NUM_RUNG_STAGES,
            reduction_factor=REDUCTION_FACTOR,
        )
    else:
        raise ValueError(f"Unsupported pruner_type='{pruner_type}'. Use 'sha' or 'hyperband'.")
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        load_if_exists=True,
    )

    print(f"Starting Optuna study '{study_name}'")
    print(f"Optuna SQLite storage: {sqlite_path}")
    print(f"Pruner type: {normalized_pruner}")
    print(f"Objective mode: {normalized_objective_mode}")
    if normalized_objective_mode == "cost_aware":
        print(f"Cost penalty per GPU-hour: {cost_penalty_per_gpu_hour:.4f}")
    print(f"ASHA promotion fraction: {PROMOTION_FRACTION:.2f} (reduction_factor={REDUCTION_FACTOR})")
    print(f"Rungs per trial: {NUM_RUNG_STAGES}")
    print(f"Total requested trials: {n_trials}")
    print(f"Parallel Optuna workers: {parallel_workers}")

    objective = _objective_factory(
        base_config,
        objective_mode=normalized_objective_mode,
        cost_penalty_per_gpu_hour=cost_penalty_per_gpu_hour,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=int(timeout_hours * 3600),
        gc_after_trial=True,
        show_progress_bar=True,
        n_jobs=max(1, parallel_workers),
    )

    _print_study_results(
        study,
        rerank_top_fraction=rerank_top_fraction,
        objective_mode=normalized_objective_mode,
    )


@app.local_entrypoint()
def main(
    n_trials: int = 120,
    timeout_hours: float = 96,
    study_name: str = "sam3-final-optuna-asha",
    rerank_top_fraction: float = 0.25,
    parallel_workers: int = 10,
    sqlite_lock_timeout_sec: int = 60,
    pruner_type: str = "hyperband",
    objective_mode: str = "cost_aware",
    cost_penalty_per_gpu_hour: float = 0.02,
):
    run_optuna_study.remote(
        n_trials=n_trials,
        timeout_hours=timeout_hours,
        study_name=study_name,
        rerank_top_fraction=rerank_top_fraction,
        parallel_workers=parallel_workers,
        sqlite_lock_timeout_sec=sqlite_lock_timeout_sec,
        pruner_type=pruner_type,
        objective_mode=objective_mode,
        cost_penalty_per_gpu_hour=cost_penalty_per_gpu_hour,
    )
