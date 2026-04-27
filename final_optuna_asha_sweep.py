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
from typing import Any

import modal
import optuna

from sam3_table.cstone_train_sam3 import (
    MODAL_ARTIFACTS_DIR,
    app,
    artifacts_vol,
    train_sam3,
)
from sam3_table.training_config import SAM3LoRAConfig

# Checkpoint filenames written by `train_sam3`. Kept in sync with
# `sam3_table.cstone_train_sam3.CHECKPOINT_NAMES` -- duplicated here so the
# salvage helper below doesn't need to import a private-ish module constant.
_RESUMABLE_CHECKPOINT_FILES = (
    "checkpoint_epoch.pt",
    "checkpoint_best.pt",
    "checkpoint_signal.pt",
)


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base (mutates and returns base)."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _commit_artifacts_volume_safely(context: str) -> None:
    """Commit the artifacts volume so other containers (e.g. a live leader
    monitor / benchmark) see the latest Optuna sqlite + checkpoint state.

    Never raises: if the commit fails (transient network hiccup, etc.) we
    log and continue so a sweep isn't killed by an ancillary sync step.
    """
    try:
        artifacts_vol.commit()
    except Exception as exc:  # pragma: no cover - best-effort sync
        print(f"[sweep] artifacts_vol.commit() failed ({context}): {exc}")


def _close_optuna_storage(study: "optuna.study.Study | None") -> None:
    """Dispose Optuna's SQLAlchemy engine so the underlying sqlite file
    is no longer held open.

    Modal's ``Volume.reload()`` refuses to operate while any file in the
    volume is open. After ``study.optimize()`` returns, Optuna's engine
    still has ``/artifacts/optuna/<study>.db`` open, which triggers
    ``ConflictError("there are open files preventing the operation: path
    optuna/<study>.db is open")``. ``engine.dispose()`` closes all pooled
    connections; later reads from the same ``study`` object will silently
    open fresh connections (which is fine -- by then the reload has
    completed).
    """
    if study is None:
        return
    import gc

    storage = getattr(study, "_storage", None)
    if storage is None:
        return

    engine = None
    for attr_chain in (
        ("engine",),
        ("_backend", "engine"),
        ("scoped_session", "bind"),
    ):
        obj = storage
        try:
            for attr in attr_chain:
                obj = getattr(obj, attr)
            engine = obj
            break
        except AttributeError:
            continue
    if engine is not None and hasattr(engine, "dispose"):
        try:
            engine.dispose()
        except Exception as exc:  # pragma: no cover - best-effort sync
            print(f"[sweep] optuna engine.dispose() failed: {exc}")
    if hasattr(storage, "remove_session"):
        try:
            storage.remove_session()
        except Exception:  # pragma: no cover - best-effort sync
            pass
    gc.collect()


def _commit_and_reload_artifacts(
    context: str,
    study: "optuna.study.Study | None" = None,
) -> None:
    """Commit pending writes, dispose any open Optuna sqlite handles, then
    reload the volume. Reload failures are logged and swallowed so a rare
    open-file edge case never crashes the sweep.
    """
    _commit_artifacts_volume_safely(context)
    _close_optuna_storage(study)
    try:
        artifacts_vol.reload()
    except Exception as exc:  # pragma: no cover - best-effort sync
        print(f"[sweep] artifacts_vol.reload() skipped ({context}): {exc}")


def _hashable_params(params: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Stable, hashable representation of an Optuna ``trial.params`` dict.

    Used to key salvage hints so we can match a re-enqueued trial back to
    its original orphaned RUNNING trial. Optuna's ``enqueue_trial(params)``
    injects exact param values, so float equality is safe here.
    """
    return tuple(sorted(params.items(), key=lambda kv: kv[0]))


def _salvage_orphaned_running_trials(
    study: "optuna.study.Study",
    num_rung_stages: int,
) -> dict[tuple[tuple[str, Any], ...], dict[str, Any]]:
    """Recover trials stuck in ``RUNNING`` state from a previous aborted run.

    When the parent sweep container is killed mid-trial (Modal budget
    cap, host preemption, manual stop, container OOM, ...), any trial
    that was actively running ends up frozen in ``RUNNING`` inside the
    Optuna sqlite DB. On the next sweep launch this function:

    1. Finds those orphaned trials.
    2. Marks them ``FAIL`` so the study state is consistent and Optuna's
       progress accounting is correct.
    3. Re-enqueues a fresh trial with the **same hyperparameters** via
       :py:meth:`optuna.study.Study.enqueue_trial`. The next call to
       ``study.optimize()`` will sample those enqueued params first.
    4. Returns a ``{params_key: {resume_output_dir, last_recorded_stage,
       original_trial_number}}`` mapping. The objective uses this to point
       ``train_sam3.remote(... resume_output_dir=...)`` at the last
       committed checkpoint dir, so the salvaged trial resumes training
       from the last completed epoch instead of restarting from epoch 0.

    Returns an empty dict when there are no orphans.
    """
    if num_rung_stages < 1:
        return {}

    salvage_hints: dict[tuple[tuple[str, Any], ...], dict[str, Any]] = {}
    salvaged_count = 0
    for trial in list(study.trials):
        if trial.state != optuna.trial.TrialState.RUNNING:
            continue

        # Find the most recent stage that recorded an output_dir; that's
        # the dir train_sam3 last wrote checkpoints into for this trial.
        last_recorded_stage = 0
        last_output_dir: str | None = None
        for s in range(num_rung_stages, 0, -1):
            recorded = trial.user_attrs.get(f"stage_{s}_output_dir")
            if isinstance(recorded, str) and recorded:
                last_recorded_stage = s
                last_output_dir = recorded
                break

        # Verify the recorded dir actually has a resumable checkpoint
        # right now. If not, fall back to a clean restart for this trial.
        if last_output_dir is not None:
            checkpoint_present = any(
                (Path(last_output_dir) / name).exists()
                for name in _RESUMABLE_CHECKPOINT_FILES
            )
            if not checkpoint_present:
                last_output_dir = None
                last_recorded_stage = 0

        # Mark the orphan as FAIL via storage internals -- Optuna's public
        # API doesn't expose a "set state on existing trial" call, but
        # `_storage.set_trial_state_values` is stable and used widely.
        try:
            study._storage.set_trial_state_values(
                trial._trial_id,
                state=optuna.trial.TrialState.FAIL,
            )
        except Exception as exc:
            print(
                f"[sweep][salvage] failed to mark trial {trial.number} FAIL: {exc}"
            )

        # Re-enqueue a fresh trial with the same hyperparameters.
        # `skip_if_exists=False` is required because the trial we just
        # marked FAIL has these exact params already in the DB.
        try:
            study.enqueue_trial(trial.params, skip_if_exists=False)
        except Exception as exc:
            print(
                f"[sweep][salvage] failed to enqueue salvage trial for "
                f"orphan {trial.number}: {exc}"
            )
            continue

        params_key = _hashable_params(trial.params)
        salvage_hints[params_key] = {
            "original_trial_number": trial.number,
            "last_recorded_stage": last_recorded_stage,
            "resume_output_dir": last_output_dir,
        }
        salvaged_count += 1
        print(
            f"[sweep][salvage] orphan trial {trial.number}: marked FAIL, "
            f"re-enqueued with identical params. Last completed stage: "
            f"{last_recorded_stage}, resume from: {last_output_dir or '<none>'}"
        )

    if salvaged_count:
        print(
            f"[sweep][salvage] re-enqueued {salvaged_count} orphaned trial(s) "
            "from a previous aborted run."
        )
    else:
        print("[sweep][salvage] no orphaned RUNNING trials detected.")
    return salvage_hints

# ASHA: keep the best 1/5 each rung.
PROMOTION_FRACTION = 0.20
REDUCTION_FACTOR = int(round(1 / PROMOTION_FRACTION))

# 500k-image regime: aggressively cheap early rungs, full budget only at the end.
SAMPLE_SCHEDULE = [1.0, 3.0, 10.0, 30.0, 100.0]
EPOCH_RATIO_SCHEDULE = [1.0 / 14.0, 2.0 / 14.0, 4.0 / 14.0, 8.0 / 14.0, 1.0]
# Use smaller validation subsets for early rungs to reduce evaluation cost.
VALID_SAMPLE_SCHEDULE = [2.0, 5.0, 10.0, 25.0, 100.0]
# Avoid expensive full-validation every epoch in later stages.
# Validation still always runs on the final epoch in trainer.
EVAL_EVERY_N_EPOCHS_SCHEDULE = [1, 1, 2, 3, 4]
# Disable step-based validation inside an epoch for sweep speed.
SWEEP_EVAL_STEPS = 10_000_000
NUM_RUNG_STAGES = len(SAMPLE_SCHEDULE)


def _build_stage_schedule(max_epochs: int) -> list[dict]:
    """Create progressively larger sample/epoch budgets by rung."""
    if (
        len(SAMPLE_SCHEDULE) != len(EPOCH_RATIO_SCHEDULE)
        or len(SAMPLE_SCHEDULE) != len(VALID_SAMPLE_SCHEDULE)
        or len(SAMPLE_SCHEDULE) != len(EVAL_EVERY_N_EPOCHS_SCHEDULE)
    ):
        raise ValueError(
            "SAMPLE_SCHEDULE, EPOCH_RATIO_SCHEDULE, VALID_SAMPLE_SCHEDULE, and "
            "EVAL_EVERY_N_EPOCHS_SCHEDULE must have same length"
        )

    stages: list[dict] = []
    prev_epochs = 0
    for sample_percent, valid_sample_percent, eval_every_n_epochs, ratio in zip(
        SAMPLE_SCHEDULE,
        VALID_SAMPLE_SCHEDULE,
        EVAL_EVERY_N_EPOCHS_SCHEDULE,
        EPOCH_RATIO_SCHEDULE,
    ):
        epoch_budget = max(1, int(round(max_epochs * ratio)))
        epoch_budget = max(epoch_budget, prev_epochs + 1)
        stages.append(
            {
                "sample_percent": sample_percent,
                "valid_sample_percent": valid_sample_percent,
                "eval_every_n_epochs": eval_every_n_epochs,
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
                "data": {
                    "sample_percent": stage_cfg["sample_percent"],
                    "valid_sample_percent": stage_cfg["valid_sample_percent"],
                    "valid_sample_seed": 42,
                },
                "num_epochs": stage_cfg["num_epochs"],
                "eval_steps": SWEEP_EVAL_STEPS,
                "eval_every_n_epochs": stage_cfg["eval_every_n_epochs"],
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
    salvage_hints: dict[tuple[tuple[str, Any], ...], dict[str, Any]] | None = None,
):
    salvage_hints = dict(salvage_hints or {})

    def objective(trial: optuna.trial.Trial) -> float:
        trial_overrides, max_epochs = _suggest_trial_overrides(trial)
        stage_schedule = _build_stage_schedule(max_epochs=max_epochs)
        normalized_mode = objective_mode.strip().lower()
        trial_start_time = time.perf_counter()
        cumulative_stage_runtime_hours = 0.0

        # If this trial's params match a salvage hint from a previous run
        # that was killed mid-flight, point stage 1 at the last committed
        # checkpoint dir so train_sam3 resumes from the last epoch instead
        # of starting over.
        params_key = _hashable_params(trial.params)
        salvage = salvage_hints.pop(params_key, None)
        if salvage is not None:
            trial.set_user_attr("salvage_origin_trial_number", salvage["original_trial_number"])
            trial.set_user_attr("salvage_last_recorded_stage", salvage["last_recorded_stage"])
            trial.set_user_attr("salvage_resume_output_dir", salvage["resume_output_dir"])
            print(
                f"[sweep][salvage] trial {trial.number} is a salvage of "
                f"{salvage['original_trial_number']}; resume_dir="
                f"{salvage['resume_output_dir'] or '<none>'}"
            )

        best_stage_val_loss = float("inf")
        final_stage_val_loss = None
        resume_output_dir = (
            salvage["resume_output_dir"] if salvage is not None else None
        )
        for stage_idx, stage_cfg in enumerate(stage_schedule, start=1):
            stage_config = _build_stage_config(
                base_config=base_config,
                trial_overrides=trial_overrides,
                stage_cfg=stage_cfg,
                trial_number=trial.number,
                stage_idx=stage_idx,
            )

            stage_start_time = time.perf_counter()
            # `fresh_run` only applies on stage 1, AND only when we don't
            # have a salvage checkpoint to resume from. With a resume hint
            # we want train_sam3 to attach to the existing run dir.
            run_result = train_sam3.remote(
                stage_config,
                fresh_run=(stage_idx == 1) and (resume_output_dir is None),
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
            trial.set_user_attr(
                f"stage_{stage_idx}_valid_sample_percent",
                stage_cfg["valid_sample_percent"],
            )
            trial.set_user_attr(
                f"stage_{stage_idx}_eval_every_n_epochs",
                stage_cfg["eval_every_n_epochs"],
            )
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

            # Flush this trial's user_attrs + intermediate value to the
            # shared volume so a live "describe current leader" / benchmark
            # caller can see this stage's result immediately, even though
            # the sweep may still be running for days.
            _commit_artifacts_volume_safely(
                f"trial={trial.number}, end of stage {stage_idx}"
            )

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
        sorted_by_score = sorted(
            completed_trials,
            key=lambda t: float(t.value) if t.value is not None else float("inf"),
        )
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
                    "objective_score": (
                        float(trial.value) if trial.value is not None else float("inf")
                    ),
                    "val_loss": (
                        float(trial.user_attrs["final_stage_val_loss"])
                        if isinstance(trial.user_attrs.get("final_stage_val_loss"), (int, float))
                        else (
                            float(trial.value) if trial.value is not None else float("inf")
                        )
                    ),
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
    base_config_dict: dict[str, object] | None = None,
    salvage_orphaned_trials: bool = True,
):
    """Run final-stage Optuna sweep (SHA/Hyperband) using shared SQLite on artifacts volume."""
    if base_config_dict is not None:
        base_config = SAM3LoRAConfig.model_validate(base_config_dict)
    else:
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

    # Recover any trials stuck in RUNNING state from a previously aborted
    # sweep (Modal budget cap, host preemption, manual stop, ...). The
    # returned hints tell the objective to point train_sam3 at the last
    # committed checkpoint dir so the salvaged trial picks up from the last
    # epoch rather than restarting from scratch.
    salvage_hints: dict[tuple[tuple[str, Any], ...], dict[str, Any]] = {}
    if salvage_orphaned_trials:
        salvage_hints = _salvage_orphaned_running_trials(
            study, num_rung_stages=NUM_RUNG_STAGES
        )
        # Persist the salvage decisions before launching new training so a
        # second crash mid-salvage doesn't lose them.
        _commit_artifacts_volume_safely("post-salvage state")
    else:
        print("[sweep][salvage] disabled by --no-salvage-orphaned-trials")

    objective = _objective_factory(
        base_config,
        objective_mode=normalized_objective_mode,
        cost_penalty_per_gpu_hour=cost_penalty_per_gpu_hour,
        salvage_hints=salvage_hints,
    )

    def _commit_after_trial(
        _study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
    ) -> None:
        """Optuna callback that flushes the artifacts volume after each
        trial finishes (COMPLETE / PRUNED / FAIL).

        This is what enables the "extract top-performing model and
        benchmark it at any time" workflow: a live monitor running e.g.
        ``modal run eval_tablebank.py --use-current-leader`` against the
        same ``artifacts-vol`` will see the most recently finalized trial
        as soon as this callback returns.
        """
        _commit_artifacts_volume_safely(
            f"trial={trial.number} state={trial.state.name}"
        )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=int(timeout_hours * 3600),
        gc_after_trial=True,
        show_progress_bar=True,
        n_jobs=max(1, parallel_workers),
        callbacks=[_commit_after_trial],
    )

    # Final sync at the end of the sweep: commit so any downstream reader
    # (deployed `describe_current_leader`, ad-hoc scripts, etc.) sees the
    # finalized state of the study, dispose the Optuna sqlite handle so
    # the volume reload below doesn't trip over open files, then reload so
    # any in-process logic that inspects the filesystem after this point
    # can see checkpoints committed from parallel training containers.
    _commit_and_reload_artifacts("end of run_optuna_study", study=study)

    _print_study_results(
        study,
        rerank_top_fraction=rerank_top_fraction,
        objective_mode=normalized_objective_mode,
    )


@app.function(
    volumes={MODAL_ARTIFACTS_DIR: artifacts_vol},
    timeout=60 * 10,
)
def describe_leader_from_study(
    study_name: str = "sam3-final-optuna-asha",
    include_running_trials: bool = False,
    include_pruned_trials: bool = True,
    sqlite_lock_timeout_sec: int = 60,
    num_rung_stages: int = NUM_RUNG_STAGES,
    require_weights_on_disk: bool = True,
) -> dict[str, Any]:
    """Self-contained leader extraction.

    Reads ``/artifacts/optuna/<study_name>.db`` directly and returns the
    best trial's weights path + hyperparameters. **Does not** depend on
    the ``tablebank-eval`` app being deployed -- this is the "give me the
    leader without running anything through eval_tablebank" path.

    Selection rules (for ``direction="minimize"``):
      * Always considers ``COMPLETE`` trials.
      * Optionally considers ``PRUNED`` trials (their last reported
        intermediate value is used as the score).
      * Optionally considers ``RUNNING`` trials (same; useful while a
        sweep is in progress).
      * Picks the trial with the lowest score whose stage output
        directory contains a ``best_lora_weights.pt`` (or
        ``last_lora_weights.pt``) file -- unless
        ``require_weights_on_disk=False``, in which case it advertises
        the conventional path even if not yet visible.

    Returns a JSON-shaped dict with the leader info plus enough metadata
    to drive a downstream evaluation or model-loading step.
    """
    artifacts_vol.reload()
    sqlite_lock_timeout_sec = max(1, int(sqlite_lock_timeout_sec))
    num_rung_stages = max(1, int(num_rung_stages))

    sqlite_path = Path(MODAL_ARTIFACTS_DIR) / "optuna" / f"{study_name}.db"
    if not sqlite_path.exists():
        raise FileNotFoundError(
            f"Optuna study DB not found at {sqlite_path}. "
            "Make sure a sweep with this study_name has started and committed."
        )

    with sqlite3.connect(str(sqlite_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.commit()

    storage_url = f"sqlite:///{sqlite_path.as_posix()}?timeout={sqlite_lock_timeout_sec}"
    study = optuna.load_study(study_name=study_name, storage=storage_url)

    candidates: list[dict[str, Any]] = []
    state_counts: dict[str, int] = {}
    for trial in study.trials:
        state_counts[trial.state.name] = state_counts.get(trial.state.name, 0) + 1

        score: float | None = None
        if (
            trial.state == optuna.trial.TrialState.COMPLETE
            and trial.value is not None
        ):
            score = float(trial.value)
        elif (
            include_pruned_trials
            and trial.state == optuna.trial.TrialState.PRUNED
            and trial.intermediate_values
        ):
            latest_step = max(trial.intermediate_values.keys())
            score = float(trial.intermediate_values[latest_step])
        elif (
            include_running_trials
            and trial.state == optuna.trial.TrialState.RUNNING
            and trial.intermediate_values
        ):
            latest_step = max(trial.intermediate_values.keys())
            score = float(trial.intermediate_values[latest_step])

        if score is None:
            continue

        # Resolve the stage output dir for this trial.
        stage_output_dir = trial.user_attrs.get(f"stage_{num_rung_stages}_output_dir")
        if not isinstance(stage_output_dir, str):
            stage_keys = sorted(
                k
                for k in trial.user_attrs.keys()
                if k.startswith("stage_") and k.endswith("_output_dir")
            )
            if stage_keys:
                stage_output_dir = trial.user_attrs[stage_keys[-1]]
        if not isinstance(stage_output_dir, str):
            continue

        weights_candidates = [
            Path(stage_output_dir) / "best_lora_weights.pt",
            Path(stage_output_dir) / "last_lora_weights.pt",
        ]
        visible_weights = next(
            (p for p in weights_candidates if p.exists()), None
        )
        if visible_weights is None and require_weights_on_disk:
            continue
        weights_path_str = (
            str(visible_weights)
            if visible_weights is not None
            else str(weights_candidates[0])
        )

        candidates.append(
            {
                "trial_number": trial.number,
                "state": trial.state.name,
                "score": score,
                "output_dir": stage_output_dir,
                "weights_path": weights_path_str,
                "weights_visible_on_disk": visible_weights is not None,
                "objective_mode": trial.user_attrs.get("objective_mode"),
                "final_stage_val_loss": trial.user_attrs.get("final_stage_val_loss"),
                "trial_runtime_hours": trial.user_attrs.get("trial_runtime_hours"),
                "salvage_origin_trial_number": trial.user_attrs.get(
                    "salvage_origin_trial_number"
                ),
                "hyperparameters": dict(trial.params),
            }
        )

    if not candidates:
        return {
            "study_name": study_name,
            "sqlite_path": str(sqlite_path),
            "total_trials": len(study.trials),
            "trial_state_counts": state_counts,
            "leader_trial_number": None,
            "leader_state": None,
            "leader_score": None,
            "weights_path": None,
            "output_dir": None,
            "hyperparameters": {},
            "reason": (
                "No trial with a usable score and weights path was found. "
                "Wait for at least one stage to finish, or relax the filters "
                "via include_running_trials / require_weights_on_disk."
            ),
        }

    best = min(candidates, key=lambda c: c["score"])
    return {
        "study_name": study_name,
        "sqlite_path": str(sqlite_path),
        "total_trials": len(study.trials),
        "trial_state_counts": state_counts,
        "leader_trial_number": best["trial_number"],
        "leader_state": best["state"],
        "leader_score": best["score"],
        "weights_path": best["weights_path"],
        "weights_visible_on_disk": best["weights_visible_on_disk"],
        "output_dir": best["output_dir"],
        "objective_mode": best["objective_mode"],
        "final_stage_val_loss": best["final_stage_val_loss"],
        "trial_runtime_hours": best["trial_runtime_hours"],
        "salvage_origin_trial_number": best["salvage_origin_trial_number"],
        "hyperparameters": best["hyperparameters"],
    }


@app.local_entrypoint()
def main(
    n_trials: int = 120,
    timeout_hours: float = 120,
    study_name: str = "sam3-final-optuna-asha",
    rerank_top_fraction: float = 0.25,
    parallel_workers: int = 10,
    sqlite_lock_timeout_sec: int = 60,
    pruner_type: str = "hyperband",
    objective_mode: str = "cost_aware",
    cost_penalty_per_gpu_hour: float = 0.02,
    auto_deploy_tablebank_eval: bool = True,
    salvage_orphaned_trials: bool = True,
):
    config_path = (
        Path(__file__).resolve().parent
        / "sam3_table"
        / "testSamples"
        / "full_lora_config.yaml"
    )
    base_config = SAM3LoRAConfig.from_yaml(config_path)

    # Auto-deploy the `tablebank-eval` Modal app so any downstream leader
    # extraction / evaluation step can resolve it via `Function.from_name`
    # without requiring users to manually run `modal deploy eval_tablebank.py`.
    if auto_deploy_tablebank_eval:
        from eval_tablebank import ensure_deployed as _ensure_tablebank_eval_deployed

        _ensure_tablebank_eval_deployed()

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
        base_config_dict=base_config.model_dump(mode="json"),
        salvage_orphaned_trials=salvage_orphaned_trials,
    )


@app.local_entrypoint()
def show_current_leader(
    study_name: str = "sam3-final-optuna-asha",
    include_running_trials: bool = False,
    include_pruned_trials: bool = True,
    sqlite_lock_timeout_sec: int = 60,
    require_weights_on_disk: bool = True,
):
    """Print the current best trial in this study as JSON.

    Self-contained -- reads the Optuna sqlite DB on ``artifacts-vol``
    directly and does not require ``tablebank-eval`` to be deployed.

    Examples:
        modal run final_optuna_asha_sweep.py::show_current_leader
        modal run final_optuna_asha_sweep.py::show_current_leader \\
            --study-name sam3-final-optuna-asha-v2
        modal run final_optuna_asha_sweep.py::show_current_leader \\
            --include-running-trials
    """
    result = describe_leader_from_study.remote(
        study_name=study_name,
        include_running_trials=include_running_trials,
        include_pruned_trials=include_pruned_trials,
        sqlite_lock_timeout_sec=sqlite_lock_timeout_sec,
        num_rung_stages=NUM_RUNG_STAGES,
        require_weights_on_disk=require_weights_on_disk,
    )
    print(json.dumps(result, indent=2, default=str))


@app.function(
    volumes={MODAL_ARTIFACTS_DIR: artifacts_vol},
    timeout=60 * 10,
)
def _read_leader_weights_bytes(
    study_name: str = "sam3-final-optuna-asha",
    include_running_trials: bool = False,
    include_pruned_trials: bool = True,
    sqlite_lock_timeout_sec: int = 60,
) -> dict[str, Any]:
    """Resolve the current leader and return its weights file as bytes
    plus minimal metadata. Used by :func:`download_leader_weights`.
    """
    leader = describe_leader_from_study.local(
        study_name=study_name,
        include_running_trials=include_running_trials,
        include_pruned_trials=include_pruned_trials,
        sqlite_lock_timeout_sec=sqlite_lock_timeout_sec,
        num_rung_stages=NUM_RUNG_STAGES,
        require_weights_on_disk=True,
    )
    weights_path = leader.get("weights_path")
    if not isinstance(weights_path, str) or not Path(weights_path).exists():
        raise FileNotFoundError(
            f"No leader weights file found for study '{study_name}'. "
            f"Got leader info: {leader}"
        )
    data = Path(weights_path).read_bytes()
    return {
        "leader_trial_number": leader["leader_trial_number"],
        "leader_score": leader["leader_score"],
        "remote_weights_path": weights_path,
        "weights_bytes": data,
        "hyperparameters": leader["hyperparameters"],
    }


@app.local_entrypoint()
def download_leader_weights(
    study_name: str = "sam3-final-optuna-asha",
    output_path: str = "leader_weights.pt",
    include_running_trials: bool = False,
    include_pruned_trials: bool = True,
    sqlite_lock_timeout_sec: int = 60,
):
    """Download the current leader's LoRA weights file to a local path.

    Self-contained -- does not require ``tablebank-eval`` to be deployed.

    Examples:
        modal run final_optuna_asha_sweep.py::download_leader_weights
        modal run final_optuna_asha_sweep.py::download_leader_weights \\
            --study-name sam3-final-optuna-asha-v2 \\
            --output-path ./best_lora_weights.pt
    """
    payload = _read_leader_weights_bytes.remote(
        study_name=study_name,
        include_running_trials=include_running_trials,
        include_pruned_trials=include_pruned_trials,
        sqlite_lock_timeout_sec=sqlite_lock_timeout_sec,
    )
    local_target = Path(output_path).resolve()
    local_target.parent.mkdir(parents=True, exist_ok=True)
    local_target.write_bytes(payload["weights_bytes"])
    summary = {
        "study_name": study_name,
        "leader_trial_number": payload["leader_trial_number"],
        "leader_score": payload["leader_score"],
        "remote_weights_path": payload["remote_weights_path"],
        "local_weights_path": str(local_target),
        "bytes_written": len(payload["weights_bytes"]),
        "hyperparameters": payload["hyperparameters"],
    }
    print(json.dumps(summary, indent=2, default=str))
