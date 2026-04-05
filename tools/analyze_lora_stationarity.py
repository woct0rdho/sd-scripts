import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from safetensors.torch import load_file
from tqdm import tqdm

from library.utils import setup_logging


setup_logging()

import logging


logger = logging.getLogger(__name__)


LOSS_KEYS = ("loss/current", "loss/average")
EFF_LR_KEY = "lr/d*eff_lr/unet"
STEP_KEYS = ("global_step", "_step")
DOWN_SUFFIX = ".lora_down.weight"
UP_SUFFIX = ".lora_up.weight"
ALPHA_SUFFIX = ".alpha"
WEIGHT_SUFFIX = ".weight"
CHECKPOINT_STEP_RE = re.compile(r"step(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate whether a LoRA run is approaching a stationary regime from "
            "W&B history and saved checkpoints."
        )
    )
    parser.add_argument(
        "--wandb-run",
        type=str,
        default=None,
        help="Path to a local .wandb run file.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing saved LoRA checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-glob",
        type=str,
        default="*.safetensors",
        help="Glob used inside --checkpoint-dir.",
    )
    parser.add_argument(
        "--loss-windows",
        type=int,
        nargs="+",
        default=[250, 500, 1000, 2000],
        help="Trailing history windows, in logged points, used for plateau analysis.",
    )
    parser.add_argument(
        "--tail-checkpoints",
        type=int,
        default=5,
        help="How many recent checkpoints to use for drift extrapolation.",
    )
    parser.add_argument(
        "--drift-thresholds",
        type=float,
        nargs="+",
        default=[0.10, 0.08, 0.05],
        help="Effective drift thresholds to project future crossing steps for.",
    )
    parser.add_argument(
        "--top-modules",
        type=int,
        default=10,
        help="How many fastest-moving LoRA modules to print for the last checkpoint interval.",
    )
    parser.add_argument(
        "--skip-spectrum",
        action="store_true",
        help="Skip SVD-based spectrum metrics to run faster.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the result as JSON instead of a human-readable report.",
    )
    return parser.parse_args()


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def read_wandb_history(wandb_run_path: Path) -> List[Dict[str, Any]]:
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal import datastore

    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_run_path))

    rows: List[Dict[str, Any]] = []
    while True:
        data = ds.scan_data()
        if data is None:
            break

        rec = wandb_internal_pb2.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") != "history":
            continue

        row: Dict[str, Any] = {}
        for item in rec.history.item:
            key = item.key if item.key else ".".join(item.nested_key)
            try:
                row[key] = json.loads(item.value_json)
            except Exception:
                row[key] = item.value_json
        rows.append(row)

    return rows


def extract_series(rows: Sequence[Dict[str, Any]], key: str) -> List[Tuple[int, float]]:
    series: List[Tuple[int, float]] = []
    for row in rows:
        step = None
        for step_key in STEP_KEYS:
            if step_key in row:
                step = _safe_float(row[step_key])
                break
        value = _safe_float(row.get(key))
        if step is None or value is None:
            continue
        series.append((int(step), value))
    return series


def linear_regression(xs: Sequence[float], ys: Sequence[float]) -> Dict[str, float]:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) < 2:
        raise ValueError("need at least two points for linear regression")

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    denom = max(1, len(xs) - 2)
    residual_std = math.sqrt(sum(r * r for r in residuals) / denom)
    return {
        "intercept": intercept,
        "slope": slope,
        "residual_std": residual_std,
    }


def compute_window_stats(series: Sequence[Tuple[int, float]], windows: Sequence[int]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for window in windows:
        if len(series) < max(2, window):
            continue

        tail = list(series[-window:])
        xs = [point[0] for point in tail]
        ys = [point[1] for point in tail]
        fit = linear_regression(xs, ys)
        total_change = fit["slope"] * (xs[-1] - xs[0])
        mean_value = sum(ys) / len(ys)
        std_value = statistics.pstdev(ys) if len(ys) > 1 else 0.0
        plateau_ratio = abs(total_change) / max(fit["residual_std"], 1e-12)
        stats[str(window)] = {
            "points": len(tail),
            "start_step": xs[0],
            "end_step": xs[-1],
            "last_value": ys[-1],
            "mean_value": mean_value,
            "std_value": std_value,
            "cv": std_value / max(abs(mean_value), 1e-12),
            "slope_per_step": fit["slope"],
            "predicted_change": total_change,
            "residual_std": fit["residual_std"],
            "plateau_ratio": plateau_ratio,
        }
    return stats


def checkpoint_sort_key(path: Path) -> Tuple[int, str]:
    match = CHECKPOINT_STEP_RE.search(path.name)
    if match:
        return int(match.group(1)), path.name
    return -1, path.name


def get_checkpoint_step(path: Path) -> Optional[int]:
    match = CHECKPOINT_STEP_RE.search(path.name)
    if match:
        return int(match.group(1))
    return None


def iter_lora_prefixes(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    return sorted(key[: -len(DOWN_SUFFIX)] for key in state_dict if key.endswith(DOWN_SUFFIX))


def get_scale(state_dict: Dict[str, torch.Tensor], prefix: str, rank: int) -> float:
    alpha_key = prefix + ALPHA_SUFFIX
    if alpha_key not in state_dict:
        return 1.0
    alpha = float(state_dict[alpha_key].float().item())
    return alpha / rank


def get_lora_factors(state_dict: Dict[str, torch.Tensor], prefix: str) -> Tuple[torch.Tensor, torch.Tensor, float]:
    down = state_dict[prefix + DOWN_SUFFIX].float()
    up = state_dict[prefix + UP_SUFFIX].float()
    scale = get_scale(state_dict, prefix, down.shape[0])
    return up, down, scale


def low_rank_fro_sq(up: torch.Tensor, down: torch.Tensor, scale: float) -> float:
    # ||scale * (up @ down)||_F^2 = scale^2 * tr((up^T up)(down down^T))
    gram_up = up.T @ up
    gram_down = down @ down.T
    return (scale * scale) * float(torch.trace(gram_up @ gram_down).item())


def low_rank_inner(
    up_a: torch.Tensor,
    down_a: torch.Tensor,
    scale_a: float,
    up_b: torch.Tensor,
    down_b: torch.Tensor,
    scale_b: float,
) -> float:
    # <scale_a * (up_a @ down_a), scale_b * (up_b @ down_b)>
    # = scale_a scale_b tr((up_a^T up_b)(down_b down_a^T))
    cross_up = up_a.T @ up_b
    cross_down = down_b @ down_a.T
    return (scale_a * scale_b) * float(torch.trace(cross_up @ cross_down).item())


def effective_singular_values(state_dict: Dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    up, down, scale = get_lora_factors(state_dict, prefix)
    scale = abs(scale)

    _, r_up = torch.linalg.qr(up, mode="reduced")
    _, r_down = torch.linalg.qr(down.T, mode="reduced")
    small = r_up @ r_down.T
    singular_values = torch.linalg.svdvals(small)
    if scale != 1.0:
        singular_values = singular_values * scale
    return singular_values


def compute_effective_rank(singular_values: torch.Tensor) -> Tuple[Optional[float], Optional[float]]:
    if singular_values.numel() == 0:
        return None, None
    energy = singular_values.pow(2)
    total_energy = float(energy.sum().item())
    if total_energy <= 0.0:
        return None, None

    prob = energy / total_energy
    entropy = -(prob * (prob + 1e-12).log()).sum().item()
    effective_rank = math.exp(entropy)
    top1_share = float((energy.max() / total_energy).item())
    return effective_rank, top1_share


def analyze_checkpoints(
    checkpoint_paths: Sequence[Path],
    compute_spectrum: bool,
    top_modules: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    prev_state_dict: Optional[Dict[str, torch.Tensor]] = None
    prev_singular_values: Optional[torch.Tensor] = None
    prev_factors: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor, float]]] = None

    progress = tqdm(checkpoint_paths, desc="Analyze checkpoints", unit="ckpt")
    for checkpoint_path in progress:
        state_dict = load_file(str(checkpoint_path))
        prefixes = iter_lora_prefixes(state_dict)
        current_factors = {prefix: get_lora_factors(state_dict, prefix) for prefix in prefixes}
        progress.set_postfix_str(checkpoint_path.name)

        raw_sq_norm = 0.0
        eff_sq_norm = 0.0
        eff_delta_sq_norm = 0.0
        eff_cos_num = 0.0
        prev_eff_sq_norm = 0.0
        current_singular_values: List[torch.Tensor] = []
        per_module_drifts: List[Tuple[str, float, float]] = []

        for key, value in state_dict.items():
            if key.endswith(WEIGHT_SUFFIX):
                tensor = value.float()
                raw_sq_norm += float((tensor * tensor).sum().item())

        for prefix in prefixes:
            up, down, scale = current_factors[prefix]
            eff_sq = low_rank_fro_sq(up, down, scale)
            eff_sq_norm += eff_sq

            if prev_state_dict is not None and prev_factors is not None:
                prev_up, prev_down, prev_scale = prev_factors[prefix]
                prev_sq = low_rank_fro_sq(prev_up, prev_down, prev_scale)
                inner = low_rank_inner(up, down, scale, prev_up, prev_down, prev_scale)
                diff_sq = max(0.0, eff_sq + prev_sq - 2.0 * inner)
                eff_delta_sq_norm += diff_sq
                eff_cos_num += inner
                prev_eff_sq_norm += prev_sq
                per_module_drifts.append((prefix, math.sqrt(diff_sq), diff_sq))

            if compute_spectrum:
                current_singular_values.append(effective_singular_values(state_dict, prefix))

        row: Dict[str, Any] = {
            "path": str(checkpoint_path),
            "step": get_checkpoint_step(checkpoint_path),
            "raw_norm": math.sqrt(raw_sq_norm),
            "effective_norm": math.sqrt(eff_sq_norm),
        }

        if prev_state_dict is not None and eff_sq_norm > 0.0 and prev_eff_sq_norm > 0.0:
            row["effective_delta_norm"] = math.sqrt(eff_delta_sq_norm)
            row["effective_rel_delta"] = row["effective_delta_norm"] / row["effective_norm"]
            row["effective_cos_to_prev"] = eff_cos_num / math.sqrt(prev_eff_sq_norm * eff_sq_norm)
            per_module_drifts.sort(key=lambda item: item[1], reverse=True)
            row["top_moving_modules"] = [
                {
                    "module": name,
                    "delta_norm": delta,
                    "delta_energy_share": diff_sq / eff_delta_sq_norm if eff_delta_sq_norm > 0.0 else None,
                }
                for name, delta, diff_sq in per_module_drifts[:top_modules]
            ]

        if compute_spectrum:
            singular_values = torch.cat(current_singular_values) if current_singular_values else torch.empty(0)
            effective_rank, top1_share = compute_effective_rank(singular_values)
            row["effective_rank"] = effective_rank
            row["top1_energy_share"] = top1_share
            if prev_singular_values is not None and singular_values.numel() == prev_singular_values.numel():
                spec_delta = torch.linalg.vector_norm(singular_values - prev_singular_values).item()
                spec_norm = torch.linalg.vector_norm(singular_values).item()
                row["spectrum_rel_change"] = spec_delta / spec_norm if spec_norm > 0.0 else None
            prev_singular_values = singular_values

        rows.append(row)
        prev_state_dict = state_dict
        prev_factors = current_factors

    return {"checkpoints": rows}


def fit_log_decay(points: Sequence[Tuple[int, float]], thresholds: Sequence[float]) -> Optional[Dict[str, Any]]:
    usable = [(step, value) for step, value in points if value > 0.0]
    if len(usable) < 2:
        return None

    xs = [point[0] for point in usable]
    ys = [math.log(point[1]) for point in usable]
    fit = linear_regression(xs, ys)

    intervals = [xs[i] - xs[i - 1] for i in range(1, len(xs))]
    step_interval = int(statistics.median(intervals)) if intervals else 1
    log_slope = fit["slope"]
    multiplier_per_interval = math.exp(log_slope * step_interval)

    projections: Dict[str, Optional[float]] = {}
    if log_slope < 0.0:
        for threshold in thresholds:
            projected_step = (math.log(threshold) - fit["intercept"]) / log_slope
            projections[f"{threshold:.4f}"] = projected_step
    else:
        for threshold in thresholds:
            projections[f"{threshold:.4f}"] = None

    return {
        "points_used": len(usable),
        "step_interval": step_interval,
        "log_slope_per_step": log_slope,
        "multiplier_per_interval": multiplier_per_interval,
        "projected_crossing_steps": projections,
    }


def heuristic_status(summary: Dict[str, Any]) -> Dict[str, Any]:
    checkpoints = summary.get("checkpoint_analysis", {}).get("checkpoints", [])
    latest_checkpoint = checkpoints[-1] if checkpoints else {}
    loss_stats = summary.get("loss_analysis", {}).get("loss/average", {})
    lr_stats = summary.get("effective_lr_analysis", {})

    plateau_500 = loss_stats.get("500", {}).get("plateau_ratio")
    plateau_2000 = loss_stats.get("2000", {}).get("plateau_ratio")
    eff_rel_delta = latest_checkpoint.get("effective_rel_delta")
    spec_rel_change = latest_checkpoint.get("spectrum_rel_change")
    lr_cv_1000 = lr_stats.get("1000", {}).get("cv")
    drift_trend = summary.get("effective_drift_trend", {})
    multiplier = drift_trend.get("multiplier_per_interval")

    status = "insufficient_data"
    reasons: List[str] = []

    if eff_rel_delta is None:
        return {"status": status, "reasons": ["No checkpoint drift data was available."]}

    if (
        plateau_500 is not None
        and plateau_500 <= 1.0
        and eff_rel_delta <= 0.10
        and (spec_rel_change is None or spec_rel_change <= 0.05)
    ):
        status = "near_stationary"
        reasons.append("recent smoothed loss is flatter than its residual noise")
        reasons.append("effective checkpoint drift is small")
        if spec_rel_change is not None:
            reasons.append("the LoRA singular spectrum is also stable")
        return {"status": status, "reasons": reasons}

    if (
        multiplier is not None
        and multiplier < 0.95
        and lr_cv_1000 is not None
        and lr_cv_1000 < 0.01
        and plateau_2000 is not None
        and plateau_2000 <= 2.5
    ):
        status = "approaching_stationary"
        reasons.append("effective checkpoint drift is decaying across recent saves")
        reasons.append("effective learning rate is stable")
        reasons.append("smoothed loss is moving slowly relative to its recent scale")
        return {"status": status, "reasons": reasons}

    status = "still_moving"
    if plateau_500 is not None and plateau_500 > 1.0:
        reasons.append("recent smoothed loss still has a directional trend")
    if eff_rel_delta > 0.10:
        reasons.append("effective LoRA weights still move materially between saves")
    if spec_rel_change is not None and spec_rel_change > 0.05:
        reasons.append("the LoRA singular spectrum is still changing")
    if not reasons:
        reasons.append("the available signals are mixed")
    return {"status": status, "reasons": reasons}


def build_summary(args: argparse.Namespace) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "inputs": {
            "wandb_run": args.wandb_run,
            "checkpoint_dir": args.checkpoint_dir,
            "checkpoint_glob": args.checkpoint_glob,
            "loss_windows": args.loss_windows,
            "tail_checkpoints": args.tail_checkpoints,
            "skip_spectrum": args.skip_spectrum,
        }
    }

    if args.wandb_run is not None:
        wandb_run_path = Path(args.wandb_run)
        rows = read_wandb_history(wandb_run_path)
        summary["wandb_history_records"] = len(rows)
        summary["loss_analysis"] = {
            key: compute_window_stats(extract_series(rows, key), args.loss_windows) for key in LOSS_KEYS
        }
        summary["effective_lr_analysis"] = compute_window_stats(extract_series(rows, EFF_LR_KEY), args.loss_windows)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_paths = sorted(checkpoint_dir.glob(args.checkpoint_glob), key=checkpoint_sort_key)
    if not checkpoint_paths:
        raise ValueError(f"No checkpoints matched {args.checkpoint_glob!r} in {checkpoint_dir}")

    summary["checkpoint_analysis"] = analyze_checkpoints(
        checkpoint_paths,
        compute_spectrum=not args.skip_spectrum,
        top_modules=args.top_modules,
    )
    summary["checkpoint_count"] = len(checkpoint_paths)

    drift_points = []
    for checkpoint in summary["checkpoint_analysis"]["checkpoints"]:
        if checkpoint.get("step") is None or checkpoint.get("effective_rel_delta") is None:
            continue
        drift_points.append((checkpoint["step"], checkpoint["effective_rel_delta"]))

    if drift_points:
        tail = drift_points[-args.tail_checkpoints :]
        summary["effective_drift_trend"] = fit_log_decay(tail, args.drift_thresholds)

    summary["heuristic_status"] = heuristic_status(summary)
    return summary


def print_human_readable(summary: Dict[str, Any]) -> None:
    print("== Stationarity Summary ==")
    print(f"status: {summary['heuristic_status']['status']}")
    for reason in summary["heuristic_status"].get("reasons", []):
        print(f"  - {reason}")

    if "wandb_history_records" in summary:
        print()
        print("== Loss Plateau ==")
        for loss_key, windows in summary.get("loss_analysis", {}).items():
            if not windows:
                continue
            print(loss_key)
            for window, stats in windows.items():
                print(
                    "  "
                    f"window={window:>4} "
                    f"last={stats['last_value']:.6f} "
                    f"change={stats['predicted_change']:+.6f} "
                    f"resid_std={stats['residual_std']:.6f} "
                    f"plateau_ratio={stats['plateau_ratio']:.3f}"
                )

        print()
        print("== Effective LR Stability ==")
        for window, stats in summary.get("effective_lr_analysis", {}).items():
            print(
                "  "
                f"window={window:>4} "
                f"mean={stats['mean_value']:.10g} "
                f"std={stats['std_value']:.10g} "
                f"cv={stats['cv']:.6g}"
            )

    checkpoints = summary["checkpoint_analysis"]["checkpoints"]
    latest = checkpoints[-1]
    print()
    print("== Latest Checkpoint Drift ==")
    print(f"step: {latest.get('step')}")
    print(f"raw_norm: {latest['raw_norm']:.6f}")
    print(f"effective_norm: {latest['effective_norm']:.6f}")
    if latest.get("effective_rel_delta") is not None:
        print(f"effective_rel_delta: {latest['effective_rel_delta']:.6f}")
        print(f"effective_cos_to_prev: {latest['effective_cos_to_prev']:.6f}")
    if latest.get("effective_rank") is not None:
        print(f"effective_rank: {latest['effective_rank']:.6f}")
        print(f"top1_energy_share: {latest['top1_energy_share']:.6f}")
    if latest.get("spectrum_rel_change") is not None:
        print(f"spectrum_rel_change: {latest['spectrum_rel_change']:.6f}")

    drift_trend = summary.get("effective_drift_trend")
    if drift_trend is not None:
        print()
        print("== Effective Drift Trend ==")
        print(f"points_used: {drift_trend['points_used']}")
        print(f"step_interval: {drift_trend['step_interval']}")
        print(f"multiplier_per_interval: {drift_trend['multiplier_per_interval']:.6f}")
        print(f"log_slope_per_step: {drift_trend['log_slope_per_step']:.8f}")
        print("projected_crossing_steps:")
        for threshold, projected_step in drift_trend["projected_crossing_steps"].items():
            if projected_step is None:
                print(f"  threshold={threshold}: no decay projection")
            else:
                print(f"  threshold={threshold}: {projected_step:.2f}")

    top_modules = latest.get("top_moving_modules")
    if top_modules:
        print()
        print("== Fastest Moving Modules (last checkpoint interval) ==")
        for module in top_modules:
            print(
                "  "
                f"{module['module']} "
                f"delta_norm={module['delta_norm']:.6f} "
                f"delta_energy_share={module['delta_energy_share']:.6f}"
            )


def main() -> None:
    args = parse_args()
    summary = build_summary(args)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_human_readable(summary)


if __name__ == "__main__":
    main()
