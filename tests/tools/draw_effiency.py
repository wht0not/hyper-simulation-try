from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean, median
from typing import Any
from xml.sax.saxutils import escape


DEFAULT_OUTPUT_DIR = Path(
    "/home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/data/efficiency"
)
DEFAULT_DATASETS = ["amem", "context", "memorybank"]
DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "amem": {
        "base_path": Path("/home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/data/amem/retrieved.json"),
        "hyper_path": Path("/home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/data/hyper_simulation/amem/answers.json"),
        "base_entries_key": "entries",
        "hyper_entries_key": "results",
        "base_time_path": ["search_time", "total"],
        "plot_mode": "compare",
        "output_prefix": "amem",
    },
    "context": {
        "base_path": Path("/home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/data/context/prepared.json"),
        "hyper_path": Path("/home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/data/hyper_simulation/context-final/answers.json"),
        "base_entries_key": "results",
        "hyper_entries_key": "results",
        "base_time_path": None,
        "plot_mode": "hyper_only",
        "output_prefix": "context",
    },
    "memorybank": {
        "base_path": Path("/home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/data/memorybank/retrieved.json"),
        "hyper_path": Path("/home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/data/hyper_simulation/memorybank/answers.json"),
        "base_entries_key": "entries",
        "hyper_entries_key": "results",
        "base_time_path": ["memorybank_context", "search_time"],
        "plot_mode": "compare",
        "drop_top_outliers_by": {
            "origin_plus_hyper_time": 2,
            "hyper_prepare_time": 4,
        },
        "output_prefix": "memorybank",
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Some generated files contain trailing commas before } or ].
        cleaned = re.sub(r",(\s*[}\]])", r"\1", text)
        payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at top level: {path}")
    return payload


def _nested_get(payload: dict[str, Any], path: list[str] | None) -> Any:
    if path is None:
        return None
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _sort_token(value: str) -> tuple[int, Any]:
    if value.isdigit():
        return (0, int(value))
    digits = "".join(ch for ch in value if ch.isdigit())
    if digits:
        return (1, value[: len(value) - len(digits)], int(digits))
    return (2, value)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * pct
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    weight = position - lower_index
    return lower + (upper - lower) * weight


def _summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p95": 0.0,
        }
    return {
        "count": len(values),
        "mean": mean(values),
        "p50": median(values),
        "min": min(values),
        "max": max(values),
        "p95": _percentile(values, 0.95),
    }


def _print_summary(label: str, values: list[float]) -> None:
    stats = _summary_stats(values)
    print(
        f"{label}: "
        f"n={int(stats['count'])}, "
        f"mean={stats['mean']:.4f}s, "
        f"p50={stats['p50']:.4f}s, "
        f"p95={stats['p95']:.4f}s, "
        f"min={stats['min']:.4f}s, "
        f"max={stats['max']:.4f}s"
    )


def _quartiles(values: list[float]) -> tuple[float, float, float]:
    return (
        _percentile(values, 0.25),
        _percentile(values, 0.5),
        _percentile(values, 0.75),
    )


def _format_seconds(value: float) -> str:
    return f"{value:.2f}s"


def _format_axis_tick(value: float, y_max: float) -> str:
    if y_max < 3:
        return f"{value:.1f}"
    if y_max < 10:
        return f"{value:.1f}"
    return f"{value:.0f}"


def _estimate_bandwidth(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    std_dev = math.sqrt(max(variance, 1e-12))
    bandwidth = 1.06 * std_dev * (len(values) ** (-0.2))
    return max(bandwidth, 1e-3)


def _kernel_density(values: list[float], sample: float, bandwidth: float) -> float:
    factor = 1.0 / (len(values) * bandwidth * math.sqrt(2.0 * math.pi))
    total = 0.0
    for value in values:
        diff = (sample - value) / bandwidth
        total += math.exp(-0.5 * diff * diff)
    return factor * total


def _scale_y(value: float, y_min: float, y_max: float, top: float, bottom: float) -> float:
    if y_max <= y_min:
        return (top + bottom) / 2.0
    ratio = (value - y_min) / (y_max - y_min)
    return bottom - ratio * (bottom - top)


def _violin_polygon_points(
    values: list[float],
    center_x: float,
    y_min: float,
    y_max: float,
    top: float,
    bottom: float,
    half_width: float,
    steps: int = 120,
) -> str:
    bandwidth = _estimate_bandwidth(values)
    samples = [
        y_min + (y_max - y_min) * index / max(steps - 1, 1)
        for index in range(steps)
    ]
    densities = [_kernel_density(values, sample, bandwidth) for sample in samples]
    max_density = max(densities) if densities else 1.0
    max_density = max(max_density, 1e-9)

    left_points: list[str] = []
    right_points: list[str] = []
    for sample, density in zip(samples, densities):
        y = _scale_y(sample, y_min, y_max, top, bottom)
        width = half_width * density / max_density
        left_points.append(f"{center_x - width:.2f},{y:.2f}")
        right_points.append(f"{center_x + width:.2f},{y:.2f}")
    return " ".join(left_points + list(reversed(right_points)))


def _svg_line(x1: float, y1: float, x2: float, y2: float, stroke: str, width: float = 1.0) -> str:
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{width:.2f}" />'
    )


def _svg_text_styled(
    x: float,
    y: float,
    text: str,
    size: int = 14,
    anchor: str = "middle",
    weight: str = "normal",
    transform: str | None = None,
) -> str:
    transform_attr = f' transform="{transform}"' if transform else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
        f'text-anchor="{anchor}" font-family="Arial, sans-serif" font-weight="{weight}"{transform_attr}>'
        f"{escape(text)}</text>"
    )


def _svg_circle(x: float, y: float, r: float, fill: str, stroke: str = "none", stroke_width: float = 0.0) -> str:
    return (
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{stroke_width:.2f}" />'
    )


def _build_summary(rows: list[dict[str, Any]], plot_mode: str) -> dict[str, Any]:
    origin_values = [row["origin_time"] for row in rows if row.get("origin_time") is not None]
    combined_values = [row["origin_plus_hyper_time"] for row in rows if row.get("origin_plus_hyper_time") is not None]
    prepared_values = [row["hyper_prepare_time"] for row in rows]
    context_sizes = [row["context_size"] for row in rows if row.get("context_size") is not None]

    summary = {
        "matched_count": len(rows),
        "metrics": {
            "hyper_prepare_time": _summary_stats(prepared_values),
        },
    }
    if plot_mode == "compare" and origin_values:
        summary["metrics"]["origin_time"] = _summary_stats(origin_values)
    if plot_mode == "compare" and combined_values:
        summary["metrics"]["origin_plus_hyper_time"] = _summary_stats(combined_values)
    if context_sizes:
        summary["metrics"]["context_size"] = {"mean": mean(context_sizes)}
    return summary


def _write_summary(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def _memorybank_items(entry: dict[str, Any]) -> list[str]:
    raw_items = entry.get("d", [])
    if not isinstance(raw_items, list):
        raw_items = [raw_items]
    return [str(item).strip() for item in raw_items if str(item).strip()]


def _entry_top_k(dataset_name: str, entry: dict[str, Any]) -> float | None:
    if dataset_name == "context":
        value = entry.get("top_k")
        return float(value) if value is not None else None
    if dataset_name == "amem":
        value = entry.get("retrieve_k")
        return float(value) if value is not None else None
    if dataset_name == "memorybank":
        return float(len(_memorybank_items(entry)))
    return None


def _entry_chunk_size(dataset_name: str, entry: dict[str, Any]) -> float | None:
    if dataset_name in {"context", "amem"}:
        value = entry.get("chunk_size")
        return float(value) if value is not None else None
    if dataset_name == "memorybank":
        items = _memorybank_items(entry)
        if not items:
            return None
        return sum(len(item) for item in items) / len(items)
    return None


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return mean(values)


def _build_category_summary(dataset_name: str, config: dict[str, Any]) -> dict[str, Any]:
    base_payload = _load_json(config["base_path"])
    base_entries = base_payload.get(config["base_entries_key"], [])
    if not isinstance(base_entries, list):
        raise ValueError(
            f"`{config['base_entries_key']}` is not a list in {config['base_path']}"
        )

    summary: dict[str, Any] = {}
    for category in [1, 2, 3, 4]:
        rows = [
            entry for entry in base_entries
            if isinstance(entry, dict) and entry.get("category") == category
        ]
        chunk_sizes = [
            value
            for entry in rows
            for value in [_entry_chunk_size(dataset_name, entry)]
            if value is not None
        ]
        category_summary: dict[str, Any] = {
            "count": len(rows),
            "chunk_size_mean": _mean_or_none(chunk_sizes),
        }
        if dataset_name == "context":
            top_ks = [
                value
                for entry in rows
                for value in [_entry_top_k(dataset_name, entry)]
                if value is not None
            ]
            category_summary["top_k_mean"] = _mean_or_none(top_ks)
        summary[f"category_{category}"] = category_summary
    return summary


def _write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_rows(dataset_name: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    base_path = config["base_path"]
    hyper_path = config["hyper_path"]
    base_payload = _load_json(base_path)
    hyper_payload = _load_json(hyper_path)

    base_entries = base_payload.get(config["base_entries_key"], [])
    hyper_entries = hyper_payload.get(config["hyper_entries_key"], [])
    if not isinstance(base_entries, list):
        raise ValueError(f"`{config['base_entries_key']}` is not a list in {base_path}")
    if not isinstance(hyper_entries, list):
        raise ValueError(f"`{config['hyper_entries_key']}` is not a list in {hyper_path}")

    base_index: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in base_entries:
        if not isinstance(entry, dict):
            continue
        sample_id = str(entry.get("sample_id", "")).strip()
        qa_id = str(entry.get("qa_id", "")).strip()
        if not sample_id or not qa_id:
            continue
        base_index[(sample_id, qa_id)] = entry

    rows: list[dict[str, Any]] = []
    for entry in hyper_entries:
        if not isinstance(entry, dict):
            continue

        sample_id = str(entry.get("sample_id", "")).strip()
        qa_id = str(entry.get("qa_id", "")).strip()
        if not sample_id or not qa_id:
            continue

        base_entry = base_index.get((sample_id, qa_id))
        if base_entry is None:
            continue

        base_time = _nested_get(base_entry, config["base_time_path"])
        hyper_time = entry.get("prepared_elapsed_seconds")
        if hyper_time is None:
            continue

        chunk_size = _entry_chunk_size(dataset_name, base_entry)
        top_k = _entry_top_k(dataset_name, base_entry)
        context_size = None
        if chunk_size is not None and top_k is not None:
            context_size = float(chunk_size) * float(top_k)

        rows.append(
            {
                "sample_id": sample_id,
                "qa_id": qa_id,
                "question": entry.get("q") or base_entry.get("q"),
                "context_size": context_size,
                "origin_time": float(base_time) if base_time is not None else None,
                "hyper_prepare_time": float(hyper_time),
                "origin_plus_hyper_time": (float(base_time) + float(hyper_time)) if base_time is not None else None,
            }
        )

    rows.sort(key=lambda item: (_sort_token(item["sample_id"]), _sort_token(item["qa_id"])))
    return rows


def _write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _drop_top_outliers(
    rows: list[dict[str, Any]],
    metric_to_count: dict[str, int],
) -> tuple[list[dict[str, Any]], list[tuple[str, dict[str, Any]]]]:
    removed: list[tuple[str, dict[str, Any]]] = []
    removed_ids: set[tuple[str, str]] = set()

    for metric_key, count in metric_to_count.items():
        if count <= 0:
            continue
        candidates = [
            row for row in rows
            if row.get(metric_key) is not None and (row["sample_id"], row["qa_id"]) not in removed_ids
        ]
        candidates.sort(key=lambda row: row[metric_key], reverse=True)
        for row in candidates[:count]:
            removed_ids.add((row["sample_id"], row["qa_id"]))
            removed.append((metric_key, row))

    filtered_rows = [row for row in rows if (row["sample_id"], row["qa_id"]) not in removed_ids]
    return filtered_rows, removed


def _draw_plot(
    series: list[tuple[str, list[float], str]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = 680
    height = 620
    margin_left = 78
    margin_right = 10
    margin_top = 26
    margin_bottom = 94
    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom

    datasets = [item[1] for item in series]
    labels = [item[0] for item in series]
    colors = [item[2] for item in series]
    if len(series) == 1:
        centers = [plot_left + (plot_right - plot_left) * 0.50]
    else:
        centers = [plot_left + (plot_right - plot_left) * 0.30, plot_left + (plot_right - plot_left) * 0.68]

    all_values = [value for values in datasets for value in values]
    data_min = min(all_values)
    data_max = max(all_values)
    padding = (data_max - data_min) * 0.08 if data_max > data_min else 1.0
    y_min = max(0.0, data_min - padding)
    y_max = data_max + padding

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
    ]

    tick_count = 6
    for tick_index in range(tick_count + 1):
        value = y_min + (y_max - y_min) * tick_index / tick_count
        y = _scale_y(value, y_min, y_max, plot_top, plot_bottom)
        parts.append(_svg_line(plot_left, y, plot_right, y, "#e6e8eb", 1.0))
        parts.append(
            _svg_text_styled(
                plot_left - 10,
                y + 5,
                _format_axis_tick(value, y_max),
                size=14,
                anchor="end",
                weight="bold",
            )
        )

    parts.append(_svg_line(plot_left, plot_top, plot_left, plot_bottom, "#222222", 1.8))
    parts.append(_svg_line(plot_left, plot_bottom, plot_right, plot_bottom, "#222222", 1.8))
    parts.append(_svg_line(plot_left, plot_top, plot_right, plot_top, "#222222", 1.8))
    parts.append(_svg_line(plot_right, plot_top, plot_right, plot_bottom, "#222222", 1.8))
    parts.append(
        _svg_text_styled(
            28,
            (plot_top + plot_bottom) / 2,
            "Time (s)",
            size=16,
            anchor="middle",
            weight="bold",
            transform=f"rotate(-90 28 {(plot_top + plot_bottom) / 2:.2f})",
        )
    )

    violin_half_width = 72.0
    box_half_width = 20.0
    cap_half_width = 14.0

    for center_x, values, label, color in zip(centers, datasets, labels, colors):
        polygon = _violin_polygon_points(
            values=values,
            center_x=center_x,
            y_min=y_min,
            y_max=y_max,
            top=plot_top,
            bottom=plot_bottom,
            half_width=violin_half_width,
        )
        parts.append(
            f'<polygon points="{polygon}" fill="{color}" fill-opacity="0.28" '
            f'stroke="{color}" stroke-width="1.0" />'
        )

        q1, q2, q3 = _quartiles(values)
        low = min(values)
        high = max(values)
        y_q1 = _scale_y(q1, y_min, y_max, plot_top, plot_bottom)
        y_q2 = _scale_y(q2, y_min, y_max, plot_top, plot_bottom)
        y_q3 = _scale_y(q3, y_min, y_max, plot_top, plot_bottom)
        y_low = _scale_y(low, y_min, y_max, plot_top, plot_bottom)
        y_high = _scale_y(high, y_min, y_max, plot_top, plot_bottom)

        parts.append(
            f'<rect x="{center_x - box_half_width:.2f}" y="{y_q3:.2f}" '
            f'width="{box_half_width * 2:.2f}" height="{max(y_q1 - y_q3, 1.0):.2f}" '
            f'fill="white" fill-opacity="0.92" stroke="#111111" stroke-width="1.1" />'
        )
        parts.append(_svg_line(center_x, y_high, center_x, y_q3, "#111111", 1.1))
        parts.append(_svg_line(center_x, y_q1, center_x, y_low, "#111111", 1.1))
        parts.append(_svg_line(center_x - cap_half_width, y_high, center_x + cap_half_width, y_high, "#111111", 1.1))
        parts.append(_svg_line(center_x - cap_half_width, y_low, center_x + cap_half_width, y_low, "#111111", 1.1))
        parts.append(_svg_line(center_x - box_half_width, y_q2, center_x + box_half_width, y_q2, "#111111", 2.0))

        stats = _summary_stats(values)
        highlight_points = [
            ("p95", stats["p95"]),
            ("mean", stats["mean"]),
            ("p50", stats["p50"]),
        ]
        label_y_positions: list[float] = []
        for _, point_value in highlight_points:
            point_y = _scale_y(point_value, y_min, y_max, plot_top, plot_bottom)
            if label_y_positions and point_y - label_y_positions[-1] < 16:
                point_y = label_y_positions[-1] + 16
            label_y_positions.append(point_y)

        marker_x = center_x + violin_half_width * 0.68
        label_x = marker_x + 10
        for (point_label, point_value), label_y in zip(highlight_points, label_y_positions):
            real_y = _scale_y(point_value, y_min, y_max, plot_top, plot_bottom)
            parts.append(_svg_line(center_x + box_half_width + 4, real_y, marker_x - 4, real_y, color, 1.2))
            parts.append(_svg_circle(marker_x, real_y, 3.2, color, stroke="white", stroke_width=0.8))
            parts.append(
                _svg_text_styled(
                    label_x,
                    label_y + 2,
                    point_label,
                    size=13,
                    anchor="start",
                    weight="bold",
                )
            )
            parts.append(
                _svg_text_styled(
                    label_x,
                    label_y + 18,
                    _format_seconds(point_value),
                    size=13,
                    anchor="start",
                    weight="bold",
                )
            )

        parts.append(_svg_text_styled(center_x, plot_bottom + 42, label, size=16, weight="bold"))

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def _output_paths(output_prefix: str) -> tuple[Path, Path, Path, Path]:
    method_output_dir = DEFAULT_OUTPUT_DIR / output_prefix
    return (
        method_output_dir / f"{output_prefix}_time_comparison.jsonl",
        method_output_dir / f"{output_prefix}_time_summary.json",
        method_output_dir / f"{output_prefix}_time_comparison.svg",
        method_output_dir / f"{output_prefix}_category_summary.json",
    )


def _process_dataset(dataset_name: str, draw_plot: bool) -> None:
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    config = DATASET_CONFIGS[dataset_name]
    output_jsonl, output_summary, output_plot, output_category_summary = _output_paths(config["output_prefix"])
    rows = _build_rows(dataset_name, config)
    if not rows:
        raise ValueError(f"No matched timing rows were found for {dataset_name}.")

    removed_outliers: list[tuple[str, dict[str, Any]]] = []
    outlier_rules = config.get("drop_top_outliers_by")
    if outlier_rules:
        rows, removed_outliers = _drop_top_outliers(rows, outlier_rules)

    origin_values = [row["origin_time"] for row in rows if row.get("origin_time") is not None]
    combined_values = [row["origin_plus_hyper_time"] for row in rows if row.get("origin_plus_hyper_time") is not None]
    prepared_values = [row["hyper_prepare_time"] for row in rows]
    context_sizes = [row["context_size"] for row in rows if row.get("context_size") is not None]
    summary = _build_summary(rows, config["plot_mode"])
    category_summary = _build_category_summary(dataset_name, config)

    _write_jsonl(rows, output_jsonl)
    _write_summary(summary, output_summary)
    _write_json(category_summary, output_category_summary)
    if draw_plot:
        if config["plot_mode"] == "hyper_only":
            plot_series = [("HyperSim", prepared_values, "#E15759")]
        else:
            plot_series = [
                ("Origin", origin_values, "#4E79A7"),
                ("HyperSim", combined_values, "#E15759"),
            ]
        _draw_plot(plot_series, output_plot)

    print(f"\n[{dataset_name}]")
    print(f"Matched rows: {len(rows)}")
    for metric_key, removed_outlier in removed_outliers:
        print(
            "Dropped outlier: "
            f"sample_id={removed_outlier['sample_id']}, "
            f"qa_id={removed_outlier['qa_id']}, "
            f"{metric_key}={removed_outlier[metric_key]:.4f}"
        )
    _print_summary("Hyper prepare only", prepared_values)
    if config["plot_mode"] == "compare":
        _print_summary("Origin", origin_values)
        _print_summary("Origin + HyperSim", combined_values)
    if context_sizes:
        print(
            "Context size: "
            f"mean={mean(context_sizes):.4f}"
        )
    else:
        print("Context size: not available")
    print(f"Saved jsonl: {output_jsonl}")
    print(f"Saved summary: {output_summary}")
    print(f"Saved category summary: {output_category_summary}")
    if draw_plot:
        print(f"Saved plot: {output_plot}")
    else:
        print("Saved plot: skipped")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate timing statistics and plots for LoCoMo retrieval settings.")
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset keys: amem,context,memorybank",
    )
    parser.add_argument(
        "--draw-plot",
        dest="draw_plot",
        action="store_true",
        help="Generate svg plots.",
    )
    parser.add_argument(
        "--no-draw-plot",
        dest="draw_plot",
        action="store_false",
        help="Skip svg plot generation.",
    )
    parser.set_defaults(draw_plot=True)
    args = parser.parse_args()

    selected = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if not selected:
        raise ValueError("No datasets selected.")

    for dataset_name in selected:
        _process_dataset(dataset_name, draw_plot=args.draw_plot)


if __name__ == "__main__":
    main()
