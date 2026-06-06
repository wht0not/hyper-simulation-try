from __future__ import annotations

import argparse
import math
from pathlib import Path
from xml.sax.saxutils import escape


DEFAULT_OUTPUT_PATH = Path(
    "/home/vincent/hyper-simulation-try/tests/tools/llm_judge_stacked.svg"
)

CHART_DATA = [
    {
        "key": "singlehop",
        "label": "singlehop",
        "items": [
            {"label": "RAG\n2048 / 6", "origin": 68.01, "hyper": 71.94},
            {"label": "RAG\n128 / 10", "origin": 62.19, "hyper": 65.52},
            {"label": "RAG\n2048 / 10", "origin": 70.99, "hyper": 76.10},
            {"label": "A-MEM\n2754 / 10", "origin": 66.35, "hyper": 70.63},
            {"label": "MemoryBank\n794 / 10", "origin": 40.55, "hyper": 46.25},
            {"label": "Full-Context\n3376 / 28.0", "origin": 80.98, "hyper": 83.35},
        ],
    },
    {
        "key": "multihop",
        "label": "multihop",
        "items": [
            {"label": "RAG\n2048 / 6", "origin": 51.77, "hyper": 55.32},
            {"label": "RAG\n128 / 10", "origin": 49.29, "hyper": 52.13},
            {"label": "RAG\n2048 / 10", "origin": 57.09, "hyper": 64.89},
            {"label": "A-MEM\n2805 / 10", "origin": 58.87, "hyper": 59.93},
            {"label": "MemoryBank\n826 / 10", "origin": 44.68, "hyper": 51.42},
            {"label": "Full-Context\n3406 / 27.5", "origin": 61.35, "hyper": 66.31},
        ],
    },
    {
        "key": "temporal_reasoning",
        "label": "Temperal reasoning",
        "items": [
            {"label": "RAG\n2048 / 6", "origin": 26.79, "hyper": 30.22},
            {"label": "RAG\n128 / 10", "origin": 28.66, "hyper": 22.74},
            {"label": "RAG\n2048 / 10", "origin": 28.66, "hyper": 29.28},
            {"label": "A-MEM\n2616 / 10", "origin": 21.81, "hyper": 30.84},
            {"label": "MemoryBank\n803 / 10", "origin": 20.25, "hyper": 22.74},
            {"label": "Full-Context\n3357 / 27.2", "origin": 26.48, "hyper": 27.41},
        ],
    },
    {
        "key": "opendomain",
        "label": "opendomain",
        "items": [
            {"label": "RAG\n2048 / 6", "origin": 31.25, "hyper": 42.71},
            {"label": "RAG\n128 / 10", "origin": 34.38, "hyper": 44.79},
            {"label": "RAG\n2048 / 10", "origin": 35.42, "hyper": 46.88},
            {"label": "A-MEM\n2736 / 10", "origin": 40.62, "hyper": 43.75},
            {"label": "MemoryBank\n819 / 10", "origin": 29.17, "hyper": 40.62},
            {"label": "Full-Context\n3428 / 27.7", "origin": 26.04, "hyper": 40.62},
        ],
    },
    {
        "key": "overall",
        "label": "overral",
        "items": [
            {"label": "RAG\n2048 / 6", "origin": 54.16, "hyper": 58.38},
            {"label": "RAG\n128 / 10", "origin": 51.10, "hyper": 52.86},
            {"label": "RAG\n2048 / 10", "origin": 57.40, "hyper": 62.47},
            {"label": "A-MEM\n2733 / 10", "origin": 46.91, "hyper": 51.29},
            {"label": "MemoryBank\n803 / 10", "origin": 33.66, "hyper": 40.26},
            {"label": "Full-Context\n3381 / 27.7", "origin": 48.71, "hyper": 54.42},
        ],
    },
]

SLOT_LABELS = [
    ["rag6-2048"],
    ["rag10-128"],
    ["rag10-2048"],
    ["amem"],
    ["memorybank"],
    ["full-context"],
]

METHOD_COLORS = [
    "#7EAEDC",
    "#B8D3EE",
    "#4F89C6",
    "#6699CF",
    "#D7E7F6",
    "#2F5F95",
]
GAIN_COLOR = "#E15759"
DROP_COLOR = "#E15759"
GRID_COLOR = "#E6E8EB"
AXIS_COLOR = "#2F3B4A"
TEXT_COLOR = "#2F3B4A"
CELL_FILL = "#F6F9FD"
CELL_STROKE = "#C9D7E6"
FONT_SIZE_DELTA = 5


def _scale_y(value: float, y_min: float, y_max: float, top: float, bottom: float) -> float:
    if y_max <= y_min:
        return bottom
    ratio = (value - y_min) / (y_max - y_min)
    return bottom - ratio * (bottom - top)


def _svg_line(x1: float, y1: float, x2: float, y2: float, stroke: str, width: float = 1.0, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{width:.2f}"{dash_attr} />'
    )


def _svg_rect(
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    stroke: str = "none",
    stroke_width: float = 0.0,
    opacity: float | None = None,
) -> str:
    opacity_attr = f' fill-opacity="{opacity:.3f}"' if opacity is not None else ""
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.2f}"{opacity_attr} />'
    )


def _svg_text(
    x: float,
    y: float,
    text: str,
    size: int = 14,
    anchor: str = "middle",
    weight: str = "normal",
    fill: str = TEXT_COLOR,
    transform: str | None = None,
) -> str:
    size += FONT_SIZE_DELTA
    transform_attr = f' transform="{transform}"' if transform else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" font-weight="{weight}" fill="{fill}"{transform_attr}>'
        f"{escape(text)}</text>"
    )


def _svg_multiline_text(
    x: float,
    y: float,
    lines: list[str],
    size: int = 12,
    anchor: str = "middle",
    weight: str = "normal",
    fill: str = TEXT_COLOR,
    transform: str | None = None,
    line_height: float = 1.18,
) -> str:
    size += FONT_SIZE_DELTA
    transform_attr = f' transform="{transform}"' if transform else ""
    parts = [
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" font-weight="{weight}" fill="{fill}"{transform_attr}>'
    ]
    for index, line in enumerate(lines):
        dy = "0" if index == 0 else f"{line_height}em"
        parts.append(f'<tspan x="{x:.2f}" dy="{dy}">{escape(line)}</tspan>')
    parts.append("</text>")
    return "".join(parts)


def _flatten_items() -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for category in CHART_DATA:
        for item in category["items"]:
            items.append(
                {
                    "category": category["label"],
                    "label": item["label"],
                    "origin": item["origin"],
                    "hyper": item["hyper"],
                }
            )
    return items


def _format_delta(origin: float, hyper: float | None) -> str:
    if hyper is None:
        return ""
    delta = hyper - origin
    return f"{delta:+.2f}"


def _build_svg(output_path: Path) -> None:
    flat_items = _flatten_items()
    max_value = max(
        max(float(item["origin"]), float(item["hyper"]) if item["hyper"] is not None else float(item["origin"]))
        for item in flat_items
    )
    y_min = 10.0
    y_max = math.ceil((max_value + 2.0) / 10.0) * 10.0

    bar_width = 32.0
    bar_gap = 16.0
    category_gap = 58.0
    margin_left = 86.0
    margin_right = 40.0
    margin_top = 76.0
    margin_bottom = 196.0
    plot_height = 560.0
    plot_top = margin_top
    plot_bottom = margin_top + plot_height

    category_positions: list[tuple[float, float, str]] = []
    bar_positions: list[tuple[float, dict[str, object]]] = []

    x = margin_left + 12.0
    for category in CHART_DATA:
        start_x = x
        for item in category["items"]:
            bar_positions.append((x, item))
            x += bar_width + bar_gap
        end_x = x - bar_gap
        category_positions.append((start_x, end_x, category["label"]))
        x += category_gap

    width = int(x - category_gap + margin_right)
    height = int(plot_bottom + margin_bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs>',
        '<pattern id="dropPattern" patternUnits="userSpaceOnUse" width="8" height="8" patternTransform="rotate(45)">',
        f'<rect width="8" height="8" fill="{DROP_COLOR}" fill-opacity="0.12" />',
        f'<line x1="0" y1="0" x2="0" y2="8" stroke="{DROP_COLOR}" stroke-width="2.2" />',
        "</pattern>",
        "</defs>",
        '<rect width="100%" height="100%" fill="white" />',
    ]

    tick_count = int((y_max - y_min) // 10)
    for tick in range(tick_count + 1):
        value = y_min + tick * 10.0
        y = _scale_y(value, y_min, y_max, plot_top, plot_bottom)
        parts.append(_svg_line(margin_left, y, width - margin_right, y, GRID_COLOR, 1.0))
        parts.append(_svg_text(margin_left - 10, y + 5, f"{value:.0f}", size=13, anchor="end", weight="bold"))

    parts.append(_svg_line(margin_left, plot_top, margin_left, plot_bottom, AXIS_COLOR, 1.8))
    parts.append(_svg_line(margin_left, plot_bottom, width - margin_right, plot_bottom, AXIS_COLOR, 1.8))
    parts.append(
        _svg_text(
            28,
            (plot_top + plot_bottom) / 2.0,
            "LLM-as-a-judge",
            size=16,
            weight="bold",
            transform=f"rotate(-90 28 {(plot_top + plot_bottom) / 2.0:.2f})",
        )
    )

    legend_x = margin_left
    legend_y = 18.0
    legend_items = [
        ("HyperSim", GAIN_COLOR),
        ("HyperSim Drop", "url(#dropPattern)"),
    ]
    for label, fill in legend_items:
        stroke = DROP_COLOR if fill == "url(#dropPattern)" else "none"
        parts.append(_svg_text(legend_x, legend_y + 13, label, size=13, anchor="start", weight="bold"))
        parts.append(_svg_rect(legend_x + 78, legend_y, 16, 16, fill, stroke=stroke, stroke_width=1.0))
        legend_x += 170

    for start_x, end_x, category_label in category_positions:
        if start_x != category_positions[0][0]:
            separator_x = start_x - category_gap / 2.0
            parts.append(_svg_line(separator_x, plot_top, separator_x, plot_bottom + 16, "#C9CDD2", 1.0, dash="5 5"))

        category_center = (start_x + end_x) / 2.0
        parts.append(_svg_text(category_center, plot_bottom + 42, category_label, size=16, weight="bold"))

    method_row_width = 1180.0
    method_cell_width = method_row_width / len(SLOT_LABELS)
    method_row_left = (width - method_row_width) / 2.0
    method_row_y = plot_bottom + 86.0
    method_row_height = 72.0
    parts.append(
        _svg_rect(
            method_row_left,
            method_row_y,
            method_row_width,
            method_row_height,
            CELL_FILL,
            stroke=AXIS_COLOR,
            stroke_width=1.4,
        )
    )
    for index, slot_lines in enumerate(SLOT_LABELS):
        cell_x = method_row_left + index * method_cell_width
        if index > 0:
            parts.append(
                _svg_line(
                    cell_x,
                    method_row_y,
                    cell_x,
                    method_row_y + method_row_height,
                    CELL_STROKE,
                    1.2,
                )
            )
        parts.append(
            _svg_multiline_text(
                cell_x + 16.0,
                method_row_y + 38.0,
                slot_lines,
                size=11,
                anchor="start",
                weight="bold",
            )
        )
        sample_bar_x = cell_x + method_cell_width - 34.0
        sample_bar_y = method_row_y + 18.0
        parts.append(
            _svg_rect(
                sample_bar_x,
                sample_bar_y + 8.0,
                14.0,
                28.0,
                METHOD_COLORS[index],
                stroke=CELL_STROKE,
                stroke_width=1.0,
                opacity=0.78,
            )
        )
        parts.append(
            _svg_rect(
                sample_bar_x,
                sample_bar_y,
                14.0,
                8.0,
                GAIN_COLOR,
                opacity=0.95,
            )
        )

    for item_index, (bar_x, item) in enumerate(bar_positions):
        origin = float(item["origin"])
        hyper = float(item["hyper"]) if item["hyper"] is not None else None
        y_origin = _scale_y(origin, y_min, y_max, plot_top, plot_bottom)
        origin_height = plot_bottom - y_origin
        method_color = METHOD_COLORS[item_index % len(SLOT_LABELS)]

        parts.append(_svg_rect(bar_x, y_origin, bar_width, origin_height, method_color, opacity=0.72))

        if hyper is not None:
            if hyper >= origin:
                y_hyper = _scale_y(hyper, y_min, y_max, plot_top, plot_bottom)
                parts.append(_svg_rect(bar_x, y_hyper, bar_width, y_origin - y_hyper, GAIN_COLOR, opacity=0.95))
            else:
                y_hyper = _scale_y(hyper, y_min, y_max, plot_top, plot_bottom)
                parts.append(_svg_rect(bar_x, y_origin, bar_width, y_hyper - y_origin, "url(#dropPattern)", stroke=DROP_COLOR, stroke_width=0.9))
                parts.append(_svg_line(bar_x, y_hyper, bar_x + bar_width, y_hyper, DROP_COLOR, 1.5))
            label_y = min(y_origin, y_hyper) - 10.0
            parts.append(
                _svg_text(
                    bar_x + bar_width / 2.0,
                    label_y,
                    _format_delta(origin, hyper),
                    size=11,
                    weight="bold",
                )
            )

    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a stacked SVG chart for LLM-as-a-judge results.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output SVG path.",
    )
    args = parser.parse_args()

    _build_svg(args.output)
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
