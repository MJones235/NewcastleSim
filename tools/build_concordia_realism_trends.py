#!/usr/bin/env python3
"""Build cross-run realism trend metrics for Station Concordia outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

STREET_EXITS = {"Grey Street", "Blackett Street", "Eldon Square"}


@dataclass
class RunMetrics:
    run_name: str
    decisions: int
    move_decisions: int
    wait_decisions: int
    weighted_entropy_all: float
    weighted_entropy_early: float
    weighted_entropy_late: float
    switch_rate_with_messages: float
    switch_rate_without_messages: float
    switch_rate_ratio: float
    brisk_ratio: float
    slow_ratio: float
    normal_ratio: float


def norm_exit(name: str | None) -> str | None:
    if not name:
        return None
    s = name.lower()
    if "grey street" in s:
        return "Grey Street"
    if "blackett street" in s:
        return "Blackett Street"
    if "eldon square" in s:
        return "Eldon Square"
    if "escalator" in s and "up" in s:
        return "Up escalator"
    if "escalator" in s and "down" in s:
        return "Down escalator"
    return name


def shannon_entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log(p, 2)
    return entropy


def weighted_entropy(rows: list[tuple[float, int, float]], min_time: float | None = None) -> float:
    filtered = []
    for t, n, h_norm in rows:
        if min_time is not None and t <= min_time:
            continue
        if min_time is None or t <= min_time:
            filtered.append((t, n, h_norm))
    if not filtered:
        return 0.0
    denom = sum(n for _, n, _ in filtered)
    if denom == 0:
        return 0.0
    return sum(n * h for _, n, h in filtered) / denom


def load_records(run_dir: Path) -> list[dict]:
    decisions_path = run_dir / "agent_decisions.json"
    if not decisions_path.exists():
        return []

    data = json.loads(decisions_path.read_text())
    records: list[dict] = []
    for agent_id, payload in data.get("agent_decisions", {}).items():
        for decision in payload.get("decisions", []):
            translated = (
                decision.get("translated") if isinstance(decision.get("translated"), dict) else {}
            )
            records.append(
                {
                    "agent": agent_id,
                    "time": float(decision.get("time", 0.0)),
                    "action_type": translated.get("action_type"),
                    "exit_name": translated.get("exit_name"),
                    "speed": translated.get("speed"),
                    "observation": decision.get("observation", ""),
                    "route_change": (
                        decision.get("route_change")
                        if isinstance(decision.get("route_change"), dict)
                        else None
                    ),
                }
            )
    return records


def compute_run_metrics(run_dir: Path) -> RunMetrics | None:
    records = load_records(run_dir)
    if not records:
        return None

    by_time_street: dict[float, Counter[str]] = defaultdict(Counter)
    move_decisions = 0
    wait_decisions = 0

    switch_with_msg = 0
    switch_without_msg = 0
    noswitch_with_msg = 0
    noswitch_without_msg = 0

    speed_counter: Counter[str] = Counter()

    for record in records:
        action_type = record.get("action_type")
        if action_type == "move":
            move_decisions += 1
        elif action_type == "wait":
            wait_decisions += 1

        normalized_exit = norm_exit(record.get("exit_name"))
        if action_type == "move" and normalized_exit in STREET_EXITS:
            by_time_street[record["time"]][normalized_exit] += 1

        observation = record.get("observation", "")
        has_msg = "What people just said to you:" in observation
        switched = record.get("route_change") is not None
        if has_msg and switched:
            switch_with_msg += 1
        elif has_msg and not switched:
            noswitch_with_msg += 1
        elif (not has_msg) and switched:
            switch_without_msg += 1
        else:
            noswitch_without_msg += 1

        speed = record.get("speed")
        if speed:
            speed_counter[speed] += 1

    entropy_rows: list[tuple[float, int, float]] = []
    for t in sorted(by_time_street):
        counter = by_time_street[t]
        h = shannon_entropy(counter)
        h_norm = h / math.log(3, 2) if sum(counter.values()) else 0.0
        entropy_rows.append((t, sum(counter.values()), h_norm))

    def weighted(rows: list[tuple[float, int, float]]) -> float:
        if not rows:
            return 0.0
        denom = sum(n for _, n, _ in rows)
        if denom == 0:
            return 0.0
        return sum(n * h for _, n, h in rows) / denom

    weighted_all = weighted(entropy_rows)
    weighted_early = weighted([row for row in entropy_rows if row[0] <= 60.0])
    weighted_late = weighted([row for row in entropy_rows if row[0] > 60.0])

    with_msg_total = switch_with_msg + noswitch_with_msg
    without_msg_total = switch_without_msg + noswitch_without_msg
    switch_rate_with = (switch_with_msg / with_msg_total) if with_msg_total else 0.0
    switch_rate_without = (switch_without_msg / without_msg_total) if without_msg_total else 0.0
    switch_ratio = (
        switch_rate_with / switch_rate_without if switch_rate_without > 0 else float("inf")
    )

    speed_total = sum(speed_counter.values())
    brisk_ratio = speed_counter.get("brisk_walk", 0) / speed_total if speed_total else 0.0
    slow_ratio = speed_counter.get("slow_walk", 0) / speed_total if speed_total else 0.0
    normal_ratio = speed_counter.get("normal_walk", 0) / speed_total if speed_total else 0.0

    return RunMetrics(
        run_name=run_dir.name,
        decisions=len(records),
        move_decisions=move_decisions,
        wait_decisions=wait_decisions,
        weighted_entropy_all=weighted_all,
        weighted_entropy_early=weighted_early,
        weighted_entropy_late=weighted_late,
        switch_rate_with_messages=switch_rate_with,
        switch_rate_without_messages=switch_rate_without,
        switch_rate_ratio=switch_ratio,
        brisk_ratio=brisk_ratio,
        slow_ratio=slow_ratio,
        normal_ratio=normal_ratio,
    )


def write_outputs(metrics: list[RunMetrics], out_csv: Path, out_md: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_name",
                "decisions",
                "move_decisions",
                "wait_decisions",
                "weighted_entropy_all",
                "weighted_entropy_early",
                "weighted_entropy_late",
                "switch_rate_with_messages",
                "switch_rate_without_messages",
                "switch_rate_ratio",
                "brisk_ratio",
                "slow_ratio",
                "normal_ratio",
            ]
        )
        for m in metrics:
            writer.writerow(
                [
                    m.run_name,
                    m.decisions,
                    m.move_decisions,
                    m.wait_decisions,
                    f"{m.weighted_entropy_all:.4f}",
                    f"{m.weighted_entropy_early:.4f}",
                    f"{m.weighted_entropy_late:.4f}",
                    f"{m.switch_rate_with_messages:.4f}",
                    f"{m.switch_rate_without_messages:.4f}",
                    "inf" if math.isinf(m.switch_rate_ratio) else f"{m.switch_rate_ratio:.4f}",
                    f"{m.brisk_ratio:.4f}",
                    f"{m.slow_ratio:.4f}",
                    f"{m.normal_ratio:.4f}",
                ]
            )

    lines = []
    lines.append("# Concordia Realism Trends")
    lines.append("")
    lines.append(
        "| Run | Entropy (all) | Entropy early | Entropy late | Switch with msg | Switch w/o msg | Switch ratio | Brisk | Slow | Normal |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in metrics:
        ratio = "inf" if math.isinf(m.switch_rate_ratio) else f"{m.switch_rate_ratio:.2f}"
        lines.append(
            "| "
            + f"{m.run_name} | {m.weighted_entropy_all:.3f} | {m.weighted_entropy_early:.3f} | {m.weighted_entropy_late:.3f}"
            + f" | {m.switch_rate_with_messages*100:.1f}% | {m.switch_rate_without_messages*100:.1f}% | {ratio}"
            + f" | {m.brisk_ratio*100:.1f}% | {m.slow_ratio*100:.1f}% | {m.normal_ratio*100:.1f}% |"
        )

    out_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Station Concordia realism trend metrics.")
    parser.add_argument(
        "--output-root",
        default="scenarios/station_concordia/output",
        help="Root directory containing run_* folders.",
    )
    parser.add_argument(
        "--csv",
        default="scenarios/station_concordia/output/realism_trends.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--markdown",
        default="scenarios/station_concordia/output/realism_trends.md",
        help="Markdown summary output path.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    run_dirs = sorted([p for p in output_root.glob("run_*") if p.is_dir()])

    metrics: list[RunMetrics] = []
    for run_dir in run_dirs:
        m = compute_run_metrics(run_dir)
        if m is not None:
            metrics.append(m)

    write_outputs(metrics, Path(args.csv), Path(args.markdown))
    print(f"Processed {len(metrics)} runs")
    print(f"Wrote CSV: {args.csv}")
    print(f"Wrote Markdown: {args.markdown}")


if __name__ == "__main__":
    main()
