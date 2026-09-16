"""Deterministic artifact, figure, table, and bundle writers."""

from __future__ import annotations

import csv
import json
import platform
import shutil
import sys
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    empty_fields: Sequence[str] = (),
) -> None:
    fields = sorted({key for row in rows for key in row}) or list(empty_fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def environment_record(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
    }


def write_latex_tables(
    root: Path,
    cohort_rows: Sequence[Mapping[str, Any]],
    claim_rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    results = root / "paper_results_table.tex"
    selected = [
        row
        for row in cohort_rows
        if str(row.get("cohort", "")).startswith("hop_")
    ]
    lines = [
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Opt./data seed & Cohort & Model & Accuracy & Pair exact \\\\",
        "\\midrule",
    ]
    for row in selected:
        model = "ERGT" if row["model"] == "native_ergt" else "Transformer"
        lines.append(
        f"{row['optimization_seed']}/{row['data_seed']} & {row['cohort']} & {model} & "
            f"{float(row['accuracy']):.3f} & "
            f"{float(row['counterfactual_pair_exact']):.3f} \\\\" 
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    results.write_text("\n".join(lines) + "\n", encoding="utf-8")

    claims = root / "paper_claims_table.tex"
    claim_lines = [
        "\\begin{tabular}{lll}",
        "\\toprule",
        "Claim & Status & Gate \\\\",
        "\\midrule",
    ]
    for row in claim_rows:
        gate = str(row.get("gate", "")).replace("_", "\\_")
        claim_lines.append(
            f"{str(row['claim']).replace('_', '\\_')} & {row['status']} & {gate} \\\\" 
        )
    claim_lines.extend(("\\bottomrule", "\\end{tabular}"))
    claims.write_text("\n".join(claim_lines) + "\n", encoding="utf-8")
    return [results, claims]


def write_figures(
    root: Path,
    cohort_rows: Sequence[Mapping[str, Any]],
    curves: Sequence[Mapping[str, Any]],
    intervention_rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for model, label in (("native_ergt", "ERGT"), ("direct_transformer", "Transformer")):
        points: dict[int, list[float]] = {}
        for row in cohort_rows:
            cohort = str(row.get("cohort", ""))
            if row.get("model") != model or not cohort.startswith("hop_"):
                continue
            hops = int(cohort.split("_")[1])
            points.setdefault(hops, []).append(float(row["accuracy"]))
        x = sorted(points)
        y = [sum(points[value]) / len(points[value]) for value in x]
        if x:
            axis.plot(x, y, marker="o", label=label)
    axis.set_xlabel("Relational horizon (hops)")
    axis.set_ylabel("Answer accuracy")
    axis.set_ylim(0.0, 1.05)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = root / "figure_horizon_accuracy.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for model, label in (("native_ergt", "ERGT"), ("direct_transformer", "Transformer")):
        selected = sorted(
            (
                int(row["step"]),
                float(row["tuning_accuracy"]),
            )
            for row in curves
            if row.get("model") == model
        )
        if selected:
            axis.plot(
                [item[0] for item in selected],
                [item[1] for item in selected],
                label=label,
            )
    axis.set_xlabel("Optimization step")
    axis.set_ylabel("Tuning accuracy")
    axis.set_ylim(0.0, 1.05)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = root / "figure_convergence.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    if intervention_rows:
        aggregate: dict[str, list[float]] = {}
        for row in intervention_rows:
            aggregate.setdefault(str(row["intervention"]), []).append(
                float(row["targeted_drop"])
            )
        names = sorted(aggregate)
        values = [sum(aggregate[name]) / len(aggregate[name]) for name in names]
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.bar(range(len(names)), values)
        axis.set_xticks(range(len(names)), names, rotation=60, ha="right")
        axis.set_ylabel("Full minus intervention accuracy")
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        path = root / "figure_causal_interventions.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)
    return paths


def bundle(root: Path, paths: Sequence[Path]) -> Path:
    output = root / "ergt_reviewer_evidence_bundle.zip"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.exists() and path != output:
                archive.write(path, arcname=path.relative_to(root))
    return output


def copy_to_colab_downloads(paths: Sequence[Path]) -> list[str]:
    if not Path("/content").exists():
        return []
    destination = Path("/content/ergt_downloads")
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for path in paths:
        if path.exists():
            target = destination / path.name
            shutil.copy2(path, target)
            copied.append(str(target))
    return copied


__all__ = [
    "bundle",
    "copy_to_colab_downloads",
    "environment_record",
    "write_csv",
    "write_figures",
    "write_json",
    "write_latex_tables",
]
