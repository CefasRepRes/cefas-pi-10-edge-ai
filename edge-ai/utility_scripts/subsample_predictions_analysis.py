#!/usr/bin/env python3
"""Particle-subsampling sufficiency analysis.

Inputs accepted:
  * Azure Blob directory containing predictions.json files
  * ordered_predictions_cache.csv, one row per ordered prediction
  * legacy subsample_counts_comparison_long.csv
  * subsampling_results_long.csv previously produced by this script

The random analysis repeatedly samples without replacement and reports the median
and 5th-95th percentile envelope. The sequential analysis retains the first chunk
and is available only when ordered particle rows are present.

We have answered two questions we set out to: 
You can get away with processing just 1% of each tenbin as long as you have random sampling throughout.
If the tenbin contains about 70% bubbles, we have evidence that it is suppressing the counts of the other classes.

"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd

LABEL_CANDIDATES = (
    "predicted_label", "prediction", "label", "class", "class_name",
    "predicted_class", "ml_class", "category",
)
GROUP_CANDIDATES = (
    "blob_name", "predictions_blob", "predictions_json", "source_blob",
    "source", "file", "filename", "predictions_file", "run_name",
)
ORDER_CANDIDATES = (
    "prediction_index", "particle_index", "record_index", "image_index",
    "sequence", "sequence_index", "row_index", "index", "ordinal",
)
TIME_CANDIDATES = ("timestamp", "datetime", "date_time", "acquisition_time", "time")
RESULT_REQUIRED = {"timestamp", "class", "fraction", "mode", "full_count", "estimate", "p05", "p95"}
LEGACY_REQUIRED = {"class", "full_count", "n_full"}


def parse_fractions(text: str) -> list[float]:
    values = sorted(set(float(x.strip()) for x in text.split(",") if x.strip()))
    if not values or any(value <= 0 or value > 1 for value in values):
        raise ValueError("All fractions must be > 0 and <= 1")
    return values


def first_present(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    lookup = {str(column).strip().lower(): str(column) for column in columns}
    return next((lookup[name] for name in candidates if name in lookup), None)


def parse_blob_url(source: str) -> Optional[Tuple[str, str, str]]:
    parsed = urlparse(source)
    if parsed.scheme not in {"http", "https"} or ".blob.core.windows.net" not in parsed.netloc:
        return None
    parts = unquote(parsed.path).lstrip("/").split("/", 1)
    if not parts or not parts[0]:
        return None
    return f"{parsed.scheme}://{parsed.netloc}", parts[0], parts[1] if len(parts) > 1 else ""


def records_from_payload(payload: Any) -> list[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("predictions", "items", "rows", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                found = records_from_payload(value)
                if found:
                    return found
        for value in payload.values():
            if isinstance(value, (list, dict)):
                found = records_from_payload(value)
                if found:
                    return found
    return []


def label_from_record(record: Dict[str, Any]) -> Optional[str]:
    for key in LABEL_CANDIDATES:
        value = record.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, dict):
            for nested in LABEL_CANDIDATES:
                inner = value.get(nested)
                if inner not in (None, ""):
                    return str(inner)
        else:
            return str(value)
    return None


def timestamp_from_text(text: str) -> pd.Timestamp:
    text = str(text).replace("\\", "/")
    patterns = (
        r"(?P<date>20\d{2}[-_]\d{2}[-_]\d{2})[/_-]?(?P<clock>\d{4,6})(?!\d)",
        r"(?P<date>20\d{6})[/_-]?(?P<clock>\d{4,6})(?!\d)",
    )
    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        if not matches:
            continue
        match = matches[-1]
        date = match.group("date").replace("_", "-")
        if "-" not in date:
            date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        clock = match.group("clock")
        clock = f"{clock[:2]}:{clock[2:4]}" + (f":{clock[4:6]}" if len(clock) >= 6 else "")
        return pd.to_datetime(f"{date} {clock}", errors="coerce")
    return pd.NaT

def hex_to_rgba(colour: str, alpha: float) -> str:
    """Convert #RRGGBB to Plotly-compatible rgba(...)."""
    value = colour.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a #RRGGBB colour, got: {colour}")
    red, green, blue = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"
    
def run_name_from_text(text: str) -> str:
    parts = [part for part in str(text).replace("\\", "/").split("/") if part]
    if "runs" in parts and parts.index("runs") + 1 < len(parts):
        return parts[parts.index("runs") + 1]
    return parts[-2] if len(parts) >= 2 else str(text)


def random_quantiles(counts: np.ndarray, n_keep: int, repeats: int,
                     rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_total = int(counts.sum())
    scale = n_total / n_keep
    draws = np.empty((repeats, len(counts)), dtype=np.float64)
    for repeat in range(repeats):
        draws[repeat] = rng.multivariate_hypergeometric(counts, n_keep) * scale
    q05, median, q95 = np.quantile(draws, [0.05, 0.50, 0.95], axis=0)
    return q05, median, q95


def hybrid_sequential_indices(n_total: int, n_keep: int, n_chunks: int = 5) -> np.ndarray:
    """Select five equally spaced contiguous chunks totalling exactly n_keep rows."""
    if n_total < 1:
        return np.empty(0, dtype=np.int64)
    n_keep = min(n_total, max(1, int(n_keep)))
    n_chunks = min(max(1, int(n_chunks)), n_total)
    stratum_edges = np.floor(np.arange(n_chunks + 1) * n_total / n_chunks).astype(np.int64)
    sample_edges = np.floor(np.arange(n_chunks + 1) * n_keep / n_chunks).astype(np.int64)
    sizes = np.diff(sample_edges)
    pieces = [np.arange(start, start + int(size), dtype=np.int64)
              for start, size in zip(stratum_edges[:-1], sizes) if size > 0]
    indices = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
    if len(indices) != n_keep or len(np.unique(indices)) != n_keep:
        raise RuntimeError(f"Invalid hybrid sequential sample: {len(indices)} rows; expected {n_keep}")
    return indices


def append_analysis(rows: list[dict], metadata: dict, ordered_labels: Sequence[str],
                    fractions: Sequence[float], repeats: int,
                    rng: np.random.Generator, allow_sequential: bool = True) -> None:
    labels = np.asarray([str(label) for label in ordered_labels if pd.notna(label) and str(label) != ""])
    if labels.size == 0:
        return
    classes, encoded = np.unique(labels, return_inverse=True)
    counts = np.bincount(encoded, minlength=len(classes)).astype(np.int64)
    n_total = len(encoded)

    for fraction in fractions:
        n_keep = min(n_total, max(1, math.ceil(n_total * fraction)))
        scale = n_total / n_keep
        q05, median, q95 = random_quantiles(counts, n_keep, repeats, rng)
        hybrid_sequential = None
        if allow_sequential:
            hybrid_indices = hybrid_sequential_indices(n_total, n_keep, n_chunks=5)
            hybrid_sequential = np.bincount(
                encoded[hybrid_indices], minlength=len(classes)
            ) * scale
        for index, class_name in enumerate(classes):
            full = int(counts[index])
            rows.append({
                **metadata, "class": class_name, "fraction": fraction, "mode": "random",
                "n_total": n_total, "n_kept": n_keep, "scale_factor": scale,
                "full_count": full, "estimate": median[index], "p05": q05[index],
                "p95": q95[index], "interval_width": q95[index] - q05[index],
                "absolute_error": median[index] - full,
                "relative_absolute_error": abs(median[index] - full) / full if full else np.nan,
                "repeats": repeats, "source_supports_sequential": allow_sequential,
            })
            if hybrid_sequential is not None:
                estimate = float(hybrid_sequential[index])
                rows.append({
                    **metadata, "class": class_name, "fraction": fraction,
                    "mode": "hybrid_sequential_five_chunks", "n_total": n_total,
                    "n_kept": n_keep, "scale_factor": scale, "full_count": full,
                    "estimate": estimate, "p05": np.nan, "p95": np.nan,
                    "interval_width": np.nan, "absolute_error": estimate - full,
                    "relative_absolute_error": abs(estimate - full) / full if full else np.nan,
                    "repeats": 1, "source_supports_sequential": True,
                })


def analyse_ordered_cache(df: pd.DataFrame, fractions: Sequence[float], repeats: int,
                          seed: int, label_column: Optional[str], group_column: Optional[str],
                          order_column: Optional[str], timestamp_column: Optional[str]) -> pd.DataFrame:
    label_column = label_column or first_present(df.columns, LABEL_CANDIDATES)
    group_column = group_column or first_present(df.columns, GROUP_CANDIDATES)
    order_column = order_column or first_present(df.columns, ORDER_CANDIDATES)
    timestamp_column = timestamp_column or first_present(df.columns, TIME_CANDIDATES)
    if label_column is None:
        raise ValueError(
            "Ordered cache detected, but no label column was recognised. "
            f"Columns found: {', '.join(map(str, df.columns))}. Use --label-column."
        )
    if group_column is None:
        # A cache containing one acquisition is still valid; preserve its physical row order.
        group_column = "__single_input_file__"
        df = df.copy()
        df[group_column] = "ordered_predictions_cache.csv"

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for group_value, group in df.groupby(group_column, sort=False, dropna=False):
        group = group.copy()
        if order_column:
            group = group.sort_values(order_column, kind="stable")
        # With no explicit order column, pandas preserves the input CSV row order.
        labels = group[label_column].dropna().astype(str).tolist()
        if not labels:
            continue
        timestamp = pd.NaT
        if timestamp_column and group[timestamp_column].notna().any():
            timestamp = pd.to_datetime(group[timestamp_column].dropna().iloc[0], dayfirst=True, errors="coerce")
        if pd.isna(timestamp):
            timestamp = timestamp_from_text(group_value)
        metadata = {
            "timestamp": timestamp,
            "run_name": run_name_from_text(group_value),
            "blob_name": str(group_value),
            "input_kind": "ordered_predictions_cache",
        }
        append_analysis(rows, metadata, labels, fractions, repeats, rng, allow_sequential=True)
        print(f"Processed ordered cache group {group_value}: {len(labels)} particles")
    if not rows:
        raise RuntimeError("No labelled rows were found in the ordered cache")
    return pd.DataFrame(rows)


def analyse_legacy(df: pd.DataFrame, fractions: Sequence[float], repeats: int,
                   seed: int) -> pd.DataFrame:
    missing = LEGACY_REQUIRED - set(df.columns)
    if missing:
        raise ValueError("Local CSV schema not recognised; missing: " + ", ".join(sorted(missing)))
    work = df.copy()
    for column, fallback in (("timestamp", pd.NaT), ("run_name", "unknown"), ("blob_name", "unknown")):
        if column not in work:
            work[column] = fallback
    work["timestamp"] = pd.to_datetime(work["timestamp"], dayfirst=True, errors="coerce")
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    keys = ["timestamp", "run_name", "blob_name", "n_full"]
    for key, group in work.groupby(keys, sort=False, dropna=False):
        classes = group["class"].astype(str).tolist()
        counts = pd.to_numeric(group["full_count"], errors="raise").to_numpy(dtype=np.int64)
        total = int(key[3])
        residual = total - int(counts.sum())
        if residual < 0:
            raise ValueError(f"Class totals exceed n_full for {key[2]}")
        if residual:
            classes.append("__unreported__")
            counts = np.append(counts, residual)
        # Expand totals to labels only conceptually; random draws are computed directly below.
        for fraction in fractions:
            n_keep = min(total, max(1, math.ceil(total * fraction)))
            scale = total / n_keep
            q05, median, q95 = random_quantiles(counts, n_keep, repeats, rng)
            for i, class_name in enumerate(classes):
                if class_name == "__unreported__":
                    continue
                full = int(counts[i])
                rows.append({
                    "timestamp": key[0], "run_name": key[1], "blob_name": key[2],
                    "input_kind": "legacy_class_counts", "class": class_name,
                    "fraction": fraction, "mode": "random", "n_total": total,
                    "n_kept": n_keep, "scale_factor": scale, "full_count": full,
                    "estimate": median[i], "p05": q05[i], "p95": q95[i],
                    "interval_width": q95[i] - q05[i],
                    "absolute_error": median[i] - full,
                    "relative_absolute_error": abs(median[i] - full) / full if full else np.nan,
                    "repeats": repeats, "source_supports_sequential": False,
                })
    return pd.DataFrame(rows)


def analyse_blob(source: str, fractions: Sequence[float], repeats: int, seed: int) -> pd.DataFrame:
    try:
        from azure.identity import InteractiveBrowserCredential
        from azure.storage.blob import BlobServiceClient
    except ImportError as exc:
        raise RuntimeError("Azure input requires: pip install azure-identity azure-storage-blob") from exc
    account, container, prefix = parse_blob_url(source) or (None, None, None)
    if account is None:
        raise ValueError("Input is not an Azure Blob URL")
    service = BlobServiceClient(account_url=account, credential=InteractiveBrowserCredential())
    client = service.get_container_client(container)
    names = sorted(blob.name for blob in client.list_blobs(name_starts_with=prefix.strip("/"))
                   if blob.name.lower().endswith("predictions.json"))
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for name in names:
        try:
            payload = json.loads(client.get_blob_client(name).download_blob().readall().decode("utf-8-sig"))
            labels = [label_from_record(record) for record in records_from_payload(payload)]
            labels = [label for label in labels if label is not None]
            append_analysis(rows, {
                "timestamp": timestamp_from_text(name), "run_name": run_name_from_text(name),
                "blob_name": name, "input_kind": "azure_predictions_json",
            }, labels, fractions, repeats, rng, allow_sequential=True)
            print(f"Processed {name}: {len(labels)} particles")
        except Exception as exc:
            print(f"Skipping {name}: {exc}")
    if not rows:
        raise RuntimeError("No usable predictions.json data were processed")
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (mode, fraction, class_name), group in results.groupby(["mode", "fraction", "class"], dropna=False):
        relative = pd.to_numeric(group["relative_absolute_error"], errors="coerce").dropna()
        records.append({
            "mode": mode, "fraction": fraction, "class": class_name,
            "timepoints": len(group),
            "median_relative_absolute_error": relative.median() if len(relative) else np.nan,
            "p95_relative_absolute_error": relative.quantile(0.95) if len(relative) else np.nan,
            "median_interval_width": pd.to_numeric(group["interval_width"], errors="coerce").median(),
        })
    return pd.DataFrame(records)


def make_report(results: pd.DataFrame, output: Path, repeats: int) -> None:
    """Write an interactive HTML report.

    The report includes:
      1. A time series for the 5% sample only.
      2. Log-log scatterplot facets split by retained percentage and sampling mode.

    This function expects the parent script to already import math, numpy as np,
    pandas as pd, and Path, and to define hex_to_rgba().
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError("Figures require: pip install plotly") from exc

    data = results.copy()
    for column in ("fraction", "full_count", "estimate", "p05", "p95"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    if "run_name" not in data.columns:
        data["run_name"] = ""
    data = data.dropna(subset=["fraction", "full_count", "estimate"])

    fractions = sorted(data["fraction"].unique())
    classes = sorted(data["class"].dropna().astype(str).unique())
    palette = ["#0078D4", "#E3008C", "#107C10", "#FF8C00", "#5C2D91", "#00B7C3"]
    class_colour = {name: palette[i % len(palette)] for i, name in enumerate(classes)}
    blocks: list[str] = []

    # Time-series output is intentionally restricted to the 5% sample.
    five = data[np.isclose(data["fraction"], 0.05)].copy()
    hybrid_at_five = bool((five["mode"] == "hybrid_sequential_five_chunks").any())
    if not five.empty:
        nrows = 2 if hybrid_at_five else 1
        titles = ["Random 5%: median and 5th-95th percentile envelope"]
        if hybrid_at_five:
            titles.append("Hybrid sequential 5%: five equally spaced chunks versus complete data")

        fig = make_subplots(
            rows=nrows,
            cols=1,
            shared_xaxes=True,
            subplot_titles=titles,
            vertical_spacing=0.10,
        )
        random = five[five["mode"] == "random"]
        hybrid = five[five["mode"] == "hybrid_sequential_five_chunks"]

        for class_name in classes:
            colour = class_colour[class_name]
            group = random[random["class"].astype(str) == class_name].sort_values("timestamp")
            if not group.empty:
                fig.add_trace(
                    go.Scatter(
                        x=group["timestamp"],
                        y=group["p95"],
                        mode="lines",
                        line={"width": 0},
                        showlegend=False,
                        legendgroup=class_name,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=group["timestamp"],
                        y=group["p05"],
                        mode="lines",
                        line={"width": 0},
                        fill="tonexty",
                        fillcolor=hex_to_rgba(colour, 0.15),
                        showlegend=False,
                        legendgroup=class_name,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=group["timestamp"],
                        y=group["full_count"],
                        mode="lines",
                        name=f"{class_name} true",
                        legendgroup=class_name,
                        line={"color": colour},
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=group["timestamp"],
                        y=group["estimate"],
                        mode="lines",
                        name=f"{class_name} random median",
                        legendgroup=class_name,
                        line={"color": colour, "dash": "dash"},
                    ),
                    row=1,
                    col=1,
                )

            if hybrid_at_five:
                group = hybrid[
                    hybrid["class"].astype(str) == class_name
                ].sort_values("timestamp")
                if not group.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=group["timestamp"],
                            y=group["full_count"],
                            mode="lines",
                            showlegend=False,
                            legendgroup=class_name,
                            line={"color": colour},
                        ),
                        row=2,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=group["timestamp"],
                            y=group["estimate"],
                            mode="lines",
                            name=f"{class_name} hybrid sequential five chunks",
                            legendgroup=class_name,
                            line={"color": colour, "dash": "dot"},
                        ),
                        row=2,
                        col=1,
                    )

        fig.update_layout(
            height=1000 if hybrid_at_five else 650,
            hovermode="x unified",
            margin={"r": 280},
            legend={"x": 1.01, "groupclick": "togglegroup"},
            title="Time-series detail for retaining 5% of particles",
        )
        fig.update_yaxes(title_text="Class count")
        fig.update_xaxes(
            title_text="Time",
            rangeslider_visible=True,
            row=nrows,
            col=1,
        )
        blocks.append(pio.to_html(fig, full_html=False, include_plotlyjs="cdn"))

    # Create one log-log facet for every percentage/mode combination.
    # Non-positive values cannot be represented on logarithmic axes and are
    # therefore excluded from the scatterplots, but remain in the source data.
    preferred_modes = ["random", "hybrid_sequential_five_chunks"]
    modes = [mode for mode in preferred_modes if (data["mode"] == mode).any()]
    modes.extend(
        mode
        for mode in data["mode"].dropna().astype(str).unique()
        if mode not in modes
    )
    facet_specs = [
        (fraction, mode)
        for fraction in fractions
        for mode in modes
        if not data[
            np.isclose(data["fraction"], fraction) & (data["mode"] == mode)
        ].empty
    ]

    if facet_specs:
        ncols = 2
        nrows = math.ceil(len(facet_specs) / ncols)
        mode_title = {
            "random": "Random",
            "hybrid_sequential_five_chunks": "Hybrid sequential",
        }
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=[
                f"{fraction:.0%} | {mode_title.get(str(mode), str(mode))}"
                for fraction, mode in facet_specs
            ],
            horizontal_spacing=0.10,
            vertical_spacing=min(0.14, 0.8 / max(nrows, 1)),
        )
        legend_seen: set[str] = set()
        excluded_nonpositive = 0

        for facet_i, (fraction, mode) in enumerate(facet_specs):
            row, col_zero_based = divmod(facet_i, ncols)
            row += 1
            col = col_zero_based + 1
            facet_all = data[
                np.isclose(data["fraction"], fraction) & (data["mode"] == mode)
            ].copy()
            valid = (facet_all["full_count"] > 0) & (facet_all["estimate"] > 0)
            excluded_nonpositive += int((~valid).sum())
            facet = facet_all[valid]

            if facet.empty:
                fig.add_annotation(
                    text="No positive count pairs",
                    x=0.5,
                    y=0.5,
                    xref=f"x{facet_i + 1} domain" if facet_i else "x domain",
                    yref=f"y{facet_i + 1} domain" if facet_i else "y domain",
                    showarrow=False,
                    row=row,
                    col=col,
                )
                fig.update_xaxes(type="log", title_text="True count", row=row, col=col)
                fig.update_yaxes(type="log", title_text="Subsampled count", row=row, col=col)
                continue

            minimum = float(
                np.nanmin(np.r_[facet["full_count"].values, facet["estimate"].values])
            )
            maximum = float(
                np.nanmax(np.r_[facet["full_count"].values, facet["estimate"].values])
            )
            lower = 10 ** (math.floor(math.log10(minimum)) - 0.05)
            upper = 10 ** (math.ceil(math.log10(maximum)) + 0.05)

            fig.add_trace(
                go.Scatter(
                    x=[lower, upper],
                    y=[lower, upper],
                    mode="lines",
                    line={"color": "#777777", "dash": "dash", "width": 1},
                    name="Perfect agreement",
                    legendgroup="identity",
                    showlegend=facet_i == 0,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

            for class_name in classes:
                group = facet[facet["class"].astype(str) == class_name]
                if group.empty:
                    continue
                showlegend = class_name not in legend_seen
                fig.add_trace(
                    go.Scatter(
                        x=group["full_count"],
                        y=group["estimate"],
                        mode="markers",
                        name=class_name,
                        legendgroup=class_name,
                        showlegend=showlegend,
                        marker={
                            "color": class_colour[class_name],
                            "symbol": "circle",
                            "size": 8,
                            "opacity": 0.72,
                            "line": {"width": 0.7, "color": "white"},
                        },
                        customdata=np.column_stack(
                            [group["run_name"].fillna("").astype(str)]
                        ),
                        hovertemplate=(
                            "Class: "
                            + class_name
                            + "<br>Run: %{customdata[0]}"
                            + "<br>True count: %{x:,.0f}"
                            + "<br>Subsampled count: %{y:,.1f}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
                if showlegend:
                    legend_seen.add(class_name)

            fig.update_xaxes(
                type="log",
                title_text="True count" if row == nrows else None,
                range=[math.log10(lower), math.log10(upper)],
                row=row,
                col=col,
            )
            fig.update_yaxes(
                type="log",
                title_text="Subsampled count" if col == 1 else None,
                range=[math.log10(lower), math.log10(upper)],
                row=row,
                col=col,
            )

        fig.update_layout(
            title="Log-log true versus subsampled class counts by retained percentage and sampling method",
            height=max(620, 390 * nrows),
            legend={"title": {"text": "Class"}, "groupclick": "togglegroup"},
            margin={"t": 110, "r": 40, "b": 70, "l": 75},
        )
        blocks.append(
            pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs=False if blocks else "cdn",
            )
        )
    else:
        excluded_nonpositive = 0

    time_note = (
        "The time-series view is restricted to 5%. The shaded band is the "
        "random-sampling 5th-95th percentile envelope."
        if not five.empty
        else "No 5% result was present. Include 0.05 in --fractions to create the time series."
    )
    exclusion_note = (
        f"Logarithmic axes require positive values. {excluded_nonpositive:,} "
        "non-positive true/subsampled count pairs were omitted from the scatterplots."
        if excluded_nonpositive
        else "All true/subsampled count pairs shown on the logarithmic axes were positive."
    )

    output.write_text(
        f"""<!doctype html>
<meta charset='utf-8'>
<title>Particle sampling sufficiency</title>
<style>
body{{font-family:Segoe UI,Arial;max-width:1500px;margin:30px auto;padding:0 25px}}
p{{max-width:1100px;line-height:1.55}}
.note{{background:#f3f8fc;border-left:5px solid #0078d4;padding:14px;margin:18px 0}}
</style>
<h1>How few particles can be sampled while preserving class counts?</h1>
<p>ML labels are fixed. Random estimates are medians from {repeats:,} samples without replacement.</p>
<div class='note'>{time_note}</div>
<div class='note'>Each retained percentage has separate random and sequential log-log scatterplots. Colour identifies class, and the grey diagonal is perfect agreement.</div>
<div class='note'>{exclusion_note}</div>
{''.join(blocks)}
""",
        encoding="utf-8",
    )



def _fit_flat_then_slope(
    x_log: np.ndarray,
    y_log: np.ndarray,
    min_segment: int,
) -> Optional[dict]:
    """Robust L1 fit: y = intercept + slope * max(0, x - breakpoint).

    Before the breakpoint the fitted response is exactly flat. After it, the
    response is linear in log-log space; a slope of -1 is inverse proportionality.
    Candidate breakpoints are observed x values satisfying the segment-size rule.
    """
    mask = np.isfinite(x_log) & np.isfinite(y_log)
    x = np.asarray(x_log[mask], dtype=float)
    y = np.asarray(y_log[mask], dtype=float)
    n = len(x)
    if n < max(2 * min_segment, 6) or np.unique(x).size < 3:
        return None

    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    candidates = np.unique(x)[1:-1]
    candidates = np.asarray([
        point for point in candidates
        if np.count_nonzero(x <= point) >= min_segment
        and np.count_nonzero(x > point) >= min_segment
    ], dtype=float)
    if not len(candidates):
        return None

    best = None
    for breakpoint in candidates:
        hinge = np.maximum(0.0, x - breakpoint)
        positive = hinge > 0
        if np.unique(hinge).size < 2 or np.count_nonzero(positive) < min_segment:
            continue
        # Theil-Sen-type post-break slope using pairwise slopes between the
        # flat-side observations (hinge=0) and post-break observations.
        left_y = y[~positive]
        right_y = y[positive]
        right_h = hinge[positive]
        slopes = ((right_y[:, None] - left_y[None, :]) / right_h[:, None]).ravel()
        slope = float(np.median(slopes))
        intercept = float(np.median(y - slope * hinge))
        fitted = intercept + slope * hinge
        loss = float(np.sum(np.abs(y - fitted)))
        candidate = {
            "breakpoint_log": float(breakpoint),
            "intercept": intercept,
            "slope": slope,
            "loss": loss,
            "x": x,
            "y": y,
            "fitted": fitted,
        }
        if best is None or candidate["loss"] < best["loss"]:
            best = candidate
    return best


def _threshold_inference(
    x: np.ndarray,
    y: np.ndarray,
    min_segment: int,
    bootstraps: int,
    permutations: int,
    rng: np.random.Generator,
) -> Optional[dict]:
    """Fit and test a flat-then-log-linear threshold relationship robustly.

    The null model is a flat median. The permutation p-value tests whether the
    optimised broken-line reduction in absolute error is stronger than expected
    when y is unrelated to x. Bootstrap percentile intervals describe uncertainty
    in the breakpoint and post-break slope.
    """
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = np.asarray(x[valid], dtype=float)
    y = np.asarray(y[valid], dtype=float)
    if len(x) < max(2 * min_segment, 6):
        return None
    x_log = np.log10(x)
    y_log = np.log10(y)
    fit = _fit_flat_then_slope(x_log, y_log, min_segment)
    if fit is None:
        return None

    null_loss = float(np.sum(np.abs(y_log - np.median(y_log))))
    observed_improvement = null_loss - fit["loss"]
    permuted_improvements = []
    for _ in range(max(0, permutations)):
        permuted = _fit_flat_then_slope(x_log, rng.permutation(y_log), min_segment)
        if permuted is not None:
            permuted_improvements.append(null_loss - permuted["loss"])
    if permuted_improvements:
        p_value = (
            1 + np.count_nonzero(np.asarray(permuted_improvements) >= observed_improvement)
        ) / (len(permuted_improvements) + 1)
    else:
        p_value = np.nan

    breakpoints = []
    slopes = []
    n = len(x_log)
    for _ in range(max(0, bootstraps)):
        indices = rng.integers(0, n, size=n)
        bootstrap_fit = _fit_flat_then_slope(x_log[indices], y_log[indices], min_segment)
        if bootstrap_fit is not None:
            breakpoints.append(10 ** bootstrap_fit["breakpoint_log"])
            slopes.append(bootstrap_fit["slope"])

    fit["breakpoint"] = 10 ** fit["breakpoint_log"]
    fit["p_value"] = float(p_value)
    fit["breakpoint_ci"] = (
        tuple(np.quantile(breakpoints, [0.025, 0.975]))
        if len(breakpoints) >= 20 else (np.nan, np.nan)
    )
    fit["slope_ci"] = (
        tuple(np.quantile(slopes, [0.025, 0.975]))
        if len(slopes) >= 20 else (np.nan, np.nan)
    )
    fit["bootstrap_successes"] = len(breakpoints)
    fit["n"] = n
    return fit


def make_inter_class_report(
    results: pd.DataFrame,
    output: Path,
    seed: int = 42,
    bootstraps: int = 1000,
    permutations: int = 999,
    min_segment: int = 5,
) -> None:
    """Write a separate additive HTML report on Bubbles inter-class effects.

    Section 2a plots true Bubbles count against true other-class counts.
    Section 2b plots composition fractions and estimates a robust flat-then-
    log-linear breakpoint for each other class. Section 2d is intentionally absent.
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError("Figures require: pip install plotly") from exc

    if min_segment < 3:
        raise ValueError("min_segment must be at least 3")
    data = results.copy()
    missing = {"class", "full_count"} - set(data.columns)
    if missing:
        raise ValueError("Inter-class analysis is missing: " + ", ".join(sorted(missing)))
    data["class"] = data["class"].astype(str)
    data["full_count"] = pd.to_numeric(data["full_count"], errors="coerce")
    if "n_total" in data.columns:
        data["n_total"] = pd.to_numeric(data["n_total"], errors="coerce")
    acquisition_columns = [
        column for column in ("timestamp", "run_name", "blob_name", "input_kind", "n_total")
        if column in data.columns
    ]
    if not acquisition_columns:
        raise ValueError("Inter-class analysis needs an acquisition identifier")
    data = data.dropna(subset=["full_count"]).drop_duplicates(
        subset=acquisition_columns + ["class"], keep="first"
    )
    data["acquisition_id"] = data[acquisition_columns].astype(str).agg(" | ".join, axis=1)
    counts = data.pivot_table(
        index="acquisition_id", columns="class", values="full_count",
        aggfunc="first", fill_value=0,
    ).astype(float)
    matches = [name for name in counts.columns if str(name).strip().casefold() == "bubbles"]
    if not matches:
        raise ValueError("Class 'bubbles' was not found. Available: " + ", ".join(map(str, counts.columns)))
    bubble_class = matches[0]
    other_classes = [name for name in counts.columns if name != bubble_class]
    if not other_classes:
        raise ValueError("No non-bubbles classes were found")

    palette = ["#0078D4", "#E3008C", "#107C10", "#FF8C00", "#5C2D91", "#00B7C3"]
    colours = {name: palette[i % len(palette)] for i, name in enumerate(other_classes)}
    ncols = min(3, len(other_classes))
    nrows = math.ceil(len(other_classes) / ncols)
    blocks = []

    # 2a: absolute true counts, with additive robust threshold inference.
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=list(map(str, other_classes)))
    excluded_counts = 0
    count_statistical_rows = []
    count_rng = np.random.default_rng(seed + 1)
    for i, class_name in enumerate(other_classes):
        row, col0 = divmod(i, ncols); row += 1; col = col0 + 1
        frame = pd.DataFrame({"x": counts[bubble_class], "y": counts[class_name]})
        valid = (frame["x"] > 0) & (frame["y"] > 0)
        excluded_counts += int((~valid).sum())
        frame = frame.loc[valid]
        fig.add_trace(go.Scatter(
            x=frame["x"], y=frame["y"], mode="markers", showlegend=False,
            marker={"color": colours[class_name], "size": 8, "opacity": 0.72},
            customdata=np.asarray(frame.index.astype(str))[:, None],
            hovertemplate="Acquisition: %{customdata[0]}<br>Bubbles: %{x:,.0f}<br>Other class: %{y:,.0f}<extra></extra>",
        ), row=row, col=col)
        count_inference = _threshold_inference(
            frame["x"].to_numpy(), frame["y"].to_numpy(), min_segment,
            bootstraps, permutations, count_rng,
        )
        if count_inference is not None:
            x_grid = np.logspace(np.log10(frame["x"].min()), np.log10(frame["x"].max()), 250)
            y_grid = 10 ** (
                count_inference["intercept"]
                + count_inference["slope"] * np.maximum(
                    0, np.log10(x_grid) - count_inference["breakpoint_log"]
                )
            )
            fig.add_trace(go.Scatter(
                x=x_grid, y=y_grid, mode="lines", showlegend=False,
                line={"color": "#202020", "width": 2},
                hovertemplate="Robust fitted relationship<extra></extra>",
            ), row=row, col=col)
            fig.add_vline(
                x=count_inference["breakpoint"], line_color="#D13438", line_width=2,
                line_dash="dash", row=row, col=col,
            )
            low_b, high_b = count_inference["breakpoint_ci"]
            low_s, high_s = count_inference["slope_ci"]
            significant_break = np.isfinite(count_inference["p_value"]) and count_inference["p_value"] < 0.05
            inverse_compatible = np.isfinite(low_s) and low_s <= -1 <= high_s
            negative_supported = np.isfinite(high_s) and high_s < 0
            interpretation = (
                "supported transition; post-break slope is compatible with inverse proportionality"
                if significant_break and inverse_compatible and negative_supported
                else "supported transition; post-break relationship is negative but not compatible with slope -1"
                if significant_break and negative_supported
                else "no statistically supported flat-to-decreasing transition"
            )
            count_statistical_rows.append({
                "class": class_name, "n": count_inference["n"],
                "breakpoint": count_inference["breakpoint"], "breakpoint_low": low_b,
                "breakpoint_high": high_b, "slope": count_inference["slope"],
                "slope_low": low_s, "slope_high": high_s,
                "p_value": count_inference["p_value"], "interpretation": interpretation,
            })
            label = (
                f"Breakpoint {count_inference['breakpoint']:,.0f}<br>"
                f"95% CI {low_b:,.0f} to {high_b:,.0f}<br>"
                f"post-break slope {count_inference['slope']:.2f}<br>"
                f"permutation p={count_inference['p_value']:.3g}"
            )
            fig.add_annotation(
                x=count_inference["breakpoint"], y=0.98,
                xref=f"x{i+1}" if i else "x",
                yref=f"y{i+1} domain" if i else "y domain", text=label,
                showarrow=False, xanchor="left", yanchor="top",
                bgcolor="rgba(255,255,255,0.82)", font={"size": 10},
            )
        else:
            count_statistical_rows.append({
                "class": class_name, "n": len(frame), "breakpoint": np.nan,
                "breakpoint_low": np.nan, "breakpoint_high": np.nan,
                "slope": np.nan, "slope_low": np.nan, "slope_high": np.nan,
                "p_value": np.nan, "interpretation": "insufficient data for threshold fit",
            })
        fig.update_xaxes(type="log", title_text="Bubbles true count" if row == nrows else None, row=row, col=col)
        fig.update_yaxes(type="log", title_text="Other-class true count" if col == 1 else None, row=row, col=col)
    fig.update_layout(
        title="2a. Bubbles true count versus true counts of other classes with robust change points",
        height=max(620, 400*nrows),
    )
    blocks.append(pio.to_html(fig, full_html=False, include_plotlyjs="cdn"))

    # 2b: compositional fractions plus robust threshold inference.
    fractions = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=list(map(str, other_classes)))
    rng = np.random.default_rng(seed)
    statistical_rows = []
    excluded_fractions = 0
    for i, class_name in enumerate(other_classes):
        row, col0 = divmod(i, ncols); row += 1; col = col0 + 1
        frame = pd.DataFrame({"x": fractions[bubble_class], "y": fractions[class_name]})
        valid = np.isfinite(frame["x"]) & np.isfinite(frame["y"]) & (frame["x"] > 0) & (frame["y"] > 0)
        excluded_fractions += int((~valid).sum())
        frame = frame.loc[valid]
        fig.add_trace(go.Scatter(
            x=frame["x"], y=frame["y"], mode="markers", showlegend=False,
            marker={"color": colours[class_name], "size": 8, "opacity": 0.68},
            customdata=np.asarray(frame.index.astype(str))[:, None],
            hovertemplate="Acquisition: %{customdata[0]}<br>Bubbles fraction: %{x:.3%}<br>Other-class fraction: %{y:.3%}<extra></extra>",
        ), row=row, col=col)
        inference = _threshold_inference(
            frame["x"].to_numpy(), frame["y"].to_numpy(), min_segment,
            bootstraps, permutations, rng,
        )
        if inference is not None:
            x_grid = np.logspace(np.log10(frame["x"].min()), np.log10(frame["x"].max()), 250)
            y_grid = 10 ** (
                inference["intercept"]
                + inference["slope"] * np.maximum(0, np.log10(x_grid) - inference["breakpoint_log"])
            )
            fig.add_trace(go.Scatter(
                x=x_grid, y=y_grid, mode="lines", showlegend=False,
                line={"color": "#202020", "width": 2},
                hovertemplate="Robust fitted relationship<extra></extra>",
            ), row=row, col=col)
            fig.add_vline(
                x=inference["breakpoint"], line_color="#D13438", line_width=2,
                line_dash="dash", row=row, col=col,
            )
            low_b, high_b = inference["breakpoint_ci"]
            low_s, high_s = inference["slope_ci"]
            significant_break = np.isfinite(inference["p_value"]) and inference["p_value"] < 0.05
            inverse_compatible = np.isfinite(low_s) and low_s <= -1 <= high_s
            negative_supported = np.isfinite(high_s) and high_s < 0
            interpretation = (
                "supported transition; post-break slope is compatible with inverse proportionality"
                if significant_break and inverse_compatible and negative_supported
                else "supported transition; post-break relationship is negative but not compatible with slope -1"
                if significant_break and negative_supported
                else "no statistically supported flat-to-decreasing transition"
            )
            statistical_rows.append({
                "class": class_name, "n": inference["n"],
                "breakpoint": inference["breakpoint"], "breakpoint_low": low_b,
                "breakpoint_high": high_b, "slope": inference["slope"],
                "slope_low": low_s, "slope_high": high_s,
                "p_value": inference["p_value"], "interpretation": interpretation,
                "bootstrap_successes": inference["bootstrap_successes"],
            })
            label = (
                f"Breakpoint {inference['breakpoint']:.2%}<br>"
                f"95% CI {low_b:.2%} to {high_b:.2%}<br>"
                f"post-break slope {inference['slope']:.2f}<br>"
                f"permutation p={inference['p_value']:.3g}"
            )
            fig.add_annotation(
                x=inference["breakpoint"], y=0.98, xref=f"x{i+1}" if i else "x",
                yref=f"y{i+1} domain" if i else "y domain", text=label,
                showarrow=False, xanchor="left", yanchor="top",
                bgcolor="rgba(255,255,255,0.82)", font={"size": 10},
            )
        else:
            statistical_rows.append({
                "class": class_name, "n": len(frame), "breakpoint": np.nan,
                "breakpoint_low": np.nan, "breakpoint_high": np.nan,
                "slope": np.nan, "slope_low": np.nan, "slope_high": np.nan,
                "p_value": np.nan, "interpretation": "insufficient data for threshold fit",
                "bootstrap_successes": 0,
            })
        fig.update_xaxes(type="log", tickformat=".1%", title_text="Bubbles fraction" if row == nrows else None, row=row, col=col)
        fig.update_yaxes(type="log", tickformat=".1%", title_text="Other-class fraction" if col == 1 else None, row=row, col=col)
    fig.update_layout(title="2b. Bubbles fraction versus other-class fraction with robust change points", height=max(620, 400*nrows))
    blocks.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    stats = pd.DataFrame(statistical_rows)
    def fmt_percent(value):
        return "NA" if not np.isfinite(value) else f"{value:.2%}"
    def fmt_number(value, digits=3):
        return "NA" if not np.isfinite(value) else f"{value:.{digits}g}"
    table_rows = []
    for record in stats.to_dict("records"):
        table_rows.append(
            "<tr>"
            f"<td>{record['class']}</td><td>{record['n']}</td>"
            f"<td>{fmt_percent(record['breakpoint'])}</td>"
            f"<td>{fmt_percent(record['breakpoint_low'])} to {fmt_percent(record['breakpoint_high'])}</td>"
            f"<td>{fmt_number(record['slope'])}</td>"
            f"<td>{fmt_number(record['slope_low'])} to {fmt_number(record['slope_high'])}</td>"
            f"<td>{fmt_number(record['p_value'])}</td>"
            f"<td>{record['interpretation']}</td>"
            "</tr>"
        )
    count_table_rows = []
    for record in count_statistical_rows:
        breakpoint = "NA" if not np.isfinite(record["breakpoint"]) else f'{record["breakpoint"]:,.0f}'
        breakpoint_low = "NA" if not np.isfinite(record["breakpoint_low"]) else f'{record["breakpoint_low"]:,.0f}'
        breakpoint_high = "NA" if not np.isfinite(record["breakpoint_high"]) else f'{record["breakpoint_high"]:,.0f}'
        count_table_rows.append(
            "<tr>"
            f"<td>{record['class']}</td><td>{record['n']}</td>"
            f"<td>{breakpoint}</td><td>{breakpoint_low} to {breakpoint_high}</td>"
            f"<td>{fmt_number(record['slope'])}</td>"
            f"<td>{fmt_number(record['slope_low'])} to {fmt_number(record['slope_high'])}</td>"
            f"<td>{fmt_number(record['p_value'])}</td>"
            f"<td>{record['interpretation']}</td>"
            "</tr>"
        )
    count_table = (
        "<h2>2a statistical results</h2><table><thead><tr>"
        "<th>Class</th><th>n</th><th>Point of no effect (Bubbles count)</th><th>95% bootstrap CI</th>"
        "<th>Post-break log-log slope</th><th>95% bootstrap CI</th><th>Permutation p</th><th>Conclusion</th>"
        "</tr></thead><tbody>" + "".join(count_table_rows) + "</tbody></table>"
    )
    table = (
        "<h2>2b statistical results</h2><table><thead><tr>"
        "<th>Class</th><th>n</th><th>Point of no effect</th><th>95% bootstrap CI</th>"
        "<th>Post-break log-log slope</th><th>95% bootstrap CI</th><th>Permutation p</th><th>Conclusion</th>"
        "</tr></thead><tbody>" + "".join(table_rows) + "</tbody></table>"
    )

    output.write_text(f"""<!doctype html>
<meta charset='utf-8'><title>Inter-class variability driven by Bubbles</title>
<style>
body{{font-family:Segoe UI,Arial;max-width:1500px;margin:30px auto;padding:0 25px}}
p{{max-width:1150px;line-height:1.55}} .note{{background:#f3f8fc;border-left:5px solid #0078d4;padding:14px;margin:18px 0}}
table{{border-collapse:collapse;width:100%;font-size:14px}} th,td{{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}} th{{background:#f3f8fc}}
</style>
<h1>Section 2: Inter-class variability driven by Bubbles</h1>
<p>This report is additive to the subsampling-sufficiency report and uses each complete-data class count once per acquisition.</p>
<div class='note'>For 2a, a flat-then-log-linear breakpoint analysis is applied to absolute true class counts. The red dashed line is the estimated Bubbles-count point of no effect; the black line is the fitted relationship.</div>
<div class='note'>For 2b, the model is flat before an estimated breakpoint and log-linear afterwards. It is fitted by minimising absolute residuals, making it less sensitive to outliers than least squares. The red dashed line is the estimated point of no effect; the black line is the fitted relationship.</div>
<div class='note'>The permutation test compares the optimised change-point model with a flat median null. A post-break log-log slope of -1 represents inverse proportionality. A claim of inverse proportionality requires p&lt;0.05, a wholly negative bootstrap slope interval, and an interval that contains -1. The breakpoint and slope intervals use {bootstraps:,} bootstrap resamples; p-values use {permutations:,} permutations.</div>
<div class='note'>Log axes omit {excluded_counts:,} non-positive count pairs in 2a and {excluded_fractions:,} non-positive or non-finite fraction pairs in 2b. The fraction denominator is the sum of available true class counts per acquisition.</div>
{''.join(blocks)}
{count_table}
{table}
""", encoding="utf-8")

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Repeated random and five-chunk hybrid-sequential particle subsampling")
#    parser.add_argument("input", nargs="?", help="Azure Blob directory or local CSV")
#    parser.add_argument("--input", help="Azure Blob directory URL or a long results CSV exported by this script", default = "subsampling_sufficiency/ordered_predictions_cache.csv")    
    parser.add_argument("--input", help="Azure Blob directory URL or a long results CSV exported by this script", default = "https://citprodc8603uksa.blob.core.windows.net/ml-prediction-results/runs/guilocalinference-2026-09-05T06-41-57Z")    
    parser.add_argument("--input-csv", help="Alternative named form for a local CSV")
    parser.add_argument("--outdir", type=Path, default=Path("subsampling_sufficiency"))
    parser.add_argument("--fractions", default="0.01,0.02,0.05,0.10,0.20,0.50")
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-column")
    parser.add_argument("--group-column")
    parser.add_argument("--order-column")
    parser.add_argument("--timestamp-column")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--breakpoint-bootstraps", type=int, default=1000)
    parser.add_argument("--breakpoint-permutations", type=int, default=999)
    parser.add_argument("--breakpoint-min-segment", type=int, default=5)
    args = parser.parse_args(argv)
    source = args.input_csv or args.input
    if not source:
        parser.error("provide an Azure Blob directory or local CSV as INPUT, or use --input-csv")
    if args.repeats < 20:
        parser.error("--repeats must be at least 20")
    fractions = parse_fractions(args.fractions)
    local = Path(source).expanduser()

    if local.is_file():
        frame = pd.read_csv(local)
        if RESULT_REQUIRED <= set(frame.columns):
            results = frame
            results["timestamp"] = pd.to_datetime(results["timestamp"], errors="coerce")
        elif LEGACY_REQUIRED <= set(frame.columns) and not first_present(frame.columns, LABEL_CANDIDATES):
            results = analyse_legacy(frame, fractions, args.repeats, args.seed)
        else:
            results = analyse_ordered_cache(
                frame, fractions, args.repeats, args.seed, args.label_column,
                args.group_column, args.order_column, args.timestamp_column,
            )
    elif parse_blob_url(source):
        results = analyse_blob(source, fractions, args.repeats, args.seed)
    else:
        raise ValueError(f"Input does not exist and is not an Azure Blob URL: {source}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    results_path = args.outdir / "subsampling_results_long.csv"
    summary_path = args.outdir / "subsampling_adequacy_summary.csv"
    report_path = args.outdir / "subsampling_sufficiency_report.html"
    inter_class_report_path = args.outdir / "inter_class_variability_report.html"
    results.to_csv(results_path, index=False)
    summarise(results).to_csv(summary_path, index=False)
    print(f"Wrote: {results_path}")
    print(f"Wrote: {summary_path}")
    if not args.no_figures:
        make_report(results, report_path, args.repeats)
        print(f"Wrote: {report_path}")
        make_inter_class_report(
            results, inter_class_report_path, seed=args.seed,
            bootstraps=args.breakpoint_bootstraps,
            permutations=args.breakpoint_permutations,
            min_segment=args.breakpoint_min_segment,
        )
        print(f"Wrote: {inter_class_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
