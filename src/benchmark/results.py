"""Save, load, and aggregate benchmark results."""

import csv
import json
import statistics
from dataclasses import asdict
from pathlib import Path

from .runner import BenchmarkRun, SampleResult
from .dataset import BenchmarkSample


def save_benchmark_run(run: BenchmarkRun, output_dir: Path | str) -> Path:
    """Save a benchmark run to JSON. Returns the saved file path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{run.run_id}.json"

    data = {
        "run_id": run.run_id,
        "timestamp": run.timestamp,
        "engines": run.engines,
        "sample_count": run.sample_count,
        "total_time_seconds": run.total_time_seconds,
        "results": [asdict(r) for r in run.results],
    }

    # Open the file with UTF-8 encoding and write the JSON with indentation
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return path


def load_benchmark_run(path: Path | str) -> BenchmarkRun:
    """Load a benchmark run from JSON."""
    # data = json.loads(Path(path).read_text())
    # results = [SampleResult(**r) for r in data["results"]]

    # data = json.loads(Path(path).read_text(encoding="utf-8"))
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    results = []
    for r in data["results"]:
        # Migrate old runs that used levenshtein instead of llm_score
        if "levenshtein" in r and "llm_score" not in r:
            r["llm_score"] = r.pop("levenshtein")
        r.setdefault("llm_score", 0.0)
        r.setdefault("llm_feedback", "")
        results.append(SampleResult(**r))
    return BenchmarkRun(
        run_id=data["run_id"],
        timestamp=data["timestamp"],
        engines=data["engines"],
        sample_count=data["sample_count"],
        total_time_seconds=data.get("total_time_seconds", 0.0),
        results=results,
    )


def list_benchmark_runs(results_dir: Path | str) -> list[dict]:
    """List available benchmark runs (newest first)."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []
    runs = []
    for f in sorted(results_dir.glob("benchmark_*.json"), reverse=True):
        try:
            # Read the file content as text with UTF-8 encoding
            # file_content = f.read_text(encoding="utf-8")
            # data = json.loads(f.read_text())
            try:
                file_content = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # If UTF-8 fails, try reading with 'windows-1252' or 'ISO-8859-1' encoding
                file_content = f.read_text(encoding="windows-1252")  # or try 'ISO-8859-1'
            data = json.loads(file_content)
            runs.append({
                "run_id": data["run_id"],
                "timestamp": data["timestamp"],
                "engines": data["engines"],
                "sample_count": data["sample_count"],
                "path": str(f),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return runs


def _agg_stats(values: list[float]) -> dict:
    """Compute mean, median, std, min, max for a list of values."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def aggregate_by_engine(run: BenchmarkRun) -> dict:
    """Aggregate metrics per engine.
    Returns: {engine_name: {cer: {mean,...}, wer: ..., llm_score: ..., time: ..., errors: int}}
    """
    by_engine: dict[str, list[SampleResult]] = {}
    for r in run.results:
        by_engine.setdefault(r.engine, []).append(r)

    agg = {}
    for engine, results in by_engine.items():
        valid = [r for r in results if r.error is None]
        agg[engine] = {
            "cer": _agg_stats([r.cer for r in valid]),
            "wer": _agg_stats([r.wer for r in valid]),
            "llm_score": _agg_stats([r.llm_score for r in valid]),
            # "levenshtein": _agg_stats([r.levenshtein for r in valid]),
            "time": {
                **_agg_stats([r.time_seconds for r in valid if r.time_seconds > 0]),
                "sum": sum(r.time_seconds for r in valid),
            },
            "total": len(results),
            "errors": len(results) - len(valid),
        }
    return agg


def aggregate_by_field(
    run: BenchmarkRun,
    samples: list[BenchmarkSample],
    field: str,
) -> dict:
    """Aggregate metrics by a sample field (format or quality).

    Returns: {field_value: {engine: {cer: {mean,...}, wer: ..., levenshtein: ...}}}
    """
    sample_map = {s.id: s for s in samples}

    # Group results by (field_value, engine)
    groups: dict[str, dict[str, list[SampleResult]]] = {}
    for r in run.results:
        sample = sample_map.get(r.sample_id)
        if not sample:
            continue
        field_value = getattr(sample, field, "UNKNOWN")
        groups.setdefault(field_value, {}).setdefault(r.engine, []).append(r)

    agg = {}
    for field_value, engines in sorted(groups.items()):
        agg[field_value] = {}
        for engine, results in engines.items():
            valid = [r for r in results if r.error is None]
            agg[field_value][engine] = {
                "cer": _agg_stats([r.cer for r in valid]),
                "wer": _agg_stats([r.wer for r in valid]),
                # "levenshtein": _agg_stats([r.levenshtein for r in valid]),
                "llm_score": _agg_stats([r.llm_score for r in valid]),
                "count": len(valid),
            }
    return agg


def export_to_csv(run: BenchmarkRun, output_path: Path | str) -> Path:
    """Export all sample results to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        # "sample_id", "engine", "cer", "wer", "levenshtein",
        "sample_id", "engine", "cer", "wer", "llm_score", "llm_feedback",
        "time_seconds", "error",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in run.results:
            writer.writerow({
                "sample_id": r.sample_id,
                "engine": r.engine,
                "cer": f"{r.cer:.4f}",
                "wer": f"{r.wer:.4f}",
                # "levenshtein": f"{r.levenshtein:.4f}",
                "llm_score": f"{r.llm_score:.2f}",
                "llm_feedback": getattr(r, "llm_feedback", ""),

                "time_seconds": f"{r.time_seconds:.2f}",
                "error": r.error or "",
            })
    return output_path
