"""Benchmark runner — executes OCR engines on benchmark samples with caching."""

import json
import time
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

from PIL import Image

from .dataset import BenchmarkSample
from .metrics import compute_all_metrics


@dataclass
class SampleResult:
    """Result of running one engine on one sample."""
    sample_id: str
    engine: str
    ocr_text: str
    cer: float
    wer: float
    levenshtein: float
    time_seconds: float
    error: str | None = None


@dataclass
class BenchmarkRun:
    """Complete results from a benchmark run."""
    run_id: str
    timestamp: str
    engines: list[str]
    sample_count: int
    total_time_seconds: float = 0.0
    results: list[SampleResult] = field(default_factory=list)


class BenchmarkRunner:
    """Runs OCR engines against benchmark samples and caches results."""

    def __init__(self, engines: list[str], cache_dir: Path | str):
        self.engine_names = engines
        self.cache_dir = Path(cache_dir)
        self._engines: dict = {}

    def _get_engine(self, name: str):
        """Lazily initialize OCR engines (same pattern as OCREnsemble)."""
        if name not in self._engines:
            if name == "tesseract":
                from ..ocr.tesseract_engine import TesseractEngine
                self._engines[name] = TesseractEngine()
            elif name == "easyocr":
                from ..ocr.easyocr_engine import EasyOCREngine
                self._engines[name] = EasyOCREngine()
            elif name == "surya":
                from ..ocr.surya_engine import SuryaEngine
                self._engines[name] = SuryaEngine()
            elif name == "ensemble":
                from ..ocr.ensemble import OCREnsemble
                self._engines[name] = OCREnsemble(
                    engines=["tesseract", "easyocr", "surya"]
                )
            else:
                raise ValueError(f"Unknown engine: {name}")
        return self._engines[name]

    def run(
        self,
        samples: list[BenchmarkSample],
        on_progress: Callable[[int, int, str], None] | None = None,
        use_cache: bool = True,
    ) -> BenchmarkRun:
        """Run benchmark on all samples with all configured engines.

        Args:
            samples: List of benchmark samples to process.
            on_progress: Callback(current, total, message) for progress updates.
            use_cache: If True, skip OCR for previously cached results.

        Returns:
            BenchmarkRun with all results.
        """
        run_id = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        total_steps = len(samples) * len(self.engine_names)
        current_step = 0
        all_results: list[SampleResult] = []
        start_time = time.perf_counter()

        for sample in samples:
            for engine_name in self.engine_names:
                current_step += 1
                if on_progress:
                    on_progress(
                        current_step,
                        total_steps,
                        f"Sample {sample.id} — {engine_name} ({current_step}/{total_steps})",
                    )

                result = self._run_single(sample, engine_name, use_cache)
                all_results.append(result)

        total_time = time.perf_counter() - start_time

        return BenchmarkRun(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            engines=self.engine_names,
            sample_count=len(samples),
            total_time_seconds=total_time,
            results=all_results,
        )

    def _run_single(
        self, sample: BenchmarkSample, engine_name: str, use_cache: bool
    ) -> SampleResult:
        """Run a single engine on a single sample."""
        reference = sample.ground_truth_markdown

        # Check cache
        if use_cache:
            cached_text = self._load_cache(sample.id, engine_name)
            if cached_text is not None:
                metrics = compute_all_metrics(cached_text, reference)
                return SampleResult(
                    sample_id=sample.id,
                    engine=engine_name,
                    ocr_text=cached_text,
                    time_seconds=0.0,  # cached
                    **metrics,
                )

        # Run OCR
        start = time.perf_counter()
        try:
            ocr_text = self._extract(engine_name, sample.image)
            elapsed = time.perf_counter() - start
        except Exception as e:
            return SampleResult(
                sample_id=sample.id,
                engine=engine_name,
                ocr_text="",
                cer=1.0,
                wer=1.0,
                levenshtein=0.0,
                time_seconds=time.perf_counter() - start,
                error=f"{type(e).__name__}: {e}",
            )

        # Save to cache
        if use_cache:
            self._save_cache(sample.id, engine_name, ocr_text)

        metrics = compute_all_metrics(ocr_text, reference)
        return SampleResult(
            sample_id=sample.id,
            engine=engine_name,
            ocr_text=ocr_text,
            time_seconds=elapsed,
            **metrics,
        )

    def _extract(self, engine_name: str, image: Image.Image) -> str:
        """Extract text using the named engine."""
        engine = self._get_engine(engine_name)
        if engine_name == "ensemble":
            result = engine.combine(image)
            return result.combined_text
        return engine.extract_text(image)

    # --- Cache helpers ---

    def _cache_path(self, sample_id: str, engine: str) -> Path:
        return self.cache_dir / engine / f"{sample_id}.json"

    def _load_cache(self, sample_id: str, engine: str) -> str | None:
        path = self._cache_path(sample_id, engine)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return data.get("ocr_text")
            except (json.JSONDecodeError, KeyError):
                return None
        return None

    def _save_cache(self, sample_id: str, engine: str, ocr_text: str):
        path = self._cache_path(sample_id, engine)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "sample_id": sample_id,
            "engine": engine,
            "ocr_text": ocr_text,
            "timestamp": datetime.now().isoformat(),
        }, ensure_ascii=False))
