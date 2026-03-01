"""Benchmark runner — executes OCR engines on benchmark samples with caching."""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from PIL import Image

from .dataset import BenchmarkSample
from .metrics import calculate_cer, calculate_wer


@dataclass
class SampleResult:
    """Result of running one engine on one sample."""
    sample_id: str
    engine: str
    ocr_text: str
    cer: float
    wer: float
    llm_score: float        # 0.0–1.0 (GPT score / 10)
    llm_feedback: str       # One-sentence GPT explanation
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

    def __init__(
        self,
        engines: list[str],
        cache_dir: Path | str,
        use_llm_eval: bool = True,
        llm_model: str = "gpt-4o-mini",
    ):
        self.engine_names = engines
        self.cache_dir = Path(cache_dir)
        self.use_llm_eval = use_llm_eval
        self.llm_model = llm_model
        self._engines: dict = {}
        self._evaluator = None

    def _get_engine(self, name: str):
        """Lazily initialize OCR engines."""
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
                self._engines[name] = OCREnsemble(engines=["tesseract", "easyocr", "surya"])
            else:
                raise ValueError(f"Unknown engine: {name}")
        return self._engines[name]

    def _get_evaluator(self):
        if self._evaluator is None:
            from .llm_evaluator import LLMEvaluator
            self._evaluator = LLMEvaluator(cache_dir=self.cache_dir, model=self.llm_model)
        return self._evaluator

    def run(
        self,
        samples: list[BenchmarkSample],
        on_progress: Callable[[int, int, str], None] | None = None,
        use_cache: bool = True,
    ) -> BenchmarkRun:
        """Run benchmark on all samples with all configured engines."""
        run_id = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # OCR + GPT eval = 2 steps per sample/engine pair
        steps_per_pair = 2 if self.use_llm_eval else 1
        total_steps = len(samples) * len(self.engine_names) * steps_per_pair
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
                        f"[OCR] Sample {sample.id} — {engine_name}",
                    )

                result = self._run_single(
                    sample, engine_name, use_cache,
                    on_progress, current_step, total_steps,
                )
                if self.use_llm_eval:
                    current_step += 1
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
        self,
        sample: BenchmarkSample,
        engine_name: str,
        use_cache: bool,
        on_progress: Callable | None,
        current_step: int,
        total_steps: int,
    ) -> SampleResult:
        """Run OCR then GPT evaluation for one sample/engine pair."""
        reference = sample.ground_truth_markdown

        # --- OCR (with cache) ---
        cached_text = self._load_ocr_cache(sample.id, engine_name) if use_cache else None
        if cached_text is not None:
            ocr_text, ocr_time, ocr_error = cached_text, 0.0, None
        else:
            start = time.perf_counter()
            try:
                ocr_text = self._extract(engine_name, sample.image)
                ocr_time = time.perf_counter() - start
                ocr_error = None
                if use_cache:
                    self._save_ocr_cache(sample.id, engine_name, ocr_text)
            except Exception as e:
                ocr_text = ""
                ocr_time = time.perf_counter() - start
                ocr_error = f"{type(e).__name__}: {e}"

        cer = calculate_cer(ocr_text, reference)
        wer = calculate_wer(ocr_text, reference)

        # --- GPT evaluation ---
        if self.use_llm_eval and not ocr_error:
            if on_progress:
                on_progress(
                    current_step + 1,
                    total_steps,
                    f"[GPT] Evaluating sample {sample.id} — {engine_name}",
                )
            llm_score, llm_feedback = self._get_evaluator().score(
                sample.id, engine_name, ocr_text, reference, use_cache=use_cache
            )
        else:
            llm_score = 0.0
            llm_feedback = ocr_error or "LLM evaluation disabled"

        return SampleResult(
            sample_id=sample.id,
            engine=engine_name,
            ocr_text=ocr_text,
            cer=cer,
            wer=wer,
            llm_score=llm_score,
            llm_feedback=llm_feedback,
            time_seconds=ocr_time,
            error=ocr_error,
        )

    def _extract(self, engine_name: str, image: Image.Image) -> str:
        engine = self._get_engine(engine_name)
        if engine_name == "ensemble":
            return engine.combine(image).combined_text
        return engine.extract_text(image)

    # --- OCR cache ---

    def _ocr_cache_path(self, sample_id: str, engine: str) -> Path:
        return self.cache_dir / engine / f"{sample_id}.json"

    def _load_ocr_cache(self, sample_id: str, engine: str) -> str | None:
        path = self._ocr_cache_path(sample_id, engine)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return data.get("ocr_text")
            except (json.JSONDecodeError, KeyError):
                return None
        return None

    def _save_ocr_cache(self, sample_id: str, engine: str, ocr_text: str):
        path = self._ocr_cache_path(sample_id, engine)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "sample_id": sample_id,
            "engine": engine,
            "ocr_text": ocr_text,
            "timestamp": datetime.now().isoformat(),
        }, ensure_ascii=False))
