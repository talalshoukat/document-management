"""Ensemble OCR - combines results from multiple OCR engines."""

from dataclasses import dataclass, field
from PIL import Image
from rapidfuzz import fuzz


@dataclass
class OCRResult:
    engine: str
    text: str
    confidence: float = 0.0
    details: list[dict] = field(default_factory=list)


@dataclass
class EnsembleResult:
    combined_text: str
    individual_results: list[OCRResult]
    consensus_score: float = 0.0


class OCREnsemble:
    """Combines output from multiple OCR engines for robust extraction."""

    def __init__(self, engines: list[str] | None = None):
        self.engine_names = engines or ["tesseract", "easyocr", "surya"]
        self._engines = {}

    def _get_engine(self, name: str):
        if name not in self._engines:
            if name == "tesseract":
                from .tesseract_engine import TesseractEngine
                self._engines[name] = TesseractEngine()
            elif name == "easyocr":
                from .easyocr_engine import EasyOCREngine
                self._engines[name] = EasyOCREngine()
            elif name == "surya":
                from .surya_engine import SuryaEngine
                self._engines[name] = SuryaEngine()
            else:
                raise ValueError(f"Unknown engine: {name}")
        return self._engines[name]

    def extract_all(self, image: Image.Image) -> list[OCRResult]:
        """Run all configured engines and collect results."""
        import traceback
        results = []
        for name in self.engine_names:
            try:
                engine = self._get_engine(name)
                text = engine.extract_text(image)
                results.append(OCRResult(engine=name, text=text, confidence=1.0))
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                tb = traceback.format_exc()
                print(f"[OCR Ensemble] {name} failed: {error_msg}\n{tb}")
                results.append(OCRResult(
                    engine=name,
                    text="",
                    confidence=0.0,
                    details=[{"error": error_msg}],
                ))
        return results

    def combine(self, image: Image.Image) -> EnsembleResult:
        """Run all engines and combine results with consensus scoring."""
        individual = self.extract_all(image)
        successful = [r for r in individual if r.text.strip()]
        if not successful:
            return EnsembleResult(
                combined_text="",
                individual_results=individual,
                consensus_score=0.0,
            )
        if len(successful) == 1:
            return EnsembleResult(
                combined_text=successful[0].text,
                individual_results=individual,
                consensus_score=0.5,
            )
        # Calculate pairwise similarity and pick the best pair
        best_score = 0.0
        best_text = successful[0].text
        for i in range(len(successful)):
            for j in range(i + 1, len(successful)):
                score = fuzz.ratio(successful[i].text, successful[j].text) / 100.0
                if score > best_score:
                    best_score = score
                    # Prefer the longer text (usually more complete)
                    if len(successful[i].text) >= len(successful[j].text):
                        best_text = successful[i].text
                    else:
                        best_text = successful[j].text
        # Build combined text line-by-line from best sources
        combined = self._merge_line_by_line(successful)
        consensus = self._calculate_consensus(successful)
        return EnsembleResult(
            combined_text=combined,
            individual_results=individual,
            consensus_score=consensus,
        )

    def _merge_line_by_line(self, results: list[OCRResult]) -> str:
        """Merge results line by line, picking the best version of each line."""
        all_lines = []
        for r in results:
            lines = r.text.strip().split("\n")
            all_lines.append(lines)
        # Use the result with the most lines as the base
        max_lines = max(len(lines) for lines in all_lines)
        merged = []
        for i in range(max_lines):
            candidates = []
            for lines in all_lines:
                if i < len(lines) and lines[i].strip():
                    candidates.append(lines[i].strip())
            if not candidates:
                continue
            # Pick the candidate most similar to all others (consensus)
            if len(candidates) == 1:
                merged.append(candidates[0])
            else:
                best = max(candidates, key=lambda c: sum(
                    fuzz.ratio(c, other) for other in candidates if other != c
                ))
                merged.append(best)
        return "\n".join(merged)

    def _calculate_consensus(self, results: list[OCRResult]) -> float:
        """Calculate overall consensus score between engines."""
        if len(results) < 2:
            return 0.5
        scores = []
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                scores.append(fuzz.ratio(results[i].text, results[j].text) / 100.0)
        return sum(scores) / len(scores)
