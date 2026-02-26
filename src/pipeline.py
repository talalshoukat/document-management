"""End-to-end document structuring pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from PIL import Image

from .ocr import OCREnsemble
from .ocr.tesseract_engine import TesseractEngine
from .llm.corrector import LLMCorrector
from .pdf_utils import pdf_to_images, pdf_bytes_to_images


@dataclass
class PageResult:
    """Result for a single page."""
    page_number: int = 1
    ocr_outputs: dict[str, str] = field(default_factory=dict)
    spatial_text: str = ""
    consensus_score: float = 0.0
    corrected_text: str = ""
    language: str = "unknown"
    structured_fields: dict = field(default_factory=dict)
    corrections_made: list[str] = field(default_factory=list)
    engine_errors: dict[str, str] = field(default_factory=dict)
    success: bool = True
    error: str | None = None


@dataclass
class PipelineResult:
    """Result from the document structuring pipeline."""
    corrected_text: str = ""
    language: str = "unknown"
    structured_fields: dict = field(default_factory=dict)
    corrections_made: list[str] = field(default_factory=list)
    ocr_outputs: dict[str, str] = field(default_factory=dict)
    spatial_text: str = ""
    consensus_score: float = 0.0
    page_count: int = 1
    pages: list[PageResult] = field(default_factory=list)
    success: bool = True
    error: str | None = None


def _process_single_image(
    image: Image.Image,
    ensemble: OCREnsemble,
    tesseract: TesseractEngine,
) -> tuple[dict[str, str], str, float, dict[str, str]]:
    """Run OCR on a single image. Returns (ocr_outputs, spatial_text, consensus, engine_errors)."""
    ensemble_result = ensemble.combine(image)
    ocr_outputs = {r.engine: r.text for r in ensemble_result.individual_results}
    consensus = ensemble_result.consensus_score
    # Collect per-engine errors
    engine_errors = {}
    for r in ensemble_result.individual_results:
        if r.details:
            for d in r.details:
                if "error" in d:
                    engine_errors[r.engine] = d["error"]

    try:
        # spatial = tesseract.render_spatial_text(image)
        spatial = ""
    except Exception:
        spatial = ""

    return ocr_outputs, spatial, consensus, engine_errors


def process_document(
    image: Image.Image | str | Path | None = None,
    pdf_bytes: bytes | None = None,
    engines: list[str] | None = None,
    model: str | None = None,
    api_key: str | None = None,
    on_progress: Callable | None = None,
) -> PipelineResult:
    """
    Process a document through the full pipeline.

    Supports single images, image paths, PDFs (via path or bytes).
    Multi-page PDFs have each page OCR'd and results merged before LLM processing.

    Stage 1: Multi-engine OCR extraction
    Stage 2: Spatial text rendering (Tesseract)
    Stage 3: LLM correction and structuring
    """
    result = PipelineResult()
    images: list[Image.Image] = []

    # --- Resolve input to list of images ---
    if pdf_bytes is not None:
        try:
            images = pdf_bytes_to_images(pdf_bytes)
            result.page_count = len(images)
        except Exception as e:
            result.success = False
            result.error = f"Failed to convert PDF: {e}"
            return result
    elif isinstance(image, (str, Path)):
        path = Path(image)
        if path.suffix.lower() == ".pdf":
            try:
                images = pdf_to_images(path)
                result.page_count = len(images)
            except Exception as e:
                result.success = False
                result.error = f"Failed to convert PDF: {e}"
                return result
        else:
            try:
                images = [Image.open(path)]
            except Exception as e:
                result.success = False
                result.error = f"Failed to load image: {e}"
                return result
    elif image is not None:
        images = [image]
    else:
        result.success = False
        result.error = "No input provided"
        return result

    if not images:
        result.success = False
        result.error = "No pages found in document"
        return result

    # --- Create engines once, reuse across pages ---
    ensemble = OCREnsemble(engines=engines)
    tesseract = TesseractEngine()
    corrector = LLMCorrector(model=model, api_key=api_key)

    # Warm up models on first call (so subsequent pages are fast)
    all_ocr: dict[str, list[str]] = {}
    all_spatial: list[str] = []
    all_corrected: list[str] = []
    consensus_scores: list[float] = []

    total = len(images)
    for page_idx, img in enumerate(images):
        page = PageResult(page_number=page_idx + 1)

        # Stage 1 & 2: OCR
        if on_progress:
            on_progress(page_idx, total, "ocr", f"Page {page_idx + 1}/{total}: Running OCR...")
        try:
            ocr_outputs, spatial, consensus, engine_errors = _process_single_image(img, ensemble, tesseract)
            page.ocr_outputs = ocr_outputs
            page.spatial_text = spatial
            page.consensus_score = consensus
            page.engine_errors = engine_errors
            for engine, text in ocr_outputs.items():
                all_ocr.setdefault(engine, []).append(text)
            all_spatial.append(spatial)
            consensus_scores.append(consensus)
        except Exception as e:
            page.success = False
            page.error = f"OCR failed: {e}"
            result.pages.append(page)
            continue

        # Stage 3: LLM correction per page
        if on_progress:
            on_progress(page_idx, total, "llm", f"Page {page_idx + 1}/{total}: AI correction & structuring...")
        try:
            llm_result = corrector.correct_and_structure(
                ocr_texts=page.ocr_outputs,
                spatial_text=page.spatial_text,
            )
            page.corrected_text = llm_result.get("corrected_text", "")
            page.language = llm_result.get("language", "unknown")
            page.structured_fields = llm_result.get("fields", {})
            page.corrections_made = llm_result.get("corrections_made", [])
            if "error" in llm_result:
                page.error = llm_result["error"]
                page.success = False
            all_corrected.append(page.corrected_text)
        except Exception as e:
            page.success = False
            page.error = f"LLM failed: {e}"

        result.pages.append(page)

    if on_progress:
        on_progress(total, total, "done", "Finalizing results...")

    # --- Aggregate results across pages ---
    page_sep = "\n\n--- Page Break ---\n\n"
    result.ocr_outputs = {
        engine: page_sep.join(pages) for engine, pages in all_ocr.items()
    }
    result.spatial_text = page_sep.join(all_spatial)
    result.corrected_text = page_sep.join(all_corrected)
    result.consensus_score = (
        sum(consensus_scores) / len(consensus_scores) if consensus_scores else 0.0
    )

    # Use first page's language or detect mixed
    languages = {p.language for p in result.pages if p.language != "unknown"}
    if len(languages) > 1:
        result.language = "mixed"
    elif languages:
        result.language = languages.pop()

    # Merge structured fields from all pages
    all_names = []
    all_numbers = []
    all_sections = []
    all_corrections = []
    for p in result.pages:
        f = p.structured_fields
        all_names.extend(f.get("names", []))
        all_numbers.extend(f.get("numbers", []))
        all_sections.extend(f.get("sections", []))
        all_corrections.extend(p.corrections_made)
    result.structured_fields = {
        "title": next((p.structured_fields.get("title") for p in result.pages if p.structured_fields.get("title")), ""),
        "date": next((p.structured_fields.get("date") for p in result.pages if p.structured_fields.get("date")), ""),
        "names": list(dict.fromkeys(all_names)),  # dedupe preserving order
        "numbers": list(dict.fromkeys(str(n) for n in all_numbers)),
        "sections": all_sections,
        "summary": next((p.structured_fields.get("summary") for p in result.pages if p.structured_fields.get("summary")), ""),
    }
    result.corrections_made = all_corrections

    # Check if any page failed
    failed = [p for p in result.pages if not p.success]
    if failed and len(failed) == len(result.pages):
        result.success = False
        result.error = f"All {len(failed)} pages failed processing"

    return result
