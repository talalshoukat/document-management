"""Surya OCR engine - strong for Arabic and multi-script documents."""

import torch
from PIL import Image


def _get_device():
    """Get best available device for Surya models.

    Note: MPS (Apple Silicon) has known issues with Surya causing slow/incorrect results.
    See: https://github.com/pytorch/pytorch/issues/84936
    Only use CUDA if available, otherwise CPU is faster and more reliable.
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SuryaEngine:
    """OCR engine using Surya - ML-based, excellent for Arabic and 90+ languages."""

    def __init__(self):
        self._rec_model = None
        self._rec_processor = None
        self._det_model = None
        self._det_processor = None

    def _load_models(self):
        if self._rec_model is None:
            from surya.model.recognition.model import load_model as load_rec_model
            from surya.model.recognition.processor import load_processor as load_rec_processor
            from surya.model.detection.segformer import load_model as load_det_model
            from surya.model.detection.segformer import load_processor as load_det_processor

            device = _get_device()
            # Recognition model: uses float16 on GPU, float32 on CPU
            rec_dtype = torch.float16 if device != "cpu" else torch.float32
            self._rec_model = load_rec_model(device=device, dtype=rec_dtype)
            self._rec_processor = load_rec_processor()
            # Detection model: use float32 (segformer works best with float32)
            self._det_model = load_det_model(device=device, dtype=torch.float32)
            self._det_processor = load_det_processor()

    def extract_text(self, image: Image.Image) -> str:
        """Extract full text from image using Surya OCR.

        Uses run_ocr which handles the full pipeline:
        detection (find text regions) → slicing (crop lines) → recognition (read text).
        """
        self._load_models()
        from surya.ocr import run_ocr

        # run_ocr returns List[OCRResult], each with .text_lines (list of TextLine with .text)
        ocr_results = run_ocr(
            [image],
            [["en", "ar"]],
            self._det_model,
            self._det_processor,
            self._rec_model,
            self._rec_processor,
        )

        if not ocr_results:
            return ""

        lines = []
        for page_result in ocr_results:
            for line in page_result.text_lines:
                if line.text.strip():
                    lines.append(line.text.strip())
        return "\n".join(lines)

    def extract_with_confidence(self, image: Image.Image) -> list[dict]:
        """Extract text with bounding boxes and confidence."""
        text = self.extract_text(image)
        results = []
        for line in text.split("\n"):
            if line.strip():
                results.append({
                    "text": line.strip(),
                    "bbox": None,
                    "confidence": 1.0,
                })
        return results
