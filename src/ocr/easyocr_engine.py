"""EasyOCR engine with Arabic + English support."""

import numpy as np
from PIL import Image


def _has_gpu() -> bool:
    """Check if MPS (Apple Silicon) or CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available() or torch.backends.mps.is_available()
    except Exception:
        return False


class EasyOCREngine:
    """OCR engine using EasyOCR - strong multilingual support."""

    def __init__(self, languages: list[str] | None = None, gpu: bool | None = None):
        self.languages = languages or ["en", "ar"]
        self._gpu = gpu if gpu is not None else _has_gpu()
        self._reader = None

    @property
    def reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(self.languages, gpu=self._gpu)
        return self._reader

    def extract_text(self, image: Image.Image) -> str:
        """Extract full text from image."""
        img_array = np.array(image)
        results = self.reader.readtext(img_array)
        if not results:
            return ""
        # Sort by vertical then horizontal position
        results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
        lines = []
        current_line = [results[0]]
        for result in results[1:]:
            prev_y = current_line[0][0][0][1]
            curr_y = result[0][0][1]
            line_height = abs(current_line[0][0][2][1] - current_line[0][0][0][1])
            if abs(curr_y - prev_y) < line_height * 0.5:
                current_line.append(result)
            else:
                current_line.sort(key=lambda r: r[0][0][0])
                lines.append(" ".join(r[1] for r in current_line))
                current_line = [result]
        if current_line:
            current_line.sort(key=lambda r: r[0][0][0])
            lines.append(" ".join(r[1] for r in current_line))
        return "\n".join(lines)

    def extract_with_confidence(self, image: Image.Image) -> list[dict]:
        """Extract text with bounding boxes and confidence scores."""
        img_array = np.array(image)
        results = self.reader.readtext(img_array)
        extracted = []
        for bbox, text, conf in results:
            extracted.append({
                "text": text,
                "bbox": bbox,
                "confidence": conf,
            })
        return extracted
