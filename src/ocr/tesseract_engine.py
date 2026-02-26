"""Tesseract OCR engine with Arabic + English support."""

from dataclasses import dataclass
from PIL import Image
import pytesseract
import os


@dataclass
class OcrWord:
    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float


class TesseractEngine:
    """OCR engine using Tesseract with multi-language support."""

    def __init__(self, languages: str = "eng+ara"):
        self.languages = languages
        cmd = os.getenv("TESSERACT_CMD")
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
        else:
            # Auto-detect tesseract binary on macOS
            for path in ["/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"]:
                if os.path.isfile(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
        # Ensure TESSDATA_PREFIX is set
        if not os.getenv("TESSDATA_PREFIX"):
            os.environ["TESSDATA_PREFIX"] = "/opt/homebrew/share/tessdata"

    def _get_config(self) -> str:
        return f"--oem 3 --psm 6 -l {self.languages}"

    def extract_words(self, image: Image.Image) -> list[OcrWord]:
        """Extract words with bounding boxes from image."""
        config = self._get_config()
        data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
        words = []
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            conf = float(data["conf"][i])
            if text and conf > 0:
                words.append(OcrWord(
                    text=text,
                    left=data["left"][i],
                    top=data["top"][i],
                    width=data["width"][i],
                    height=data["height"][i],
                    confidence=conf,
                ))
        return words

    def extract_text(self, image: Image.Image) -> str:
        """Extract full text from image."""
        words = self.extract_words(image)
        if not words:
            return ""
        # Group words by line (similar top position)
        words.sort(key=lambda w: (w.top, w.left))
        lines = []
        current_line = [words[0]]
        for word in words[1:]:
            if abs(word.top - current_line[0].top) < current_line[0].height * 0.5:
                current_line.append(word)
            else:
                current_line.sort(key=lambda w: w.left)
                lines.append(" ".join(w.text for w in current_line))
                current_line = [word]
        if current_line:
            current_line.sort(key=lambda w: w.left)
            lines.append(" ".join(w.text for w in current_line))
        return "\n".join(lines)

    def render_spatial_text(self, image: Image.Image, target_width: int = 100) -> str:
        """Render text preserving spatial layout (from Abwab approach)."""
        words = self.extract_words(image)
        if not words:
            return ""
        img_w, img_h = image.size
        scale_x = target_width / img_w
        scale_y = scale_x  # uniform scaling
        grid_h = int(img_h * scale_y) + 1
        grid_w = target_width + 20
        grid = [[" "] * grid_w for _ in range(grid_h)]
        for word in words:
            col = int(word.left * scale_x)
            row = int(word.top * scale_y)
            for ch in word.text:
                if 0 <= row < grid_h and 0 <= col < grid_w:
                    if grid[row][col] != " ":
                        col += 1
                        if col >= grid_w:
                            break
                    grid[row][col] = ch
                    col += 1
        lines = ["".join(row).rstrip() for row in grid]
        # Remove empty lines at start/end, compact consecutive empty lines
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        result = []
        prev_empty = False
        for line in lines:
            if not line:
                if not prev_empty:
                    result.append("")
                prev_empty = True
            else:
                result.append(line)
                prev_empty = False
        return "\n".join(result)
