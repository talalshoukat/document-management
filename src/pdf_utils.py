"""PDF to image conversion utilities."""

from pathlib import Path
from PIL import Image
import pypdfium2 as pdfium


def pdf_to_images(pdf_path: str | Path, dpi: int = 300) -> list[Image.Image]:
    """Convert a PDF file to a list of PIL Images (one per page)."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    images = []
    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(scale=dpi / 72)
        images.append(bitmap.to_pil())
    return images


def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = 300) -> list[Image.Image]:
    """Convert PDF bytes to a list of PIL Images (one per page)."""
    pdf = pdfium.PdfDocument(pdf_bytes)
    images = []
    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(scale=dpi / 72)
        images.append(bitmap.to_pil())
    return images
