"""LLM-based text correction and structured data extraction."""

import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CORRECTION_SYSTEM_PROMPT = """You are a document text correction and structuring specialist.
You handle documents in Arabic, English, or a mix of both languages.

You will receive OCR text from multiple engines. Your job is to:

1. CORRECT the text:
   - Fix OCR errors by cross-referencing multiple OCR outputs
   - Fix misspellings and garbled characters
   - Properly handle Arabic text (right-to-left), including diacritics
   - Properly handle mixed Arabic-English text
   - Maintain the original document structure (paragraphs, sections)

2. EXTRACT structured fields from the corrected text.

Return a JSON object with:
{
  "corrected_text": "The full corrected document text, preserving paragraphs and structure",
  "language": "detected language: arabic, english, or mixed",
  "fields": {
    "title": "Document title if present",
    "date": "Any date found in the document",
    "names": ["List of person or organization names found"],
    "numbers": ["List of important numbers (IDs, amounts, phone numbers)"],
    "sections": [
      {
        "heading": "Section heading if any",
        "content": "Section text content"
      }
    ],
    "summary": "Brief 1-2 sentence summary of the document content"
  },
  "corrections_made": ["List of notable corrections made to the OCR text"]
}

Important rules:
- Preserve the original language. Do not translate.
- If text is in Arabic, keep it in Arabic.
- If text is mixed, preserve both languages where they appear.
- Collapse multiple spaces into single spaces.
- Join lines that were incorrectly split by OCR.
- The corrected_text should be clean, readable, and faithful to the original document.
"""


class LLMCorrector:
    """Uses LLM to correct OCR text and extract structured data."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def correct_and_structure(
        self,
        ocr_texts: dict[str, str],
        spatial_text: str = "",
    ) -> dict:
        """
        Correct OCR text and extract structured fields.

        Args:
            ocr_texts: Dict mapping engine name to extracted text
            spatial_text: Optional spatial text rendering from Tesseract
        """
        # Build the user message with all OCR sources
        parts = []
        for engine, text in ocr_texts.items():
            if text.strip():
                parts.append(f"=== OCR Output from {engine} ===\n{text}")
        if spatial_text.strip():
            parts.append(f"=== Spatial Layout (preserves document positioning) ===\n{spatial_text}")
        if not parts:
            return {
                "corrected_text": "",
                "language": "unknown",
                "fields": {
                    "title": "",
                    "date": "",
                    "names": [],
                    "numbers": [],
                    "sections": [],
                    "summary": "",
                },
                "corrections_made": [],
            }
        user_message = (
            "Please correct the following OCR outputs and extract structured data.\n"
            "Cross-reference the multiple OCR sources to determine the most accurate text.\n\n"
            + "\n\n".join(parts)
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=4000,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            result = json.loads(response.choices[0].message.content)
            # Ensure all expected keys exist
            result.setdefault("corrected_text", "")
            result.setdefault("language", "unknown")
            result.setdefault("fields", {})
            result["fields"].setdefault("title", "")
            result["fields"].setdefault("date", "")
            result["fields"].setdefault("names", [])
            result["fields"].setdefault("numbers", [])
            result["fields"].setdefault("sections", [])
            result["fields"].setdefault("summary", "")
            result.setdefault("corrections_made", [])
            return result
        except json.JSONDecodeError:
            return {
                "corrected_text": parts[0] if parts else "",
                "language": "unknown",
                "fields": {"title": "", "date": "", "names": [], "numbers": [], "sections": [], "summary": ""},
                "corrections_made": [],
            }
        except Exception as e:
            return {
                "corrected_text": "",
                "language": "unknown",
                "fields": {"title": "", "date": "", "names": [], "numbers": [], "sections": [], "summary": ""},
                "corrections_made": [],
                "error": str(e),
            }
