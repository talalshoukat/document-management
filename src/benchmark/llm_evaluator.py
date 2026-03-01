"""GPT-based OCR quality evaluator.

Uses OpenAI to score OCR output against ground truth on a 0–10 scale,
with structured feedback explaining the rating. Results are cached per
(sample_id, engine) so the API is only called once per unique pair.
"""

import json
from pathlib import Path
from datetime import datetime

import openai


SYSTEM_PROMPT = """You are an expert OCR quality evaluator specializing in Arabic, English, and mixed-language documents.

You will receive:
1. GROUND TRUTH — the exact correct text from the document
2. OCR OUTPUT — what an OCR engine extracted from the document image

Score the OCR output from 0 to 10 based on how accurately it captures the ground truth:

  10  Perfect — identical content, no meaningful differences
   8–9  Excellent — minor issues only (extra spaces, punctuation, formatting)
   6–7  Good — most content correct, a few wrong/missing words
   4–5  Fair — key information present but notable errors throughout
   2–3  Poor — many wrong characters/words, significant content missing
   0–1  Unusable — output is mostly garbage or completely empty

For Arabic text, pay special attention to:
- Correct Arabic characters (similar-looking letters easily confused: ر/ز, ح/ج/خ, etc.)
- Word completeness (no missing letters due to ligature errors)
- Numbers: Arabic-Indic (٠١٢٣) vs Western (0123) — both acceptable
- Diacritics (harakat): minor if missing, major if wrong character used
- Right-to-left order preserved

Respond ONLY with a JSON object — no extra text:
{"score": <integer 0-10>, "feedback": "<one concise sentence explaining the score>"}"""


class LLMEvaluator:
    """Scores OCR output against ground truth using GPT."""

    def __init__(self, cache_dir: Path | str, model: str = "gpt-4o-mini"):
        self.cache_dir = Path(cache_dir) / "llm_scores"
        self.model = model
        self._client: openai.OpenAI | None = None

    def _get_client(self) -> openai.OpenAI:
        if self._client is None:
            self._client = openai.OpenAI()
        return self._client

    def score(
        self,
        sample_id: str,
        engine: str,
        ocr_text: str,
        ground_truth: str,
        use_cache: bool = True,
    ) -> tuple[float, str]:
        """Score OCR output against ground truth.

        Returns:
            (score_0_to_1, feedback_string)
            Score is normalized to 0–1 (divides GPT's 0–10 by 10).
        """
        # Check cache first
        if use_cache:
            cached = self._load_cache(sample_id, engine)
            if cached:
                return cached["score"] / 10.0, cached["feedback"]

        raw_score, feedback = self._call_gpt(ocr_text, ground_truth)

        if use_cache:
            self._save_cache(sample_id, engine, raw_score, feedback)

        return raw_score / 10.0, feedback

    def _call_gpt(self, ocr_text: str, ground_truth: str) -> tuple[int, str]:
        """Call GPT and parse the JSON response. Returns (raw_score 0-10, feedback)."""
        # Truncate to keep within token limits
        gt_snippet = ground_truth[:3000]
        ocr_snippet = ocr_text[:3000] if ocr_text.strip() else "(empty — no text extracted)"

        user_msg = f"GROUND TRUTH:\n{gt_snippet}\n\nOCR OUTPUT:\n{ocr_snippet}"

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            data = json.loads(response.choices[0].message.content)
            raw_score = max(0, min(10, int(data.get("score", 0))))
            feedback = str(data.get("feedback", ""))
            return raw_score, feedback
        except Exception as e:
            return 0, f"Evaluation failed: {e}"

    # --- Cache helpers ---

    def _cache_path(self, sample_id: str, engine: str) -> Path:
        return self.cache_dir / engine / f"{sample_id}.json"

    def _load_cache(self, sample_id: str, engine: str) -> dict | None:
        path = self._cache_path(sample_id, engine)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, KeyError):
                return None
        return None

    def _save_cache(self, sample_id: str, engine: str, raw_score: int, feedback: str):
        path = self._cache_path(sample_id, engine)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "sample_id": sample_id,
            "engine": engine,
            "score": raw_score,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat(),
        }, ensure_ascii=False), encoding="utf-8")
