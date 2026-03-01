"""OCR benchmark metrics: CER, WER, and Levenshtein similarity."""

from jiwer import cer, wer
from rapidfuzz import fuzz


def calculate_cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate between OCR output and ground truth.

    Returns 0.0 for perfect match, higher values for more errors.
    """
    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0
    if not hypothesis.strip():
        return 1.0
    return cer(reference, hypothesis)


def calculate_wer(hypothesis: str, reference: str) -> float:
    """Word Error Rate between OCR output and ground truth.

    Returns 0.0 for perfect match, higher values for more errors.
    """
    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0
    if not hypothesis.strip():
        return 1.0
    return wer(reference, hypothesis)


def calculate_levenshtein_similarity(hypothesis: str, reference: str) -> float:
    """Normalized Levenshtein similarity (0.0 = no match, 1.0 = identical)."""
    return fuzz.ratio(hypothesis, reference) / 100.0


def compute_all_metrics(hypothesis: str, reference: str) -> dict:
    """Compute all OCR metrics at once.

    Returns dict with keys: cer, wer, levenshtein.
    """
    return {
        "cer": calculate_cer(hypothesis, reference),
        "wer": calculate_wer(hypothesis, reference),
        "levenshtein": calculate_levenshtein_similarity(hypothesis, reference),
    }
