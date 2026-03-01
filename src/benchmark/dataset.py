"""Load and manage the OmniAI OCR benchmark dataset from HuggingFace."""

import json
import os
from dataclasses import dataclass
from PIL import Image


@dataclass
class BenchmarkSample:
    """A single sample from the OmniAI OCR benchmark."""
    id: str
    image: Image.Image
    ground_truth_markdown: str
    ground_truth_json: str
    json_schema: str
    format: str       # TABLE, CHART, DELIVERY_NOTE, etc.
    quality: str      # HIGH_QUALITY, CLEAN, LOW_QUALITY, PHOTO


class BenchmarkDataset:
    """Loads the getomni-ai/ocr-benchmark dataset from HuggingFace.

    Set HF_TOKEN in your .env file or environment to avoid rate limits.
    Get a free token at https://huggingface.co/settings/tokens
    """

    DATASET_NAME = "getomni-ai/ocr-benchmark"

    def __init__(self):
        self._dataset = None
        self._samples: list[BenchmarkSample] | None = None

    def _ensure_loaded(self):
        if self._dataset is None:
            from datasets import load_dataset
            token = os.getenv("HF_TOKEN")
            self._dataset = load_dataset(
                self.DATASET_NAME, split="test", token=token,
            )

    def load(self, max_samples: int | None = None) -> list[BenchmarkSample]:
        """Load benchmark samples from HuggingFace.

        Args:
            max_samples: Limit number of samples loaded. None = all 1000.

        Returns:
            List of BenchmarkSample objects.
        """
        self._ensure_loaded()

        count = len(self._dataset)
        if max_samples is not None:
            count = min(max_samples, count)

        samples = []
        for i in range(count):
            item = self._dataset[i]
            # Parse metadata to extract format and quality tags
            fmt, quality = self._parse_metadata(item.get("metadata", ""))
            samples.append(BenchmarkSample(
                id=str(item.get("id", i)),
                image=item["image"],
                ground_truth_markdown=item.get("true_markdown_output", ""),
                ground_truth_json=item.get("true_json_output", ""),
                json_schema=item.get("json_schema", ""),
                format=fmt,
                quality=quality,
            ))

        self._samples = samples
        return samples

    def _parse_metadata(self, metadata_str: str) -> tuple[str, str]:
        """Extract format and quality from metadata JSON string."""
        fmt = "UNKNOWN"
        quality = "UNKNOWN"
        if not metadata_str:
            return fmt, quality
        try:
            meta = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
            fmt = meta.get("format", "UNKNOWN")
            quality = meta.get("documentQuality", "UNKNOWN")
        except (json.JSONDecodeError, TypeError):
            pass
        return fmt, quality

    def get_available_formats(self, samples: list[BenchmarkSample]) -> list[str]:
        """Get unique document format tags from loaded samples."""
        return sorted({s.format for s in samples if s.format != "UNKNOWN"})

    def get_available_qualities(self, samples: list[BenchmarkSample]) -> list[str]:
        """Get unique quality tags from loaded samples."""
        return sorted({s.quality for s in samples if s.quality != "UNKNOWN"})

    def filter_samples(
        self,
        samples: list[BenchmarkSample],
        formats: list[str] | None = None,
        qualities: list[str] | None = None,
    ) -> list[BenchmarkSample]:
        """Filter samples by format and/or quality tags."""
        filtered = samples
        if formats:
            filtered = [s for s in filtered if s.format in formats]
        if qualities:
            filtered = [s for s in filtered if s.quality in qualities]
        return filtered
