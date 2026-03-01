"""Load and manage the OmniAI OCR benchmark dataset from HuggingFace.

Samples are saved locally on first download so subsequent runs load instantly
from disk without hitting HuggingFace rate limits.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

_DEFAULT_LOCAL_DIR = Path(__file__).parent.parent.parent / "data" / "benchmark_dataset"


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
    """Loads the getomni-ai/ocr-benchmark dataset.

    On first use, downloads all 1,000 samples from HuggingFace and saves them
    locally (images as PNG + an index.json). All subsequent runs load from
    the local copy — no internet required and no rate-limit issues.

    Set HF_TOKEN in your .env file or environment to avoid HuggingFace rate
    limits during the initial download.
    """

    DATASET_NAME = "getomni-ai/ocr-benchmark"

    def __init__(self, local_dir: Path | str | None = None):
        self.local_dir = Path(local_dir) if local_dir else _DEFAULT_LOCAL_DIR
        self._index: list[dict] | None = None

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def index_path(self) -> Path:
        return self.local_dir / "index.json"

    @property
    def images_dir(self) -> Path:
        return self.local_dir / "images"

    # ------------------------------------------------------------------
    # Local storage helpers
    # ------------------------------------------------------------------

    def is_saved_locally(self) -> bool:
        """Return True if the dataset has been downloaded and saved locally."""
        return self.index_path.exists()

    def local_sample_count(self) -> int:
        """Return the number of samples saved locally."""
        if not self.is_saved_locally():
            return 0
        if self._index is None:
            self._index = json.loads(self.index_path.read_text(encoding="utf-8"))
        return len(self._index)

    def download_and_save(
        self,
        on_progress: "Callable[[int, int, str], None] | None" = None,
    ) -> int:
        """Download all samples from HuggingFace and persist them locally.

        Returns the number of samples saved.
        """
        from datasets import load_dataset

        token = os.getenv("HF_TOKEN")
        if on_progress:
            on_progress(0, 1, "Connecting to HuggingFace…")

        ds = load_dataset(self.DATASET_NAME, split="test", token=token)
        total = len(ds)

        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        index: list[dict] = []
        for i, item in enumerate(ds):
            sample_id = str(item.get("id", i))
            img_path = self.images_dir / f"{sample_id}.png"
            img: Image.Image = item["image"]
            img.save(str(img_path), format="PNG")

            fmt, quality = self._parse_metadata(item.get("metadata", ""))
            index.append({
                "id": sample_id,
                "image_file": img_path.name,
                "ground_truth_markdown": item.get("true_markdown_output", ""),
                "ground_truth_json": item.get("true_json_output", ""),
                "json_schema": item.get("json_schema", ""),
                "format": fmt,
                "quality": quality,
            })

            if on_progress:
                on_progress(i + 1, total, f"Saved sample {i + 1}/{total}")

        self.index_path.write_text(
            json.dumps(index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._index = index
        return len(index)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, max_samples: int | None = None) -> list[BenchmarkSample]:
        """Load benchmark samples from the local copy (downloading first if needed).

        Args:
            max_samples: Limit number of samples loaded. None = all.

        Returns:
            List of BenchmarkSample objects.
        """
        if not self.is_saved_locally():
            self.download_and_save()

        if self._index is None:
            self._index = json.loads(self.index_path.read_text(encoding="utf-8"))

        entries = self._index
        if max_samples is not None:
            entries = entries[:max_samples]

        samples = []
        for entry in entries:
            img_path = self.images_dir / entry["image_file"]
            image = Image.open(str(img_path)).copy()
            samples.append(BenchmarkSample(
                id=entry["id"],
                image=image,
                ground_truth_markdown=entry["ground_truth_markdown"],
                ground_truth_json=entry["ground_truth_json"],
                json_schema=entry["json_schema"],
                format=entry["format"],
                quality=entry["quality"],
            ))
        return samples

    # ------------------------------------------------------------------
    # Filters / helpers
    # ------------------------------------------------------------------

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
