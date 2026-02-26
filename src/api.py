"""FastAPI backend for document structuring."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from .pipeline import process_document

app = FastAPI(title="Document Structuring API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"
UPLOADS_DIR = Path(__file__).parent.parent / "data" / "uploads"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/tiff", "image/bmp", "image/webp", "application/pdf"}
MAX_SIZE = 50 * 1024 * 1024  # 50MB (PDFs can be larger)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/extract")
async def extract(file: UploadFile = File(...), engines: str = "tesseract,easyocr"):
    """Extract and structure document content from an uploaded image."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}. Supported: {ALLOWED_TYPES}")

    contents = await file.read()
    if len(contents) > MAX_SIZE:
        raise HTTPException(400, f"File too large. Max size: {MAX_SIZE // (1024*1024)}MB")

    # Save upload
    doc_id = str(uuid.uuid4())[:8]
    upload_path = UPLOADS_DIR / f"{doc_id}_{file.filename}"
    upload_path.write_bytes(contents)

    # Process
    engine_list = [e.strip() for e in engines.split(",") if e.strip()]
    is_pdf = file.content_type == "application/pdf" or (file.filename and file.filename.lower().endswith(".pdf"))
    if is_pdf:
        result = process_document(pdf_bytes=contents, engines=engine_list)
    else:
        image = Image.open(io.BytesIO(contents))
        result = process_document(image=image, engines=engine_list)

    # Build response
    pages_list = []
    for p in result.pages:
        pages_list.append({
            "page_number": p.page_number,
            "ocr_outputs": p.ocr_outputs,
            "spatial_text": p.spatial_text,
            "consensus_score": p.consensus_score,
            "corrected_text": p.corrected_text,
            "language": p.language,
            "structured_fields": p.structured_fields,
            "corrections_made": p.corrections_made,
            "success": p.success,
            "error": p.error,
        })
    response = {
        "id": doc_id,
        "filename": file.filename,
        "timestamp": datetime.now().isoformat(),
        "success": result.success,
        "error": result.error,
        "language": result.language,
        "corrected_text": result.corrected_text,
        "structured_fields": result.structured_fields,
        "corrections_made": result.corrections_made,
        "ocr_outputs": result.ocr_outputs,
        "spatial_text": result.spatial_text,
        "consensus_score": result.consensus_score,
        "page_count": result.page_count,
        "pages": pages_list,
    }

    # Save result
    result_path = RESULTS_DIR / f"{doc_id}.json"
    result_path.write_text(json.dumps(response, ensure_ascii=False, indent=2))

    return response


@app.get("/results")
async def list_results():
    """List all processed document results."""
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        data = json.loads(f.read_text())
        results.append({
            "id": data.get("id"),
            "filename": data.get("filename"),
            "timestamp": data.get("timestamp"),
            "success": data.get("success"),
            "language": data.get("language"),
        })
    return results


@app.get("/results/{doc_id}")
async def get_result(doc_id: str):
    """Get a specific document result."""
    result_path = RESULTS_DIR / f"{doc_id}.json"
    if not result_path.exists():
        raise HTTPException(404, f"Result not found: {doc_id}")
    return json.loads(result_path.read_text())
