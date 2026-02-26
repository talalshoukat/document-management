#!/bin/bash
# Run the FastAPI backend
conda activate docstructure
export TESSDATA_PREFIX=/opt/homebrew/share/tessdata
cd "$(dirname "$0")"
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
