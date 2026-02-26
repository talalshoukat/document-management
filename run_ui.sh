#!/bin/bash
# Run the Streamlit UI
conda activate docstructure
export TESSDATA_PREFIX=/opt/homebrew/share/tessdata
cd "$(dirname "$0")"
streamlit run ui/app.py --server.port 8501
