"""Streamlit UI for Document Structuring System."""

import sys
import os
from pathlib import Path

# Add project root to path and load env
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Ensure TESSDATA_PREFIX is set
if not os.getenv("TESSDATA_PREFIX"):
    os.environ["TESSDATA_PREFIX"] = "/opt/homebrew/share/tessdata"

import streamlit as st
from PIL import Image
import json
import io

st.set_page_config(
    page_title="Document Structuring System",
    page_icon="📄",
    layout="wide",
)

st.markdown("""
<style>
    .rtl-text { direction: rtl; text-align: right; font-family: 'Arial', sans-serif; }
    .ocr-output { background-color: #1e1e1e; color: #d4d4d4; padding: 12px;
                  border-radius: 6px; font-family: monospace; white-space: pre-wrap;
                  max-height: 400px; overflow-y: auto; }
    .correction-chip { background-color: #2d5a27; color: white; padding: 4px 10px;
                       border-radius: 12px; display: inline-block; margin: 2px; font-size: 0.85em; }
    .field-card { background-color: #f8f9fa; color: #1a1a1a; padding: 12px; border-radius: 8px;
                  border-left: 4px solid #0066cc; margin-bottom: 8px; }
    .section-card { background-color: #f0f4f8; color: #1a1a1a; padding: 10px; border-radius: 6px;
                    margin-bottom: 6px; }
    .page-header { background-color: #e8edf2; color: #1a1a1a; padding: 8px 14px; border-radius: 6px;
                   font-weight: bold; margin: 16px 0 8px 0; }
</style>
""", unsafe_allow_html=True)

st.title("Document Structuring System")
st.caption("Upload Arabic, English, or mixed-language documents for OCR extraction and AI-powered structuring")


# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    engines = st.multiselect(
        "OCR Engines",
        ["tesseract", "easyocr", "surya"],
        default=["tesseract", "easyocr"],
        help="Select which OCR engines to use. More engines = better accuracy but slower. Surya is slower but better for Arabic.",
    )
    if not engines:
        st.warning("Select at least one OCR engine.")
        engines = ["tesseract"]

    st.divider()
    st.header("History")
    results_dir = Path(__file__).parent.parent / "data" / "results"
    if results_dir.exists():
        result_files = sorted(results_dir.glob("*.json"), reverse=True)
        if result_files:
            for rf in result_files[:10]:
                data = json.loads(rf.read_text())
                if st.button(f"📄 {data.get('filename', 'unknown')} ({data.get('id', '')})", key=rf.stem):
                    st.session_state["loaded_result"] = data
        else:
            st.info("No processed documents yet.")


# --- Main area ---
uploaded_file = st.file_uploader(
    "Upload a document (image or PDF)",
    type=["png", "jpg", "jpeg", "tiff", "bmp", "webp", "pdf"],
    help="Supported formats: PNG, JPG, TIFF, BMP, WebP, PDF",
)


def display_page_result(page: dict, page_image=None, key_prefix: str = ""):
    """Display a single page's result in side-by-side layout."""
    col_left, col_right = st.columns(2)

    with col_left:
        if page_image is not None:
            st.image(page_image, use_container_width=True)

        # OCR outputs for this page
        ocr_outputs = page.get("ocr_outputs", {})
        if ocr_outputs:
            tab_names = list(ocr_outputs.keys())
            if page.get("spatial_text"):
                tab_names.append("Spatial Layout")
            tabs = st.tabs(tab_names)
            for i, engine in enumerate(list(ocr_outputs.keys())):
                with tabs[i]:
                    text = ocr_outputs[engine]
                    if text:
                        st.markdown(f'<div class="ocr-output">{text}</div>', unsafe_allow_html=True)
                    else:
                        st.warning(f"{engine} returned no text.")
            if page.get("spatial_text"):
                with tabs[-1]:
                    st.markdown(f'<div class="ocr-output">{page["spatial_text"]}</div>', unsafe_allow_html=True)

        score = page.get("consensus_score", 0)
        st.metric("OCR Consensus", f"{score:.0%}", label_visibility="visible")

        # Show per-engine errors
        engine_errors = page.get("engine_errors", {})
        if engine_errors:
            for eng, err in engine_errors.items():
                st.warning(f"**{eng}** error: {err}")

    with col_right:
        # Error
        error = page.get("error")
        if error:
            st.error(f"Error: {error}")

        # Language
        lang = page.get("language", "unknown")
        lang_emoji = {"arabic": "🇸🇦", "english": "🇬🇧", "mixed": "🌐"}.get(lang, "❓")
        st.caption(f"Language: **{lang}** {lang_emoji}")

        # Corrected text
        st.markdown("**Corrected Text**")
        corrected = page.get("corrected_text", "")
        if corrected:
            css_class = "rtl-text" if lang == "arabic" else ""
            st.markdown(f'<div class="{css_class}" style="background:#f8f9fa; color:#1a1a1a; padding:12px; border-radius:6px;">{corrected}</div>', unsafe_allow_html=True)
        else:
            st.warning("No corrected text available.")

        # Structured fields
        fields = page.get("structured_fields", {})
        if fields:
            st.markdown("**Extracted Fields**")
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                if fields.get("title"):
                    st.markdown(f'<div class="field-card"><strong>Title</strong><br>{fields["title"]}</div>', unsafe_allow_html=True)
            with f_col2:
                if fields.get("date"):
                    st.markdown(f'<div class="field-card"><strong>Date</strong><br>{fields["date"]}</div>', unsafe_allow_html=True)

            names = fields.get("names", [])
            if names:
                st.markdown(f'<div class="field-card"><strong>Names</strong><br>{", ".join(names)}</div>', unsafe_allow_html=True)

            numbers = fields.get("numbers", [])
            if numbers:
                st.markdown(f'<div class="field-card"><strong>Numbers</strong><br>{", ".join(str(n) for n in numbers)}</div>', unsafe_allow_html=True)

            if fields.get("summary"):
                st.markdown(f'<div class="field-card"><strong>Summary</strong><br>{fields["summary"]}</div>', unsafe_allow_html=True)

            sections = fields.get("sections", [])
            if sections:
                for sec in sections:
                    heading = sec.get("heading", "Untitled Section")
                    content = sec.get("content", "")
                    st.markdown(f'<div class="section-card"><strong>{heading}</strong><br>{content}</div>', unsafe_allow_html=True)

        # Corrections
        corrections = page.get("corrections_made", [])
        if corrections:
            st.markdown("**Corrections Applied**")
            for c in corrections:
                st.markdown(f'<div class="correction-chip">{c}</div>', unsafe_allow_html=True)


def display_result(result: dict, original_images=None):
    """Display the full processing result, page by page for multi-page docs."""
    images = []
    if original_images is not None:
        images = original_images if isinstance(original_images, list) else [original_images]

    pages = result.get("pages", [])
    page_count = result.get("page_count", 1)

    # Show error if entire pipeline failed
    error = result.get("error")
    if error:
        st.error(f"Pipeline error: {error}")

    if pages and page_count > 1:
        # --- Multi-page: show page-by-page report ---
        st.subheader(f"Document Report ({page_count} pages)")

        # Overall summary
        lang = result.get("language", "unknown")
        lang_emoji = {"arabic": "🇸🇦", "english": "🇬🇧", "mixed": "🌐"}.get(lang, "❓")
        score = result.get("consensus_score", 0)
        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1:
            st.metric("Total Pages", page_count)
        with s_col2:
            st.metric("Language", f"{lang} {lang_emoji}")
        with s_col3:
            st.metric("Avg Consensus", f"{score:.0%}")

        st.divider()

        # Page-by-page tabs
        page_tabs = st.tabs([f"Page {p.get('page_number', i+1)}" for i, p in enumerate(pages)])
        for i, page in enumerate(pages):
            with page_tabs[i]:
                page_img = images[i] if i < len(images) else None
                display_page_result(page, page_image=page_img, key_prefix=f"p{i}_")

        st.divider()

        # Download full JSON
        st.download_button(
            "Download Full Structured JSON",
            data=json.dumps(result, ensure_ascii=False, indent=2),
            file_name=f"document_{result.get('id', 'result')}.json",
            mime="application/json",
        )

    elif pages and page_count == 1:
        # --- Single page: show the one page result directly ---
        page = pages[0]
        page_img = images[0] if images else None
        display_page_result(page, page_image=page_img, key_prefix="single_")

        st.divider()
        st.download_button(
            "Download Structured JSON",
            data=json.dumps(result, ensure_ascii=False, indent=2),
            file_name=f"document_{result.get('id', 'result')}.json",
            mime="application/json",
        )

    else:
        # --- Fallback: legacy result without pages (e.g. from history) ---
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Input Document")
            for i, img in enumerate(images):
                st.image(img, caption=f"Page {i+1}" if len(images) > 1 else None, use_container_width=True)
            ocr_outputs = result.get("ocr_outputs", {})
            if ocr_outputs:
                tabs = st.tabs(list(ocr_outputs.keys()) + (["Spatial Layout"] if result.get("spatial_text") else []))
                for i, (engine, text) in enumerate(ocr_outputs.items()):
                    with tabs[i]:
                        if text:
                            st.markdown(f'<div class="ocr-output">{text}</div>', unsafe_allow_html=True)
                        else:
                            st.warning(f"{engine} returned no text.")
                if result.get("spatial_text"):
                    with tabs[-1]:
                        st.markdown(f'<div class="ocr-output">{result["spatial_text"]}</div>', unsafe_allow_html=True)

        with col_right:
            st.subheader("Structured Output")
            lang = result.get("language", "unknown")
            corrected = result.get("corrected_text", "")
            if corrected:
                css_class = "rtl-text" if lang == "arabic" else ""
                st.markdown(f'<div class="{css_class}" style="background:#f8f9fa; color:#1a1a1a; padding:12px; border-radius:6px;">{corrected}</div>', unsafe_allow_html=True)
            fields = result.get("structured_fields", {})
            if fields:
                st.json(fields)

        st.divider()
        st.download_button(
            "Download Structured JSON",
            data=json.dumps(result, ensure_ascii=False, indent=2),
            file_name=f"document_{result.get('id', 'result')}.json",
            mime="application/json",
        )


def pipeline_result_to_dict(result, filename: str) -> dict:
    """Convert a PipelineResult dataclass to a serializable dict."""
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
            "engine_errors": getattr(p, "engine_errors", {}),
            "success": p.success,
            "error": p.error,
        })
    return {
        "id": "live",
        "filename": filename,
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


# Handle uploaded file
if uploaded_file is not None:
    is_pdf = uploaded_file.name.lower().endswith(".pdf")
    preview_images = []

    if is_pdf:
        from src.pdf_utils import pdf_bytes_to_images
        pdf_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        preview_images = pdf_bytes_to_images(pdf_bytes, dpi=150)
    else:
        preview_images = [Image.open(uploaded_file)]
        uploaded_file.seek(0)

    if st.button("Process Document", type="primary", use_container_width=True):
        from src.pipeline import process_document

        progress_bar = st.progress(0)
        status_text = st.empty()

        def on_progress(current, total, stage, message):
            # Each page has 2 steps: ocr + llm. Total steps = total * 2 + 1 (finalize)
            total_steps = total * 2 + 1
            if stage == "ocr":
                step = current * 2
            elif stage == "llm":
                step = current * 2 + 1
            else:  # done
                step = total_steps
            progress_bar.progress(min(step / total_steps, 1.0))
            status_text.markdown(f"**{message}**")

        status_text.markdown("**Starting pipeline...**")
        if is_pdf:
            result = process_document(pdf_bytes=pdf_bytes, engines=engines, on_progress=on_progress)
        else:
            result = process_document(image=preview_images[0], engines=engines, on_progress=on_progress)

        progress_bar.progress(1.0)
        status_text.markdown("**Done!**")
        st.session_state["current_result"] = pipeline_result_to_dict(result, uploaded_file.name)
        progress_bar.empty()
        status_text.empty()

    # Display current result
    if "current_result" in st.session_state:
        display_result(st.session_state["current_result"], original_images=preview_images)
    else:
        # Show preview
        if is_pdf:
            st.info(f"PDF with {len(preview_images)} page(s)")
        for i, img in enumerate(preview_images):
            st.image(img, caption=f"Page {i+1}" if len(preview_images) > 1 else "Uploaded document", use_container_width=True)

# Handle loaded result from history
elif "loaded_result" in st.session_state:
    result = st.session_state["loaded_result"]
    uploads_dir = Path(__file__).parent.parent / "data" / "uploads"
    img = None
    for f in uploads_dir.glob(f"{result.get('id', '')}*"):
        try:
            img = Image.open(f)
            break
        except Exception:
            pass
    display_result(result, original_images=img)
else:
    st.info("Upload a document image or PDF to get started, or select a previous result from the sidebar.")
