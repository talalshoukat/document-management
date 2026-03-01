"""Streamlit page: OCR Benchmark against OmniAI dataset."""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if not os.getenv("TESSDATA_PREFIX"):
    os.environ["TESSDATA_PREFIX"] = "/opt/homebrew/share/tessdata"

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import streamlit as st
import pandas as pd
import json
import time

st.set_page_config(page_title="OCR Benchmark", page_icon="📊", layout="wide")

st.title("OCR Benchmark")
st.caption("Benchmark OCR engines against the OmniAI 1,000-document dataset")

# ---------------------------------------------------------------------------
# Sidebar — Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    engines = st.multiselect(
        "Engines to benchmark",
        ["tesseract", "easyocr", "surya", "ensemble"],
        default=["tesseract", "easyocr"],
        help="Surya is slow on CPU (~30-60s/image). Ensemble runs all 3 engines.",
    )
    if not engines:
        st.warning("Select at least one engine.")
        engines = ["tesseract"]

    sample_count = st.radio(
        "Sample count",
        [10, 50, 100, 500, 1000],
        index=0,
        help="Number of samples from the dataset. Start small to test.",
    )

    use_cache = st.checkbox("Use cache", value=True, help="Skip OCR for previously cached results.")
    use_llm_eval = st.checkbox(
        "GPT evaluation", value=True,
        help="Score OCR quality using GPT-4o-mini. Uses your OPENAI_API_KEY. Results are cached.",
    )

    st.divider()

    # Filters (populated after dataset is loaded)
    format_filter = st.session_state.get("format_filter", [])
    quality_filter = st.session_state.get("quality_filter", [])

    st.divider()
    st.header("Previous Runs")
    results_dir = PROJECT_ROOT / "data" / "benchmark_results"
    from src.benchmark.results import list_benchmark_runs, load_benchmark_run
    prev_runs = list_benchmark_runs(results_dir)
    if prev_runs:
        for run_info in prev_runs[:10]:
            label = f"{run_info['timestamp'][:16]} — {', '.join(run_info['engines'])} ({run_info['sample_count']} samples)"
            if st.button(label, key=run_info["run_id"]):
                loaded = load_benchmark_run(run_info["path"])
                st.session_state["benchmark_run"] = loaded
                # Load sample images from local dataset for the inspector
                from src.benchmark.dataset import BenchmarkDataset
                _ds = BenchmarkDataset()
                if _ds.is_saved_locally():
                    _ids = {r.sample_id for r in loaded.results}
                    _all = _ds.load(max_samples=None)
                    st.session_state["benchmark_samples"] = [s for s in _all if s.id in _ids]
                else:
                    st.session_state["benchmark_samples"] = None
    else:
        st.info("No previous runs.")


# ---------------------------------------------------------------------------
# Dataset Status & Download
# ---------------------------------------------------------------------------
from src.benchmark.dataset import BenchmarkDataset

dataset = BenchmarkDataset()

if dataset.is_saved_locally():
    local_count = dataset.local_sample_count()
    st.success(f"Dataset ready — {local_count} samples saved locally")
else:
    st.warning("Dataset not downloaded yet. Download it once, then all future runs load instantly from disk.")
    if st.button("Download Dataset (1,000 samples from HuggingFace)", type="secondary"):
        with st.status("Downloading dataset...", expanded=True) as dl_status:
            progress_bar = st.progress(0)
            status_text = st.empty()

            def dl_progress(current: int, total: int, message: str):
                if total > 0:
                    progress_bar.progress(current / total)
                status_text.text(message)

            try:
                count = dataset.download_and_save(on_progress=dl_progress)
                progress_bar.empty()
                status_text.empty()
                dl_status.update(label=f"Downloaded {count} samples!", state="complete")
                st.rerun()
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                dl_status.update(label="Download failed", state="error")
                st.error(f"Download failed: {e}\n\nAdd `HF_TOKEN=hf_xxx` to your .env file to avoid rate limits.")
                st.stop()

# ---------------------------------------------------------------------------
# Main — Run Benchmark
# ---------------------------------------------------------------------------
run_col1, run_col2 = st.columns([3, 1])
with run_col2:
    run_clicked = st.button(
        "Run Benchmark", type="primary", use_container_width=True,
        disabled=not dataset.is_saved_locally(),
    )

if run_clicked:
    from src.benchmark.runner import BenchmarkRunner
    from src.benchmark.results import save_benchmark_run

    with st.status("Loading samples...", expanded=True) as status:
        samples = dataset.load(max_samples=sample_count)

        # Populate filter options in session
        avail_formats = dataset.get_available_formats(samples)
        avail_qualities = dataset.get_available_qualities(samples)

        # Apply filters
        if format_filter:
            samples = dataset.filter_samples(samples, formats=format_filter)
        if quality_filter:
            samples = dataset.filter_samples(samples, qualities=quality_filter)

        if not samples:
            st.error("No samples match the current filters.")
            st.stop()

        status.update(label=f"Loaded {len(samples)} samples. Running benchmark...")

        progress_bar = st.progress(0)
        progress_text = st.empty()

        cache_dir = PROJECT_ROOT / "data" / "benchmark_cache"
        runner = BenchmarkRunner(engines=engines, cache_dir=cache_dir, use_llm_eval=use_llm_eval)

        def on_progress(current: int, total: int, message: str):
            progress_bar.progress(current / total if total else 1.0)
            progress_text.text(message)

        run = runner.run(samples, on_progress=on_progress, use_cache=use_cache)

        # Save
        save_benchmark_run(run, results_dir)

        progress_bar.empty()
        progress_text.empty()
        status.update(label=f"Done! {len(run.results)} results in {run.total_time_seconds:.1f}s", state="complete")

    st.session_state["benchmark_run"] = run
    st.session_state["benchmark_samples"] = samples


# ---------------------------------------------------------------------------
# Display Results
# ---------------------------------------------------------------------------
run = st.session_state.get("benchmark_run")
if run is None:
    st.info("Configure engines and sample count in the sidebar, then click **Run Benchmark**.")
    st.stop()

from src.benchmark.results import aggregate_by_engine, aggregate_by_field, export_to_csv

samples = st.session_state.get("benchmark_samples")

tab_overview, tab_format, tab_quality, tab_details, tab_export = st.tabs(
    ["Overview", "By Document Type", "By Quality", "Sample Details", "Export"]
)

# ---- Tab 1: Overview ----
with tab_overview:
    st.subheader("Engine Comparison")
    agg = aggregate_by_engine(run)

    # Summary table
    rows = []
    for engine, stats in agg.items():
        rows.append({
            "Engine": engine,
            "Avg CER": stats["cer"]["mean"],
            "Avg WER": stats["wer"]["mean"],
            "Avg GPT Score": stats["llm_score"]["mean"],
            "Median CER": stats["cer"]["median"],
            "Median WER": stats["wer"]["median"],
            "Total Time (s)": stats["time"]["sum"],
            "Errors": stats["errors"],
        })
    df_summary = pd.DataFrame(rows)

    st.dataframe(
        df_summary.style.format({
            "Avg CER": "{:.3f}",
            "Avg WER": "{:.3f}",
            "Avg GPT Score": "{:.2f}",
            "Median CER": "{:.3f}",
            "Median WER": "{:.3f}",
            "Total Time (s)": "{:.1f}",
        }).highlight_min(subset=["Avg CER", "Avg WER"], color="#c6efce")
          .highlight_max(subset=["Avg GPT Score"], color="#c6efce"),
        use_container_width=True,
        hide_index=True,
    )

    # Bar charts
    col1, col2, col3 = st.columns(3)
    chart_df = df_summary.set_index("Engine")
    with col1:
        st.markdown("**Character Error Rate** (lower is better)")
        st.bar_chart(chart_df["Avg CER"])
    with col2:
        st.markdown("**Word Error Rate** (lower is better)")
        st.bar_chart(chart_df["Avg WER"])
    with col3:
        st.markdown("**GPT Score** (higher is better, 0–1)")
        st.bar_chart(chart_df["Avg GPT Score"])

    # Summary stats
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Samples", run.sample_count)
    m2.metric("Total Time", f"{run.total_time_seconds:.1f}s")
    m3.metric("Engines", ", ".join(run.engines))


# ---- Tab 2: By Document Type ----
with tab_format:
    if samples is None:
        st.info("Format breakdown is only available for the current run (not loaded runs).")
    else:
        st.subheader("Metrics by Document Format")
        fmt_agg = aggregate_by_field(run, samples, "format")
        if not fmt_agg:
            st.warning("No format metadata available.")
        else:
            # Build table: rows = format, columns per engine
            fmt_rows = []
            for fmt_val, engine_data in fmt_agg.items():
                for eng, stats in engine_data.items():
                    fmt_rows.append({
                        "Format": fmt_val,
                        "Engine": eng,
                        "Avg CER": stats["cer"]["mean"],
                        "Avg WER": stats["wer"]["mean"],
                        "Avg GPT Score": stats["llm_score"]["mean"],
                        "Samples": stats["count"],
                    })
            df_fmt = pd.DataFrame(fmt_rows)
            st.dataframe(
                df_fmt.style.format({
                    "Avg CER": "{:.3f}",
                    "Avg WER": "{:.3f}",
                    "Avg GPT Score": "{:.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            # Pivot for chart
            for metric, label in [("Avg CER", "CER by Format"), ("Avg GPT Score", "GPT Score by Format")]:
                pivot = df_fmt.pivot(index="Format", columns="Engine", values=metric)
                st.markdown(f"**{label}**")
                st.bar_chart(pivot)


# ---- Tab 3: By Quality ----
with tab_quality:
    if samples is None:
        st.info("Quality breakdown is only available for the current run.")
    else:
        st.subheader("Metrics by Document Quality")
        qual_agg = aggregate_by_field(run, samples, "quality")
        if not qual_agg:
            st.warning("No quality metadata available.")
        else:
            qual_rows = []
            for qual_val, engine_data in qual_agg.items():
                for eng, stats in engine_data.items():
                    qual_rows.append({
                        "Quality": qual_val,
                        "Engine": eng,
                        "Avg CER": stats["cer"]["mean"],
                        "Avg WER": stats["wer"]["mean"],
                        "Avg GPT Score": stats["llm_score"]["mean"],
                        "Samples": stats["count"],
                    })
            df_qual = pd.DataFrame(qual_rows)
            st.dataframe(
                df_qual.style.format({
                    "Avg CER": "{:.3f}",
                    "Avg WER": "{:.3f}",
                    "Avg GPT Score": "{:.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            for metric, label in [("Avg CER", "CER by Quality"), ("Avg GPT Score", "GPT Score by Quality")]:
                pivot = df_qual.pivot(index="Quality", columns="Engine", values=metric)
                st.markdown(f"**{label}**")
                st.bar_chart(pivot)


# ---- Tab 4: Sample Details ----
with tab_details:
    st.subheader("Per-Sample Results")

    # Build detail table
    detail_rows = []
    for r in run.results:
        detail_rows.append({
            "Sample ID": r.sample_id,
            "Engine": r.engine,
            "CER": r.cer,
            "WER": r.wer,
            "GPT Score": r.llm_score,
            "GPT Feedback": r.llm_feedback,
            "Time (s)": r.time_seconds,
            "Error": r.error or "",
        })
    df_detail = pd.DataFrame(detail_rows)

    # Filters
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        filter_engine = st.selectbox("Filter by engine", ["All"] + run.engines, key="detail_engine")
    with f_col2:
        sort_by = st.selectbox("Sort by", ["CER", "WER", "GPT Score", "Time (s)"], key="detail_sort")

    filtered_df = df_detail
    if filter_engine != "All":
        filtered_df = filtered_df[filtered_df["Engine"] == filter_engine]
    ascending = sort_by != "GPT Score"
    filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)

    st.dataframe(
        filtered_df.style.format({
            "CER": "{:.4f}",
            "WER": "{:.4f}",
            "GPT Score": "{:.2f}",
            "Time (s)": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Sample inspector
    st.divider()
    st.markdown("**Inspect Sample**")
    sample_ids = sorted(set(r.sample_id for r in run.results))
    selected_id = st.selectbox("Select sample ID", sample_ids, key="inspect_sample")

    if selected_id and samples:
        sample = next((s for s in samples if s.id == selected_id), None)
        if sample:
            inspect_col1, inspect_col2 = st.columns(2)
            with inspect_col1:
                st.markdown("**Document Image**")
                st.image(sample.image, use_container_width=True)
                st.markdown("**Ground Truth (Markdown)**")
                st.code(sample.ground_truth_markdown[:2000], language="markdown")
            with inspect_col2:
                # Show OCR outputs per engine
                sample_results = [r for r in run.results if r.sample_id == selected_id]
                for r in sample_results:
                    score_str = f"GPT: {r.llm_score:.2f}" if r.llm_score else ""
                    with st.expander(
                        f"{r.engine} — CER: {r.cer:.3f} | WER: {r.wer:.3f} | {score_str}"
                    ):
                        if r.error:
                            st.error(r.error)
                        else:
                            if r.llm_feedback:
                                st.info(f"GPT feedback: {r.llm_feedback}")
                            st.code(r.ocr_text[:2000], language="text")
    elif selected_id:
        st.info("Sample images not available for loaded runs. Run a fresh benchmark to inspect samples.")


# ---- Tab 5: Export ----
with tab_export:
    st.subheader("Export Results")

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        # JSON download
        run_json = json.dumps({
            "run_id": run.run_id,
            "timestamp": run.timestamp,
            "engines": run.engines,
            "sample_count": run.sample_count,
            "total_time_seconds": run.total_time_seconds,
            "results": [
                {
                    "sample_id": r.sample_id,
                    "engine": r.engine,
                    "cer": r.cer,
                    "wer": r.wer,
                    "llm_score": r.llm_score,
                    "llm_feedback": r.llm_feedback,
                    "time_seconds": r.time_seconds,
                    "error": r.error,
                }
                for r in run.results
            ],
        }, ensure_ascii=False, indent=2)
        st.download_button(
            "Download JSON",
            data=run_json,
            file_name=f"{run.run_id}.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_e2:
        # CSV download
        csv_path = PROJECT_ROOT / "data" / "benchmark_results" / f"{run.run_id}.csv"
        export_to_csv(run, csv_path)
        st.download_button(
            "Download CSV",
            data=csv_path.read_text(),
            file_name=f"{run.run_id}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Raw JSON viewer
    with st.expander("View raw JSON"):
        st.json(json.loads(run_json))
