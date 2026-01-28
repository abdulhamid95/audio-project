"""AI Audio Refiner - Streamlit Web Application with Interactive Review."""

import os
import tempfile
import shutil
import streamlit as st
from faster_whisper import WhisperModel

from utils import convert_input
from processor import AudioProcessor, RepetitionAnalysis, DeletionRange


@st.cache_resource
def load_whisper_model():
    """Load and cache the Whisper model (large-v3 for Arabic accuracy)."""
    return WhisperModel("large-v3", device="cpu", compute_type="int8")


def init_session_state():
    """Initialize session state variables."""
    if "phase" not in st.session_state:
        st.session_state.phase = "upload"  # upload -> review -> complete
    if "analysis" not in st.session_state:
        st.session_state.analysis = None
    if "prepared_audio_path" not in st.session_state:
        st.session_state.prepared_audio_path = None
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = None
    if "review_decisions" not in st.session_state:
        st.session_state.review_decisions = {}
    if "final_audio" not in st.session_state:
        st.session_state.final_audio = None
    if "output_filename" not in st.session_state:
        st.session_state.output_filename = None


def reset_state():
    """Reset to initial state."""
    if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
        shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
    st.session_state.phase = "upload"
    st.session_state.analysis = None
    st.session_state.prepared_audio_path = None
    st.session_state.temp_dir = None
    st.session_state.review_decisions = {}
    st.session_state.final_audio = None
    st.session_state.output_filename = None


def render_upload_phase():
    """Render the upload and initial processing phase."""
    st.markdown("Upload raw audio recordings and let AI clean them up automatically.")

    # Sidebar Controls
    with st.sidebar:
        st.header("Processing Options")

        language = st.selectbox(
            "Audio Language",
            options=["ar", "en", "auto"],
            index=0,
            format_func=lambda x: {"ar": "Arabic (العربية)", "en": "English", "auto": "Auto-detect"}[x],
            help="Select the language of your audio. Arabic is optimized for best results."
        )

        remove_noise = st.checkbox(
            "Remove Noise",
            value=True,
            help="Apply noise reduction using the first 0.5s as noise profile."
        )

        silence_threshold = st.slider(
            "Silence Threshold (dB)",
            min_value=-60,
            max_value=-20,
            value=-40,
            step=1,
            help="Audio below this level is considered silence."
        )

        st.divider()
        st.caption(
            "Silences longer than 800ms will be truncated to 100ms "
            "to maintain natural pacing."
        )
        if language == "ar":
            st.caption("Arabic text normalization enabled (diacritics, Alef variants, etc.)")

        st.divider()
        st.subheader("Repetition Detection")
        st.caption("""
        **Three-Tier System:**
        - **>85%** similarity: Auto-remove
        - **70-85%**: You review and decide
        - **<70%**: Keep both (different content)
        """)

    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=["mp3", "wav", "aup3"],
        help="Supported: MP3, WAV, Audacity Project (.aup3)"
    )

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        st.caption(f"**{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

        if st.button("Analyze Audio", type="primary", use_container_width=True):
            # Load model
            with st.spinner("Loading AI model (large-v3)..."):
                model = load_whisper_model()

            # Create persistent temp directory
            temp_dir = tempfile.mkdtemp()
            st.session_state.temp_dir = temp_dir
            st.session_state.output_filename = f"{os.path.splitext(uploaded_file.name)[0]}_cleaned.wav"

            # Save uploaded file
            input_path = os.path.join(temp_dir, uploaded_file.name)
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Processing status
            with st.status("Analyzing audio...", expanded=True) as status:
                log_container = st.container()
                progress_bar = st.progress(0)

                def log_message(msg: str):
                    with log_container:
                        st.write(msg)

                def update_progress(value: float):
                    progress_bar.progress(value)

                try:
                    # Convert to WAV
                    log_message("Converting to WAV format...")
                    wav_path = convert_input(input_path, temp_dir)
                    log_message("Conversion complete.")
                    update_progress(0.1)

                    # Phase 1: Prepare and analyze
                    processor = AudioProcessor(
                        whisper_model=model,
                        log_callback=log_message
                    )

                    prepared_path, analysis = processor.process_phase1_prepare(
                        input_path=wav_path,
                        temp_dir=temp_dir,
                        remove_noise=remove_noise,
                        language=language if language != "auto" else None,
                        progress_callback=update_progress,
                    )

                    # Store in session state
                    st.session_state.prepared_audio_path = prepared_path
                    st.session_state.analysis = analysis
                    st.session_state.silence_threshold = silence_threshold
                    st.session_state.language = language

                    status.update(label="Analysis complete!", state="complete")

                except Exception as e:
                    status.update(label="Analysis failed", state="error")
                    st.error(str(e))
                    reset_state()
                    return

            # Transition to review phase
            st.session_state.phase = "review"
            st.rerun()


def render_review_phase():
    """Render the review phase for uncertain repetitions."""
    analysis: RepetitionAnalysis = st.session_state.analysis

    st.markdown("### Analysis Results")

    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Segments", len(analysis.segments))
    with col2:
        st.metric("Auto-Remove", len(analysis.auto_delete), help="High confidence (>85%)")
    with col3:
        st.metric("Needs Review", len(analysis.needs_review), help="Uncertain (70-85%)")

    # Auto-delete section
    if analysis.auto_delete:
        with st.expander(f"Auto-Remove ({len(analysis.auto_delete)} segments)", expanded=False):
            st.caption("These segments will be automatically removed (>85% similarity):")
            for i, deletion in enumerate(analysis.auto_delete):
                st.text(f"{i+1}. [{deletion.start:.1f}s - {deletion.end:.1f}s] {deletion.reason}")

    # Review section - THE MAIN INTERACTIVE PART
    if analysis.needs_review:
        st.markdown("---")
        st.markdown("### Potential Repetitions Found")
        st.caption("Review these uncertain matches (70-85% similarity). Select which segment to **keep**.")

        for idx, candidate in enumerate(analysis.needs_review):
            with st.container():
                st.markdown(f"#### Pair {idx + 1} — {candidate.similarity:.0%} Similar")

                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("**Segment A** (earlier)")
                    st.text(f"[{candidate.segment_a.start:.1f}s - {candidate.segment_a.end:.1f}s]")
                    st.info(candidate.segment_a.text)
                    st.caption(f"Normalized: _{candidate.normalized_a}_")

                with col_b:
                    st.markdown("**Segment B** (later)")
                    st.text(f"[{candidate.segment_b.start:.1f}s - {candidate.segment_b.end:.1f}s]")
                    st.info(candidate.segment_b.text)
                    st.caption(f"Normalized: _{candidate.normalized_b}_")

                # Determine which is longer/more complete
                len_a = len(candidate.normalized_a)
                len_b = len(candidate.normalized_b)
                if len_b > len_a:
                    recommended = "keep_b"
                    rec_label = "B is longer/more complete"
                elif len_a > len_b:
                    recommended = "keep_a"
                    rec_label = "A is longer/more complete"
                else:
                    recommended = "keep_b"  # Default to later take
                    rec_label = "Same length - keeping later take"

                # Selection
                key = f"decision_{idx}"
                default_idx = 0 if recommended == "keep_b" else 1

                decision = st.radio(
                    f"Decision for Pair {idx + 1}:",
                    options=["keep_b", "keep_a", "keep_both"],
                    index=default_idx,
                    format_func=lambda x: {
                        "keep_b": "Delete A, Keep B (recommended)" if recommended == "keep_b" else "Delete A, Keep B",
                        "keep_a": "Delete B, Keep A (recommended)" if recommended == "keep_a" else "Delete B, Keep A",
                        "keep_both": "Keep Both (not a repetition)"
                    }[x],
                    key=key,
                    horizontal=True,
                    help=rec_label
                )

                st.session_state.review_decisions[idx] = decision
                st.markdown("---")

    else:
        st.success("No uncertain matches to review! All repetitions were high-confidence.")

    # Action buttons
    col_back, col_process = st.columns(2)

    with col_back:
        if st.button("Start Over", use_container_width=True):
            reset_state()
            st.rerun()

    with col_process:
        if st.button("Apply Changes & Generate Audio", type="primary", use_container_width=True):
            process_final_audio()


def process_final_audio():
    """Apply all deletions and generate final audio."""
    analysis: RepetitionAnalysis = st.session_state.analysis

    # Collect all deletions
    all_deletions: list[DeletionRange] = list(analysis.auto_delete)

    # Add user-confirmed deletions from review
    for idx, candidate in enumerate(analysis.needs_review):
        decision = st.session_state.review_decisions.get(idx, "keep_b")

        if decision == "keep_b":
            # Delete segment A
            all_deletions.append(candidate.get_deletion_for("a"))
        elif decision == "keep_a":
            # Delete segment B
            all_deletions.append(candidate.get_deletion_for("b"))
        # "keep_both" = no deletion

    # Load model
    with st.spinner("Finalizing audio..."):
        model = load_whisper_model()

        with st.status("Generating final audio...", expanded=True) as status:
            log_container = st.container()
            progress_bar = st.progress(0)

            def log_message(msg: str):
                with log_container:
                    st.write(msg)

            def update_progress(value: float):
                progress_bar.progress(value)

            try:
                processor = AudioProcessor(
                    whisper_model=model,
                    log_callback=log_message
                )

                output_path = os.path.join(st.session_state.temp_dir, "final_output.wav")

                processor.process_phase2_finalize(
                    input_path=st.session_state.prepared_audio_path,
                    output_path=output_path,
                    deletions=all_deletions,
                    silence_threshold_db=float(st.session_state.silence_threshold),
                    progress_callback=update_progress,
                )

                # Read final audio
                with open(output_path, "rb") as f:
                    st.session_state.final_audio = f.read()

                status.update(label="Complete!", state="complete")

            except Exception as e:
                status.update(label="Failed", state="error")
                st.error(str(e))
                return

    # Transition to complete phase
    st.session_state.phase = "complete"
    st.rerun()


def render_complete_phase():
    """Render the completion phase with download."""
    st.success("Audio processing complete!")

    st.subheader("Final Result")
    st.audio(st.session_state.final_audio, format="audio/wav")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Download Cleaned Audio",
            data=st.session_state.final_audio,
            file_name=st.session_state.output_filename,
            mime="audio/wav",
            type="primary",
            use_container_width=True,
        )

    with col2:
        if st.button("Process Another File", use_container_width=True):
            reset_state()
            st.rerun()


def main():
    st.set_page_config(
        page_title="AI Audio Refiner",
        page_icon="🎙️",
        layout="wide",
    )

    st.title("AI Audio Refiner")

    init_session_state()

    # Render based on current phase
    if st.session_state.phase == "upload":
        render_upload_phase()
    elif st.session_state.phase == "review":
        render_review_phase()
    elif st.session_state.phase == "complete":
        render_complete_phase()


if __name__ == "__main__":
    main()
