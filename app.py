"""AI Audio Refiner - Streamlit Web Application with Interactive Review."""

import io
import os
import tempfile
import shutil
import streamlit as st
from faster_whisper import WhisperModel
from pydub import AudioSegment

from utils import convert_input
from processor import (
    AudioProcessor,
    RepetitionAnalysis,
    DeletionRange,
    VideoSegment,
    check_ffmpeg,
    create_timestamps_map,
    process_video_ffmpeg,
    get_video_duration,
)


def extract_audio_segment(audio_path: str, start: float, end: float) -> bytes:
    """Extract a segment from an audio file and return as bytes for playback."""
    audio = AudioSegment.from_wav(audio_path)
    start_ms = int(start * 1000)
    end_ms = int(end * 1000)
    segment = audio[start_ms:end_ms]

    buffer = io.BytesIO()
    segment.export(buffer, format="wav")
    buffer.seek(0)
    return buffer.read()


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
    # Video processing state
    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    if "final_video" not in st.session_state:
        st.session_state.final_video = None
    if "video_output_filename" not in st.session_state:
        st.session_state.video_output_filename = None
    if "all_deletions" not in st.session_state:
        st.session_state.all_deletions = []


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
    # Video state
    st.session_state.video_path = None
    st.session_state.final_video = None
    st.session_state.video_output_filename = None
    st.session_state.all_deletions = []


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
                        silence_threshold_db=float(silence_threshold),
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
        st.metric("Auto-Remove", len(analysis.auto_delete), help="High confidence (>75%)")
    with col3:
        st.metric("Needs Review", len(analysis.needs_review), help="Uncertain (50-75%)")

    # Auto-delete section
    if analysis.auto_delete:
        with st.expander(f"Auto-Remove ({len(analysis.auto_delete)} segments)", expanded=False):
            st.caption("These segments will be automatically removed (>75% similarity):")
            for i, deletion in enumerate(analysis.auto_delete):
                st.text(f"{i+1}. [{deletion.start:.1f}s - {deletion.end:.1f}s] {deletion.reason}")

    # Review section - THE MAIN INTERACTIVE PART
    if analysis.needs_review:
        st.markdown("---")
        st.markdown("### Review Potential Repetitions (50%-75% Match)")
        st.caption("Review these uncertain matches. Select which segment to **keep** (longer/more complete is recommended).")

        for idx, candidate in enumerate(analysis.needs_review):
            with st.container():
                st.markdown(f"#### Pair {idx + 1} — {candidate.similarity:.0%} Similar")

                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("**Segment A** (earlier)")
                    st.text(f"[{candidate.segment_a.start:.1f}s - {candidate.segment_a.end:.1f}s]")
                    st.info(candidate.segment_a.text)
                    # Play button for segment A
                    if st.session_state.prepared_audio_path:
                        audio_a = extract_audio_segment(
                            st.session_state.prepared_audio_path,
                            candidate.segment_a.start,
                            candidate.segment_a.end
                        )
                        st.audio(audio_a, format="audio/wav")
                    st.caption(f"Normalized: _{candidate.normalized_a}_")

                with col_b:
                    st.markdown("**Segment B** (later)")
                    st.text(f"[{candidate.segment_b.start:.1f}s - {candidate.segment_b.end:.1f}s]")
                    st.info(candidate.segment_b.text)
                    # Play button for segment B
                    if st.session_state.prepared_audio_path:
                        audio_b = extract_audio_segment(
                            st.session_state.prepared_audio_path,
                            candidate.segment_b.start,
                            candidate.segment_b.end
                        )
                        st.audio(audio_b, format="audio/wav")
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

    # Save deletions for video processing
    st.session_state.all_deletions = all_deletions

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
    """Render the completion phase with download and video processing."""
    st.success("Audio processing complete!")

    # Audio Result Section
    st.subheader("🎵 Final Audio")
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

    # Video Processing Section
    st.markdown("---")
    st.subheader("🎬 Video Processing (Optional)")

    # Check FFmpeg availability
    ffmpeg_available = check_ffmpeg()

    if not ffmpeg_available:
        st.warning("FFmpeg not found. Install FFmpeg to enable video processing with speed ramping.")
    else:
        st.info(
            "Upload a screen recording to sync with the edited audio. "
            "Sections where audio was removed will play at **10x speed** instead of being cut."
        )

        # Show final video if already processed
        if st.session_state.final_video:
            st.success("Video processing complete!")
            st.video(st.session_state.final_video)

            st.download_button(
                label="Download Processed Video (MP4)",
                data=st.session_state.final_video,
                file_name=st.session_state.video_output_filename,
                mime="video/mp4",
                type="primary",
                use_container_width=True,
            )
        else:
            # Video upload
            video_file = st.file_uploader(
                "Upload Video File",
                type=["mp4", "mov", "avi", "mkv", "webm"],
                help="Upload the screen recording that corresponds to this audio."
            )

            if video_file is not None:
                st.video(video_file)
                st.caption(f"**{video_file.name}** ({video_file.size / (1024*1024):.1f} MB)")

                # Video processing options
                col_preset, col_bitrate = st.columns(2)
                with col_preset:
                    preset = st.selectbox(
                        "Encoding Speed",
                        options=["ultrafast", "fast", "medium", "slow"],
                        index=1,
                        help="Faster = larger file, slower = better compression"
                    )
                with col_bitrate:
                    bitrate = st.selectbox(
                        "Video Quality",
                        options=["3000k", "5000k", "8000k", "10000k"],
                        index=1,
                        help="Higher = better quality, larger file"
                    )

                speedup = st.slider(
                    "Speed-up Factor",
                    min_value=2,
                    max_value=20,
                    value=10,
                    help="How fast to play removed sections (10x = 10 seconds becomes 1 second)"
                )

                if st.button("Process Video", type="primary", use_container_width=True):
                    process_video(video_file, preset, bitrate, speedup)


def process_video(video_file, preset: str, bitrate: str, speedup: int):
    """Process video with speed ramping based on audio deletions."""
    temp_dir = st.session_state.temp_dir

    with st.status("Processing video...", expanded=True) as status:
        log_container = st.container()

        def log_message(msg: str):
            with log_container:
                st.write(msg)

        try:
            # Save uploaded video
            log_message("Saving video file...")
            video_input_path = os.path.join(temp_dir, video_file.name)
            with open(video_input_path, "wb") as f:
                f.write(video_file.getbuffer())

            # Get video duration
            log_message("Analyzing video...")
            video_duration = get_video_duration(video_input_path)
            if video_duration <= 0:
                raise RuntimeError("Could not determine video duration.")
            log_message(f"Video duration: {video_duration:.1f} seconds")

            # Create timestamps map from deletions
            log_message("Building segment map...")
            deletions = st.session_state.all_deletions
            timestamps_map = create_timestamps_map(deletions, video_duration)

            normal_count = sum(1 for s in timestamps_map if s.speed == "normal")
            speedup_count = sum(1 for s in timestamps_map if s.speed == "speedup")
            log_message(f"  Normal segments: {normal_count}")
            log_message(f"  Speed-up segments: {speedup_count} (will play at {speedup}x)")

            # Save processed audio to file for FFmpeg
            audio_path = os.path.join(temp_dir, "final_audio.wav")
            with open(audio_path, "wb") as f:
                f.write(st.session_state.final_audio)

            # Output path
            output_filename = f"{os.path.splitext(video_file.name)[0]}_edited.mp4"
            output_path = os.path.join(temp_dir, output_filename)

            # Process video
            log_message("Starting FFmpeg encoding (this may take a while)...")
            process_video_ffmpeg(
                video_input=video_input_path,
                audio_input=audio_path,
                output_path=output_path,
                timestamps_map=timestamps_map,
                speedup_factor=float(speedup),
                video_bitrate=bitrate,
                preset=preset,
                log_callback=log_message
            )

            # Read output video
            with open(output_path, "rb") as f:
                st.session_state.final_video = f.read()
            st.session_state.video_output_filename = output_filename

            status.update(label="Video processing complete!", state="complete")

        except Exception as e:
            status.update(label="Video processing failed", state="error")
            st.error(str(e))
            return

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
