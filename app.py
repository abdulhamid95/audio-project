"""AI Audio Refiner - Streamlit Web Application."""

import os
import tempfile
import streamlit as st
from faster_whisper import WhisperModel

from utils import convert_input
from processor import AudioProcessor


@st.cache_resource
def load_whisper_model():
    """Load and cache the Whisper model (large-v3 for Arabic accuracy)."""
    return WhisperModel("large-v3", device="cpu", compute_type="int8")


def main():
    st.set_page_config(
        page_title="AI Audio Refiner",
        page_icon="🎙️",
        layout="centered",
    )

    st.title("AI Audio Refiner")
    st.markdown(
        "Upload raw audio recordings and let AI clean them up automatically."
    )

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

        remove_repetitions = st.checkbox(
            "Remove Repetitions",
            value=True,
            help="Use AI to detect and remove repeated phrases (keeps the last take)."
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
            st.caption("🇸🇦 Arabic text normalization is enabled (diacritics, Alef variants, etc.)")

    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=["mp3", "wav", "aup3"],
        help="Supported: MP3, WAV, Audacity Project (.aup3)"
    )

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        st.caption(f"**{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

        if st.button("Process Audio", type="primary", use_container_width=True):
            # Load model
            with st.spinner("Loading AI model..."):
                model = load_whisper_model()

            # Create temp directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded file
                input_path = os.path.join(temp_dir, uploaded_file.name)
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Processing status container
                with st.status("Processing audio...", expanded=True) as status:
                    log_container = st.container()
                    progress_bar = st.progress(0)
                    logs = []

                    def log_message(msg: str):
                        logs.append(msg)
                        with log_container:
                            st.write(msg)

                    def update_progress(value: float):
                        progress_bar.progress(value)

                    try:
                        # Step 1: Convert to WAV
                        log_message("Converting to WAV format...")
                        wav_path = convert_input(input_path, temp_dir)
                        log_message("Conversion complete.")
                        update_progress(0.1)

                        # Step 2: Process
                        output_path = os.path.join(temp_dir, "processed_output.wav")
                        processor = AudioProcessor(
                            whisper_model=model,
                            log_callback=log_message
                        )

                        processor.process(
                            input_path=wav_path,
                            output_path=output_path,
                            remove_noise=remove_noise,
                            remove_repetitions=remove_repetitions,
                            silence_threshold_db=float(silence_threshold),
                            language=language if language != "auto" else None,
                            progress_callback=update_progress,
                        )

                        status.update(label="Processing complete!", state="complete")

                    except RuntimeError as e:
                        status.update(label="Processing failed", state="error")
                        st.error(str(e))
                        return

                # Results
                st.success("Audio processing complete!")

                # Audio preview
                st.subheader("Result")
                with open(output_path, "rb") as f:
                    audio_bytes = f.read()
                st.audio(audio_bytes, format="audio/wav")

                # Download button
                output_filename = f"{os.path.splitext(uploaded_file.name)[0]}_cleaned.wav"
                st.download_button(
                    label="Download Cleaned Audio",
                    data=audio_bytes,
                    file_name=output_filename,
                    mime="audio/wav",
                    type="primary",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
