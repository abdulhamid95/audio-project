"""AI Tutorial Audio Refiner - Streamlit Web Application."""

import os
import tempfile
import streamlit as st

from processor import process_audio_file


def main():
    st.set_page_config(
        page_title="AI Tutorial Audio Refiner",
        page_icon="🎙️",
        layout="centered",
    )

    st.title("AI Tutorial Audio Refiner")
    st.markdown(
        "Upload your tutorial recordings to automatically clean them up with "
        "noise reduction, repetition removal, and silence trimming."
    )

    # Sidebar settings
    with st.sidebar:
        st.header("Processing Settings")

        st.subheader("Silence Removal")
        remove_silence = st.checkbox("Remove Long Silences", value=True)
        silence_threshold = st.slider(
            "Silence Threshold (dB)",
            min_value=-60.0,
            max_value=-20.0,
            value=-40.0,
            step=1.0,
            help="Audio below this level is considered silence. Lower values detect quieter sounds as silence.",
            disabled=not remove_silence,
        )

        st.subheader("Repetition Removal")
        remove_repetitions = st.checkbox(
            "Remove Repetitions",
            value=True,
            help="Uses AI to detect and remove repeated phrases, keeping only the final take.",
        )

        st.subheader("Noise Reduction")
        noise_intensity = st.slider(
            "Noise Reduction Intensity",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.1,
            help="0 = no noise reduction, 1 = maximum noise reduction",
        )

    # File upload
    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=["mp3", "wav", "aup3"],
        help="Supported formats: MP3, WAV, Audacity Project (.aup3)",
    )

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        st.caption(f"Uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

        if st.button("Process Audio", type="primary", use_container_width=True):
            # Create temp directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded file
                input_path = os.path.join(temp_dir, uploaded_file.name)
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Progress tracking
                progress_bar = st.progress(0, text="Starting processing...")
                status_text = st.empty()

                def update_progress(progress: float, message: str):
                    progress_bar.progress(progress, text=message)
                    status_text.text(message)

                # Process the audio
                result = process_audio_file(
                    input_path=input_path,
                    output_dir=temp_dir,
                    remove_silence=remove_silence,
                    silence_threshold_db=silence_threshold,
                    remove_repetitions_enabled=remove_repetitions,
                    noise_reduction_intensity=noise_intensity,
                    progress_callback=update_progress,
                )

                if result.error:
                    st.error(f"Processing failed: {result.error}")
                else:
                    progress_bar.progress(1.0, text="Complete!")
                    st.success("Processing complete!")

                    # Show statistics
                    st.subheader("Processing Summary")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Repetitions Removed", f"{result.repetitions_removed:.1f}s")
                    with col2:
                        st.metric("Silence Trimmed", f"{result.silence_trimmed:.1f}s")

                    st.markdown("**Stages completed:**")
                    for stage in result.stages_completed:
                        st.markdown(f"- {stage}")

                    # Audio preview
                    st.subheader("Preview Result")
                    with open(result.output_path, "rb") as f:
                        audio_bytes = f.read()
                    st.audio(audio_bytes, format="audio/wav")

                    # Download button
                    output_filename = f"{os.path.splitext(uploaded_file.name)[0]}_cleaned.wav"
                    st.download_button(
                        label="Download Processed Audio",
                        data=audio_bytes,
                        file_name=output_filename,
                        mime="audio/wav",
                        type="primary",
                        use_container_width=True,
                    )


if __name__ == "__main__":
    main()
