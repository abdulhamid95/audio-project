"""Audio processing pipeline for the web application."""

import os
import tempfile
from dataclasses import dataclass
from typing import Callable

from audio_editor import (
    convert_to_wav,
    reduce_noise,
    transcribe_audio,
    detect_repetitions,
    remove_repetitions,
    trim_silences,
)


@dataclass
class ProcessingResult:
    """Result of audio processing."""
    output_path: str
    stages_completed: list[str]
    repetitions_removed: float  # seconds
    silence_trimmed: float  # seconds
    error: str | None = None


def process_audio_file(
    input_path: str,
    output_dir: str,
    remove_silence: bool = True,
    silence_threshold_db: float = -40.0,
    remove_repetitions_enabled: bool = True,
    noise_reduction_intensity: float = 0.8,
    progress_callback: Callable[[float, str], None] | None = None,
) -> ProcessingResult:
    """
    Process an audio file through the cleaning pipeline.

    Args:
        input_path: Path to the uploaded audio file.
        output_dir: Directory to save output files.
        remove_silence: Whether to trim long silences.
        silence_threshold_db: Threshold for silence detection (dB).
        remove_repetitions_enabled: Whether to detect and remove repetitions.
        noise_reduction_intensity: Noise reduction strength (0.0 to 1.0).
        progress_callback: Function(progress: float, message: str) for updates.

    Returns:
        ProcessingResult with output path and statistics.
    """
    stages_completed = []
    repetitions_removed = 0.0
    silence_trimmed = 0.0

    def update_progress(progress: float, message: str):
        if progress_callback:
            progress_callback(progress, message)

    try:
        current_file = input_path

        # Stage 1: Convert to WAV (0-20%)
        update_progress(0.05, "Converting to WAV format...")
        wav_path = convert_to_wav(current_file, output_dir)
        current_file = wav_path
        stages_completed.append("Conversion to WAV")
        update_progress(0.20, "Conversion complete")

        # Stage 2: Noise Reduction (20-40%)
        if noise_reduction_intensity > 0:
            update_progress(0.25, "Applying noise reduction...")
            denoised_path = os.path.join(output_dir, "denoised.wav")
            current_file = reduce_noise(
                current_file,
                denoised_path,
                prop_decrease=noise_reduction_intensity,
            )
            stages_completed.append(f"Noise Reduction ({noise_reduction_intensity:.0%})")
            update_progress(0.40, "Noise reduction complete")

        # Stage 3: Repetition Removal (40-70%)
        if remove_repetitions_enabled:
            update_progress(0.45, "Transcribing audio with AI...")
            segments = transcribe_audio(current_file, model_size="base")
            update_progress(0.55, f"Found {len(segments)} segments, detecting repetitions...")

            cuts = detect_repetitions(segments)

            if cuts:
                update_progress(0.60, f"Removing {len(cuts)} repetitions...")
                no_rep_path = os.path.join(output_dir, "no_repetitions.wav")
                current_file, repetitions_removed = remove_repetitions(
                    current_file,
                    no_rep_path,
                    cuts,
                )
                stages_completed.append(f"Repetition Removal ({repetitions_removed:.1f}s removed)")
            else:
                stages_completed.append("Repetition Detection (none found)")

            update_progress(0.70, "Repetition processing complete")

        # Stage 4: Silence Trimming (70-90%)
        if remove_silence:
            update_progress(0.75, "Detecting and trimming silences...")
            trimmed_path = os.path.join(output_dir, "trimmed.wav")
            current_file, silence_trimmed = trim_silences(
                current_file,
                trimmed_path,
                silence_threshold_db=silence_threshold_db,
            )
            stages_completed.append(f"Silence Trimming ({silence_trimmed:.1f}s removed)")
            update_progress(0.90, "Silence trimming complete")

        # Stage 5: Final output (90-100%)
        update_progress(0.95, "Finalizing output...")
        output_path = os.path.join(output_dir, "processed_output.wav")

        # Copy to final location with proper format
        from pydub import AudioSegment
        final_audio = AudioSegment.from_wav(current_file)
        final_audio.export(output_path, format="wav")

        stages_completed.append("Output saved")
        update_progress(1.0, "Processing complete!")

        return ProcessingResult(
            output_path=output_path,
            stages_completed=stages_completed,
            repetitions_removed=repetitions_removed,
            silence_trimmed=silence_trimmed,
        )

    except Exception as e:
        return ProcessingResult(
            output_path="",
            stages_completed=stages_completed,
            repetitions_removed=repetitions_removed,
            silence_trimmed=silence_trimmed,
            error=str(e),
        )
