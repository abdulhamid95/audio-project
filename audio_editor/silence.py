"""Silence detection and trimming for audio files."""

import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment


def detect_silence_regions(
    audio_path: str,
    silence_threshold_db: float = -40.0,
    min_silence_ms: int = 800,
) -> list[tuple[float, float]]:
    """
    Detect regions of silence in an audio file.

    Args:
        audio_path: Path to the audio file.
        silence_threshold_db: Volume threshold below which audio is considered silence (in dB).
        min_silence_ms: Minimum duration of silence to detect (milliseconds).

    Returns:
        List of (start_seconds, end_seconds) tuples for silent regions.
    """
    sample_rate, audio_data = wavfile.read(audio_path)

    # Convert to float
    if audio_data.dtype == np.int16:
        audio_float = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_float = audio_data.astype(np.float32) / 2147483648.0
    else:
        audio_float = audio_data.astype(np.float32)

    # Handle stereo
    if len(audio_float.shape) > 1:
        audio_float = np.mean(audio_float, axis=1)

    # Convert threshold from dB to linear
    threshold_linear = 10 ** (silence_threshold_db / 20)

    # Calculate RMS in windows
    window_size = int(sample_rate * 0.02)  # 20ms windows
    hop_size = window_size // 2

    silence_regions = []
    in_silence = False
    silence_start = 0

    for i in range(0, len(audio_float) - window_size, hop_size):
        window = audio_float[i:i + window_size]
        rms = np.sqrt(np.mean(window ** 2))

        current_time = i / sample_rate

        if rms < threshold_linear:
            if not in_silence:
                in_silence = True
                silence_start = current_time
        else:
            if in_silence:
                silence_end = current_time
                duration_ms = (silence_end - silence_start) * 1000

                if duration_ms >= min_silence_ms:
                    silence_regions.append((silence_start, silence_end))

                in_silence = False

    # Handle silence at the end
    if in_silence:
        silence_end = len(audio_float) / sample_rate
        duration_ms = (silence_end - silence_start) * 1000

        if duration_ms >= min_silence_ms:
            silence_regions.append((silence_start, silence_end))

    return silence_regions


def trim_silences(
    input_path: str,
    output_path: str,
    min_silence_ms: int = 800,
    target_silence_ms: int = 300,
    silence_threshold_db: float = -40.0,
) -> tuple[str, float]:
    """
    Trim long silences in audio to a target duration.

    Args:
        input_path: Path to input audio file.
        output_path: Path for output audio file.
        min_silence_ms: Only trim silences longer than this (milliseconds).
        target_silence_ms: Trim long silences to this duration (milliseconds).
        silence_threshold_db: Volume threshold for silence detection (dB).

    Returns:
        Tuple of (output_path, duration_trimmed_seconds)
    """
    # Detect silent regions
    silence_regions = detect_silence_regions(
        input_path,
        silence_threshold_db=silence_threshold_db,
        min_silence_ms=min_silence_ms,
    )

    if not silence_regions:
        # No long silences to trim, just copy the file
        audio = AudioSegment.from_wav(input_path)
        audio.export(output_path, format="wav")
        return output_path, 0.0

    # Load audio with pydub for manipulation
    audio = AudioSegment.from_wav(input_path)
    original_duration = len(audio) / 1000.0

    # Calculate how much to trim from each silence
    # We want to keep target_silence_ms of each silence
    target_silence_seconds = target_silence_ms / 1000.0

    # Build result by processing silences from end to start
    # (to avoid recalculating positions after cuts)
    result = audio
    total_trimmed = 0.0

    for start, end in reversed(silence_regions):
        silence_duration = end - start
        trim_amount = silence_duration - target_silence_seconds

        if trim_amount > 0:
            # Calculate the portion to remove
            # Keep half of target silence at start, half at end
            keep_start = target_silence_seconds / 2
            keep_end = target_silence_seconds / 2

            trim_start_ms = int((start + keep_start) * 1000)
            trim_end_ms = int((end - keep_end) * 1000)

            # Remove the middle portion
            result = result[:trim_start_ms] + result[trim_end_ms:]
            total_trimmed += (trim_end_ms - trim_start_ms) / 1000.0

    # Export result
    result.export(output_path, format="wav")

    return output_path, total_trimmed
