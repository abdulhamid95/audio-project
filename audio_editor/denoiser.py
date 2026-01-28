"""Noise reduction for audio files."""

import numpy as np
from scipy.io import wavfile
import noisereduce as nr


def reduce_noise(
    input_path: str,
    output_path: str,
    noise_sample_seconds: float = 0.5,
    prop_decrease: float = 0.8,
    stationary: bool = True,
) -> str:
    """
    Apply noise reduction to an audio file.

    Args:
        input_path: Path to input WAV file.
        output_path: Path for output WAV file.
        noise_sample_seconds: Duration of audio at start to use as noise profile.
                            Set to 0 to use automatic noise estimation.
        prop_decrease: Proportion to reduce noise by (0.0 to 1.0).
        stationary: If True, assumes stationary noise (consistent background noise).
                   If False, uses non-stationary noise reduction (better for varying noise).

    Returns:
        Path to the denoised audio file.
    """
    # Load the audio file
    sample_rate, audio_data = wavfile.read(input_path)

    # Convert to float for processing
    if audio_data.dtype == np.int16:
        audio_float = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_float = audio_data.astype(np.float32) / 2147483648.0
    else:
        audio_float = audio_data.astype(np.float32)

    # Handle stereo by converting to mono if needed
    if len(audio_float.shape) > 1:
        audio_float = np.mean(audio_float, axis=1)

    # Extract noise sample from the beginning of the audio
    if noise_sample_seconds > 0:
        noise_samples = int(noise_sample_seconds * sample_rate)
        noise_clip = audio_float[:noise_samples]
    else:
        noise_clip = None

    # Apply noise reduction
    if stationary:
        reduced_audio = nr.reduce_noise(
            y=audio_float,
            sr=sample_rate,
            y_noise=noise_clip,
            prop_decrease=prop_decrease,
            stationary=True,
        )
    else:
        reduced_audio = nr.reduce_noise(
            y=audio_float,
            sr=sample_rate,
            prop_decrease=prop_decrease,
            stationary=False,
        )

    # Normalize to prevent clipping
    max_val = np.max(np.abs(reduced_audio))
    if max_val > 0:
        reduced_audio = reduced_audio / max_val * 0.95

    # Convert back to int format for saving
    # Use 16-bit for wider compatibility
    reduced_int = (reduced_audio * 32767).astype(np.int16)

    # Save the denoised audio
    wavfile.write(output_path, sample_rate, reduced_int)

    return output_path
