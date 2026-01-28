"""Helper functions for audio file handling."""

import os
import subprocess
from pathlib import Path


def convert_input(file_path: str, output_dir: str) -> str:
    """
    Convert input audio file to standardized WAV format.

    Handles .aup3, .mp3, .wav files using ffmpeg.

    Args:
        file_path: Path to input audio file.
        output_dir: Directory to save converted file.

    Returns:
        Path to standardized WAV file.

    Raises:
        RuntimeError: If conversion fails.
    """
    file_path = os.path.abspath(file_path)
    ext = Path(file_path).suffix.lower()
    base_name = Path(file_path).stem
    output_path = os.path.join(output_dir, f"{base_name}_converted.wav")

    # FFmpeg command for standardized output: 16kHz mono (optimal for Whisper)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", file_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            if ext == ".aup3":
                raise RuntimeError(
                    "Could not convert .aup3 file. Audacity project files require "
                    "special handling. Please open the file in Audacity and export "
                    "it as a WAV file (File -> Export -> Export as WAV)."
                )
            raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")

        return output_path

    except subprocess.TimeoutExpired:
        raise RuntimeError("Conversion timed out. File may be too large or corrupted.")
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg not found. Please install FFmpeg:\n"
            "  - Ubuntu/Debian: sudo apt install ffmpeg\n"
            "  - macOS: brew install ffmpeg\n"
            "  - Windows: Download from https://ffmpeg.org/download.html"
        )
