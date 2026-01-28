"""Convert .aup3 files to WAV format."""

import os
import sqlite3
import subprocess
import tempfile
from pathlib import Path


def is_aup3_file(file_path: str) -> bool:
    """Check if the file is an Audacity .aup3 project file."""
    return Path(file_path).suffix.lower() == ".aup3"


def extract_from_aup3_sqlite(aup3_path: str, output_path: str) -> bool:
    """
    Attempt to extract audio from .aup3 SQLite database.

    .aup3 files are SQLite databases containing audio data in BLOB format.
    This function tries to extract and reconstruct the audio.

    Returns True if successful, False otherwise.
    """
    try:
        conn = sqlite3.connect(aup3_path)
        cursor = conn.cursor()

        # Check if this is a valid aup3 file with audio data
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        if "sampleblocks" not in tables:
            conn.close()
            return False

        # Get project info
        cursor.execute("SELECT doc FROM project LIMIT 1")
        project_doc = cursor.fetchone()

        if not project_doc:
            conn.close()
            return False

        conn.close()
        # For complex .aup3 reconstruction, fall back to ffmpeg
        # Direct SQLite extraction requires parsing Audacity's internal format
        return False

    except (sqlite3.Error, Exception):
        return False


def convert_with_ffmpeg(input_path: str, output_path: str) -> str:
    """
    Convert audio file to high-quality WAV using ffmpeg.

    Output: 48kHz, 24-bit PCM WAV
    """
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", input_path,
        "-ar", "48000",  # 48kHz sample rate
        "-sample_fmt", "s32",  # 24-bit (stored in 32-bit container)
        "-ac", "1",  # Mono for speech
        "-c:a", "pcm_s24le",  # 24-bit PCM
        output_path
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")

    return output_path


def convert_to_wav(input_path: str, output_dir: str | None = None) -> str:
    """
    Convert input audio file to WAV format suitable for processing.

    Args:
        input_path: Path to input audio file (.aup3, .mp3, .wav, etc.)
        output_dir: Directory for output file. If None, uses same directory as input.

    Returns:
        Path to the WAV file ready for processing.
    """
    input_path = os.path.abspath(input_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Determine output path
    if output_dir is None:
        output_dir = os.path.dirname(input_path)

    base_name = Path(input_path).stem
    output_path = os.path.join(output_dir, f"{base_name}_working.wav")

    # If already a WAV file, check if it needs conversion for consistency
    if Path(input_path).suffix.lower() == ".wav":
        # Still convert to ensure consistent format (48kHz, 24-bit, mono)
        convert_with_ffmpeg(input_path, output_path)
        return output_path

    # Handle .aup3 files
    if is_aup3_file(input_path):
        # Try direct SQLite extraction first
        if extract_from_aup3_sqlite(input_path, output_path):
            return output_path

        # Fall back to ffmpeg (works if ffmpeg has Audacity support)
        try:
            convert_with_ffmpeg(input_path, output_path)
            return output_path
        except RuntimeError:
            raise RuntimeError(
                f"Could not convert .aup3 file. Please export as WAV from Audacity first, "
                f"or ensure ffmpeg is installed with proper codec support."
            )

    # For other formats, use ffmpeg
    convert_with_ffmpeg(input_path, output_path)
    return output_path
