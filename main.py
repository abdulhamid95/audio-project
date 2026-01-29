#!/usr/bin/env python3
"""
AI Audio Editor for Tutorial Recordings

Cleans up tutorial recordings with:
- Noise reduction
- Smart repetition removal (using AI transcription)
- Silence trimming
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

from audio_editor import (
    convert_to_wav,
    reduce_noise,
    transcribe_audio,
    detect_repetitions,
    remove_repetitions,
    trim_silences,
)


def log(message: str, verbose: bool = True) -> None:
    """Print a log message if verbose mode is enabled."""
    if verbose:
        print(f"[*] {message}")


def process_audio(
    input_path: str,
    output_path: str | None = None,
    skip_denoise: bool = False,
    skip_repetitions: bool = False,
    skip_silence: bool = False,
    similarity_threshold: float = 0.80,
    min_silence_ms: int = 800,
    target_silence_ms: int = 300,
    whisper_model: str = "base",
    noise_sample_seconds: float = 0.5,
    verbose: bool = True,
) -> str:
    """
    Process an audio file through the complete pipeline.

    Args:
        input_path: Path to input audio file (.aup3, .wav, .mp3, etc.)
        output_path: Path for output file. If None, creates one automatically.
        skip_denoise: Skip noise reduction step.
        skip_repetitions: Skip repetition detection and removal.
        skip_silence: Skip silence trimming.
        similarity_threshold: Threshold for repetition detection (0-1).
        min_silence_ms: Minimum silence duration to trim.
        target_silence_ms: Target duration for trimmed silences.
        whisper_model: Whisper model size for transcription.
        noise_sample_seconds: Seconds of audio to use as noise profile.
        verbose: Print progress messages.

    Returns:
        Path to the processed output file.
    """
    input_path = os.path.abspath(input_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Determine output path
    if output_path is None:
        input_stem = Path(input_path).stem
        input_dir = os.path.dirname(input_path)
        output_path = os.path.join(input_dir, f"{input_stem}_cleaned.wav")

    output_path = os.path.abspath(output_path)

    # Create temp directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        current_file = input_path

        # Step 1: Convert to WAV
        log("Converting to WAV format...", verbose)
        wav_path = os.path.join(temp_dir, "01_converted.wav")
        current_file = convert_to_wav(current_file, temp_dir)
        os.rename(current_file, wav_path)
        current_file = wav_path
        log(f"  Converted: {current_file}", verbose)

        # Step 2: Noise reduction
        if not skip_denoise:
            log("Applying noise reduction...", verbose)
            denoised_path = os.path.join(temp_dir, "02_denoised.wav")
            current_file = reduce_noise(
                current_file,
                denoised_path,
                noise_sample_seconds=noise_sample_seconds,
            )
            log("  Noise reduction complete", verbose)

        # Step 3: Silence trimming (BEFORE transcription for speed)
        # Removing silence first reduces the audio duration Whisper needs to process
        if not skip_silence:
            log("Trimming long silences...", verbose)
            trimmed_path = os.path.join(temp_dir, "03_trimmed.wav")
            current_file, trimmed = trim_silences(
                current_file,
                trimmed_path,
                min_silence_ms=min_silence_ms,
                target_silence_ms=target_silence_ms,
            )
            log(f"  Trimmed {trimmed:.1f} seconds of silence", verbose)

        # Step 4: Transcription and repetition removal
        if not skip_repetitions:
            log(f"Transcribing audio (model: {whisper_model})...", verbose)
            segments = transcribe_audio(current_file, model_size=whisper_model)
            log(f"  Found {len(segments)} segments", verbose)

            log("Detecting repetitions...", verbose)
            cuts = detect_repetitions(
                segments,
                similarity_threshold=similarity_threshold,
            )

            if cuts:
                log(f"  Found {len(cuts)} repetitions to remove:", verbose)
                for cut in cuts:
                    log(f"    {cut.start:.1f}s - {cut.end:.1f}s: {cut.reason[:60]}", verbose)

                log("Removing repetitions...", verbose)
                no_rep_path = os.path.join(temp_dir, "04_no_repetitions.wav")
                current_file, removed = remove_repetitions(
                    current_file,
                    no_rep_path,
                    cuts,
                )
                log(f"  Removed {removed:.1f} seconds of repetitions", verbose)
            else:
                log("  No repetitions detected", verbose)

        # Copy final result to output path
        log(f"Saving output to: {output_path}", verbose)

        # Use pydub to ensure proper WAV format
        from pydub import AudioSegment
        final_audio = AudioSegment.from_wav(current_file)
        final_audio.export(output_path, format="wav")

    log("Processing complete!", verbose)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="AI-powered audio editor for cleaning up tutorial recordings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s recording.wav
  %(prog)s recording.aup3 -o cleaned.wav
  %(prog)s recording.wav --skip-denoise --similarity 0.85
  %(prog)s recording.wav --whisper-model small --min-silence 1000
        """,
    )

    parser.add_argument(
        "input",
        help="Input audio file (.aup3, .wav, .mp3, etc.)",
    )

    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: <input>_cleaned.wav)",
    )

    parser.add_argument(
        "--skip-denoise",
        action="store_true",
        help="Skip noise reduction step",
    )

    parser.add_argument(
        "--skip-repetitions",
        action="store_true",
        help="Skip repetition detection and removal",
    )

    parser.add_argument(
        "--skip-silence",
        action="store_true",
        help="Skip silence trimming",
    )

    parser.add_argument(
        "--similarity",
        type=float,
        default=0.80,
        help="Similarity threshold for repetition detection (0-1, default: 0.80)",
    )

    parser.add_argument(
        "--min-silence",
        type=int,
        default=800,
        help="Minimum silence duration to trim in ms (default: 800)",
    )

    parser.add_argument(
        "--target-silence",
        type=int,
        default=300,
        help="Target duration for trimmed silences in ms (default: 300)",
    )

    parser.add_argument(
        "--whisper-model",
        choices=["tiny", "base", "small", "medium", "large-v2"],
        default="base",
        help="Whisper model size (default: base)",
    )

    parser.add_argument(
        "--noise-sample",
        type=float,
        default=0.5,
        help="Seconds of audio at start to use as noise profile (default: 0.5)",
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    try:
        output = process_audio(
            input_path=args.input,
            output_path=args.output,
            skip_denoise=args.skip_denoise,
            skip_repetitions=args.skip_repetitions,
            skip_silence=args.skip_silence,
            similarity_threshold=args.similarity,
            min_silence_ms=args.min_silence,
            target_silence_ms=args.target_silence,
            whisper_model=args.whisper_model,
            noise_sample_seconds=args.noise_sample,
            verbose=not args.quiet,
        )

        if not args.quiet:
            print(f"\nOutput saved to: {output}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
