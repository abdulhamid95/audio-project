"""AI transcription using faster-whisper."""

from dataclasses import dataclass
from faster_whisper import WhisperModel


def detect_device() -> tuple[str, str]:
    """
    Detect available hardware and return optimal device/compute settings.

    Returns:
        Tuple of (device, compute_type) for WhisperModel initialization.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"


@dataclass
class Segment:
    """A transcribed segment with timing information."""
    text: str
    start: float  # Start time in seconds
    end: float    # End time in seconds

    def __repr__(self) -> str:
        return f"Segment({self.start:.2f}-{self.end:.2f}: '{self.text}')"


def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
    language: str | None = None,
    device: str = "auto",
) -> list[Segment]:
    """
    Transcribe audio file using faster-whisper with word-level timestamps.

    Args:
        audio_path: Path to the audio file.
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large-v2').
                   'base' offers good balance of speed and accuracy.
        language: Language code (e.g., 'en', 'ar'). If None, auto-detects.
        device: Device to use ('cpu', 'cuda', or 'auto').

    Returns:
        List of Segment objects with text and timestamps.
    """
    # Auto-detect optimal device and compute type
    if device == "auto":
        detected_device, compute_type = detect_device()
    elif device == "cuda":
        detected_device = "cuda"
        compute_type = "float16"
    else:
        detected_device = "cpu"
        compute_type = "int8"

    # Initialize the model with optimal settings
    model = WhisperModel(
        model_size,
        device=detected_device,
        compute_type=compute_type,
    )

    # Build transcription kwargs with optimized settings
    transcribe_kwargs = {
        "language": language,
        "word_timestamps": True,
        "vad_filter": True,  # Filter out silence
        "beam_size": 1,  # Faster decoding with greedy search
    }

    # Add Arabic-specific prompt for better recognition
    if language == "ar":
        transcribe_kwargs["initial_prompt"] = "اللغة العربية الفصحى"

    # Transcribe with optimized settings
    segments_generator, info = model.transcribe(audio_path, **transcribe_kwargs)

    # Convert to our Segment format
    segments = []
    for segment in segments_generator:
        # Use word-level timestamps if available for more precision
        if segment.words:
            # Group words into natural phrases (by segment)
            text = segment.text.strip()
            start = segment.words[0].start
            end = segment.words[-1].end
        else:
            text = segment.text.strip()
            start = segment.start
            end = segment.end

        if text:  # Only add non-empty segments
            segments.append(Segment(text=text, start=start, end=end))

    return segments


def get_words_with_timestamps(
    audio_path: str,
    model_size: str = "base",
    language: str | None = None,
) -> list[dict]:
    """
    Get individual words with their timestamps.

    Useful for fine-grained repetition detection.

    Returns:
        List of dicts: [{'word': str, 'start': float, 'end': float}, ...]
    """
    # Auto-detect optimal device and compute type
    detected_device, compute_type = detect_device()
    model = WhisperModel(model_size, device=detected_device, compute_type=compute_type)

    # Build transcription kwargs with optimized settings
    transcribe_kwargs = {
        "language": language,
        "word_timestamps": True,
        "beam_size": 1,  # Faster decoding
    }

    # Add Arabic-specific prompt for better recognition
    if language == "ar":
        transcribe_kwargs["initial_prompt"] = "اللغة العربية الفصحى"

    segments_generator, _ = model.transcribe(audio_path, **transcribe_kwargs)

    words = []
    for segment in segments_generator:
        if segment.words:
            for word in segment.words:
                words.append({
                    'word': word.word.strip(),
                    'start': word.start,
                    'end': word.end,
                })

    return words
