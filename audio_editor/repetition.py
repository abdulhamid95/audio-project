"""Detect and remove repetitions in audio based on transcription."""

from dataclasses import dataclass
from rapidfuzz import fuzz
from pydub import AudioSegment

from .transcriber import Segment


@dataclass
class TimeRange:
    """A time range to cut from the audio."""
    start: float  # Start time in seconds
    end: float    # End time in seconds
    reason: str   # Why this range is being cut

    def __repr__(self) -> str:
        return f"Cut({self.start:.2f}-{self.end:.2f}: {self.reason})"


def calculate_similarity(text_a: str, text_b: str) -> float:
    """
    Calculate similarity between two text segments.

    Uses fuzzy matching to handle slight variations in repeated phrases.

    Returns:
        Similarity score from 0.0 to 1.0
    """
    # Normalize texts
    text_a = text_a.lower().strip()
    text_b = text_b.lower().strip()

    # Use token sort ratio to handle word order variations
    # and partial ratio to handle incomplete repetitions
    ratio = fuzz.ratio(text_a, text_b) / 100.0
    token_ratio = fuzz.token_sort_ratio(text_a, text_b) / 100.0

    # Take the higher of the two scores
    return max(ratio, token_ratio)


def detect_repetitions(
    segments: list[Segment],
    similarity_threshold: float = 0.80,
    max_gap_seconds: float = 2.0,
) -> list[TimeRange]:
    """
    Detect repeated phrases in transcribed segments.

    Algorithm:
    1. Compare consecutive segments using fuzzy matching
    2. Group similar consecutive segments (within gap threshold)
    3. Mark all but the LAST segment in each group for removal

    The last segment is kept because in tutorial recordings, speakers
    typically repeat a phrase immediately after making a mistake,
    with the final attempt being the correct one.

    Args:
        segments: List of transcribed Segment objects.
        similarity_threshold: Minimum similarity (0-1) to consider as repetition.
        max_gap_seconds: Maximum time gap between segments to consider them consecutive.

    Returns:
        List of TimeRange objects representing audio to cut.
    """
    if len(segments) < 2:
        return []

    cuts = []
    i = 0

    while i < len(segments) - 1:
        current = segments[i]
        group = [current]

        # Look ahead for similar segments
        j = i + 1
        while j < len(segments):
            next_seg = segments[j]

            # Check if segments are close enough in time
            time_gap = next_seg.start - group[-1].end
            if time_gap > max_gap_seconds:
                break

            # Check similarity with all segments in the current group
            is_similar = any(
                calculate_similarity(seg.text, next_seg.text) >= similarity_threshold
                for seg in group
            )

            if is_similar:
                group.append(next_seg)
                j += 1
            else:
                break

        # If we found a group of repetitions, mark all but the last for cutting
        if len(group) > 1:
            for seg in group[:-1]:  # Keep the last one
                cuts.append(TimeRange(
                    start=seg.start,
                    end=seg.end,
                    reason=f"Repetition of: '{seg.text[:50]}...'" if len(seg.text) > 50 else f"Repetition of: '{seg.text}'"
                ))
            i = j  # Skip past the group
        else:
            i += 1

    return cuts


def merge_overlapping_cuts(cuts: list[TimeRange], padding: float = 0.1) -> list[TimeRange]:
    """
    Merge overlapping or adjacent cut ranges.

    Args:
        cuts: List of TimeRange objects.
        padding: Extra time (seconds) to add around cuts for smoother transitions.

    Returns:
        Merged list of TimeRange objects.
    """
    if not cuts:
        return []

    # Sort by start time
    sorted_cuts = sorted(cuts, key=lambda x: x.start)

    merged = []
    current = TimeRange(
        start=max(0, sorted_cuts[0].start - padding),
        end=sorted_cuts[0].end + padding,
        reason=sorted_cuts[0].reason
    )

    for cut in sorted_cuts[1:]:
        cut_start = max(0, cut.start - padding)
        cut_end = cut.end + padding

        # Check for overlap
        if cut_start <= current.end:
            # Merge
            current.end = max(current.end, cut_end)
            current.reason = f"{current.reason}; {cut.reason}"
        else:
            merged.append(current)
            current = TimeRange(start=cut_start, end=cut_end, reason=cut.reason)

    merged.append(current)
    return merged


def remove_repetitions(
    input_path: str,
    output_path: str,
    cuts: list[TimeRange],
) -> tuple[str, float]:
    """
    Remove specified time ranges from audio file.

    Args:
        input_path: Path to input audio file.
        output_path: Path for output audio file.
        cuts: List of TimeRange objects to remove.

    Returns:
        Tuple of (output_path, duration_removed_seconds)
    """
    audio = AudioSegment.from_wav(input_path)
    original_duration = len(audio) / 1000.0  # Convert ms to seconds

    if not cuts:
        audio.export(output_path, format="wav")
        return output_path, 0.0

    # Merge overlapping cuts
    merged_cuts = merge_overlapping_cuts(cuts)

    # Sort cuts by start time (should already be sorted after merge)
    merged_cuts.sort(key=lambda x: x.start)

    # Build the output audio by keeping segments between cuts
    result = AudioSegment.empty()
    current_pos = 0.0

    for cut in merged_cuts:
        # Keep audio before this cut
        if cut.start > current_pos:
            start_ms = int(current_pos * 1000)
            end_ms = int(cut.start * 1000)
            result += audio[start_ms:end_ms]

        # Move position past the cut
        current_pos = cut.end

    # Keep audio after the last cut
    if current_pos < original_duration:
        start_ms = int(current_pos * 1000)
        result += audio[start_ms:]

    # Export result
    result.export(output_path, format="wav")

    duration_removed = original_duration - (len(result) / 1000.0)
    return output_path, duration_removed
