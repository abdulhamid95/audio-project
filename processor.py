"""Core audio processing logic - The Brain."""

import os
import re
import string
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable

import numpy as np
from scipy.io import wavfile
import noisereduce as nr
from pydub import AudioSegment
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


def create_optimized_whisper_model(model_size: str = "base") -> WhisperModel:
    """
    Create a WhisperModel with optimal settings for available hardware.

    Args:
        model_size: Whisper model size (tiny, base, small, medium, large-v2, etc.)

    Returns:
        Configured WhisperModel instance.
    """
    device, compute_type = detect_device()
    return WhisperModel(model_size, device=device, compute_type=compute_type)


@dataclass
class Segment:
    """A transcribed segment with timing."""
    text: str
    start: float
    end: float
    index: int = 0  # Position in original segment list


@dataclass
class DeletionRange:
    """A time range marked for deletion."""
    start: float
    end: float
    reason: str


@dataclass
class RepetitionCandidate:
    """A potential repetition pair for review."""
    segment_a: Segment  # The earlier (potentially incomplete) segment
    segment_b: Segment  # The later (potentially complete) segment
    similarity: float   # Similarity ratio (0.0 - 1.0)
    normalized_a: str   # Normalized text of segment A
    normalized_b: str   # Normalized text of segment B
    recommended_delete: str  # "a" or "b" - which one to delete (shorter/incomplete)

    def get_deletion_for(self, choice: str) -> DeletionRange:
        """Get deletion range for the chosen segment."""
        if choice == "a":
            return DeletionRange(
                start=self.segment_a.start,
                end=self.segment_a.end,
                reason=f"User confirmed: '{self.segment_a.text[:30]}...'"
            )
        else:
            return DeletionRange(
                start=self.segment_b.start,
                end=self.segment_b.end,
                reason=f"User confirmed: '{self.segment_b.text[:30]}...'"
            )


@dataclass
class RepetitionAnalysis:
    """Results of repetition analysis with three tiers."""
    auto_delete: list[DeletionRange]      # Tier 1: >85% - auto delete
    needs_review: list[RepetitionCandidate]  # Tier 2: 70-85% - needs user review
    segments: list[Segment]               # All transcribed segments


class AudioProcessor:
    """
    Audio processing pipeline implementing "The Last Take Strategy".

    Pipeline:
    A. Noise Reduction
    B. Smart Repetition Removal
    C. Silence Removal
    """

    def __init__(
        self,
        whisper_model: WhisperModel,
        log_callback: Callable[[str], None] | None = None
    ):
        """
        Initialize the processor.

        Args:
            whisper_model: Pre-loaded Whisper model instance.
            log_callback: Function to call with log messages.
        """
        self.model = whisper_model
        self.log = log_callback or (lambda x: None)

    def process(
        self,
        input_path: str,
        output_path: str,
        remove_noise: bool = True,
        remove_repetitions: bool = True,
        silence_threshold_db: float = -40.0,
        language: str | None = "ar",
        progress_callback: Callable[[float], None] | None = None,
    ) -> str:
        """
        Run the complete processing pipeline (non-interactive mode).

        Pipeline order optimized for speed:
        A. Noise Reduction
        B. Silence Removal (BEFORE transcription to reduce Whisper workload)
        C. Repetition Detection & Removal

        Args:
            input_path: Path to input WAV file.
            output_path: Path for output WAV file.
            remove_noise: Whether to apply noise reduction.
            remove_repetitions: Whether to detect and remove repetitions.
            silence_threshold_db: Threshold for silence detection.
            language: Language code for transcription (default "ar" for Arabic).
            progress_callback: Function(progress: float) for updates.

        Returns:
            Path to processed audio file.
        """
        update = progress_callback or (lambda x: None)
        current_path = input_path

        # Step A: Noise Reduction (0-25%)
        if remove_noise:
            self.log("Applying noise reduction...")
            update(0.1)
            denoised_path = output_path.replace(".wav", "_denoised.wav")
            current_path = self._reduce_noise(current_path, denoised_path)
            self.log("Noise reduction complete.")
            update(0.25)

        # Step B: Silence Removal (25-40%) - BEFORE transcription for speed
        self.log("Detecting and trimming silences...")
        update(0.3)
        trimmed_path = output_path.replace(".wav", "_trimmed.wav")
        current_path = self._trim_silences(
            current_path,
            trimmed_path,
            threshold_db=silence_threshold_db
        )
        self.log("Silence trimming complete.")
        update(0.4)

        # Step C: Repetition Removal (40-100%)
        if remove_repetitions:
            lang_str = language if language else "auto-detect"
            self.log(f"Transcribing audio with Whisper AI (language: {lang_str})...")
            update(0.45)
            segments = self._transcribe(current_path, language=language)
            self.log(f"Found {len(segments)} speech segments.")
            update(0.7)

            self.log("Analyzing for repetitions and false starts...")
            deletions = self._detect_repetitions(segments)

            if deletions:
                self.log(f"Found {len(deletions)} sections to remove:")
                for d in deletions:
                    self.log(f"  - {d.reason}")

                current_path = self._remove_segments(current_path, output_path, deletions)
                self.log("Repetitions removed.")
            else:
                self.log("No repetitions detected.")
                # Copy to final output path if no deletions
                audio = AudioSegment.from_wav(current_path)
                audio.export(output_path, format="wav")
            update(1.0)
        else:
            # No repetition removal, copy to final output
            audio = AudioSegment.from_wav(current_path)
            audio.export(output_path, format="wav")
            update(1.0)

        return output_path

    def process_phase1_prepare(
        self,
        input_path: str,
        temp_dir: str,
        remove_noise: bool = True,
        silence_threshold_db: float = -40.0,
        language: str | None = "ar",
        progress_callback: Callable[[float], None] | None = None,
    ) -> tuple[str, RepetitionAnalysis | None]:
        """
        Phase 1: Prepare audio and analyze repetitions (interactive mode).

        Pipeline order optimized for speed:
        A. Noise Reduction
        B. Silence Removal (BEFORE transcription to reduce Whisper workload)
        C. Transcription & Analysis

        Returns:
            Tuple of (prepared_audio_path, repetition_analysis or None)
        """
        update = progress_callback or (lambda x: None)
        current_path = input_path

        # Step A: Noise Reduction (0-20%)
        if remove_noise:
            self.log("Applying noise reduction...")
            update(0.1)
            denoised_path = os.path.join(temp_dir, "denoised.wav")
            current_path = self._reduce_noise(current_path, denoised_path)
            self.log("Noise reduction complete.")
            update(0.2)

        # Step B: Silence Removal (20-35%) - BEFORE transcription for speed
        self.log("Detecting and trimming silences...")
        update(0.25)
        trimmed_path = os.path.join(temp_dir, "trimmed.wav")
        current_path = self._trim_silences(
            current_path,
            trimmed_path,
            threshold_db=silence_threshold_db
        )
        self.log("Silence trimming complete.")
        update(0.35)

        # Step C: Transcription and Analysis (35-60%)
        lang_str = language if language else "auto-detect"
        self.log(f"Transcribing audio with Whisper AI (language: {lang_str})...")
        update(0.4)
        segments = self._transcribe(current_path, language=language)
        self.log(f"Found {len(segments)} speech segments.")
        update(0.55)

        self.log("Analyzing for repetitions (three-tier system)...")
        analysis = self.analyze_repetitions(segments)

        self.log(f"Analysis complete:")
        self.log(f"  - Tier 1 (auto-delete): {len(analysis.auto_delete)} segments")
        self.log(f"  - Tier 2 (needs review): {len(analysis.needs_review)} pairs")
        update(0.6)

        return current_path, analysis

    def process_phase2_finalize(
        self,
        input_path: str,
        output_path: str,
        deletions: list[DeletionRange],
        progress_callback: Callable[[float], None] | None = None,
    ) -> str:
        """
        Phase 2: Apply deletions and finalize audio (interactive mode).

        Note: Silence removal is now done in phase 1 (before transcription) for speed.

        Args:
            input_path: Path to prepared audio (from phase 1, already silence-trimmed).
            output_path: Path for final output.
            deletions: Combined list of all confirmed deletions.
            progress_callback: Function(progress: float) for updates.

        Returns:
            Path to processed audio file.
        """
        update = progress_callback or (lambda x: None)
        current_path = input_path

        # Apply deletions (60-100%)
        if deletions:
            self.log(f"Removing {len(deletions)} confirmed segments...")
            update(0.7)
            current_path = self._remove_segments(current_path, output_path, deletions)
            self.log("Segments removed.")
            update(1.0)
        else:
            self.log("No segments to remove.")
            # Copy to final output
            audio = AudioSegment.from_wav(current_path)
            audio.export(output_path, format="wav")
            update(1.0)

        return output_path

    def _reduce_noise(
        self,
        input_path: str,
        output_path: str,
        noise_sample_seconds: float = 0.5
    ) -> str:
        """Apply noise reduction using noisereduce."""
        sample_rate, audio_data = wavfile.read(input_path)

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

        # Use first N seconds as noise profile
        noise_samples = int(noise_sample_seconds * sample_rate)
        noise_clip = audio_float[:noise_samples]

        # Apply noise reduction
        reduced = nr.reduce_noise(
            y=audio_float,
            sr=sample_rate,
            y_noise=noise_clip,
            prop_decrease=0.8,
            stationary=True,
        )

        # Normalize
        max_val = np.max(np.abs(reduced))
        if max_val > 0:
            reduced = reduced / max_val * 0.95

        # Save
        reduced_int = (reduced * 32767).astype(np.int16)
        wavfile.write(output_path, sample_rate, reduced_int)

        return output_path

    def _transcribe(self, audio_path: str, language: str | None = "ar") -> list[Segment]:
        """Transcribe audio using Whisper with optimized settings.

        Args:
            audio_path: Path to audio file.
            language: Language code (default "ar" for Arabic), None for auto-detect.
        """
        transcribe_kwargs = {
            "word_timestamps": True,
            "vad_filter": True,
            "beam_size": 1,  # Faster decoding with greedy search
        }
        if language:
            transcribe_kwargs["language"] = language
            # Arabic-specific prompt for better recognition
            if language == "ar":
                transcribe_kwargs["initial_prompt"] = "اللغة العربية الفصحى"

        segments_gen, info = self.model.transcribe(audio_path, **transcribe_kwargs)
        self.log(f"Detected language: {info.language} (probability: {info.language_probability:.0%})")

        segments = []
        for i, seg in enumerate(segments_gen):
            text = seg.text.strip()
            if text:
                start = seg.words[0].start if seg.words else seg.start
                end = seg.words[-1].end if seg.words else seg.end
                segments.append(Segment(text=text, start=start, end=end))
                # Streaming log: update UI as each segment is processed
                self.log(f"  Segment {i+1}: {text[:40]}{'...' if len(text) > 40 else ''}")

        return segments

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison, optimized for Arabic language.

        Arabic-specific normalization:
        - Remove Arabic diacritics (Tashkeel/Harakat)
        - Normalize Alef variants (أ, إ, آ → ا)
        - Normalize Teh Marbuta (ة → ه)
        - Remove common prefixes (و, ف) from word beginnings

        Also handles English:
        - Remove punctuation
        - Lowercase
        """
        # Remove Arabic diacritics (Tashkeel/Harakat)
        # Range: \u064B-\u065F (Fathatan to Waslah)
        # Also: \u0670 (Superscript Alef), \u06D6-\u06DC (Quranic marks)
        arabic_diacritics = re.compile(r'[\u064B-\u065F\u0670\u06D6-\u06DC]')
        text = arabic_diacritics.sub('', text)

        # Normalize Alef variants (أ إ آ ٱ → ا)
        text = re.sub(r'[أإآٱ]', 'ا', text)

        # Normalize Teh Marbuta (ة → ه)
        text = text.replace('ة', 'ه')

        # Normalize Alef Maksura (ى → ي)
        text = text.replace('ى', 'ي')

        # Remove all punctuation (English and Arabic)
        # Arabic punctuation: ، ؛ ؟ etc.
        arabic_punctuation = '،؛؟«»ـ'
        all_punctuation = string.punctuation + arabic_punctuation
        text = text.translate(str.maketrans("", "", all_punctuation))

        # Lowercase (for any English mixed in)
        text = text.lower().strip()

        # Collapse multiple spaces into one
        text = re.sub(r'\s+', ' ', text)

        # Strip common Arabic prefixes from words (و, ف, ال, ب, ك, ل)
        # These are often attached and can cause matching issues
        words = text.split()
        normalized_words = []
        for word in words:
            # Strip single-letter prefixes و (and), ف (so/then)
            if len(word) > 2 and word[0] in 'وف':
                word = word[1:]
            # Strip بال، كال، فال، وال (common prefix combinations with ال)
            if len(word) > 3 and word[:3] in ['بال', 'كال', 'فال', 'وال']:
                word = word[3:]
            # Strip ال (the) - definite article
            elif len(word) > 2 and word[:2] == 'ال':
                word = word[2:]
            normalized_words.append(word)

        text = ' '.join(normalized_words)

        return text

    def analyze_repetitions(
        self,
        segments: list[Segment],
        high_threshold: float = 0.85,
        low_threshold: float = 0.70,
        lookahead_window: int = 3,
        debug: bool = True
    ) -> RepetitionAnalysis:
        """
        Analyze repetitions using three-tier confidence system.

        Tiers:
        - Tier 1 (>85%): High confidence - auto-delete the shorter/earlier segment
        - Tier 2 (70-85%): Uncertain - needs user review
        - Tier 3 (<70%): Different - keep both

        Args:
            segments: List of transcribed segments.
            high_threshold: Threshold for auto-delete (default 0.85).
            low_threshold: Threshold for review (default 0.70).
            lookahead_window: How many segments ahead to compare (default 3).
            debug: Print comparison details for debugging.

        Returns:
            RepetitionAnalysis with auto_delete, needs_review, and segments.
        """
        # Add index to segments
        for i, seg in enumerate(segments):
            seg.index = i

        if len(segments) < 2:
            return RepetitionAnalysis(auto_delete=[], needs_review=[], segments=segments)

        auto_delete = []
        needs_review = []
        processed_indices = set()

        if debug:
            print(f"\n{'='*60}")
            print("REPETITION DETECTION - THREE TIER ANALYSIS")
            print(f"{'='*60}")
            print(f"Total segments: {len(segments)}")
            print(f"Tier 1 (Auto-delete): >{high_threshold:.0%}")
            print(f"Tier 2 (Review): {low_threshold:.0%}-{high_threshold:.0%}")
            print(f"Tier 3 (Keep): <{low_threshold:.0%}")
            print(f"Lookahead window: {lookahead_window}")
            print(f"{'='*60}\n")

        for i in range(len(segments)):
            if i in processed_indices:
                continue

            curr = segments[i]
            curr_text = self._normalize_text(curr.text)
            curr_words = curr_text.split()

            for offset in range(1, lookahead_window + 1):
                j = i + offset
                if j >= len(segments):
                    break
                if j in processed_indices:
                    continue

                next_seg = segments[j]
                next_text = self._normalize_text(next_seg.text)
                next_words = next_text.split()

                # Calculate similarity
                ratio = SequenceMatcher(None, curr_text, next_text).ratio()

                # Check for false start / prefix match (boost ratio if detected)
                is_false_start = False
                if len(curr_text) > 3 and next_text.startswith(curr_text):
                    is_false_start = True
                    ratio = max(ratio, high_threshold + 0.01)  # Treat as high confidence
                elif len(curr_words) <= 2 and len(curr_words) >= 1 and len(next_words) > len(curr_words):
                    if curr_words == next_words[:len(curr_words)]:
                        is_false_start = True
                        ratio = max(ratio, high_threshold + 0.01)
                elif len(curr_text) < len(next_text) * 0.6 and len(curr_words) <= 4:
                    if curr_words and next_words and curr_words == next_words[:len(curr_words)]:
                        is_false_start = True
                        ratio = max(ratio, high_threshold + 0.01)

                if debug:
                    print(f"Comparing Seg[{i}] vs Seg[{j}]:")
                    print(f"  A: '{curr.text[:50]}{'...' if len(curr.text) > 50 else ''}'")
                    print(f"  B: '{next_seg.text[:50]}{'...' if len(next_seg.text) > 50 else ''}'")
                    print(f"  Normalized A: '{curr_text}'")
                    print(f"  Normalized B: '{next_text}'")
                    print(f"  -> Ratio: {ratio:.2%}" + (" (false start)" if is_false_start else ""))

                # Determine which segment to delete (shorter/incomplete one)
                # Usually the first one, but if second is shorter, delete that
                if len(curr_text) <= len(next_text):
                    recommended_delete = "a"  # Delete earlier/shorter
                else:
                    recommended_delete = "b"  # Delete later if it's shorter

                # Tier 1: High confidence (>85%) - auto delete
                if ratio > high_threshold:
                    if debug:
                        print(f"  -> TIER 1: Auto-delete (segment {recommended_delete.upper()})")
                    processed_indices.add(i)

                    # Delete the shorter/incomplete segment
                    if recommended_delete == "a":
                        auto_delete.append(DeletionRange(
                            start=curr.start,
                            end=curr.end,
                            reason=f"Auto: {ratio:.0%} match - '{curr.text[:35]}...'"
                        ))
                    else:
                        auto_delete.append(DeletionRange(
                            start=next_seg.start,
                            end=next_seg.end,
                            reason=f"Auto: {ratio:.0%} match - '{next_seg.text[:35]}...'"
                        ))
                        processed_indices.add(j)
                    break

                # Tier 2: Uncertain (70-85%) - needs review
                elif ratio >= low_threshold:
                    if debug:
                        print(f"  -> TIER 2: Needs review")
                    processed_indices.add(i)
                    needs_review.append(RepetitionCandidate(
                        segment_a=curr,
                        segment_b=next_seg,
                        similarity=ratio,
                        normalized_a=curr_text,
                        normalized_b=next_text,
                        recommended_delete=recommended_delete
                    ))
                    break

                # Tier 3: Different (<70%) - keep both
                else:
                    if debug:
                        print(f"  -> TIER 3: Keep both")

                if debug:
                    print()

        if debug:
            print(f"{'='*60}")
            print(f"ANALYSIS COMPLETE")
            print(f"  Auto-delete (Tier 1): {len(auto_delete)} segments")
            print(f"  Needs review (Tier 2): {len(needs_review)} pairs")
            print(f"{'='*60}\n")

        return RepetitionAnalysis(
            auto_delete=auto_delete,
            needs_review=needs_review,
            segments=segments
        )

    def _detect_repetitions(
        self,
        segments: list[Segment],
        similarity_threshold: float = 0.80,
        lookahead_window: int = 3,
        debug: bool = True
    ) -> list[DeletionRange]:
        """
        Legacy method - detects repetitions and returns deletions directly.
        Use analyze_repetitions() for interactive workflow.
        """
        analysis = self.analyze_repetitions(
            segments,
            high_threshold=similarity_threshold,
            low_threshold=0.70,
            lookahead_window=lookahead_window,
            debug=debug
        )
        # In legacy mode, auto-delete everything above the threshold
        # and also auto-confirm all review candidates
        deletions = list(analysis.auto_delete)
        for candidate in analysis.needs_review:
            deletions.append(candidate.get_deletion_for(candidate.recommended_delete))
        return deletions

    def _remove_segments(
        self,
        input_path: str,
        output_path: str,
        deletions: list[DeletionRange]
    ) -> str:
        """Remove marked segments from audio."""
        audio = AudioSegment.from_wav(input_path)

        # Sort by start time
        sorted_dels = sorted(deletions, key=lambda x: x.start)

        # Build output by keeping non-deleted parts
        result = AudioSegment.empty()
        current_pos = 0.0

        for d in sorted_dels:
            # Keep audio before this deletion
            if d.start > current_pos:
                start_ms = int(current_pos * 1000)
                end_ms = int(d.start * 1000)
                result += audio[start_ms:end_ms]
            current_pos = max(current_pos, d.end)

        # Keep audio after last deletion
        if current_pos * 1000 < len(audio):
            result += audio[int(current_pos * 1000):]

        result.export(output_path, format="wav")
        return output_path

    def _trim_silences(
        self,
        input_path: str,
        output_path: str,
        threshold_db: float = -40.0,
        min_silence_ms: int = 800,
        target_silence_ms: int = 100
    ) -> str:
        """
        Detect silences > min_silence_ms and truncate to target_silence_ms.

        This keeps pacing natural instead of completely removing silences.
        """
        sample_rate, audio_data = wavfile.read(input_path)

        # Convert to float
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_float = audio_data.astype(np.float32) / 2147483648.0
        else:
            audio_float = audio_data.astype(np.float32)

        if len(audio_float.shape) > 1:
            audio_float = np.mean(audio_float, axis=1)

        # Detect silence regions
        threshold_linear = 10 ** (threshold_db / 20)
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
                    duration_ms = (current_time - silence_start) * 1000
                    if duration_ms >= min_silence_ms:
                        silence_regions.append((silence_start, current_time))
                    in_silence = False

        # Handle trailing silence
        if in_silence:
            end_time = len(audio_float) / sample_rate
            duration_ms = (end_time - silence_start) * 1000
            if duration_ms >= min_silence_ms:
                silence_regions.append((silence_start, end_time))

        if not silence_regions:
            # No long silences, just copy
            audio = AudioSegment.from_wav(input_path)
            audio.export(output_path, format="wav")
            return output_path

        # Trim silences using pydub
        audio = AudioSegment.from_wav(input_path)
        target_sec = target_silence_ms / 1000.0
        result = audio

        # Process from end to start to avoid position shifts
        for start, end in reversed(silence_regions):
            silence_duration = end - start
            trim_amount = silence_duration - target_sec

            if trim_amount > 0:
                # Keep half of target at start, half at end
                keep_start = target_sec / 2
                keep_end = target_sec / 2

                trim_start_ms = int((start + keep_start) * 1000)
                trim_end_ms = int((end - keep_end) * 1000)

                result = result[:trim_start_ms] + result[trim_end_ms:]

        result.export(output_path, format="wav")
        return output_path
